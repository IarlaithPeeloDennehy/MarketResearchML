"""
channels.py — single source of truth for encoder input channels.

CRITICAL INVARIANT: the offline trainer (scripts/pretrain_encoder.py) and
production inference (embedding_features.point_in_time_embedding) MUST build the
encoder's input channels through THIS module. If the two diverged, the frozen
encoder would receive inputs from a different distribution than it trained on and
silently produce garbage embeddings. Everything channel-related lives here so
there is exactly one implementation to keep in sync — guarded by a train/inference
parity test.

A "channel" is one daily time series describing a stock. The encoder takes a
(T, C) window of standardized channels. Channels:

  Base (price only — v1):
    logret                  daily log return
    vol                     trailing 20d realized vol of logret
  Group A (volume + cross-asset/sector context — derivable from cached data):
    rel_volume              log(volume / trailing-median volume)
    log_dollar_vol          log(close * volume)
    excess_ret_vs_market    logret − SPY logret
    rel_strength_vs_sector  logret − sector-SPDR logret
  Group B (Finnhub series we currently discard — point-in-time gated):
    earn_surprise_pulse     decaying pulse of clipped earnings surprise%, only
                            active AFTER each announcement date
    analyst_net_buy_ts      (buy−sell)/total from recommendation_trends, ffilled
                            from each trend's publish month

All look-ahead safety is enforced here: every channel value at day t depends only
on information available at day t. Missing source data for a channel (e.g. a UK/IE
ticker with no Finnhub series, or an unmapped sector) yields a neutral all-zero
channel — the encoder still runs (graceful degradation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Channel groups / specs ──────────────────────────────────────────────────
BASE_CHANNELS = ["logret", "vol"]
GROUP_A = ["rel_volume", "log_dollar_vol", "excess_ret_vs_market", "rel_strength_vs_sector"]
GROUP_B = ["earn_surprise_pulse", "analyst_net_buy_ts"]

# Default spec for the first richer retrain (phase 1 = base + Group A).
# Group B is added to a checkpoint's spec in phase 2. The ACTIVE spec at
# inference always comes from the loaded checkpoint's meta["channels"].
DEFAULT_SPEC = BASE_CHANNELS + GROUP_A
ALL_CHANNELS = BASE_CHANNELS + GROUP_A + GROUP_B

# ── Reference universe for market / sector context ──────────────────────────
MARKET_ETF = "SPY"
# finnhubIndustry (company_profile2) → SPDR sector ETF. Unmapped → None (neutral).
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Communications": "XLC",
    "Media": "XLC",
    "Telecommunication": "XLC",
    "Financial Services": "XLF",
    "Banking": "XLF",
    "Insurance": "XLF",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Pharmaceuticals": "XLV",
    "Biotechnology": "XLV",
    "Energy": "XLE",
    "Oil & Gas": "XLE",
    "Industrials": "XLI",
    "Machinery": "XLI",
    "Aerospace & Defense": "XLI",
    "Consumer Discretionary": "XLY",
    "Retail": "XLY",
    "Automobiles": "XLY",
    "Consumer Staples": "XLP",
    "Food Products": "XLP",
    "Beverages": "XLP",
    "Consumer products": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Chemicals": "XLB",
}
REFERENCE_TICKERS = [MARKET_ETF] + sorted(set(SECTOR_ETF_MAP.values()))

# Earnings-surprise pulse parameters
_EARN_LAG_DAYS = 45        # period_end → assumed announcement (when no explicit date)
_EARN_DECAY_DAYS = 21.0    # e^-(days_since/decay): ~1 month half-ish life
_EARN_WINDOW_DAYS = 63     # ignore events older than this


def sector_to_etf(sector: str | None) -> str | None:
    """Map a finnhubIndustry string to a SPDR sector ETF, else None (neutral)."""
    if not sector:
        return None
    if sector in SECTOR_ETF_MAP:
        return SECTOR_ETF_MAP[sector]
    # loose contains-match for taxonomy drift
    low = sector.lower()
    for key, etf in SECTOR_ETF_MAP.items():
        if key.lower() in low or low in key.lower():
            return etf
    return None


# ── small trailing helpers (vectorized — the offline trainer builds millions
#    of windows, so per-element Python loops here would dominate runtime) ──────
def _trailing_std(arr: np.ndarray, win: int) -> np.ndarray:
    # population std (ddof=0) over an expanding-then-rolling window; <2 obs → 0
    s = pd.Series(arr, dtype="float64").rolling(win, min_periods=2).std(ddof=0)
    return s.fillna(0.0).to_numpy()


def _trailing_median(arr: np.ndarray, win: int) -> np.ndarray:
    s = pd.Series(arr, dtype="float64").rolling(win, min_periods=1).median()
    return s.fillna(0.0).to_numpy()


def _logret(close: np.ndarray) -> np.ndarray:
    close = np.asarray(close, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(np.where(close > 0, close, np.nan)))
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def _align(ref: pd.Series | None, dates: pd.DatetimeIndex) -> np.ndarray | None:
    """Reindex a reference Close series onto `dates` (forward-fill). None-safe."""
    if ref is None or len(ref) == 0:
        return None
    try:
        s = ref.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        return s.reindex(dates, method="ffill").to_numpy(dtype=np.float64)
    except Exception:
        return None


# ── earnings / analyst per-date series (point-in-time) ──────────────────────
def _earnings_pulse(dates: pd.DatetimeIndex, earnings_hist: list | None) -> np.ndarray:
    out = np.zeros(len(dates), dtype=np.float64)
    if not earnings_hist:
        return out
    events = []  # (announce_ts, clipped_surprise)
    for e in earnings_hist:
        sp = e.get("surprisePercent")
        per = e.get("period") or e.get("date")
        if sp is None or per is None:
            continue
        try:
            base = pd.Timestamp(per)
        except Exception:
            continue
        # explicit announcement date if present, else period_end + lag
        adate = e.get("announce_date")
        announce = pd.Timestamp(adate) if adate else base + pd.Timedelta(days=_EARN_LAG_DAYS)
        events.append((announce, float(np.clip(sp / 50.0, -1.0, 1.0))))
    if not events:
        return out
    events.sort()
    ev_dates = np.array([t.value for t, _ in events])
    ev_vals = np.array([v for _, v in events])
    for i, d in enumerate(dates):
        # most recent announcement strictly on/before this date
        j = np.searchsorted(ev_dates, d.value, side="right") - 1
        if j < 0:
            continue
        days_since = (d.value - ev_dates[j]) / 8.64e13  # ns → days
        if 0 <= days_since <= _EARN_WINDOW_DAYS:
            out[i] = ev_vals[j] * float(np.exp(-days_since / _EARN_DECAY_DAYS))
    return out


def _analyst_net_buy(dates: pd.DatetimeIndex, analyst_hist: list | None) -> np.ndarray:
    out = np.zeros(len(dates), dtype=np.float64)
    if not analyst_hist:
        return out
    rows = []  # (publish_ts, net_ratio)
    for r in analyst_hist:
        per = r.get("period")
        if not per:
            continue
        try:
            ts = pd.Timestamp(per)
        except Exception:
            continue
        buy = (r.get("buy") or 0) + (r.get("strongBuy") or 0)
        sell = (r.get("sell") or 0) + (r.get("strongSell") or 0)
        hold = r.get("hold") or 0
        total = buy + sell + hold
        rows.append((ts, (buy - sell) / total if total else 0.0))
    if not rows:
        return out
    rows.sort()
    r_dates = np.array([t.value for t, _ in rows])
    r_vals = np.array([v for _, v in rows])
    for i, d in enumerate(dates):
        j = np.searchsorted(r_dates, d.value, side="right") - 1
        if j >= 0:
            out[i] = r_vals[j]
    return out


# ── main builder ────────────────────────────────────────────────────────────
def build_window_channels(
    close: pd.Series | np.ndarray,
    spec: list[str],
    *,
    end_idx: int | None = None,
    window: int = 252,
    volume: pd.Series | np.ndarray | None = None,
    market_close: pd.Series | None = None,
    sector_close: pd.Series | None = None,
    earnings_hist: list | None = None,
    analyst_hist: list | None = None,
) -> np.ndarray:
    """Build a (T, C) raw (un-standardized) channel matrix for the window ending
    at ``end_idx`` (inclusive). T = min(window, available_returns); C = len(spec).

    Point-in-time: only ``close[:end_idx+1]`` and history/reference data dated
    on/before each window day are used. Missing sources → neutral (zero) channel.
    """
    is_series = hasattr(close, "iloc")
    if end_idx is not None:
        close_w = close.iloc[: end_idx + 1] if is_series else np.asarray(close)[: end_idx + 1]
    else:
        close_w = close

    n_keep = window + 1  # +1 because diff() loses one
    if is_series:
        close_w = close_w.iloc[-n_keep:]
        dates = pd.DatetimeIndex(close_w.index)
        close_arr = close_w.to_numpy(dtype=np.float64)
    else:
        close_arr = np.asarray(close_w, dtype=np.float64)[-n_keep:]
        dates = None

    if close_arr.size < 2:
        return np.zeros((0, len(spec)), dtype=np.float32)

    # date index in returns space (drop the first bar)
    ret_dates = dates[1:] if dates is not None else None
    logret = _logret(close_arr)                       # length L-1
    T = logret.size

    # volume aligned to close window
    vol_arr = None
    if volume is not None:
        if hasattr(volume, "reindex") and dates is not None:
            vser = volume.sort_index()
            vser = vser[~vser.index.duplicated(keep="last")]
            vol_arr = vser.reindex(dates, method="ffill").to_numpy(dtype=np.float64)
        else:
            v = np.asarray(volume, dtype=np.float64)
            vol_arr = v[-n_keep:] if v.size >= n_keep else None
        if vol_arr is not None:
            vol_arr = np.nan_to_num(vol_arr[1:], nan=0.0)   # align to returns space

    # Align reference Close to the FULL window dates (length L) so their logret
    # (length L-1 = T) lines up with the stock's logret.
    mkt = _align(market_close, dates) if dates is not None else None
    sec = _align(sector_close, dates) if dates is not None else None

    cols: list[np.ndarray] = []
    for name in spec:
        if name == "logret":
            c = logret
        elif name == "vol":
            c = _trailing_std(logret, 20)
        elif name == "rel_volume":
            if vol_arr is None:
                c = np.zeros(T)
            else:
                med = _trailing_median(vol_arr, 20)
                with np.errstate(divide="ignore", invalid="ignore"):
                    c = np.log(np.where((vol_arr > 0) & (med > 0), vol_arr / med, 1.0))
                c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
        elif name == "log_dollar_vol":
            if vol_arr is None:
                c = np.zeros(T)
            else:
                dv = close_arr[1:] * vol_arr
                c = np.log(np.where(dv > 0, dv, 1.0))
        elif name == "excess_ret_vs_market":
            c = logret - _logret(mkt) if mkt is not None else np.zeros(T)
        elif name == "rel_strength_vs_sector":
            c = logret - _logret(sec) if sec is not None else np.zeros(T)
        elif name == "earn_surprise_pulse":
            c = _earnings_pulse(ret_dates, earnings_hist) if ret_dates is not None else np.zeros(T)
        elif name == "analyst_net_buy_ts":
            c = _analyst_net_buy(ret_dates, analyst_hist) if ret_dates is not None else np.zeros(T)
        else:
            c = np.zeros(T)
        # guarantee length T
        c = np.asarray(c, dtype=np.float64)
        if c.size != T:
            c = np.resize(np.nan_to_num(c), T)
        cols.append(np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0))

    mat = np.stack(cols, axis=1).astype(np.float32)   # (T, C)
    return mat


def standardize_window(mat: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score. Shared by trainer and inference so the
    encoder always sees identically-scaled inputs. Constant/NaN cols → 0."""
    if mat.size == 0:
        return mat
    mu = np.nanmean(mat, axis=0, keepdims=True)
    sd = np.nanstd(mat, axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    out = (mat - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
