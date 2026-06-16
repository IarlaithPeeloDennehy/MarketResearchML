"""
embedding_features.py
─────────────────────
Bridge between the frozen price encoder (encoder.py) and the existing feature
pipeline (model.py / backtest.py / feature_engineering.py).

Responsibilities:
  * Hold the process-wide loaded encoder (set once at startup by main.py).
  * Turn a point-in-time price window into ``emb_0 .. emb_{D-1}`` columns.
  * Standardize embedding columns cross-sectionally (per period snapshot), to
    match the rank-normalization philosophy of the price features.
  * Cache embeddings in-memory (keyed by ticker, last_bar_date, encoder_version)
    so the many windows in a backtest don't re-run the encoder repeatedly.
    Nothing is persisted to disk — embeddings are ms/ticker to recompute.

When no encoder is loaded (torch missing, USE_ENCODER off, or no checkpoint),
EVERY function here degrades to a no-op: ``embedding_columns()`` returns ``[]``
and ``point_in_time_embedding()`` returns ``{}``. Callers then behave exactly
as they did before pre-training existed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import channels
from .encoder import LoadedEncoder

logger = logging.getLogger(__name__)

# ── Process-wide encoder reference ──────────────────────────────────────────
_ENCODER: Optional[LoadedEncoder] = None

# In-memory embedding cache: {(ticker, date_key, version): np.ndarray}.
# Cleared on every set_encoder() call. Deliberately NOT persisted to disk —
# embeddings are ms/ticker to recompute, and a per-file disk cache would waste
# blocks + leave stale encoder-version dirs on the small Render disk.
_MEM_CACHE: dict[tuple, np.ndarray] = {}
_MEM_CACHE_MAX = 50_000

# Cache directories (mirror data_fetcher; same CACHE_DIR env).
_cache_root = os.environ.get("CACHE_DIR")
_CACHE = Path(_cache_root) if _cache_root else Path(__file__).parent.parent / "cache"
_PRICE_DIR = _CACHE / "prices"
_INFO_DIR = _CACHE / "info"

# Per-process input caches (avoid re-reading parquet/json for every window).
# Cleared on set_encoder(). Reference ETF series are loaded once and reused.
_REF_CACHE: dict[str, Optional[pd.Series]] = {}
_VOL_CACHE: dict[str, Optional[pd.Series]] = {}
_SECTOR_CACHE: dict[str, Optional[str]] = {}
_HIST_CACHE: dict[str, tuple] = {}


def _reset_input_caches() -> None:
    _MEM_CACHE.clear()
    _REF_CACHE.clear()
    _VOL_CACHE.clear()
    _SECTOR_CACHE.clear()
    _HIST_CACHE.clear()


def set_encoder(encoder: Optional[LoadedEncoder]) -> None:
    """Install (or clear) the process-wide encoder. Called once at startup."""
    global _ENCODER
    _ENCODER = encoder
    _reset_input_caches()
    if encoder is not None:
        logger.info(
            f"Embedding features ENABLED — encoder v{encoder.version}, "
            f"{encoder.embed_dim} dims"
        )
    else:
        logger.info("Embedding features DISABLED — price-ranks-only mode.")


def get_encoder() -> Optional[LoadedEncoder]:
    return _ENCODER


def encoder_enabled() -> bool:
    return _ENCODER is not None


def embedding_columns() -> list[str]:
    """Column names the encoder contributes, or [] when disabled."""
    if _ENCODER is None:
        return []
    return [f"emb_{i}" for i in range(_ENCODER.embed_dim)]


def _date_key(prices: pd.Series, end_idx: Optional[int]) -> str:
    """Stable key for the window end (prefers the index date, falls back to idx)."""
    try:
        idx = end_idx if end_idx is not None else len(prices) - 1
        label = prices.index[min(idx, len(prices) - 1)]
        return str(label)
    except Exception:
        return str(end_idx)


# ── input loaders (by ticker, cached per process, invalidated on file mtime) ─
# The per-ticker caches key on the source file's mtime so that when the 7-day
# price/series cache refreshes under a long-running server, the loaders pick up
# the new bars instead of serving a stale series.
def _stem(ticker: str) -> str:
    from .feature_engineering import safe_ticker_filename
    return safe_ticker_filename(ticker)


def _cached(cache: dict, key: str, path: Path, loader):
    """Return loader() result, cached and invalidated by path mtime."""
    mtime = path.stat().st_mtime if path.exists() else None
    hit = cache.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    value = loader() if mtime is not None else None
    cache[key] = (mtime, value)
    return value


def _load_volume(ticker: str) -> Optional[pd.Series]:
    p = _PRICE_DIR / f"{_stem(ticker)}.parquet"

    def _read():
        try:
            df = pd.read_parquet(p)
            return df["Volume"].sort_index() if "Volume" in df.columns else None
        except Exception:
            return None

    return _cached(_VOL_CACHE, ticker, p, _read)


def _load_reference(etf: Optional[str]) -> Optional[pd.Series]:
    if not etf:
        return None
    p = _PRICE_DIR / f"{_stem(etf)}.parquet"

    def _read():
        try:
            df = pd.read_parquet(p)
            return df["Close"].sort_index() if "Close" in df.columns else None
        except Exception:
            return None

    return _cached(_REF_CACHE, etf, p, _read)


def _load_sector(ticker: str) -> Optional[str]:
    p = _INFO_DIR / f"{_stem(ticker)}.json"

    def _read():
        try:
            import json
            with open(p) as f:
                return (json.load(f) or {}).get("sector")
        except Exception:
            return None

    return _cached(_SECTOR_CACHE, ticker, p, _read)


def _load_history(ticker: str) -> tuple:
    from . import series_cache
    p = series_cache._path(ticker)

    def _read():
        try:
            return series_cache.load(ticker)
        except Exception:
            return (None, None)

    return _cached(_HIST_CACHE, ticker, p, _read) or (None, None)


def point_in_time_embedding(
    prices: pd.Series | np.ndarray,
    end_idx: Optional[int] = None,
    ticker: Optional[str] = None,
) -> dict[str, float]:
    """Embed the trailing price window ending at ``end_idx`` (inclusive).

    Returns ``{emb_0: .., emb_{D-1}: ..}`` or ``{}`` when the encoder is off.

    All channels are built through the shared ``channels.build_window_channels``
    (the single source of truth shared with the offline trainer). Channel inputs
    (volume, sector, market/sector reference ETFs, earnings/analyst history) are
    loaded by ticker from the cache; missing sources yield neutral channels.

    Point-in-time guarantee: only ``prices[:end_idx+1]`` plus reference/history
    data dated on/before each window day are used.

    Cached in-memory only (keyed by ticker + window-end date + encoder version);
    nothing is written to disk.
    """
    enc = _ENCODER
    if enc is None:
        return {}

    is_series = hasattr(prices, "iloc")

    # ── in-memory cache lookup ──────────────────────────────────────────────
    cache_key = None
    if ticker is not None and is_series:
        cache_key = (ticker, _date_key(prices, end_idx), enc.version)
        hit = _MEM_CACHE.get(cache_key)
        if hit is not None:
            return {f"emb_{i}": float(v) for i, v in enumerate(hit)}

    spec = enc.channels

    # Load only the inputs the active channel spec actually needs.
    volume = market = sector_ref = earnings = analyst = None
    if ticker is not None and is_series:
        if any(c in spec for c in ("rel_volume", "log_dollar_vol")):
            volume = _load_volume(ticker)
        if "excess_ret_vs_market" in spec:
            market = _load_reference(channels.MARKET_ETF)
        if "rel_strength_vs_sector" in spec:
            sector_ref = _load_reference(channels.sector_to_etf(_load_sector(ticker)))
        if any(c in spec for c in channels.GROUP_B):
            earnings, analyst = _load_history(ticker)

    mat = channels.build_window_channels(
        prices, spec, end_idx=end_idx, window=enc.window,
        volume=volume, market_close=market, sector_close=sector_ref,
        earnings_hist=earnings, analyst_hist=analyst,
    )
    emb = enc.embed_channels(mat)

    if cache_key is not None:
        _store_mem(cache_key, emb)

    return {f"emb_{i}": float(v) for i, v in enumerate(emb)}


def _store_mem(key: tuple, arr: np.ndarray) -> None:
    if len(_MEM_CACHE) >= _MEM_CACHE_MAX:
        # cheap eviction: drop an arbitrary item
        _MEM_CACHE.pop(next(iter(_MEM_CACHE)))
    _MEM_CACHE[key] = arr


def standardize_emb_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally z-score embedding columns in-place (per snapshot).

    Mirrors the cross-sectional rank-normalization applied to price features:
    removes market-wide level shifts so the head sees relative structure.
    Constant/all-NaN columns become 0. No-op when no emb columns are present.
    """
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        return df
    for c in emb_cols:
        col = pd.to_numeric(df[c], errors="coerce")
        mu = col.mean()
        sd = col.std()
        if not np.isfinite(sd) or sd < 1e-8:
            df[c] = 0.0
        else:
            df[c] = ((col - mu) / sd).fillna(0.0)
    return df
