"""
Tests for the shared channel builder (channels.py) — the single source of truth
both the offline trainer and production inference use. The highest-risk failure
mode is look-ahead leakage or train/inference drift, so these focus on
determinism, point-in-time safety, and graceful neutral fallback.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import channels as ch


def _series(n=300, seed=0, end="2024-12-31"):
    rng = np.random.default_rng(seed)
    px = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    return pd.Series(px, index=pd.bdate_range(end=end, periods=n))


def _vol(n=300, seed=1, end="2024-12-31"):
    rng = np.random.default_rng(seed)
    v = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.Series(v, index=pd.bdate_range(end=end, periods=n))


# ── shape / determinism ─────────────────────────────────────────────────────
def test_column_count_matches_spec():
    s = _series()
    mat = ch.build_window_channels(s, ch.ALL_CHANNELS, end_idx=len(s) - 1, window=120)
    assert mat.shape[1] == len(ch.ALL_CHANNELS)
    assert mat.shape[0] == 120          # window returns
    assert mat.dtype == np.float32


def test_deterministic():
    s = _series()
    a = ch.build_window_channels(s, ch.DEFAULT_SPEC, end_idx=250, window=120)
    b = ch.build_window_channels(s, ch.DEFAULT_SPEC, end_idx=250, window=120)
    np.testing.assert_array_equal(a, b)


# ── point-in-time (the core safety property) ────────────────────────────────
def test_future_bars_do_not_change_past_window():
    s = _series(n=300)
    full = ch.build_window_channels(s, ch.DEFAULT_SPEC, end_idx=150, window=120)
    truncated = ch.build_window_channels(s.iloc[:180], ch.DEFAULT_SPEC, end_idx=150, window=120)
    np.testing.assert_array_equal(full, truncated)


def test_volume_channels_used_when_present_neutral_when_absent():
    s = _series()
    spec = ["logret", "rel_volume", "log_dollar_vol"]
    with_vol = ch.build_window_channels(s, spec, end_idx=250, window=120, volume=_vol())
    without = ch.build_window_channels(s, spec, end_idx=250, window=120, volume=None)
    # volume channels are non-trivial with data, all-zero without
    assert np.any(with_vol[:, 1] != 0) or np.any(with_vol[:, 2] != 0)
    assert np.all(without[:, 1] == 0) and np.all(without[:, 2] == 0)
    # logret channel identical either way (volume must not affect it)
    np.testing.assert_array_equal(with_vol[:, 0], without[:, 0])


def test_excess_vs_market_zero_when_market_equals_stock():
    s = _series()
    spec = ["excess_ret_vs_market"]
    mat = ch.build_window_channels(s, spec, end_idx=250, window=120, market_close=s)
    # stock minus itself ≈ 0
    assert np.allclose(mat[:, 0], 0.0, atol=1e-6)


# ── earnings pulse causality ────────────────────────────────────────────────
def test_earnings_pulse_zero_before_announcement():
    dates = pd.bdate_range(end="2024-12-31", periods=200)
    announce_period = dates[100]                     # period_end
    hist = [{"period": str(announce_period.date()), "surprisePercent": 50.0}]
    pulse = ch._earnings_pulse(dates, hist)
    announce = announce_period + pd.Timedelta(days=ch._EARN_LAG_DAYS)
    before = dates < announce
    after = (dates >= announce) & (dates <= announce + pd.Timedelta(days=10))
    assert np.all(pulse[before] == 0.0), "pulse must be zero before announcement date"
    assert np.any(pulse[after] > 0.0), "pulse must fire after announcement"


def test_future_earnings_do_not_leak():
    s = _series(n=200)
    spec = ["earn_surprise_pulse"]
    end = 150
    end_date = s.index[end]
    past_evt = {"period": str((s.index[80]).date()), "surprisePercent": 40.0}
    future_evt = {"period": str((end_date + pd.Timedelta(days=120)).date()), "surprisePercent": 90.0}
    only_past = ch.build_window_channels(s, spec, end_idx=end, window=120,
                                         earnings_hist=[past_evt])
    with_future = ch.build_window_channels(s, spec, end_idx=end, window=120,
                                           earnings_hist=[past_evt, future_evt])
    np.testing.assert_array_equal(only_past, with_future)


# ── analyst trend ───────────────────────────────────────────────────────────
def test_analyst_net_buy_ffill_and_pit():
    dates = pd.bdate_range(end="2024-12-31", periods=120)
    p = dates[60]
    hist = [{"period": str(p.date()), "strongBuy": 3, "buy": 5, "hold": 1, "sell": 1, "strongSell": 0}]
    ts = ch._analyst_net_buy(dates, hist)
    assert np.all(ts[dates < p] == 0.0)            # nothing before publish
    expected = (3 + 5 - 1 - 0) / (3 + 5 + 1 + 1)   # (buy-sell)/total
    assert np.allclose(ts[dates >= p], expected)


# ── sector mapping ──────────────────────────────────────────────────────────
def test_sector_to_etf():
    assert ch.sector_to_etf("Technology") == "XLK"
    assert ch.sector_to_etf("Financial Services") == "XLF"
    assert ch.sector_to_etf("Banking") == "XLF"
    assert ch.sector_to_etf("Totally Unknown Sector") is None
    assert ch.sector_to_etf(None) is None
    assert ch.MARKET_ETF in ch.REFERENCE_TICKERS
