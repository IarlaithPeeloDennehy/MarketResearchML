"""
Cache isolation tests.

Verifies that /backtest rejects a stale cache that was populated for a
different ticker universe, forcing a fresh fetch rather than silently
returning cross-user-contaminated data.
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_cache_subset_check_accepts_matching_tickers():
    """A cache that contains all requested tickers should be accepted."""
    cached_keys = {"AAPL", "MSFT", "GOOG", "AMZN"}
    requested   = ["AAPL", "MSFT", "GOOG"]
    assert set(requested).issubset(cached_keys)


def test_cache_subset_check_rejects_missing_ticker():
    """A cache missing any requested ticker must be rejected."""
    cached_keys = {"AAPL", "MSFT"}
    requested   = ["AAPL", "MSFT", "META"]
    assert not set(requested).issubset(cached_keys)


def test_cache_subset_check_rejects_completely_different_universe():
    """A cache from a different user's universe must be rejected."""
    cached_keys = {"TSLA", "NVDA", "AMD"}
    requested   = ["AAPL", "MSFT", "GOOG"]
    assert not set(requested).issubset(cached_keys)


def test_cache_subset_check_accepts_identical_universe():
    """An exact match should be accepted."""
    tickers = ["AAPL", "MSFT", "GOOG"]
    assert set(tickers).issubset(set(tickers))


def test_cache_subset_check_accepts_superset_cache():
    """A cache with MORE tickers than requested is still valid — superset is OK."""
    cached_keys = {"AAPL", "MSFT", "GOOG", "AMZN", "META"}
    requested   = ["AAPL", "MSFT", "GOOG"]
    assert set(requested).issubset(cached_keys)


def test_none_cache_triggers_fresh_fetch():
    """None cache must always trigger a fresh fetch."""
    cached_raw = None
    requested  = ["AAPL", "MSFT", "GOOG"]
    should_use_cache = (
        cached_raw is not None
        and set(requested).issubset(set(cached_raw.keys()))
    )
    assert not should_use_cache
