"""
series_cache.py — persistence for Finnhub time-series we used to discard.

`data_fetcher._save_info` only keeps scalar fields, so earnings and analyst
*history* (lists of dated records) were dropped before caching. Group B channels
(`earn_surprise_pulse`, `analyst_net_buy_ts`) need that history, point-in-time.

This module stores/loads it as `$CACHE_DIR/series/{ticker}.json`:
    {"earnings": [{"period": "...", "surprisePercent": ..}, ...],
     "analyst":  [{"period": "...", "buy":.., "strongBuy":.., "hold":..,
                   "sell":.., "strongSell":..}, ...]}
Tiny files (a few KB), so this is fine on the 1 GB disk.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .feature_engineering import safe_ticker_filename

_cache_root = os.environ.get("CACHE_DIR")
_CACHE = Path(_cache_root) if _cache_root else Path(__file__).parent.parent / "cache"
_SERIES_DIR = _CACHE / "series"


def _path(ticker: str) -> Path:
    return _SERIES_DIR / f"{safe_ticker_filename(ticker)}.json"


def save(ticker: str, earnings_hist: list | None, analyst_hist: list | None) -> None:
    """Persist earnings + analyst history for a ticker. Non-fatal on error."""
    try:
        _SERIES_DIR.mkdir(parents=True, exist_ok=True)
        with open(_path(ticker), "w") as f:
            json.dump(
                {"earnings": earnings_hist or [], "analyst": analyst_hist or []},
                f,
            )
    except Exception:
        pass


def load(ticker: str) -> tuple[list | None, list | None]:
    """Return (earnings_hist, analyst_hist), or (None, None) if uncached."""
    p = _path(ticker)
    if not p.exists():
        return None, None
    try:
        with open(p) as f:
            d = json.load(f) or {}
        return d.get("earnings"), d.get("analyst")
    except Exception:
        return None, None
