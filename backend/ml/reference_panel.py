"""
reference_panel.py
──────────────────
Builds a cached "market reference panel": the 7 raw price-derived feature
values (PRICE_FEATURE_COLS) for every anchor ticker already in the price cache.

Why this exists
  The model is trained cross-sectionally, but at inference time
  feature_engineering.build_features ranks each feature only *within the user's
  request*. A 5-stock watchlist therefore gets ranked against itself, so a
  mediocre stock can look great among four worse ones and scores are not
  comparable across runs. Ranking the user's stocks against this broad panel
  instead (see feature_engineering.apply_reference_ranks) makes every score
  market-relative and independent of which other stocks the user happened to
  pick.

  Only the 7 price features need a reference distribution because the trained
  model's feature set is exactly PRICE_FEATURE_COLS (fundamentals are 0.5-filled
  in historical training rows and never enter the model). All 7 are computable
  from the cached price parquets alone — no fundamentals required.

The panel is cheap to compute and is cached to disk with a short TTL, mirroring
the daily public-insights snapshot pattern. It is fully optional: if it can't be
built (cold start, empty cache) callers fall back to within-universe ranking.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .feature_engineering import (
    PRICE_FEATURE_COLS,
    _compute_rsi,
    _momentum,
    _realised_vol,
    safe_ticker_filename,
)
from .model import PRICE_DIR, _ANCHOR_TICKERS

logger = logging.getLogger(__name__)

REFERENCE_DIR  = PRICE_DIR.parent / "reference"
REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
_PANEL_PATH    = REFERENCE_DIR / "panel.parquet"
_MAX_AGE_HOURS = 24
_MIN_NAMES     = 20   # below this the panel isn't broad enough to be useful


def _price_features(close: pd.Series) -> dict | None:
    """The 7 PRICE_FEATURE_COLS values from a single close series (today's snapshot).
    Reuses the same helpers feature_engineering uses so values match build_features."""
    close = close.dropna()
    if len(close) < 30:
        return None
    hi = close.iloc[-252:].max()
    return {
        "mom_1m":            _momentum(close, 21),
        "mom_3m":            _momentum(close, 63),
        "mom_6m":            _momentum(close, 126),
        "mom_12m":           _momentum(close, 252),
        "vol_60d":           _realised_vol(close, 60),
        "rsi_14":            _compute_rsi(close, 14),
        "price_vs_52w_high": float(close.iloc[-1] / hi - 1) if hi > 0 else np.nan,
    }


def build_reference_panel() -> pd.DataFrame | None:
    """Compute the panel from every anchor ticker already in the price cache.
    Never fetches — uses only what is on disk. Returns None if too few names."""
    rows = []
    for ticker in _ANCHOR_TICKERS:
        path = PRICE_DIR / f"{safe_ticker_filename(ticker)}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
            if "Close" not in df.columns:
                continue
            feats = _price_features(df["Close"].sort_index())
            if feats is None:
                continue
            feats["ticker"] = ticker
            rows.append(feats)
        except Exception as exc:
            logger.warning(f"reference_panel: skipping {ticker} ({exc})")

    if len(rows) < _MIN_NAMES:
        logger.info(
            f"reference_panel: only {len(rows)} cached anchors "
            f"(need {_MIN_NAMES}) — panel not built yet"
        )
        return None

    panel = pd.DataFrame(rows)[["ticker", *PRICE_FEATURE_COLS]]
    try:
        panel.to_parquet(_PANEL_PATH)
        logger.info(f"reference_panel: built + cached from {len(panel)} anchors")
    except Exception as exc:
        logger.warning(f"reference_panel: could not cache panel ({exc})")
    return panel


def _is_fresh() -> bool:
    if not _PANEL_PATH.exists():
        return False
    age_h = (datetime.now(timezone.utc).timestamp() - _PANEL_PATH.stat().st_mtime) / 3600
    return age_h < _MAX_AGE_HOURS


def load_reference_panel() -> pd.DataFrame | None:
    """Return the cached panel, rebuilding it if missing or stale.
    Returns None (and callers fall back to within-universe ranking) if it can't
    be built — never raises into the request path."""
    try:
        if _is_fresh():
            panel = pd.read_parquet(_PANEL_PATH)
            if len(panel) >= _MIN_NAMES:
                return panel
        return build_reference_panel()
    except Exception as exc:
        logger.warning(f"reference_panel: load failed, falling back ({exc})")
        return None
