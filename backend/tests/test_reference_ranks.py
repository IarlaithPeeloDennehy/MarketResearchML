"""
Reference-panel de-biasing tests.

apply_reference_ranks must make a stock's price ranks depend only on the stock
and the fixed market reference panel — NOT on which other stocks share the
request. This is what removes the inference-time universe-selection bias.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import (
    PRICE_FEATURE_COLS,
    apply_reference_ranks,
)

PRICE_RANK_COLS = [f"{c}_rank" for c in PRICE_FEATURE_COLS]


def _panel(n: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"ticker": [f"REF{i}" for i in range(n)]}
    for c in PRICE_FEATURE_COLS:
        data[c] = rng.normal(0, 1, n)
    return pd.DataFrame(data)


def _stock_row(ticker: str, values: dict) -> pd.DataFrame:
    row = {"ticker": ticker}
    row.update({c: values.get(c, 0.0) for c in PRICE_FEATURE_COLS})
    return pd.DataFrame([row])


def test_reference_rank_is_universe_independent():
    """A stock's reference ranks are identical scored alone vs in a watchlist."""
    panel = _panel()
    target = {c: 0.3 for c in PRICE_FEATURE_COLS}

    solo = apply_reference_ranks(_stock_row("AAA", target), panel)

    others = pd.concat([
        _stock_row("AAA", target),
        _stock_row("BBB", {c: 2.0 for c in PRICE_FEATURE_COLS}),
        _stock_row("CCC", {c: -2.0 for c in PRICE_FEATURE_COLS}),
    ], ignore_index=True)
    multi = apply_reference_ranks(others, panel)
    multi_aaa = multi[multi["ticker"] == "AAA"]

    for rc in PRICE_RANK_COLS:
        assert abs(float(solo[rc].iloc[0]) - float(multi_aaa[rc].iloc[0])) < 1e-9


def test_reference_rank_matches_percentile():
    """Rank == fraction of reference observations <= the value (ascending)."""
    panel = _panel(n=100, seed=7)
    val = 0.5
    df = apply_reference_ranks(_stock_row("AAA", {c: val for c in PRICE_FEATURE_COLS}), panel)
    for c in PRICE_FEATURE_COLS:
        expected = float((panel[c].to_numpy() <= val).mean())
        assert abs(float(df[f"{c}_rank"].iloc[0]) - expected) < 1e-9


def test_missing_panel_is_noop():
    """No panel -> df returned unchanged (graceful within-universe fallback)."""
    df = _stock_row("AAA", {c: 1.0 for c in PRICE_FEATURE_COLS})
    df["mom_12m_rank"] = 0.42
    out = apply_reference_ranks(df.copy(), None)
    assert "mom_12m_rank" in out.columns
    assert float(out["mom_12m_rank"].iloc[0]) == 0.42


def test_higher_value_ranks_higher():
    """Monotonic: a higher feature value gets a >= reference rank."""
    panel = _panel(n=80, seed=3)
    low = apply_reference_ranks(_stock_row("L", {c: -1.0 for c in PRICE_FEATURE_COLS}), panel)
    high = apply_reference_ranks(_stock_row("H", {c: 1.0 for c in PRICE_FEATURE_COLS}), panel)
    for c in PRICE_FEATURE_COLS:
        assert float(high[f"{c}_rank"].iloc[0]) >= float(low[f"{c}_rank"].iloc[0])
