"""
ML training data integrity tests.

Verifies that _labels_for_horizon produces rows containing ONLY price-derived
rank features and none of the fundamental columns (pe_ratio, roe, etc.) that
would constitute look-ahead bias.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

# Make sure the backend package is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.feature_engineering import PRICE_FEATURE_COLS, FEATURE_COLS
from ml.model import NUMKTEnsemble


FUNDAMENTAL_COLS = [c for c in FEATURE_COLS if c not in PRICE_FEATURE_COLS]
PRICE_RANK_COLS  = [f"{c}_rank" for c in PRICE_FEATURE_COLS]
FUNDAMENTAL_RANK_COLS = [f"{c}_rank" for c in FUNDAMENTAL_COLS]


def _write_price_parquet(tmp_path: Path, ticker: str, n: int = 400, seed: int = 0) -> None:
    """Write a minimal price parquet that model.py can load."""
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    df = pd.DataFrame({"Close": prices, "Open": prices, "High": prices, "Low": prices}, index=idx)
    # safe_ticker_filename converts . and - to _
    safe = ticker.replace(".", "_").replace("-", "_")
    df.to_parquet(tmp_path / f"{safe}.parquet")


def test_no_fundamental_features_in_training_rows(tmp_path):
    """_labels_for_horizon must not produce any fundamental-derived columns."""
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    for i, t in enumerate(tickers):
        _write_price_parquet(tmp_path, t, n=400, seed=i)

    model = NUMKTEnsemble()
    with patch("ml.model.PRICE_DIR", tmp_path):
        rows, labels, fwd = model._labels_for_horizon(tickers, horizon=63)

    assert len(rows) > 0, "Expected at least one training row"

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    # No fundamental columns should appear — either raw or ranked
    all_fundamental = set(FUNDAMENTAL_COLS) | set(FUNDAMENTAL_RANK_COLS)
    leaked = all_keys.intersection(all_fundamental)
    assert not leaked, (
        f"Fundamental features found in training rows — look-ahead bias detected: {leaked}"
    )

    # Price rank columns must be present
    missing_price = set(PRICE_RANK_COLS) - all_keys
    assert not missing_price, f"Missing expected price rank columns: {missing_price}"

    # Only price-derived columns and housekeeping keys are allowed
    allowed = set(PRICE_RANK_COLS) | set(PRICE_FEATURE_COLS) | {"_time_idx"}
    unexpected = all_keys - allowed
    assert not unexpected, f"Unexpected non-price columns in training rows: {unexpected}"


def test_labels_for_horizon_returns_empty_with_too_few_tickers(tmp_path):
    """Should return empty lists when fewer than 3 tickers have price data."""
    tickers = ["AAPL", "MSFT"]
    for i, t in enumerate(tickers):
        _write_price_parquet(tmp_path, t, n=400, seed=i)

    model = NUMKTEnsemble()
    with patch("ml.model.PRICE_DIR", tmp_path):
        rows, labels, fwd = model._labels_for_horizon(tickers, horizon=63)

    assert rows == [] and labels == [] and fwd == []


def test_labels_are_binary(tmp_path):
    """All labels must be 0 or 1 (above/below median return)."""
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    for i, t in enumerate(tickers):
        _write_price_parquet(tmp_path, t, n=500, seed=i)

    model = NUMKTEnsemble()
    with patch("ml.model.PRICE_DIR", tmp_path):
        _, labels, _ = model._labels_for_horizon(tickers, horizon=63)

    assert len(labels) > 0
    assert all(lbl in (0, 1) for lbl in labels), "Labels must be binary (0 or 1)"


def test_all_rank_values_in_unit_interval(tmp_path):
    """Rank values must be in [0, 1] (percentile ranks)."""
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    for i, t in enumerate(tickers):
        _write_price_parquet(tmp_path, t, n=400, seed=i)

    model = NUMKTEnsemble()
    with patch("ml.model.PRICE_DIR", tmp_path):
        rows, _, _ = model._labels_for_horizon(tickers, horizon=63)

    for row in rows:
        for key, val in row.items():
            if key.endswith("_rank"):
                assert 0.0 <= val <= 1.0, (
                    f"Rank out of bounds: {key}={val}"
                )


def test_anchor_universe_size_and_uniqueness():
    from ml.model import _ANCHOR_TICKERS
    assert len(_ANCHOR_TICKERS) >= 40, "Anchor universe too small for stable head training"
    assert len(set(_ANCHOR_TICKERS)) == len(_ANCHOR_TICKERS), "Duplicate tickers in anchor list"
