"""Shared pytest fixtures."""
import numpy as np
import pandas as pd
import pytest


def _make_price_series(n: int = 500, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    idx = pd.bdate_range(end="2024-12-31", periods=n)
    return pd.Series(prices, index=idx, name="Close")


def _make_price_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    return _make_price_series(n, seed).rename("Close").to_frame()


@pytest.fixture
def price_series_factory():
    return _make_price_series


@pytest.fixture
def price_df_factory():
    return _make_price_df
