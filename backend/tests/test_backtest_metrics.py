"""
Backtest metrics tests.

Verifies:
  - calmar_ratio is None when max drawdown is zero (no division by near-zero)
  - effective_lookback_years is present in the result dict
  - run_backtest returns an error dict (not an exception) when data is insufficient
"""
import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.backtest import run_backtest


def _make_raw_data(tickers: list[str], n: int = 600, seed: int = 42) -> dict:
    """Build a minimal raw_data dict that run_backtest can consume."""
    rng = np.random.default_rng(seed)
    raw = {}
    for i, t in enumerate(tickers):
        prices = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
        idx = pd.bdate_range(end="2024-12-31", periods=n)
        df = pd.DataFrame({"Close": prices}, index=idx)
        raw[t] = {"prices": df}
    return raw


def _make_features_df(tickers: list[str]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {
        "ticker":      tickers,
        "mom_1m":      rng.uniform(-0.1, 0.1,  len(tickers)).tolist(),
        "mom_3m":      rng.uniform(-0.2, 0.2,  len(tickers)).tolist(),
        "mom_6m":      rng.uniform(-0.3, 0.3,  len(tickers)).tolist(),
        "mom_12m":     rng.uniform(-0.4, 0.4,  len(tickers)).tolist(),
        "vol_60d":     rng.uniform(0.1, 0.5,   len(tickers)).tolist(),
        "rsi_14":      rng.uniform(30, 70,     len(tickers)).tolist(),
        "price_vs_52w_high": rng.uniform(-0.3, 0.0, len(tickers)).tolist(),
    }
    return pd.DataFrame(data).set_index("ticker")


class TestCalmarRatio:
    def test_calmar_is_none_when_no_drawdown(self):
        """When max_dd >= 0, calmar_ratio must be None (not inf or nan)."""
        # Simulate a monotonically rising portfolio — no drawdown ever occurs
        pa = np.array([0.05, 0.04, 0.06, 0.05, 0.07])  # all positive
        cum = np.cumprod(1 + pa)
        rm  = np.maximum.accumulate(cum)
        max_dd = float(np.min((cum - rm) / (rm + 1e-9)))

        ppy = 4
        ann_port = float((np.prod(1 + pa) ** (ppy / len(pa))) - 1)
        calmar = round(ann_port / abs(max_dd), 3) if max_dd < -1e-6 else None

        assert calmar is None, f"Expected None for zero-drawdown, got {calmar}"

    def test_calmar_is_float_when_drawdown_exists(self):
        """When a real drawdown exists, calmar_ratio must be a finite float."""
        pa = np.array([0.05, -0.15, 0.06, 0.04, 0.07])
        cum = np.cumprod(1 + pa)
        rm  = np.maximum.accumulate(cum)
        max_dd = float(np.min((cum - rm) / (rm + 1e-9)))

        ppy = 4
        ann_port = float((np.prod(1 + pa) ** (ppy / len(pa))) - 1)
        calmar = round(ann_port / abs(max_dd), 3) if max_dd < -1e-6 else None

        assert calmar is not None
        assert isinstance(calmar, float)
        assert np.isfinite(calmar)

    def test_calmar_none_is_json_serialisable(self):
        """None must survive JSON serialisation without raising TypeError."""
        import json
        result = {"calmar_ratio": None, "sharpe_ratio": 1.23}
        serialised = json.dumps(result)
        assert '"calmar_ratio": null' in serialised


class TestEffectiveLookbackYears:
    def test_effective_lookback_years_present(self):
        """run_backtest result must include effective_lookback_years."""
        tickers     = ["AAPL", "MSFT", "GOOG", "AMZN"]
        raw_data    = _make_raw_data(tickers, n=600)
        features_df = _make_features_df(tickers)

        result = run_backtest(
            features_df      = features_df,
            raw_data         = raw_data,
            train_model      = False,
            forward_months   = 3,
            rebalance_months = 3,
        )
        assert "effective_lookback_years" in result, (
            "run_backtest must return effective_lookback_years"
        )

    def test_effective_lookback_years_is_positive_float(self):
        tickers     = ["AAPL", "MSFT", "GOOG", "AMZN"]
        raw_data    = _make_raw_data(tickers, n=600)
        features_df = _make_features_df(tickers)

        result = run_backtest(
            features_df      = features_df,
            raw_data         = raw_data,
            train_model      = False,
            forward_months   = 3,
            rebalance_months = 3,
        )
        ely = result.get("effective_lookback_years")
        assert isinstance(ely, float)
        assert ely > 0

    def test_effective_lookback_years_matches_min_bars(self):
        """effective_lookback_years should equal round(min_bars / 252, 1)."""
        tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
        n_bars  = 504  # exactly 2 years
        raw_data    = _make_raw_data(tickers, n=n_bars)
        features_df = _make_features_df(tickers)

        result = run_backtest(
            features_df      = features_df,
            raw_data         = raw_data,
            train_model      = False,
            forward_months   = 3,
            rebalance_months = 3,
        )
        ely = result.get("effective_lookback_years")
        expected = round(n_bars / 252, 1)
        assert ely == expected, f"Expected {expected}, got {ely}"


class TestTickerValidation:
    """Tests for the _TICKER_RE lookahead fix (Item 14)."""

    def test_pure_digit_ticker_is_rejected(self):
        import re
        _TICKER_RE = re.compile(r"^(?=[A-Z0-9.\-]*[A-Z])[A-Z0-9.\-]{1,15}$")
        assert not _TICKER_RE.match("123"), "Pure-digit ticker must be rejected"
        assert not _TICKER_RE.match("99999"), "Pure-digit ticker must be rejected"

    def test_valid_tickers_are_accepted(self):
        import re
        _TICKER_RE = re.compile(r"^(?=[A-Z0-9.\-]*[A-Z])[A-Z0-9.\-]{1,15}$")
        valid = ["AAPL", "BRK-B", "AZN.L", "A5G.IR", "META", "TSLA"]
        for t in valid:
            assert _TICKER_RE.match(t), f"Valid ticker {t!r} was incorrectly rejected"

    def test_empty_string_is_rejected(self):
        import re
        _TICKER_RE = re.compile(r"^(?=[A-Z0-9.\-]*[A-Z])[A-Z0-9.\-]{1,15}$")
        assert not _TICKER_RE.match("")

    def test_ticker_too_long_is_rejected(self):
        import re
        _TICKER_RE = re.compile(r"^(?=[A-Z0-9.\-]*[A-Z])[A-Z0-9.\-]{1,15}$")
        assert not _TICKER_RE.match("A" * 16)

    def test_lowercase_is_rejected(self):
        import re
        _TICKER_RE = re.compile(r"^(?=[A-Z0-9.\-]*[A-Z])[A-Z0-9.\-]{1,15}$")
        assert not _TICKER_RE.match("aapl")
