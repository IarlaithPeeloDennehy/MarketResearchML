"""
Tests for the model edge monitoring framework.

Two layers:
  * Pure tests (always run): PSI math, IC classifier, recommendation fusion,
    per-period IC, label maturation from the price cache, and the graceful
    no-op / exception-safety guarantees.
  * Store-backed tests (skipped when sqlmodel is unavailable): baseline →
    log → drift → mature → IC → recommendation against a throwaway SQLite DB,
    plus storage-failure safety.
"""
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring import psi, config
from monitoring import alerts, recommendation, ic_monitor, metrics_store
import monitoring


# ── a minimal fake model (only the attributes monitoring reads) ─────────────
class FakeModel:
    def __init__(self, feature_names, train_ic=0.10, oof=None):
        self.trained_at = "2020-01-01T00:00:00"
        self.is_fitted = True
        self.cv_ic = train_ic
        self._feature_names = feature_names
        self._oof_proba = oof if oof is not None else np.linspace(0.2, 0.8, 50)

    def predict_proba(self, X):
        return np.clip(np.asarray(X, dtype=float).mean(axis=1), 0.01, 0.99)


# ════════════════════════════════════════════════════════════════════════════
# Pure: PSI
# ════════════════════════════════════════════════════════════════════════════
def test_psi_zero_for_identical_distribution():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 5000)
    edges = psi.histogram_edges(base)
    props = psi.proportions(base, edges)
    same = psi.proportions(base, edges)
    assert psi.psi_value(props, same) < 1e-9


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, 5000)
    edges = psi.histogram_edges(base)
    base_props = psi.proportions(base, edges)
    small = psi.proportions(rng.normal(0.3, 1, 5000), edges)
    large = psi.proportions(rng.normal(2.0, 1, 5000), edges)
    psi_small = psi.psi_value(base_props, small)
    psi_large = psi.psi_value(base_props, large)
    assert 0 < psi_small < psi_large


def test_psi_status_thresholds():
    assert psi.psi_status(0.05) == "stable"
    assert psi.psi_status(0.15) == "warning"
    assert psi.psi_status(0.40) == "critical"


def test_distribution_stats_handles_missing():
    vals = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
    d = psi.distribution_stats(vals)
    assert d["missing_rate"] == pytest.approx(0.4)
    assert d["p50"] == pytest.approx(2.0)
    assert psi.distribution_stats([np.nan, np.nan])["missing_rate"] == 1.0


# ════════════════════════════════════════════════════════════════════════════
# Pure: IC classifier + recommendation fusion
# ════════════════════════════════════════════════════════════════════════════
def test_classify_ic_truth_table():
    train = 0.10
    assert alerts.classify_ic(0.09, train, [0.09]) == "ok"        # >70%
    assert alerts.classify_ic(0.06, train, [0.06]) == "WARNING"   # <70%, >50%
    assert alerts.classify_ic(0.04, train, [0.04]) == "CRITICAL"  # <50%
    # N consecutive below critical → RETRAIN
    assert alerts.classify_ic(0.04, train, [0.04, 0.03, 0.02]) == "RETRAIN"
    assert alerts.classify_ic(None, train, []) == "ok"
    assert alerts.classify_ic(0.04, 0.0, [0.04]) == "ok"          # no valid train IC


def test_decide_healthy():
    out = recommendation.decide(
        {"level": "ok", "ratio": 1.0},
        {"critical_features": [], "warning_features": [], "prediction_status": "stable"},
    )
    assert out["status"] == "healthy" and out["confidence"] == 1.0


def test_decide_watch():
    out = recommendation.decide(
        {"level": "WARNING", "ratio": 0.65},
        {"critical_features": [], "warning_features": ["roe_rank"], "prediction_status": "stable"},
    )
    assert out["status"] == "watch"
    assert 0 < out["confidence"] <= 1.0


def test_decide_retrain_multi_signal_confidence():
    out = recommendation.decide(
        {"level": "RETRAIN", "ratio": 0.3},
        {"critical_features": ["mom_12m_rank"], "warning_features": [],
         "prediction_status": "critical", "prediction_psi": 0.4},
    )
    assert out["status"] == "retrain"
    assert out["confidence"] == pytest.approx(1.0)   # all 3 signal families agree
    assert len(out["reasons"]) >= 3


# ════════════════════════════════════════════════════════════════════════════
# Pure: per-period IC + label maturation from the price cache
# ════════════════════════════════════════════════════════════════════════════
def test_period_ics_cross_sectional():
    # one period, predictions perfectly rank-correlated with realized returns
    rows = [
        types.SimpleNamespace(period_key="2024-01-01", predicted_at=datetime(2024, 1, 1),
                              prediction=p, realized_return=p)
        for p in (0.1, 0.4, 0.6, 0.9)
    ]
    out = ic_monitor._period_ics(rows)
    assert len(out) == 1
    assert out[0]["spearman"] == pytest.approx(1.0)


def test_realized_return_from_price_cache(tmp_path, monkeypatch):
    import ml.model as m
    monkeypatch.setattr(m, "PRICE_DIR", tmp_path)
    idx = pd.bdate_range(end="2024-12-31", periods=200)
    close = pd.Series(np.linspace(100, 200, 200), index=idx)   # +100% over the window
    pd.DataFrame({"Close": close}).to_parquet(tmp_path / "AAPL.parquet")
    predicted_at = idx[50]
    ret = ic_monitor._realized_return("AAPL", predicted_at, horizon_days=10)
    expected = close.iloc[60] / close.iloc[50] - 1
    assert ret == pytest.approx(expected, rel=1e-6)
    # horizon beyond available bars → None (label not yet matured)
    assert ic_monitor._realized_return("AAPL", idx[195], horizon_days=10) is None


# ════════════════════════════════════════════════════════════════════════════
# Pure: graceful no-op + exception safety (no DB configured)
# ════════════════════════════════════════════════════════════════════════════
def test_entrypoints_noop_without_store():
    assert metrics_store.available() is False
    model = FakeModel(["mom_12m_rank", "rsi_14_rank"])
    df = pd.DataFrame({"mom_12m_rank": [0.2, 0.8], "rsi_14_rank": [0.5, 0.5], "ticker": ["A", "B"]})
    results = [{"ticker": "A", "fundamental_score": 60.0}, {"ticker": "B", "fundamental_score": 40.0}]
    # none of these may raise, even with no DB. With no data the recommendation
    # is simply "healthy" (nothing wrong to report).
    monitoring.capture_training_baseline(model, df)
    monitoring.record_analyse(model, df, results)
    out = monitoring.mature_and_evaluate(model)
    assert out is None or out["status"] in ("healthy", "watch", "retrain")
    # inputs must be unchanged (monitoring is side-effect-free on its inputs)
    assert list(df["ticker"]) == ["A", "B"]
    assert results[0]["fundamental_score"] == 60.0


def test_monitoring_survives_bad_model():
    class Broken:
        @property
        def trained_at(self):
            raise RuntimeError("boom")
    # wrappers must swallow everything
    monitoring.capture_training_baseline(Broken(), pd.DataFrame())
    monitoring.record_analyse(Broken(), pd.DataFrame(), [{"ticker": "A", "fundamental_score": 1}])
    assert monitoring.mature_and_evaluate(Broken()) is None


def test_disabled_flag(monkeypatch):
    monkeypatch.setattr(config, "ENABLED", False)
    model = FakeModel(["mom_12m_rank"])
    monitoring.capture_training_baseline(model, pd.DataFrame({"mom_12m_rank": [0.1]}))
    assert monitoring.dashboard(model) == {}


# ════════════════════════════════════════════════════════════════════════════
# Store-backed (skipped when sqlmodel is unavailable)
# ════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def store(tmp_path):
    pytest.importorskip("sqlmodel")
    from sqlmodel import SQLModel, create_engine
    import monitoring.models  # noqa: F401 — register tables
    engine = create_engine(f"sqlite:///{tmp_path/'mon.db'}")
    SQLModel.metadata.create_all(engine)
    metrics_store.set_engine(engine)
    try:
        yield engine
    finally:
        metrics_store.set_engine(None)


def _feature_df():
    rng = np.random.default_rng(3)
    return pd.DataFrame({
        "mom_12m_rank": rng.random(60),
        "rsi_14_rank": rng.random(60),
        "ticker": [f"T{i}" for i in range(60)],
    })


def test_baseline_roundtrip(store):
    model = FakeModel(["mom_12m_rank", "rsi_14_rank"], train_ic=0.12)
    assert monitoring.capture_training_baseline(model, _feature_df()) is True
    b = metrics_store.get_baseline(metrics_store.model_version(model))
    assert b is not None
    assert b.train_ic == pytest.approx(0.12)
    assert "mom_12m_rank" in b.feature_stats


def test_record_analyse_persists_predictions_and_drift(store):
    model = FakeModel(["mom_12m_rank", "rsi_14_rank"])
    monitoring.capture_training_baseline(model, _feature_df())
    df = pd.DataFrame({"mom_12m_rank": [0.2, 0.8, 0.5], "rsi_14_rank": [0.5, 0.5, 0.9],
                       "ticker": ["A", "B", "C"]})
    results = [{"ticker": "A", "fundamental_score": 70.0},
               {"ticker": "B", "fundamental_score": 30.0},
               {"ticker": "C", "fundamental_score": 55.0}]
    monitoring.record_analyse(model, df, results, period_key="2024-06-01")
    version = metrics_store.model_version(model)
    assert len(metrics_store.get_latest_drift(version)) == 2          # two features
    assert metrics_store.get_latest_prediction_drift(version) is not None


def test_end_to_end_mature_ic_and_recommend(store, tmp_path, monkeypatch):
    import ml.model as m
    monkeypatch.setattr(m, "PRICE_DIR", tmp_path)
    model = FakeModel(["mom_12m_rank"], train_ic=0.10)
    monitoring.capture_training_baseline(model, _feature_df())
    version = metrics_store.model_version(model)

    # write price parquets and insert past-dated predictions (so they're due)
    idx = pd.bdate_range(end="2024-12-31", periods=200)
    predicted_at = idx[50]
    rng = np.random.default_rng(5)
    rows = []
    for i, tk in enumerate(["A", "B", "C", "D"]):
        # realized return increases with i; prediction also increases → positive IC
        end_price = 100 * (1 + 0.05 * i)
        close = pd.Series(np.linspace(100, end_price, 200), index=idx)
        pd.DataFrame({"Close": close}).to_parquet(tmp_path / f"{tk}.parquet")
        rows.append({"model_version": version, "period_key": "2024-01", "ticker": tk,
                     "prediction": 0.2 + 0.2 * i, "horizon_days": 10,
                     "predicted_at": predicted_at})
    metrics_store.insert_predictions(rows)

    assert ic_monitor.mature_labels() == 4
    ic_rows = ic_monitor.compute_rolling_ic(model)
    assert ic_rows and ic_rows[0]["spearman_ic"] == pytest.approx(1.0)

    rec = recommendation.recommend(model)
    assert rec["status"] in ("healthy", "watch", "retrain")
    dash = monitoring.dashboard(model)
    assert dash["model_version"] == version and "rolling_ic" in dash


def test_storage_failure_is_safe(store):
    """A broken engine must not raise out of any monitoring call."""
    from sqlalchemy import create_engine
    broken = create_engine("sqlite:////nonexistent/path/cannot/open.db")
    metrics_store.set_engine(broken)
    model = FakeModel(["mom_12m_rank"])
    # all of these hit the broken DB and must swallow the error (never raise)
    assert monitoring.capture_training_baseline(model, _feature_df()) in (False, None)
    monitoring.record_analyse(model, _feature_df(), [{"ticker": "A", "fundamental_score": 50}])
    out = monitoring.mature_and_evaluate(model)
    assert out is None or isinstance(out, dict)
