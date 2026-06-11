"""
Model edge monitoring — IC tracking, feature/prediction drift, retraining
recommendations.

This package is FULLY DECOUPLED from model logic: it imports from `ml`, never
the reverse, and every public entry point below is wrapped so a monitoring or
storage failure logs and returns — it can never raise into inference or training.

Public entry points (the only things callers/hooks should use):
  capture_training_baseline(model, feature_df)   — call after training
  record_analyse(model, feature_df, results, period_key=None) — call after /analyse
  mature_and_evaluate(model)                      — background: mature labels →
                                                    rolling IC → recommendation → prune
  dashboard(model)                                — read-only dashboard payload
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from . import config
from . import baseline as _baseline
from . import drift_monitor as _drift
from . import ic_monitor as _ic
from . import recommendation as _rec
from . import dashboard_data as _dash
from . import metrics_store

logger = logging.getLogger(__name__)

model_version = metrics_store.model_version  # re-export


def _today_key() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def capture_training_baseline(model, feature_df) -> bool:
    if not config.ENABLED:
        return False
    try:
        return bool(_baseline.capture_training_baseline(model, feature_df))
    except Exception as exc:
        logger.warning(f"[monitoring] capture_training_baseline hook failed (non-fatal): {exc}")
        return False


def record_analyse(model, feature_df, results: list, period_key: Optional[str] = None) -> None:
    """Log predictions + compute feature/prediction drift for one /analyse batch.
    Drift needs no labels, so it's available immediately; IC matures later."""
    if not config.ENABLED:
        return
    try:
        pk = period_key or _today_key()
        n = _ic.log_predictions(model, results, pk)
        predictions = [
            (r.get("fundamental_score") or r.get("composite_score"))
            for r in (results or [])
            if (r.get("fundamental_score") or r.get("composite_score")) is not None
        ]
        predictions = [p / 100.0 for p in predictions]
        _drift.compute_feature_drift(model, feature_df, pk)
        _drift.compute_prediction_drift(model, predictions, pk)
        logger.debug(f"[monitoring] recorded {n} predictions + drift for period {pk}")
    except Exception as exc:
        logger.warning(f"[monitoring] record_analyse hook failed (non-fatal): {exc}")


def mature_and_evaluate(model) -> Optional[dict]:
    """Background pass: fill realized returns, recompute rolling IC, refresh the
    retraining recommendation, and prune old rows. Returns the recommendation."""
    if not config.ENABLED:
        return None
    try:
        _ic.mature_labels()
        _ic.compute_rolling_ic(model)
        rec = _rec.recommend(model, persist=True)
        metrics_store.prune(config.RETENTION_DAYS)
        logger.info(f"[monitoring] evaluation complete: {rec['status']} "
                    f"(confidence {rec['confidence']})")
        return rec
    except Exception as exc:
        logger.warning(f"[monitoring] mature_and_evaluate failed (non-fatal): {exc}")
        return None


def dashboard(model) -> dict:
    if not config.ENABLED:
        return {}
    try:
        return _dash.get_dashboard_data(model)
    except Exception as exc:
        logger.warning(f"[monitoring] dashboard failed (non-fatal): {exc}")
        return {}
