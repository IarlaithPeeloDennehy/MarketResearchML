"""
alerts.py — turn IC and drift metrics into WARNING / CRITICAL / RETRAIN signals.

IC alerts (relative to the training IC baseline):
  WARNING  : rolling IC < IC_WARNING_RATIO  × train_IC   (default 0.70)
  CRITICAL : rolling IC < IC_CRITICAL_RATIO × train_IC    (default 0.50)
  RETRAIN  : rolling IC below the critical threshold for N consecutive windows

Drift alerts:
  CRITICAL : any feature PSI ≥ PSI_CRITICAL (0.25)
  WARNING  : any feature PSI ≥ PSI_WARNING  (0.10)
"""
from __future__ import annotations

import logging

from . import config, metrics_store

logger = logging.getLogger(__name__)


def classify_ic(rolling_ic, train_ic, recent_means: list,
                n_consecutive: int = None) -> str:
    """Pure IC classifier → 'ok' | 'WARNING' | 'CRITICAL' | 'RETRAIN'.

    recent_means: most-recent-first list of rolling IC means (for the RETRAIN
    'N consecutive windows below critical' rule).
    """
    n_consecutive = n_consecutive or config.IC_RETRAIN_CONSECUTIVE
    if rolling_ic is None or train_ic is None or train_ic <= 0:
        return "ok"
    crit_thr = config.IC_CRITICAL_RATIO * train_ic
    warn_thr = config.IC_WARNING_RATIO * train_ic
    consecutive_bad = (
        len(recent_means) >= n_consecutive
        and all((m is not None and m < crit_thr) for m in recent_means[:n_consecutive])
    )
    if consecutive_bad:
        return "RETRAIN"
    if rolling_ic < crit_thr:
        return "CRITICAL"
    if rolling_ic < warn_thr:
        return "WARNING"
    return "ok"


def evaluate_ic(model) -> dict:
    """Evaluate IC health for the primary rolling window. Persists an alert row
    when WARNING/CRITICAL/RETRAIN. Returns a structured state dict."""
    version = metrics_store.model_version(model)
    baseline = metrics_store.get_baseline(version)
    train_ic = baseline.train_ic if baseline else None

    recent = metrics_store.get_recent_ic(version, config.PRIMARY_WINDOW, limit=config.IC_RETRAIN_CONSECUTIVE)
    state = {"level": "ok", "train_ic": train_ic, "rolling_ic": None,
             "ratio": None, "window": config.PRIMARY_WINDOW}
    if not recent or train_ic is None or train_ic <= 0:
        return state

    latest = recent[0]            # most recent (desc order)
    rolling_ic = latest.ic_mean
    state["rolling_ic"] = rolling_ic
    if rolling_ic is None:
        return state
    state["ratio"] = rolling_ic / train_ic

    level = classify_ic(rolling_ic, train_ic, [r.ic_mean for r in recent])
    state["level"] = level
    if level != "ok":
        metrics_store.insert_alert(version, level, "ic", {
            "rolling_ic": rolling_ic, "train_ic": train_ic, "ratio": ratio,
            "window": config.PRIMARY_WINDOW,
            "consecutive_checked": len(recent),
        })
    return state


def evaluate_drift(model) -> dict:
    """Summarise the latest feature + prediction drift and persist alerts."""
    version = metrics_store.model_version(model)
    drift_rows = metrics_store.get_latest_drift(version)
    pred_drift = metrics_store.get_latest_prediction_drift(version)

    critical = [d.feature for d in drift_rows if d.psi >= config.PSI_CRITICAL]
    warning = [d.feature for d in drift_rows if config.PSI_WARNING <= d.psi < config.PSI_CRITICAL]
    max_psi = max((d.psi for d in drift_rows), default=0.0)

    pred_psi = pred_drift.psi if pred_drift else 0.0
    pred_status = pred_drift.status if pred_drift else "stable"

    if critical:
        metrics_store.insert_alert(version, "CRITICAL", "psi",
                                   {"critical_features": critical, "max_psi": max_psi})
    elif warning:
        metrics_store.insert_alert(version, "WARNING", "psi",
                                   {"warning_features": warning, "max_psi": max_psi})
    if pred_status == "critical":
        metrics_store.insert_alert(version, "CRITICAL", "pred_drift", {"psi": pred_psi})

    return {
        "critical_features": critical, "warning_features": warning,
        "max_feature_psi": max_psi, "prediction_psi": pred_psi,
        "prediction_status": pred_status,
    }
