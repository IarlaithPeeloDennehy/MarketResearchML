"""
recommendation.py — fuse IC + drift signals into a retraining recommendation.

Output: {"status": "healthy|watch|retrain", "confidence": float, "reasons": [...]}

  retrain : persistent IC degradation (RETRAIN) OR a critical-PSI feature OR
            material prediction drift
  watch   : IC deterioration (WARNING/CRITICAL) OR any moderate drift
  healthy : none of the above

`confidence` is how many of the three independent signal families
(IC, feature drift, prediction drift) agree with the verdict — so one noisy
signal can't, on its own, scream "retrain".
"""
from __future__ import annotations

import logging

from . import metrics_store
from .alerts import evaluate_ic, evaluate_drift

logger = logging.getLogger(__name__)


def decide(ic: dict, drift: dict) -> dict:
    """Pure fusion of IC + drift state → recommendation. No DB access, so it's
    directly unit-testable."""
    reasons: list[str] = []
    retrain_signals = 0
    watch_signals = 0
    total_signals = 3   # IC, feature drift, prediction drift

    # 1) IC
    ic_level = ic.get("level", "ok")
    if ic_level == "RETRAIN":
        retrain_signals += 1
        reasons.append(f"IC below {ic.get('ratio'):.2f}× training IC for consecutive windows")
    elif ic_level == "CRITICAL":
        retrain_signals += 1
        reasons.append(f"Rolling IC critically low ({ic.get('ratio'):.2f}× training IC)")
    elif ic_level == "WARNING":
        watch_signals += 1
        reasons.append(f"Rolling IC deteriorating ({ic.get('ratio'):.2f}× training IC)")

    # 2) feature drift
    if drift.get("critical_features"):
        retrain_signals += 1
        reasons.append(f"Critical feature drift (PSI≥0.25): {', '.join(drift['critical_features'][:5])}")
    elif drift.get("warning_features"):
        watch_signals += 1
        reasons.append(f"Moderate feature drift on {len(drift['warning_features'])} feature(s)")

    # 3) prediction drift
    if drift.get("prediction_status") == "critical":
        retrain_signals += 1
        reasons.append(f"Prediction distribution drift (PSI={drift.get('prediction_psi'):.2f})")
    elif drift.get("prediction_status") == "warning":
        watch_signals += 1
        reasons.append("Mild prediction distribution drift")

    if retrain_signals > 0:
        status = "retrain"
        confidence = round(retrain_signals / total_signals, 2)
    elif watch_signals > 0:
        status = "watch"
        confidence = round(watch_signals / total_signals, 2)
    else:
        status = "healthy"
        confidence = 1.0
        reasons.append("IC stable and no material drift")

    return {"status": status, "confidence": confidence, "reasons": reasons}


def recommend(model, persist: bool = True) -> dict:
    """Evaluate IC + drift for `model` and produce a recommendation."""
    result = decide(evaluate_ic(model), evaluate_drift(model))
    if persist:
        metrics_store.insert_alert(metrics_store.model_version(model),
                                   result["status"], "recommendation", result)
    return result
