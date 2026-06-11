"""
drift_monitor.py — feature and prediction distribution drift vs the training
baseline. Available immediately (no labels required), so this is the earliest
warning signal that the world has moved away from what the model trained on.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import psi, metrics_store

logger = logging.getLogger(__name__)


def compute_feature_drift(model, feature_df: pd.DataFrame, period_key: str) -> list[dict]:
    """PSI + distribution stats for each feature vs its training baseline.
    Returns the rows written (empty if no baseline exists yet)."""
    version = metrics_store.model_version(model)
    baseline = metrics_store.get_baseline(version)
    if baseline is None or not baseline.feature_stats:
        return []

    rows = []
    for feature, base in baseline.feature_stats.items():
        if feature not in feature_df.columns:
            continue
        col = pd.to_numeric(feature_df[feature], errors="coerce").to_numpy()
        stats = psi.distribution_stats(col)
        edges = base.get("edges") or []
        base_props = base.get("props") or []
        live_props = psi.proportions(col, edges) if edges else []
        psi_v = psi.psi_value(base_props, live_props) if (base_props and live_props) else 0.0
        rows.append({
            "model_version": version, "period_key": period_key, "feature": feature,
            "psi": float(psi_v), "status": psi.psi_status(psi_v),
            "mean": stats["mean"], "std": stats["std"], "missing_rate": stats["missing_rate"],
            "p5": stats["p5"], "p25": stats["p25"], "p50": stats["p50"],
            "p75": stats["p75"], "p95": stats["p95"],
        })
    metrics_store.insert_drift(rows)
    return rows


def compute_prediction_drift(model, predictions, period_key: str) -> dict:
    """PSI of the live prediction distribution vs the training prediction dist."""
    version = metrics_store.model_version(model)
    baseline = metrics_store.get_baseline(version)
    preds = np.asarray([p for p in predictions if p is not None], dtype=float)
    if baseline is None or not baseline.prediction_stats or preds.size == 0:
        return {}
    base = baseline.prediction_stats
    stats = psi.distribution_stats(preds)
    edges = base.get("edges") or []
    base_props = base.get("props") or []
    live_props = psi.proportions(preds, edges) if edges else []
    psi_v = psi.psi_value(base_props, live_props) if (base_props and live_props) else 0.0
    row = {
        "model_version": version, "period_key": period_key,
        "psi": float(psi_v), "status": psi.psi_status(psi_v),
        "mean": stats["mean"], "std": stats["std"],
        "p5": stats["p5"], "p25": stats["p25"], "p50": stats["p50"],
        "p75": stats["p75"], "p95": stats["p95"],
        "hist": psi.histogram(preds),
    }
    metrics_store.insert_prediction_drift(row)
    return row
