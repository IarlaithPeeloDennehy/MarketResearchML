"""
baseline.py — capture the training-time distributions monitoring compares against.

Called once after a model finishes training (from backtest.run_backtest). For
each feature the model uses we freeze: distribution stats, PSI bin edges, and the
baseline bin proportions. We also freeze the training prediction distribution
(the OOF probabilities) and the training IC (the reference for IC alerts).

Everything is wrapped — a baseline-capture failure must never affect training.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config, psi, metrics_store

logger = logging.getLogger(__name__)


def _feature_baseline(feature_df: pd.DataFrame, feature_names: list[str]) -> dict:
    stats = {}
    for f in feature_names:
        if f not in feature_df.columns:
            continue
        col = pd.to_numeric(feature_df[f], errors="coerce").to_numpy()
        d = psi.distribution_stats(col)
        edges = psi.histogram_edges(col)
        props = psi.proportions(col, edges) if edges else []
        d["edges"] = edges
        d["props"] = props
        stats[f] = d
    return stats


def _prediction_baseline(model, feature_df: pd.DataFrame) -> dict:
    proba = getattr(model, "_oof_proba", None)
    if proba is None or len(proba) == 0:
        # fall back to in-sample predictions on the training matrix
        try:
            cols = [c for c in model._feature_names if c in feature_df.columns]
            if cols:
                X = feature_df[cols].to_numpy()
                proba = model.predict_proba(X)
        except Exception:
            proba = None
    if proba is None or len(proba) == 0:
        return {}
    proba = np.asarray(proba, dtype=float)
    d = psi.distribution_stats(proba)
    edges = psi.histogram_edges(proba)
    d["edges"] = edges
    d["props"] = psi.proportions(proba, edges) if edges else []
    d["hist"] = psi.histogram(proba)
    return d


def capture_training_baseline(model, feature_df: pd.DataFrame) -> bool:
    """Persist feature + prediction baselines and the training IC for `model`.

    Safe to call always — no-ops when monitoring is disabled, the model isn't
    fitted, or no DB is configured. Never raises.
    """
    if not config.ENABLED:
        return False
    try:
        if model is None or not getattr(model, "is_fitted", False):
            return False
        feature_names = list(getattr(model, "_feature_names", []) or [])
        if feature_df is None or feature_df.empty or not feature_names:
            return False

        version = metrics_store.model_version(model)
        feature_stats = _feature_baseline(feature_df, feature_names)
        prediction_stats = _prediction_baseline(model, feature_df)
        train_ic = getattr(model, "cv_ic", None)

        ok = metrics_store.save_baseline(version, train_ic, feature_stats, prediction_stats)
        if ok:
            logger.info(
                f"[monitoring] baseline captured for {version}: "
                f"{len(feature_stats)} features, train_IC={train_ic}"
            )
        return ok
    except Exception as exc:
        logger.warning(f"[monitoring] capture_training_baseline failed (non-fatal): {exc}")
        return False
