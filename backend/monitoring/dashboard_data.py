"""
dashboard_data.py — read-only aggregators that shape monitoring metrics into
JSON-friendly structures for charts. Builds NO UI; just the data a dashboard
(or an API endpoint) would render.
"""
from __future__ import annotations

import logging

from . import config, metrics_store
from .recommendation import recommend

logger = logging.getLogger(__name__)


def get_dashboard_data(model) -> dict:
    version = metrics_store.model_version(model)
    baseline = metrics_store.get_baseline(version)

    # 1) rolling IC time series per window (chronological)
    rolling_ic = {}
    for window in config.ROLLING_WINDOWS:
        rows = list(reversed(metrics_store.get_recent_ic(version, window, limit=200)))
        rolling_ic[str(window)] = [
            {"computed_at": str(r.computed_at), "ic_mean": r.ic_mean,
             "spearman": r.spearman_ic, "ic_ir": r.ic_ir, "n_obs": r.n_obs}
            for r in rows
        ]

    # 2) IC information ratio (latest, primary window)
    primary = metrics_store.get_recent_ic(version, config.PRIMARY_WINDOW, limit=1)
    ic_ir = primary[0].ic_ir if primary else None

    # 3) feature PSI heatmap (latest period)
    feature_psi = [
        {"feature": d.feature, "psi": d.psi, "status": d.status}
        for d in metrics_store.get_latest_drift(version)
    ]

    # 4) prediction distribution drift
    pd_row = metrics_store.get_latest_prediction_drift(version)
    prediction_drift = None
    if pd_row is not None:
        prediction_drift = {
            "psi": pd_row.psi, "status": pd_row.status,
            "mean": pd_row.mean, "std": pd_row.std,
            "quantiles": {"p5": pd_row.p5, "p25": pd_row.p25, "p50": pd_row.p50,
                          "p75": pd_row.p75, "p95": pd_row.p95},
            "histogram": pd_row.hist,
            "baseline_histogram": (baseline.prediction_stats or {}).get("hist") if baseline else None,
        }

    # 5) alert history
    alerts = [
        {"created_at": str(a.created_at), "level": a.level, "kind": a.kind,
         "detail": a.detail}
        for a in metrics_store.get_recent_alerts(limit=50)
    ]

    # 6) retraining recommendation (computed on demand, not persisted)
    recommendation = recommend(model, persist=False)

    return {
        "model_version": version,
        "train_ic": baseline.train_ic if baseline else None,
        "rolling_ic": rolling_ic,
        "ic_information_ratio": ic_ir,
        "feature_psi": feature_psi,
        "prediction_drift": prediction_drift,
        "alerts": alerts,
        "recommendation": recommendation,
    }
