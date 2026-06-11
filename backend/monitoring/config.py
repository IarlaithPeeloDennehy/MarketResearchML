"""
Monitoring thresholds — all env-overridable so they can be tuned without code
changes. Defaults match the spec.
"""
from __future__ import annotations

import os


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _i(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Master switch. When off, every monitoring entry point is a no-op.
ENABLED = os.environ.get("MONITORING_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")

# ── Information Coefficient ─────────────────────────────────────────────────
IC_WARNING_RATIO  = _f("MON_IC_WARNING_RATIO", 0.70)   # rolling IC < 0.70×train → WARNING
IC_CRITICAL_RATIO = _f("MON_IC_CRITICAL_RATIO", 0.50)  # rolling IC < 0.50×train → CRITICAL
IC_RETRAIN_CONSECUTIVE = _i("MON_IC_RETRAIN_CONSECUTIVE", 3)  # N consecutive sub-threshold windows → RETRAIN
ROLLING_WINDOWS = [20, 60, 120]
PRIMARY_WINDOW  = _i("MON_PRIMARY_WINDOW", 60)         # window the alert/recommendation engine reads

# ── Drift (PSI) ─────────────────────────────────────────────────────────────
PSI_WARNING  = _f("MON_PSI_WARNING", 0.10)   # 0.10 ≤ PSI < 0.25 → Warning
PSI_CRITICAL = _f("MON_PSI_CRITICAL", 0.25)  # PSI ≥ 0.25 → Critical
PSI_BINS     = _i("MON_PSI_BINS", 10)
PRED_DRIFT_PSI_CRITICAL = _f("MON_PRED_DRIFT_PSI_CRITICAL", 0.25)

# ── Labels / horizon ────────────────────────────────────────────────────────
# Default forward horizon (trading days) used to mature predictions into
# realized returns. Matches the model's 12-month forward training horizon.
DEFAULT_HORIZON_DAYS = _i("MON_HORIZON_DAYS", 252)

# Retention for monitoring rows (days), mirroring the activity_events purge.
RETENTION_DAYS = _i("MON_RETENTION_DAYS", 730)
