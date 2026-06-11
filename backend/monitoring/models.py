"""
SQLModel tables for model edge monitoring (additive — no existing table touched).

  monitoring_baseline          — per model_version: training feature/prediction
                                 distribution baselines + reference (training) IC
  monitoring_predictions       — logged predictions awaiting realized labels
  monitoring_ic                — rolling IC stats computed after labels mature
  monitoring_drift             — per-period feature distribution + PSI vs baseline
  monitoring_prediction_drift  — per-period prediction distribution + PSI
  monitoring_alerts            — alert + retraining-recommendation history

All keyed by ``model_version`` so metrics are attributable across retrains.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel, Column
from sqlalchemy import JSON, Text


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MonitoringBaseline(SQLModel, table=True):
    __tablename__ = "monitoring_baseline"

    id:            Optional[int] = Field(default=None, primary_key=True)
    model_version: str           = Field(index=True, max_length=128)
    created_at:    datetime       = Field(default_factory=_now)
    train_ic:      Optional[float] = Field(default=None)
    n_features:    int            = Field(default=0)
    # {feature: {mean,std,missing_rate,p5..p95, edges:[...], props:[...]}}
    feature_stats: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    # {mean,std,p5..p95, edges:[...], props:[...], hist:{edges,counts}}
    prediction_stats: Optional[dict] = Field(default=None, sa_column=Column(JSON))


class MonitoringPrediction(SQLModel, table=True):
    __tablename__ = "monitoring_predictions"

    id:            Optional[int] = Field(default=None, primary_key=True)
    model_version: str           = Field(index=True, max_length=128)
    period_key:    str           = Field(index=True, max_length=32)   # batch grouping, e.g. YYYY-MM-DD
    predicted_at:  datetime       = Field(default_factory=_now)
    ticker:        str           = Field(max_length=32)
    prediction:    float
    horizon_days:  int           = Field(default=252)
    realized_return: Optional[float] = Field(default=None)
    matured:       bool          = Field(default=False, index=True)
    matured_at:    Optional[datetime] = Field(default=None)


class MonitoringIC(SQLModel, table=True):
    __tablename__ = "monitoring_ic"

    id:            Optional[int] = Field(default=None, primary_key=True)
    model_version: str           = Field(index=True, max_length=128)
    computed_at:   datetime       = Field(default_factory=_now)
    window:        int            # rolling window size (20/60/120)
    spearman_ic:   Optional[float] = Field(default=None)
    pearson_ic:    Optional[float] = Field(default=None)
    ic_mean:       Optional[float] = Field(default=None)
    ic_std:        Optional[float] = Field(default=None)
    ic_ir:         Optional[float] = Field(default=None)
    n_obs:         int            = Field(default=0)


class MonitoringDrift(SQLModel, table=True):
    __tablename__ = "monitoring_drift"

    id:            Optional[int] = Field(default=None, primary_key=True)
    model_version: str           = Field(index=True, max_length=128)
    period_key:    str           = Field(index=True, max_length=32)
    computed_at:   datetime       = Field(default_factory=_now)
    feature:       str           = Field(max_length=64)
    psi:           float          = Field(default=0.0)
    status:        str           = Field(default="stable", max_length=16)
    mean:          Optional[float] = Field(default=None)
    std:           Optional[float] = Field(default=None)
    missing_rate:  Optional[float] = Field(default=None)
    p5:            Optional[float] = Field(default=None)
    p25:           Optional[float] = Field(default=None)
    p50:           Optional[float] = Field(default=None)
    p75:           Optional[float] = Field(default=None)
    p95:           Optional[float] = Field(default=None)


class MonitoringPredictionDrift(SQLModel, table=True):
    __tablename__ = "monitoring_prediction_drift"

    id:            Optional[int] = Field(default=None, primary_key=True)
    model_version: str           = Field(index=True, max_length=128)
    period_key:    str           = Field(index=True, max_length=32)
    computed_at:   datetime       = Field(default_factory=_now)
    psi:           float          = Field(default=0.0)
    status:        str           = Field(default="stable", max_length=16)
    mean:          Optional[float] = Field(default=None)
    std:           Optional[float] = Field(default=None)
    p5:            Optional[float] = Field(default=None)
    p25:           Optional[float] = Field(default=None)
    p50:           Optional[float] = Field(default=None)
    p75:           Optional[float] = Field(default=None)
    p95:           Optional[float] = Field(default=None)
    hist:          Optional[dict] = Field(default=None, sa_column=Column(JSON))


class MonitoringAlert(SQLModel, table=True):
    __tablename__ = "monitoring_alerts"

    id:            Optional[int] = Field(default=None, primary_key=True)
    model_version: str           = Field(index=True, max_length=128)
    created_at:    datetime       = Field(default_factory=_now)
    level:         str           = Field(max_length=16)   # WARNING/CRITICAL/RETRAIN/healthy/watch
    kind:          str           = Field(max_length=24)   # ic / psi / pred_drift / recommendation
    period_key:    Optional[str] = Field(default=None, max_length=32)
    detail:        Optional[dict] = Field(default=None, sa_column=Column(JSON))
