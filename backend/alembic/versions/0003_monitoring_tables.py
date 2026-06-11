"""Model edge monitoring tables (additive — no existing table touched)

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-11
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _existing_tables() -> set:
    conn = op.get_bind()
    return set(sa.inspect(conn).get_table_names())


def upgrade() -> None:
    existing = _existing_tables()

    if "monitoring_baseline" not in existing:
        op.create_table(
            "monitoring_baseline",
            sa.Column("id",               sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("model_version",    sa.Text,    nullable=False),
            sa.Column("created_at",       sa.DateTime(timezone=True), nullable=False),
            sa.Column("train_ic",         sa.Float,   nullable=True),
            sa.Column("n_features",       sa.Integer, nullable=False, server_default="0"),
            sa.Column("feature_stats",    sa.JSON,    nullable=True),
            sa.Column("prediction_stats", sa.JSON,    nullable=True),
        )
        op.create_index("ix_monitoring_baseline_model_version", "monitoring_baseline", ["model_version"])

    if "monitoring_predictions" not in existing:
        op.create_table(
            "monitoring_predictions",
            sa.Column("id",              sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("model_version",   sa.Text,    nullable=False),
            sa.Column("period_key",      sa.Text,    nullable=False),
            sa.Column("predicted_at",    sa.DateTime(timezone=True), nullable=False),
            sa.Column("ticker",          sa.Text,    nullable=False),
            sa.Column("prediction",      sa.Float,   nullable=False),
            sa.Column("horizon_days",    sa.Integer, nullable=False, server_default="252"),
            sa.Column("realized_return", sa.Float,   nullable=True),
            sa.Column("matured",         sa.Boolean, nullable=False, server_default="false"),
            sa.Column("matured_at",      sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index("ix_monitoring_predictions_model_version", "monitoring_predictions", ["model_version"])
        op.create_index("ix_monitoring_predictions_period_key", "monitoring_predictions", ["period_key"])
        op.create_index("ix_monitoring_predictions_matured", "monitoring_predictions", ["matured"])

    if "monitoring_ic" not in existing:
        op.create_table(
            "monitoring_ic",
            sa.Column("id",            sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("model_version", sa.Text,    nullable=False),
            sa.Column("computed_at",   sa.DateTime(timezone=True), nullable=False),
            sa.Column("window",        sa.Integer, nullable=False),
            sa.Column("spearman_ic",   sa.Float,   nullable=True),
            sa.Column("pearson_ic",    sa.Float,   nullable=True),
            sa.Column("ic_mean",       sa.Float,   nullable=True),
            sa.Column("ic_std",        sa.Float,   nullable=True),
            sa.Column("ic_ir",         sa.Float,   nullable=True),
            sa.Column("n_obs",         sa.Integer, nullable=False, server_default="0"),
        )
        op.create_index("ix_monitoring_ic_model_version", "monitoring_ic", ["model_version"])

    if "monitoring_drift" not in existing:
        op.create_table(
            "monitoring_drift",
            sa.Column("id",            sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("model_version", sa.Text,    nullable=False),
            sa.Column("period_key",    sa.Text,    nullable=False),
            sa.Column("computed_at",   sa.DateTime(timezone=True), nullable=False),
            sa.Column("feature",       sa.Text,    nullable=False),
            sa.Column("psi",           sa.Float,   nullable=False, server_default="0"),
            sa.Column("status",        sa.Text,    nullable=False, server_default="stable"),
            sa.Column("mean",          sa.Float,   nullable=True),
            sa.Column("std",           sa.Float,   nullable=True),
            sa.Column("missing_rate",  sa.Float,   nullable=True),
            sa.Column("p5",            sa.Float,   nullable=True),
            sa.Column("p25",           sa.Float,   nullable=True),
            sa.Column("p50",           sa.Float,   nullable=True),
            sa.Column("p75",           sa.Float,   nullable=True),
            sa.Column("p95",           sa.Float,   nullable=True),
        )
        op.create_index("ix_monitoring_drift_model_version", "monitoring_drift", ["model_version"])
        op.create_index("ix_monitoring_drift_period_key", "monitoring_drift", ["period_key"])

    if "monitoring_prediction_drift" not in existing:
        op.create_table(
            "monitoring_prediction_drift",
            sa.Column("id",            sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("model_version", sa.Text,    nullable=False),
            sa.Column("period_key",    sa.Text,    nullable=False),
            sa.Column("computed_at",   sa.DateTime(timezone=True), nullable=False),
            sa.Column("psi",           sa.Float,   nullable=False, server_default="0"),
            sa.Column("status",        sa.Text,    nullable=False, server_default="stable"),
            sa.Column("mean",          sa.Float,   nullable=True),
            sa.Column("std",           sa.Float,   nullable=True),
            sa.Column("p5",            sa.Float,   nullable=True),
            sa.Column("p25",           sa.Float,   nullable=True),
            sa.Column("p50",           sa.Float,   nullable=True),
            sa.Column("p75",           sa.Float,   nullable=True),
            sa.Column("p95",           sa.Float,   nullable=True),
            sa.Column("hist",          sa.JSON,    nullable=True),
        )
        op.create_index("ix_monitoring_prediction_drift_model_version", "monitoring_prediction_drift", ["model_version"])
        op.create_index("ix_monitoring_prediction_drift_period_key", "monitoring_prediction_drift", ["period_key"])

    if "monitoring_alerts" not in existing:
        op.create_table(
            "monitoring_alerts",
            sa.Column("id",            sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("model_version", sa.Text,    nullable=False),
            sa.Column("created_at",    sa.DateTime(timezone=True), nullable=False),
            sa.Column("level",         sa.Text,    nullable=False),
            sa.Column("kind",          sa.Text,    nullable=False),
            sa.Column("period_key",    sa.Text,    nullable=True),
            sa.Column("detail",        sa.JSON,    nullable=True),
        )
        op.create_index("ix_monitoring_alerts_model_version", "monitoring_alerts", ["model_version"])
        op.create_index("ix_monitoring_alerts_created_at", "monitoring_alerts", ["created_at"])


def downgrade() -> None:
    for tbl in (
        "monitoring_alerts",
        "monitoring_prediction_drift",
        "monitoring_drift",
        "monitoring_ic",
        "monitoring_predictions",
        "monitoring_baseline",
    ):
        op.drop_table(tbl)
