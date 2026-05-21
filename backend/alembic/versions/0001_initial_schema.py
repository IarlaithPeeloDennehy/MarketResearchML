"""Initial schema — users, sessions, preferences, analyses, activity

Revision ID: 0001
Revises:
Create Date: 2026-05-19
"""

from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _existing_tables() -> set:
    conn = op.get_bind()
    return set(sa.inspect(conn).get_table_names())


def upgrade() -> None:
    existing = _existing_tables()

    # ── users ──────────────────────────────────────────────────────────────
    if "users" not in existing:
        op.create_table(
            "users",
            sa.Column("id",            sa.Text,    primary_key=True),
            sa.Column("email",         sa.Text,    nullable=False),
            sa.Column("password_hash", sa.Text,    nullable=False),
            sa.Column("display_name",  sa.Text,    nullable=True),
            sa.Column("created_at",    sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("is_active",     sa.Boolean, nullable=False, server_default="true"),
        )
        op.create_index("ix_users_email", "users", ["email"], unique=True)

    # ── sessions ───────────────────────────────────────────────────────────
    if "sessions" not in existing:
        op.create_table(
            "sessions",
            sa.Column("id",         sa.Text,    primary_key=True),
            sa.Column("user_id",    sa.Text,    sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("token_hash", sa.Text,    nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("ip_address", sa.Text,    nullable=True),
            sa.Column("user_agent", sa.Text,    nullable=True),
            sa.Column("revoked",    sa.Boolean, nullable=False, server_default="false"),
        )
        op.create_index("ix_sessions_user_id",    "sessions", ["user_id"])
        op.create_index("ix_sessions_token_hash", "sessions", ["token_hash"])

    # ── user_preferences ───────────────────────────────────────────────────
    if "user_preferences" not in existing:
        op.create_table(
            "user_preferences",
            sa.Column("user_id",         sa.Text,    sa.ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("default_profile", sa.Text,    nullable=False, server_default="quality"),
            sa.Column("default_risk",    sa.Text,    nullable=False, server_default="medium"),
            sa.Column("lookback_years",  sa.Integer, nullable=False, server_default="5"),
            sa.Column("default_tickers", sa.JSON,    nullable=True),
            sa.Column("updated_at",      sa.DateTime(timezone=True), nullable=False),
        )

    # ── saved_analyses ─────────────────────────────────────────────────────
    if "saved_analyses" not in existing:
        op.create_table(
            "saved_analyses",
            sa.Column("id",         sa.Text,    primary_key=True),
            sa.Column("user_id",    sa.Text,    sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("name",       sa.Text,    nullable=False),
            sa.Column("profile",    sa.Text,    nullable=False),
            sa.Column("risk",       sa.Text,    nullable=False),
            sa.Column("tickers",    sa.JSON,    nullable=False),
            sa.Column("results",    sa.JSON,    nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_saved_analyses_user_id", "saved_analyses", ["user_id"])

    # ── activity_events ────────────────────────────────────────────────────
    if "activity_events" not in existing:
        op.create_table(
            "activity_events",
            sa.Column("id",         sa.BigInteger, primary_key=True, autoincrement=True),
            sa.Column("user_id",    sa.Text,       sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
            sa.Column("event_type", sa.Text,       nullable=False),
            sa.Column("payload",    sa.JSON,       nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("ip_address", sa.Text,       nullable=True),
        )
        op.create_index("ix_activity_events_user_id",    "activity_events", ["user_id"])
        op.create_index("ix_activity_events_created_at", "activity_events", ["created_at"])


def downgrade() -> None:
    op.drop_table("activity_events")
    op.drop_table("saved_analyses")
    op.drop_table("user_preferences")
    op.drop_table("sessions")
    op.drop_table("users")
