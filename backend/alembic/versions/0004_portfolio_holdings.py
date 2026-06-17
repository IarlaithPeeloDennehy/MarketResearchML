"""Add user_preferences.portfolio JSON column (additive — guarded)

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-17
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _columns(table: str) -> set:
    conn = op.get_bind()
    insp = sa.inspect(conn)
    if table not in insp.get_table_names():
        return set()
    return {c["name"] for c in insp.get_columns(table)}


def upgrade() -> None:
    # Idempotent add-column (mirrors the _existing_tables guard used in 0003):
    # only add when the table exists and the column is missing.
    if "user_preferences" not in sa.inspect(op.get_bind()).get_table_names():
        return
    if "portfolio" not in _columns("user_preferences"):
        op.add_column("user_preferences", sa.Column("portfolio", sa.JSON, nullable=True))


def downgrade() -> None:
    if "portfolio" in _columns("user_preferences"):
        op.drop_column("user_preferences", "portfolio")
