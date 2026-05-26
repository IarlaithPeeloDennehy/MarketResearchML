"""sessions.token_hash: add UNIQUE constraint

Revision ID: 0002
Revises: 0001
Create Date: 2026-05-26
"""
from alembic import op
import sqlalchemy as sa

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop the plain (non-unique) index created in 0001
    op.drop_index("ix_sessions_token_hash", table_name="sessions")
    # Create a new unique index — also serves as the lookup index
    op.create_index(
        "ix_sessions_token_hash",
        "sessions",
        ["token_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_sessions_token_hash", table_name="sessions")
    op.create_index(
        "ix_sessions_token_hash",
        "sessions",
        ["token_hash"],
        unique=False,
    )
