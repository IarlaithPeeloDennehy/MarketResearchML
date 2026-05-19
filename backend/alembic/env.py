"""
Alembic migration environment.

Reads DATABASE_URL from the environment (same Render variable the app uses),
converting the postgres:// prefix to postgresql:// if necessary.

All SQLModel table models must be imported here so their metadata is visible
to autogenerate. Import auth.models to register the tables.
"""

import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Make sure the backend package root is on sys.path so imports work
# when alembic is invoked from the backend/ directory.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Import models so SQLModel.metadata is populated
import auth.models  # noqa: F401
from sqlmodel import SQLModel

config = context.config

# Alembic logging config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata


def _get_url() -> str:
    raw = os.environ.get("DATABASE_URL", "")
    if not raw:
        raise RuntimeError(
            "DATABASE_URL environment variable is not set. "
            "Set it before running alembic."
        )
    return raw.replace("postgres://", "postgresql://", 1)


def run_migrations_offline() -> None:
    url = _get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    cfg = config.get_section(config.config_ini_section, {})
    cfg["sqlalchemy.url"] = _get_url()

    connectable = engine_from_config(
        cfg,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,  # single-use connection for migrations
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
