"""
Database engine and table creation.

DATABASE_URL is read from the environment. On Render, it arrives as
  postgres://user:pass@host:5432/dbname
SQLAlchemy requires the postgresql:// scheme, so we convert it.

If DATABASE_URL is not set the engine is None and the app runs in
"no-database" mode — all ML/analysis endpoints still work.
"""

import os
import logging
from sqlmodel import SQLModel, create_engine

logger = logging.getLogger(__name__)

_raw_url = os.environ.get("DATABASE_URL", "")

if _raw_url:
    # Render supplies postgres://, SQLAlchemy needs postgresql://
    DATABASE_URL = _raw_url.replace("postgres://", "postgresql://", 1)
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,   # drops stale connections before use
        pool_size=5,
        max_overflow=10,
        echo=False,
    )
    logger.info("Database engine created")
else:
    DATABASE_URL = None
    engine = None
    logger.info("DATABASE_URL not set — running without database")


def create_db_and_tables():
    """Create all tables defined via SQLModel. Called on startup when no migration tool runs."""
    if engine is None:
        return
    SQLModel.metadata.create_all(engine)
