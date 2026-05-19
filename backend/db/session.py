"""
FastAPI dependency that yields a database session.

Usage in a route:
    from db.session import get_db
    from sqlmodel import Session

    @app.get("/example")
    def example(db: Session = Depends(get_db)):
        ...

If the database is not configured the dependency raises 503 immediately,
so ML-only deployments (no DATABASE_URL) still work for all other endpoints.
"""

from typing import Generator
from fastapi import HTTPException
from sqlmodel import Session
from db.base import engine


def get_db() -> Generator[Session, None, None]:
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL to enable this feature.",
        )
    with Session(engine) as session:
        yield session
