"""
SQLModel table definitions.

Tables
──────
  users              — account credentials + profile
  sessions           — server-side session tokens (httpOnly cookies)
  user_preferences   — per-user UI / analysis settings (1-to-1 with users)
  saved_analyses     — analyses the user has bookmarked
  activity_events    — append-only audit log (logins, analyses, etc.)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel, Column
from sqlalchemy import JSON, Text, String


# ── helpers ────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _uuid() -> str:
    return str(uuid4())


# ── users ──────────────────────────────────────────────────────────────────

class User(SQLModel, table=True):
    __tablename__ = "users"

    id:            str      = Field(default_factory=_uuid, primary_key=True)
    email:         str      = Field(unique=True, index=True, max_length=254)
    password_hash: str      = Field(sa_column=Column(Text, nullable=False))
    display_name:  Optional[str] = Field(default=None, max_length=100)
    created_at:    datetime = Field(default_factory=_now)
    last_login_at: Optional[datetime] = Field(default=None)
    is_active:     bool     = Field(default=True)


# ── sessions ───────────────────────────────────────────────────────────────

class UserSession(SQLModel, table=True):
    __tablename__ = "sessions"

    id:         str      = Field(default_factory=_uuid, primary_key=True)
    user_id:    str      = Field(foreign_key="users.id", index=True)
    token_hash: str      = Field(sa_column=Column(String(64), unique=True, index=True, nullable=False))  # SHA-256 hex
    created_at: datetime = Field(default_factory=_now)
    expires_at: datetime
    ip_address: Optional[str] = Field(default=None, max_length=45)
    user_agent: Optional[str] = Field(default=None, sa_column=Column(Text))
    revoked:    bool     = Field(default=False)


# ── user_preferences ───────────────────────────────────────────────────────

class UserPreferences(SQLModel, table=True):
    __tablename__ = "user_preferences"

    user_id:        str  = Field(foreign_key="users.id", primary_key=True)
    default_profile: str = Field(default="quality", max_length=50)
    default_risk:    str = Field(default="medium",  max_length=50)
    lookback_years:  int = Field(default=5)
    # Stored as JSON array, e.g. ["AAPL", "MSFT", "GOOG"]
    default_tickers: Optional[list] = Field(default=None, sa_column=Column(JSON))
    # Current holdings for the diversification view, e.g.
    # [{"ticker": "AAPL", "weight": 25.0}, {"ticker": "JNJ", "weight": null}]
    portfolio:       Optional[list] = Field(default=None, sa_column=Column(JSON))
    updated_at:      datetime = Field(default_factory=_now)


# ── saved_analyses ─────────────────────────────────────────────────────────

class SavedAnalysis(SQLModel, table=True):
    __tablename__ = "saved_analyses"

    id:         str      = Field(default_factory=_uuid, primary_key=True)
    user_id:    str      = Field(foreign_key="users.id", index=True)
    name:       str      = Field(max_length=200)
    profile:    str      = Field(max_length=50)
    risk:       str      = Field(max_length=50)
    # JSON arrays / objects — avoid PostgreSQL ARRAY for SQLModel compat
    tickers:    list     = Field(sa_column=Column(JSON, nullable=False))
    results:    dict     = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=_now)


# ── activity_events ────────────────────────────────────────────────────────

class ActivityEvent(SQLModel, table=True):
    __tablename__ = "activity_events"

    # BIGSERIAL — better insert performance than UUID for append-only tables
    id:         Optional[int] = Field(default=None, primary_key=True)
    user_id:    str           = Field(foreign_key="users.id", index=True)
    event_type: str           = Field(max_length=50)   # e.g. "login", "analyse", "backtest"
    payload:    Optional[dict] = Field(default=None, sa_column=Column(JSON))
    created_at: datetime      = Field(default_factory=_now)
    ip_address: Optional[str] = Field(default=None, max_length=45)
