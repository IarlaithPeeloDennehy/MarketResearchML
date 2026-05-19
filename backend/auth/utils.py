"""
Auth utilities — password hashing and session token helpers.

Design decisions
────────────────
- passlib[bcrypt] for password hashing (industry standard, auto-salted)
- Session tokens are 32 random bytes → 64-char hex string (256 bits of entropy)
- Only the SHA-256 hash of the token is stored in the DB — same principle as
  password hashing: a DB leak doesn't expose live session tokens
- Token lifetime: 30 days for "remember me", 24 hours otherwise
"""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt as _bcrypt

SESSION_TTL_SHORT = timedelta(hours=24)   # standard session
SESSION_TTL_LONG  = timedelta(days=30)    # "remember me"


# ── passwords ──────────────────────────────────────────────────────────────
# bcrypt truncates at 72 bytes. SHA-256 pre-hashing collapses any password
# to a 64-char hex string, lifting that limit without weakening security.

def _prehash(plain: str) -> str:
    return hashlib.sha256(plain.encode("utf-8")).hexdigest()


def hash_password(plain: str) -> str:
    return _bcrypt.hashpw(_prehash(plain).encode(), _bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt.checkpw(_prehash(plain).encode(), hashed.encode())


# ── session tokens ─────────────────────────────────────────────────────────

def generate_token() -> str:
    """Return a 256-bit cryptographically random hex token (64 chars)."""
    return secrets.token_hex(32)


def hash_token(token: str) -> str:
    """SHA-256 hex digest of a session token — this is what gets stored."""
    return hashlib.sha256(token.encode()).hexdigest()


def token_expiry(remember_me: bool = False) -> datetime:
    ttl = SESSION_TTL_LONG if remember_me else SESSION_TTL_SHORT
    return datetime.now(timezone.utc) + ttl
