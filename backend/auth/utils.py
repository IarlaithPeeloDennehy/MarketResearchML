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

from passlib.context import CryptContext

# bcrypt rounds=12 is the passlib default — ~250ms on modern hardware,
# which is slow enough to deter brute force but fast enough for login UX
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SESSION_TTL_SHORT = timedelta(hours=24)   # standard session
SESSION_TTL_LONG  = timedelta(days=30)    # "remember me"


# ── passwords ──────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


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
