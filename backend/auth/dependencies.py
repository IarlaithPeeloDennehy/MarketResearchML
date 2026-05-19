"""
FastAPI dependencies for authentication.

get_current_user  — raises 401 if the request carries no valid session cookie
get_optional_user — returns None instead of raising (for endpoints that work
                    both authenticated and anonymous)

Cookie name: "session"
  - HttpOnly: JS cannot read it (XSS protection)
  - SameSite=Lax: sent on same-origin navigations, blocked on cross-site POSTs
  - Secure flag is set by the auth router when the request comes over HTTPS

Flow
────
1. Read "session" cookie from the request
2. SHA-256 hash the raw token value
3. Look up the hash in the sessions table
4. Verify the session is not expired and not revoked
5. Verify the owning user is active
6. Return the User row
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import Cookie, Depends, HTTPException, Request
from sqlmodel import Session, select

from auth.models import User, UserSession
from auth.utils import hash_token
from db.session import get_db


def _resolve_user(
    session_cookie: Optional[str],
    db: Session,
) -> Optional[User]:
    """Core lookup — returns the User or None. Raises nothing."""
    if not session_cookie:
        return None

    token_hash = hash_token(session_cookie)

    stmt = select(UserSession).where(
        UserSession.token_hash == token_hash,
        UserSession.revoked == False,  # noqa: E712
    )
    user_session = db.exec(stmt).first()

    if user_session is None:
        return None

    # Check expiry (expires_at is stored as UTC-aware)
    if user_session.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        return None

    user = db.get(User, user_session.user_id)
    if user is None or not user.is_active:
        return None

    return user


def get_current_user(
    request: Request,
    session: Optional[str] = Cookie(default=None),
    db: Session = Depends(get_db),
) -> User:
    """Dependency that requires an authenticated user. Raises 401 otherwise."""
    user = _resolve_user(session, db)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated. Please log in.",
            headers={"WWW-Authenticate": "Cookie"},
        )
    return user


def get_optional_user(
    request: Request,
    session: Optional[str] = Cookie(default=None),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Dependency that returns the current user or None (no 401)."""
    return _resolve_user(session, db)
