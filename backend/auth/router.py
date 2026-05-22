"""
Auth routes: /auth/signup  /auth/login  /auth/logout  /auth/me

Cookie strategy
───────────────
  Name:     session
  HttpOnly: true  — JS cannot read it (XSS protection)
  SameSite: Lax   — sent on same-origin navigations, blocked on cross-site POST
  Secure:   true when request arrives over HTTPS (checked via X-Forwarded-Proto
            set by Render's proxy, or request.url.scheme)
  Max-Age:  matches token TTL (24h or 30d)
  Path:     /     — cookie sent to every endpoint

Rate limiting
─────────────
  Applied via the slowapi limiter injected from main.py.
  Signup:  5 requests / minute per IP
  Login:   10 requests / minute per IP
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlmodel import Session, select

from auth.models import User, UserPreferences, UserSession, ActivityEvent
from auth.utils import (
    generate_token, hash_password, hash_token,
    token_expiry, verify_password,
)
from auth.dependencies import get_current_user
from db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

# Shares the same limiter instance registered on app.state in main.py
limiter = Limiter(key_func=get_remote_address)

_COOKIE_NAME = "session"


# ── helpers ────────────────────────────────────────────────────────────────

def _is_secure(request: Request) -> bool:
    """True when the request arrived over HTTPS (Render sets X-Forwarded-Proto)."""
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return proto == "https"


def _set_session_cookie(response: Response, token: str, remember_me: bool, request: Request):
    expiry = token_expiry(remember_me)
    max_age = int((expiry - datetime.now(timezone.utc)).total_seconds())
    response.set_cookie(
        key=_COOKIE_NAME,
        value=token,
        httponly=True,
        samesite="lax",
        secure=_is_secure(request),
        max_age=max_age,
        path="/",
    )


def _get_client_ip(request: Request) -> str | None:
    """Return the client IP from X-Forwarded-For (first entry only) or the
    direct connection address. Capped at 45 chars to cover the longest valid
    IPv6 address and prevent forged headers from writing junk to the database."""
    header = request.headers.get("x-forwarded-for")
    if header:
        ip = header.split(",")[0].strip()
    elif request.client:
        ip = request.client.host
    else:
        return None
    return ip[:45]


def _log_event(db: Session, user_id: str, event_type: str,
               request: Request, payload: Optional[dict] = None):
    ip = _get_client_ip(request)
    db.add(ActivityEvent(
        user_id=user_id,
        event_type=event_type,
        payload=payload,
        ip_address=ip,
    ))


# ── schemas ────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    email:        str
    password:     str
    display_name: Optional[str] = None

    @field_validator("email")
    @classmethod
    def email_lower(cls, v: str) -> str:
        v = v.strip().lower()
        if "@" not in v or "." not in v.split("@")[-1]:
            raise ValueError("Please enter a valid email address.")
        if len(v) > 254:
            raise ValueError("Email address is too long.")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v


class LoginRequest(BaseModel):
    email:       str
    password:    str
    remember_me: bool = False

    @field_validator("email")
    @classmethod
    def email_lower(cls, v: str) -> str:
        return v.strip().lower()


class UserOut(BaseModel):
    id:           str
    email:        str
    display_name: Optional[str]
    created_at:   datetime


# ── POST /auth/signup ──────────────────────────────────────────────────────

@router.post("/signup", status_code=201)
@limiter.limit("5/minute")
def signup(
    body: SignupRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    # Check for duplicate email
    existing = db.exec(select(User).where(User.email == body.email)).first()
    if existing:
        raise HTTPException(400, "An account with that email already exists.")

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        display_name=body.display_name,
    )
    db.add(user)
    db.flush()  # get user.id before creating dependent rows

    # Create default preferences row
    db.add(UserPreferences(user_id=user.id))

    # Create session
    token = generate_token()
    db.add(UserSession(
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=token_expiry(remember_me=False),
        ip_address=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    ))

    _log_event(db, user.id, "signup", request)
    db.commit()

    _set_session_cookie(response, token, remember_me=False, request=request)

    return {"status": "ok", "user": UserOut(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
    )}


# ── POST /auth/login ───────────────────────────────────────────────────────

@router.post("/login")
@limiter.limit("10/minute")
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    user = db.exec(select(User).where(User.email == body.email)).first()

    # Constant-time rejection — verify even on miss to prevent timing attacks
    dummy_hash = "$2b$12$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    candidate_hash = user.password_hash if user else dummy_hash
    valid = verify_password(body.password, candidate_hash)

    if not user or not valid or not user.is_active:
        raise HTTPException(401, "Invalid email or password.")

    # Create new session
    token = generate_token()
    db.add(UserSession(
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=token_expiry(body.remember_me),
        ip_address=_get_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    ))

    user.last_login_at = datetime.now(timezone.utc)
    _log_event(db, user.id, "login", request)
    db.commit()

    _set_session_cookie(response, token, body.remember_me, request)

    return {"status": "ok", "user": UserOut(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
        created_at=user.created_at,
    )}


# ── POST /auth/logout ──────────────────────────────────────────────────────

@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    raw_token = request.cookies.get(_COOKIE_NAME)
    if raw_token:
        token_hash = hash_token(raw_token)
        stmt = select(UserSession).where(UserSession.token_hash == token_hash)
        user_session = db.exec(stmt).first()
        if user_session:
            user_session.revoked = True
            _log_event(db, user_session.user_id, "logout", request)
            db.commit()

    response.delete_cookie(_COOKIE_NAME, path="/")
    return {"status": "ok"}


# ── GET /auth/me ───────────────────────────────────────────────────────────

@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {"status": "ok", "user": UserOut(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        created_at=current_user.created_at,
    )}


# ── PATCH /auth/me ─────────────────────────────────────────────────────────

class UpdateMeRequest(BaseModel):
    display_name: Optional[str] = None

    @field_validator("display_name")
    @classmethod
    def name_clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.strip()
        if len(v) > 100:
            raise ValueError("Display name must be 100 characters or fewer.")
        return v or None   # empty string → None


@router.patch("/me")
def update_me(
    body: UpdateMeRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if body.display_name is not None:
        current_user.display_name = body.display_name
    _log_event(db, current_user.id, "profile_updated", request,
               {"fields": list(body.model_dump(exclude_none=True).keys())})
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return {"status": "ok", "user": UserOut(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        created_at=current_user.created_at,
    )}


# ── POST /auth/change-password ─────────────────────────────────────────────

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password:     str

    @field_validator("new_password")
    @classmethod
    def new_pwd_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("New password must be at least 8 characters.")
        return v


@router.post("/change-password")
@limiter.limit("5/minute")
def change_password(
    body: ChangePasswordRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Requires the current password to prevent account takeover via a stolen cookie.
    Revokes all OTHER active sessions so stolen sessions stop working immediately.
    Keeps the caller's current session alive — user stays logged in.
    """
    if not verify_password(body.current_password, current_user.password_hash):
        raise HTTPException(400, "Current password is incorrect.")

    if body.current_password == body.new_password:
        raise HTTPException(400, "New password must differ from the current one.")

    # Identify the caller's session so we can preserve it
    raw_token    = request.cookies.get(_COOKIE_NAME, "")
    current_hash = hash_token(raw_token) if raw_token else None

    # Revoke every other active session
    stmt = select(UserSession).where(
        UserSession.user_id == current_user.id,
        UserSession.revoked  == False,  # noqa: E712
    )
    for s in db.exec(stmt).all():
        if s.token_hash != current_hash:
            s.revoked = True

    current_user.password_hash = hash_password(body.new_password)
    _log_event(db, current_user.id, "password_changed", request)
    db.add(current_user)
    db.commit()

    return {"status": "ok", "message": "Password changed. Other sessions have been signed out."}


# ── DELETE /auth/me ────────────────────────────────────────────────────────

class DeleteAccountRequest(BaseModel):
    password: str   # must confirm identity before destructive action


@router.delete("/me", status_code=200)
@limiter.limit("3/minute")
def delete_account(
    body: DeleteAccountRequest,
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Permanently delete the authenticated user's account.

    Requires password confirmation so a stolen session cookie alone cannot
    trigger deletion. All related rows (sessions, preferences, saved analyses,
    activity events) are removed by ON DELETE CASCADE in the migration.
    """
    if not verify_password(body.password, current_user.password_hash):
        raise HTTPException(400, "Incorrect password.")

    # Log before deletion so the event is captured (will cascade-delete anyway,
    # but gives a final audit entry if we ever restore from backup).
    _log_event(db, current_user.id, "account_deleted", request)
    db.flush()   # write the event before the user row is gone

    db.delete(current_user)
    db.commit()

    response.delete_cookie(_COOKIE_NAME, path="/")
    return {"status": "ok", "message": "Account permanently deleted."}
