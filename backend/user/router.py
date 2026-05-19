"""
User-facing data endpoints.

/user/preferences  GET  — fetch the current user's preferences
/user/preferences  PUT  — update preferences (partial updates accepted)

/user/analyses     GET  — list saved analyses (newest first)
/user/analyses     POST — save a new analysis (from the last /analyse run)
/user/analyses/{id} DELETE — delete one saved analysis

All endpoints require an authenticated session (get_current_user dependency).
"""

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from auth.dependencies import get_current_user
from auth.models import (
    ActivityEvent, SavedAnalysis, User, UserPreferences,
)
from db.session import get_db

router = APIRouter(prefix="/user", tags=["user"])


# ── shared log helper ──────────────────────────────────────────────────────

def _log(db: Session, user_id: str, event: str, payload: Optional[dict] = None):
    db.add(ActivityEvent(user_id=user_id, event_type=event, payload=payload))


# ══════════════════════════════════════════════════════════════════════════
# PREFERENCES
# ══════════════════════════════════════════════════════════════════════════

class PreferencesOut(BaseModel):
    default_profile:  str
    default_risk:     str
    lookback_years:   int
    default_tickers:  Optional[List[str]]
    updated_at:       datetime


class PreferencesIn(BaseModel):
    default_profile:  Optional[str]   = None
    default_risk:     Optional[str]   = None
    lookback_years:   Optional[int]   = None
    default_tickers:  Optional[List[str]] = None


_VALID_PROFILES = {"quality", "growth", "value", "momentum", "income"}
_VALID_RISKS    = {"low", "medium", "high"}


@router.get("/preferences", response_model=PreferencesOut)
def get_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    prefs = db.get(UserPreferences, current_user.id)
    if prefs is None:
        # Lazily create the row if it was somehow missing
        prefs = UserPreferences(user_id=current_user.id)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)
    return PreferencesOut(
        default_profile=prefs.default_profile,
        default_risk=prefs.default_risk,
        lookback_years=prefs.lookback_years,
        default_tickers=prefs.default_tickers,
        updated_at=prefs.updated_at,
    )


@router.put("/preferences", response_model=PreferencesOut)
def update_preferences(
    body: PreferencesIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if body.default_profile and body.default_profile not in _VALID_PROFILES:
        raise HTTPException(400, f"profile must be one of {sorted(_VALID_PROFILES)}")
    if body.default_risk and body.default_risk not in _VALID_RISKS:
        raise HTTPException(400, f"risk must be one of {sorted(_VALID_RISKS)}")
    if body.lookback_years is not None and not (1 <= body.lookback_years <= 10):
        raise HTTPException(400, "lookback_years must be between 1 and 10")
    if body.default_tickers is not None and len(body.default_tickers) > 40:
        raise HTTPException(400, "Maximum 40 default tickers")

    prefs = db.get(UserPreferences, current_user.id)
    if prefs is None:
        prefs = UserPreferences(user_id=current_user.id)
        db.add(prefs)

    if body.default_profile  is not None: prefs.default_profile  = body.default_profile
    if body.default_risk     is not None: prefs.default_risk     = body.default_risk
    if body.lookback_years   is not None: prefs.lookback_years   = body.lookback_years
    if body.default_tickers  is not None: prefs.default_tickers  = body.default_tickers
    prefs.updated_at = datetime.now(timezone.utc)

    _log(db, current_user.id, "preferences_updated", body.model_dump(exclude_none=True))
    db.commit()
    db.refresh(prefs)

    return PreferencesOut(
        default_profile=prefs.default_profile,
        default_risk=prefs.default_risk,
        lookback_years=prefs.lookback_years,
        default_tickers=prefs.default_tickers,
        updated_at=prefs.updated_at,
    )


# ══════════════════════════════════════════════════════════════════════════
# SAVED ANALYSES
# ══════════════════════════════════════════════════════════════════════════

class AnalysisOut(BaseModel):
    id:         str
    name:       str
    profile:    str
    risk:       str
    tickers:    List[str]
    created_at: datetime
    # results omitted from the list view to keep payload small


class AnalysisDetailOut(AnalysisOut):
    results: dict   # full payload included on individual fetch


class SaveAnalysisIn(BaseModel):
    name:    str
    profile: str
    risk:    str
    tickers: List[str]
    results: dict


@router.get("/analyses/{analysis_id}", response_model=AnalysisDetailOut)
def get_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = db.get(SavedAnalysis, analysis_id)
    if row is None or row.user_id != current_user.id:
        raise HTTPException(404, "Analysis not found.")
    return AnalysisDetailOut(
        id=row.id, name=row.name, profile=row.profile,
        risk=row.risk, tickers=row.tickers,
        created_at=row.created_at, results=row.results,
    )


@router.get("/analyses", response_model=List[AnalysisOut])
def list_analyses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt = (
        select(SavedAnalysis)
        .where(SavedAnalysis.user_id == current_user.id)
        .order_by(SavedAnalysis.created_at.desc())
        .limit(100)
    )
    rows = db.exec(stmt).all()
    return [
        AnalysisOut(
            id=r.id, name=r.name, profile=r.profile,
            risk=r.risk, tickers=r.tickers, created_at=r.created_at,
        )
        for r in rows
    ]


@router.post("/analyses", response_model=AnalysisDetailOut, status_code=201)
def save_analysis(
    body: SaveAnalysisIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not body.name.strip():
        raise HTTPException(400, "name must not be empty")
    if len(body.tickers) < 1:
        raise HTTPException(400, "at least one ticker required")

    row = SavedAnalysis(
        user_id=current_user.id,
        name=body.name.strip(),
        profile=body.profile,
        risk=body.risk,
        tickers=body.tickers,
        results=body.results,
    )
    db.add(row)
    _log(db, current_user.id, "analysis_saved", {"name": body.name, "n_tickers": len(body.tickers)})
    db.commit()
    db.refresh(row)

    return AnalysisDetailOut(
        id=row.id, name=row.name, profile=row.profile,
        risk=row.risk, tickers=row.tickers,
        created_at=row.created_at, results=row.results,
    )


@router.delete("/analyses/{analysis_id}", status_code=204)
def delete_analysis(
    analysis_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    row = db.get(SavedAnalysis, analysis_id)
    if row is None or row.user_id != current_user.id:
        # Return 404 whether not found or belongs to another user —
        # don't reveal that the record exists at all
        raise HTTPException(404, "Analysis not found.")
    _log(db, current_user.id, "analysis_deleted", {"analysis_id": analysis_id})
    db.delete(row)
    db.commit()
