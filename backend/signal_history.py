"""
signal_history.py
─────────────────
Lightweight per-user signal history for STATEFUL hold discipline.

When a logged-in user runs /analyse we record the per-ticker signal they were
shown, and on a later /analyse we read back the most recent prior signal for
each requested ticker. scoring._apply_hysteresis then uses that to soften a
fresh SELL on a name the user recently held as a BUY (anti-panic-sell).

Storage reuses the existing `ActivityEvent` table (JSON payload, 90-day
retention) — no new table / migration. Everything here is wrapped so a missing
DB or query error can never break /analyse: reads return {} and writes no-op.

The parsing/building logic is split into pure functions (build_signal_payload,
extract_prior_signals) so it is unit-testable without a database.
"""
from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

ANALYSE_EVENT = "analyse_signals"
_MAX_EVENTS   = 60   # how many recent analyse events to scan for prior signals


# ── pure helpers (no DB) ─────────────────────────────────────────────────────

def build_signal_payload(results: list[dict]) -> dict:
    """Compact {ticker: {signal, score}} payload from score_universe results."""
    signals = {}
    for r in results or []:
        t = r.get("ticker")
        if not t or not r.get("signal"):
            continue
        signals[str(t)] = {"signal": r.get("signal"), "score": r.get("composite_score")}
    return {"signals": signals}


def _fmt_date(dt) -> str | None:
    try:
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        return dt.strftime("%d %b %Y")   # e.g. "12 Jun 2026" (Windows-safe)
    except Exception:
        return None


def extract_prior_signals(events, tickers) -> dict:
    """Most recent prior signal per requested ticker.

    `events` is an iterable of (payload_dict, created_at) ordered MOST RECENT
    FIRST. Returns {ticker: {signal, score, date}} for tickers found.
    """
    want = {str(t) for t in (tickers or [])}
    out: dict[str, dict] = {}
    for payload, created_at in events:
        if not want:
            break
        sig_map  = (payload or {}).get("signals") or {}
        date_str = _fmt_date(created_at)
        for t in list(want):
            entry = sig_map.get(t)
            if entry and entry.get("signal"):
                out[t] = {"signal": entry.get("signal"),
                          "score":  entry.get("score"),
                          "date":   date_str}
                want.discard(t)
    return out


# ── DB wrappers (graceful: never raise into /analyse) ────────────────────────

def load_prior_signals(db, user_id: str, tickers: list[str]) -> dict:
    """Read the user's most recent prior signal for each ticker. {} on any error."""
    try:
        from sqlmodel import select
        from auth.models import ActivityEvent
        rows = db.exec(
            select(ActivityEvent)
            .where(ActivityEvent.user_id == user_id)
            .where(ActivityEvent.event_type == ANALYSE_EVENT)
            .order_by(ActivityEvent.created_at.desc())
            .limit(_MAX_EVENTS)
        ).all()
        return extract_prior_signals([(r.payload, r.created_at) for r in rows], tickers)
    except Exception as exc:
        logger.warning(f"[signal_history] load skipped (non-fatal): {exc}")
        return {}


def record_signals(db, user_id: str, results: list[dict], ip: str | None = None) -> None:
    """Persist this analyse's per-ticker signals. No-op on any error."""
    try:
        from auth.models import ActivityEvent
        payload = build_signal_payload(results)
        if not payload.get("signals"):
            return
        db.add(ActivityEvent(user_id=user_id, event_type=ANALYSE_EVENT,
                             payload=payload, ip_address=ip))
        db.commit()
    except Exception as exc:
        logger.warning(f"[signal_history] record skipped (non-fatal): {exc}")
        try:
            db.rollback()
        except Exception:
            pass
