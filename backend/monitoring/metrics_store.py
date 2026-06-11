"""
metrics_store.py — persistence layer for monitoring.

All access goes through here so the rest of monitoring never touches the DB
directly. Every function is exception-safe: on any failure (no DB configured,
connection error, write error) it logs and returns a safe default — monitoring
must never raise into inference or training.

Uses the app's existing Postgres engine (db.base.engine). Tests can inject a
throwaway SQLite engine via ``set_engine``.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

# sqlmodel + the table models are only importable where the DB stack is present
# (it is in production). Locally / in pure-logic tests they may be absent — the
# store then simply reports unavailable and every function no-ops. This mirrors
# the app's existing "no DATABASE_URL → run without DB" behaviour.
try:
    from sqlmodel import Session, select
    from .models import (
        MonitoringBaseline,
        MonitoringPrediction,
        MonitoringIC,
        MonitoringDrift,
        MonitoringPredictionDrift,
        MonitoringAlert,
    )
    _STORE_OK = True
except Exception:  # pragma: no cover - exercised in envs without sqlmodel
    Session = select = None
    MonitoringBaseline = MonitoringPrediction = MonitoringIC = None
    MonitoringDrift = MonitoringPredictionDrift = MonitoringAlert = None
    _STORE_OK = False

logger = logging.getLogger(__name__)

_ENGINE_OVERRIDE = None   # tests set this; production uses db.base.engine


def set_engine(engine) -> None:
    """Override the engine (used by tests)."""
    global _ENGINE_OVERRIDE
    _ENGINE_OVERRIDE = engine


def _engine():
    if not _STORE_OK:
        return None
    if _ENGINE_OVERRIDE is not None:
        return _ENGINE_OVERRIDE
    try:
        from db.base import engine
        return engine
    except Exception:
        return None


def available() -> bool:
    return _engine() is not None


@contextmanager
def _session():
    eng = _engine()
    if eng is None:
        yield None
        return
    s = Session(eng)
    try:
        yield s
    finally:
        s.close()


def model_version(model) -> str:
    """Stable version string for a trained model (its training timestamp).

    Forward-compatible: if an encoder is loaded its version is appended so
    head+encoder combinations are distinguishable.
    """
    if model is None:
        return "unknown"
    base = getattr(model, "trained_at", None) or "untrained"
    version = str(base)
    try:  # optional — only present once the encoder feature is merged
        from ml.embedding_features import get_encoder
        enc = get_encoder()
        if enc is not None:
            version = f"{version}+enc:{enc.version}"
    except Exception:
        pass
    return version[:128]


# ── baseline ────────────────────────────────────────────────────────────────
def save_baseline(model_version_: str, train_ic, feature_stats: dict,
                  prediction_stats: dict) -> bool:
    try:
        with _session() as s:
            if s is None:
                return False
            row = MonitoringBaseline(
                model_version=model_version_,
                train_ic=(float(train_ic) if train_ic is not None else None),
                n_features=len(feature_stats or {}),
                feature_stats=feature_stats,
                prediction_stats=prediction_stats,
            )
            s.add(row)
            s.commit()
            return True
    except Exception as exc:
        logger.warning(f"[monitoring] save_baseline failed: {exc}")
        return False


def get_baseline(model_version_: str) -> Optional[MonitoringBaseline]:
    try:
        with _session() as s:
            if s is None:
                return None
            stmt = (select(MonitoringBaseline)
                    .where(MonitoringBaseline.model_version == model_version_)
                    .order_by(MonitoringBaseline.created_at.desc()))
            return s.exec(stmt).first()
    except Exception as exc:
        logger.warning(f"[monitoring] get_baseline failed: {exc}")
        return None


# ── predictions / labels ─────────────────────────────────────────────────────
def insert_predictions(rows: list[dict]) -> int:
    try:
        with _session() as s:
            if s is None or not rows:
                return 0
            objs = [MonitoringPrediction(**r) for r in rows]
            s.add_all(objs)
            s.commit()
            return len(objs)
    except Exception as exc:
        logger.warning(f"[monitoring] insert_predictions failed: {exc}")
        return 0


def get_due_unmatured(now: Optional[datetime] = None, limit: int = 5000) -> list[MonitoringPrediction]:
    """Predictions whose horizon has elapsed but realized return isn't filled."""
    now = now or datetime.now(timezone.utc)
    try:
        with _session() as s:
            if s is None:
                return []
            stmt = (select(MonitoringPrediction)
                    .where(MonitoringPrediction.matured == False)  # noqa: E712
                    .limit(limit))
            rows = s.exec(stmt).all()
            due = []
            for r in rows:
                pa = r.predicted_at
                if pa.tzinfo is None:
                    pa = pa.replace(tzinfo=timezone.utc)
                if pa + timedelta(days=int(r.horizon_days * 1.45)) <= now:
                    # ~1.45 calendar days per trading day
                    due.append(r)
            return due
    except Exception as exc:
        logger.warning(f"[monitoring] get_due_unmatured failed: {exc}")
        return []


def update_matured(updates: list[tuple]) -> int:
    """updates: list of (prediction_id, realized_return)."""
    try:
        with _session() as s:
            if s is None or not updates:
                return 0
            n = 0
            now = datetime.now(timezone.utc)
            for pid, ret in updates:
                obj = s.get(MonitoringPrediction, pid)
                if obj is not None:
                    obj.realized_return = float(ret)
                    obj.matured = True
                    obj.matured_at = now
                    s.add(obj)
                    n += 1
            s.commit()
            return n
    except Exception as exc:
        logger.warning(f"[monitoring] update_matured failed: {exc}")
        return 0


def get_matured(model_version_: str, limit: int = 100000) -> list[MonitoringPrediction]:
    try:
        with _session() as s:
            if s is None:
                return []
            stmt = (select(MonitoringPrediction)
                    .where(MonitoringPrediction.model_version == model_version_)
                    .where(MonitoringPrediction.matured == True)  # noqa: E712
                    .order_by(MonitoringPrediction.predicted_at)
                    .limit(limit))
            return s.exec(stmt).all()
    except Exception as exc:
        logger.warning(f"[monitoring] get_matured failed: {exc}")
        return []


# ── IC / drift / alerts ───────────────────────────────────────────────────────
def insert_ic(rows: list[dict]) -> int:
    try:
        with _session() as s:
            if s is None or not rows:
                return 0
            s.add_all([MonitoringIC(**r) for r in rows])
            s.commit()
            return len(rows)
    except Exception as exc:
        logger.warning(f"[monitoring] insert_ic failed: {exc}")
        return 0


def get_recent_ic(model_version_: str, window: int, limit: int = 50) -> list[MonitoringIC]:
    try:
        with _session() as s:
            if s is None:
                return []
            stmt = (select(MonitoringIC)
                    .where(MonitoringIC.model_version == model_version_)
                    .where(MonitoringIC.window == window)
                    .order_by(MonitoringIC.computed_at.desc())
                    .limit(limit))
            return s.exec(stmt).all()
    except Exception as exc:
        logger.warning(f"[monitoring] get_recent_ic failed: {exc}")
        return []


def insert_drift(rows: list[dict]) -> int:
    try:
        with _session() as s:
            if s is None or not rows:
                return 0
            s.add_all([MonitoringDrift(**r) for r in rows])
            s.commit()
            return len(rows)
    except Exception as exc:
        logger.warning(f"[monitoring] insert_drift failed: {exc}")
        return 0


def insert_prediction_drift(row: dict) -> bool:
    try:
        with _session() as s:
            if s is None:
                return False
            s.add(MonitoringPredictionDrift(**row))
            s.commit()
            return True
    except Exception as exc:
        logger.warning(f"[monitoring] insert_prediction_drift failed: {exc}")
        return False


def get_latest_drift(model_version_: str) -> list[MonitoringDrift]:
    """Most recent period's per-feature drift rows."""
    try:
        with _session() as s:
            if s is None:
                return []
            latest = (select(MonitoringDrift.period_key)
                      .where(MonitoringDrift.model_version == model_version_)
                      .order_by(MonitoringDrift.computed_at.desc())
                      .limit(1))
            pk = s.exec(latest).first()
            if pk is None:
                return []
            stmt = (select(MonitoringDrift)
                    .where(MonitoringDrift.model_version == model_version_)
                    .where(MonitoringDrift.period_key == pk))
            return s.exec(stmt).all()
    except Exception as exc:
        logger.warning(f"[monitoring] get_latest_drift failed: {exc}")
        return []


def get_latest_prediction_drift(model_version_: str) -> Optional[MonitoringPredictionDrift]:
    try:
        with _session() as s:
            if s is None:
                return None
            stmt = (select(MonitoringPredictionDrift)
                    .where(MonitoringPredictionDrift.model_version == model_version_)
                    .order_by(MonitoringPredictionDrift.computed_at.desc())
                    .limit(1))
            return s.exec(stmt).first()
    except Exception as exc:
        logger.warning(f"[monitoring] get_latest_prediction_drift failed: {exc}")
        return None


def insert_alert(model_version_: str, level: str, kind: str,
                 detail: dict, period_key: Optional[str] = None) -> bool:
    try:
        with _session() as s:
            if s is None:
                return False
            s.add(MonitoringAlert(model_version=model_version_, level=level,
                                  kind=kind, detail=detail, period_key=period_key))
            s.commit()
            return True
    except Exception as exc:
        logger.warning(f"[monitoring] insert_alert failed: {exc}")
        return False


def get_recent_alerts(limit: int = 50) -> list[MonitoringAlert]:
    try:
        with _session() as s:
            if s is None:
                return []
            stmt = (select(MonitoringAlert)
                    .order_by(MonitoringAlert.created_at.desc())
                    .limit(limit))
            return s.exec(stmt).all()
    except Exception as exc:
        logger.warning(f"[monitoring] get_recent_alerts failed: {exc}")
        return []


def prune(days: int) -> None:
    """Delete monitoring rows older than `days` (mirrors activity_events purge)."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        with _session() as s:
            if s is None:
                return
            for model, ts in (
                (MonitoringPrediction, MonitoringPrediction.predicted_at),
                (MonitoringIC, MonitoringIC.computed_at),
                (MonitoringDrift, MonitoringDrift.computed_at),
                (MonitoringPredictionDrift, MonitoringPredictionDrift.computed_at),
                (MonitoringAlert, MonitoringAlert.created_at),
            ):
                for obj in s.exec(select(model).where(ts < cutoff)).all():
                    s.delete(obj)
            s.commit()
    except Exception as exc:
        logger.warning(f"[monitoring] prune failed: {exc}")
