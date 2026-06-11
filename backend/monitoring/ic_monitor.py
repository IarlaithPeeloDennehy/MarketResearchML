"""
ic_monitor.py — Information Coefficient tracking with delayed labels.

Flow:
  1. log_predictions()    — at inference, store (period, ticker, prediction).
  2. mature_labels()      — later, once the forward horizon has elapsed, fill the
                            realized return from the price cache.
  3. compute_rolling_ic() — from matured rows, compute per-period cross-sectional
                            Spearman/Pearson IC, then 20/60/120-period rolling
                            mean / std / information ratio.

IC cannot be computed at prediction time (the target is a *forward* return), so
until the first horizon matures these tables are simply empty — by design.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from . import config, metrics_store

logger = logging.getLogger(__name__)


def log_predictions(model, results: list[dict], period_key: str,
                    horizon_days: int = None) -> int:
    """Persist this batch of predictions for later IC scoring.
    `results` are score_universe outputs; we log the raw model probability."""
    horizon_days = horizon_days or config.DEFAULT_HORIZON_DAYS
    version = metrics_store.model_version(model)
    rows = []
    for r in results or []:
        ticker = r.get("ticker")
        # fundamental_score is the model probability ×100; fall back to composite
        score = r.get("fundamental_score")
        if score is None:
            score = r.get("composite_score")
        if ticker is None or score is None:
            continue
        rows.append({
            "model_version": version, "period_key": period_key,
            "ticker": str(ticker), "prediction": float(score) / 100.0,
            "horizon_days": int(horizon_days),
        })
    return metrics_store.insert_predictions(rows)


# ── label maturation (realized forward returns from the price cache) ─────────
def _load_close(ticker: str):
    try:
        from ml.model import PRICE_DIR
        from ml.feature_engineering import safe_ticker_filename
        p = PRICE_DIR / f"{safe_ticker_filename(ticker)}.parquet"
        if not p.exists():
            return None
        df = pd.read_parquet(p)
        return df["Close"].sort_index() if "Close" in df.columns else None
    except Exception:
        return None


def _realized_return(ticker: str, predicted_at, horizon_days: int):
    s = _load_close(ticker)
    if s is None or len(s) == 0:
        return None
    ts = pd.Timestamp(predicted_at)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    try:
        start = int(s.index.searchsorted(ts))
    except Exception:
        return None
    end = start + int(horizon_days)
    if start >= len(s) or end >= len(s):
        return None  # not enough future bars yet → stays unmatured
    p0, p1 = float(s.iloc[start]), float(s.iloc[end])
    return (p1 / p0 - 1) if p0 > 0 else None


def mature_labels() -> int:
    """Fill realized returns for predictions whose horizon has elapsed."""
    due = metrics_store.get_due_unmatured()
    updates = []
    for row in due:
        ret = _realized_return(row.ticker, row.predicted_at, row.horizon_days)
        if ret is not None:
            updates.append((row.id, ret))
    n = metrics_store.update_matured(updates)
    if n:
        logger.info(f"[monitoring] matured {n} predictions into realized returns")
    return n


# ── rolling IC ───────────────────────────────────────────────────────────────
def _period_ics(matured: list) -> list[dict]:
    """Per-period cross-sectional Spearman/Pearson IC, time-ordered."""
    if not matured:
        return []
    df = pd.DataFrame([{
        "period_key": m.period_key, "predicted_at": m.predicted_at,
        "prediction": m.prediction, "realized": m.realized_return,
    } for m in matured])
    out = []
    for pk, grp in df.groupby("period_key"):
        g = grp.dropna(subset=["prediction", "realized"])
        if len(g) < 2 or g["prediction"].nunique() < 2 or g["realized"].nunique() < 2:
            continue
        sp, _ = spearmanr(g["prediction"], g["realized"])
        pe, _ = pearsonr(g["prediction"], g["realized"])
        out.append({
            "period_key": pk,
            "ts": g["predicted_at"].min(),
            "spearman": float(sp) if np.isfinite(sp) else 0.0,
            "pearson": float(pe) if np.isfinite(pe) else 0.0,
        })
    out.sort(key=lambda x: x["ts"])
    return out


def compute_rolling_ic(model) -> list[dict]:
    """Compute and persist 20/60/120-period rolling IC stats. Returns the rows."""
    version = metrics_store.model_version(model)
    matured = metrics_store.get_matured(version)
    periods = _period_ics(matured)
    if not periods:
        return []

    sp_series = [p["spearman"] for p in periods]
    latest_sp = periods[-1]["spearman"]
    latest_pe = periods[-1]["pearson"]

    rows = []
    for window in config.ROLLING_WINDOWS:
        recent = sp_series[-window:]
        if not recent:
            continue
        arr = np.asarray(recent, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        rows.append({
            "model_version": version, "window": window,
            "spearman_ic": latest_sp, "pearson_ic": latest_pe,
            "ic_mean": mean, "ic_std": std,
            "ic_ir": float(mean / std) if std > 1e-9 else 0.0,
            "n_obs": len(recent),
        })
    metrics_store.insert_ic(rows)
    return rows
