"""
insights.py
───────────
Builds a public, login-free "Proof & Theses" snapshot that turns the cold-start
landing page into an instant, credible demonstration of the model:

  - Verified track record — honest out-of-sample numbers from the walk-forward
    backtest (holdout IC, hit rate, net alpha, Sharpe, max drawdown).
  - Per-stock theses — a balanced bull AND bear case for each headline stock,
    a confidence band, and the model's *historically realized* hit-rate at that
    confidence (so each individual call ties back to the verified track record).

The snapshot is computed on a daily cadence and stored as JSON. The public
GET /api/insights endpoint serves the latest snapshot. Deep per-stock analysis,
custom universes, and saving stay gated behind signup — that gate is the
conversion funnel.

Reuses the existing model, feature, scoring, and backtest machinery — no new ML.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .data_fetcher import fetch_multiple_stocks
from .feature_engineering import build_features
from .model import NUMKTEnsemble
from .backtest import run_backtest
from .scoring import score_universe, build_thesis

logger = logging.getLogger(__name__)

# Snapshot storage — uses the same persistent cache root as prices/models.
_cache_root = os.environ.get("CACHE_DIR")
_base = Path(_cache_root) if _cache_root else Path(__file__).parent.parent / "cache"
INSIGHTS_DIR  = _base / "insights"
SNAPSHOT_PATH = INSIGHTS_DIR / "snapshot.json"

SNAPSHOT_MAX_AGE_HOURS = 24

# Curated public universe — liquid US/UK/IE names. Kept modest to respect
# Yahoo's rate limits and the 2s-per-fetch cache layer in data_fetcher.
INSIGHTS_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META",
    "JPM",  "JNJ",  "PG",   "XOM",   "WMT",  "V",
    "UNH",  "HD",   "CVX",  "BRK-B",
    "SHEL.L", "AZN.L", "HSBA.L", "BP.L",
    "RYAAY", "CRH", "A5G.IR",
]


# ── Confidence bands ─────────────────────────────────────────────────────────
# composite_score is on a 0–100 scale (score_universe rounds probability*100).

def _confidence_band(score: float) -> str:
    if score >= 70: return "High"
    if score >= 55: return "Moderate"
    if score >= 38: return "Low"
    return "Bearish"


def _band_hit_rates(period_examples: list | None) -> dict:
    """Realized out-of-sample hit-rate per confidence band.

    Buckets every holdout prediction by its model_score band and measures how
    often that band actually beat the universe (actual_top). This is what powers
    each thesis's "right X% of the time at this confidence" line.
    """
    buckets: dict[str, list[int]] = {"High": [], "Moderate": [], "Low": [], "Bearish": []}
    for period in period_examples or []:
        for st in period.get("stocks", []):
            ms = st.get("model_score")
            at = st.get("actual_top")
            if ms is None or at is None:
                continue
            buckets[_confidence_band(float(ms))].append(1 if at else 0)

    out = {}
    for label, vals in buckets.items():
        out[label] = {
            "hit_rate": round(100 * sum(vals) / len(vals), 1) if vals else None,
            "n":        len(vals),
        }
    return out


# ── Snapshot build ───────────────────────────────────────────────────────────

async def build_snapshot() -> dict:
    """Compute and persist the public insights snapshot. Returns the snapshot."""
    raw_data, _ = await fetch_multiple_stocks(INSIGHTS_UNIVERSE, lookback_years=6)
    if not raw_data:
        raise RuntimeError("insights: no data fetched for universe")

    features_df = build_features(raw_data, profile="quality")
    if features_df.empty:
        raise RuntimeError("insights: feature matrix is empty")

    # Prefer the model trained on real returns; fall back to a synthetic fit
    # (same path /analyse uses when no trained model exists yet).
    model = NUMKTEnsemble.load("default")
    if model is None:
        model = NUMKTEnsemble()
        model.fit(features_df)
        logger.info("insights: no saved model — using synthetic fit for snapshot")

    results = score_universe(
        model=model, features_df=features_df, raw_data=raw_data,
        profile="quality", risk="medium",
    )

    # Verified track record — honest walk-forward backtest. Never trains/saves
    # here (train_model=False) so the public page can't mutate the live model.
    track_record: dict = {}
    bands: dict = {}
    try:
        metrics  = run_backtest(
            features_df=features_df, raw_data=raw_data, profile="quality",
            forward_months=12, rebalance_months=3,
            train_model=False, model_name="default",
        )
        training = metrics.get("training", {}) or {}
        holdout_ic = training.get("holdout_ic")
        track_record = {
            "hit_rate":       metrics.get("hit_rate"),
            "ic":             holdout_ic if holdout_ic is not None else metrics.get("ic_mean"),
            "ic_is_holdout":  holdout_ic is not None,
            "ann_alpha":      metrics.get("ann_alpha_net"),
            "sharpe":         metrics.get("sharpe_ratio"),
            "max_drawdown":   metrics.get("max_drawdown"),
            "n_periods":      metrics.get("n_periods"),
            "interpretation": metrics.get("interpretation"),
        }
        bands = _band_hit_rates(training.get("period_examples"))
    except Exception as exc:
        logger.warning(f"insights: backtest unavailable, track record omitted ({exc})")

    rows_by_ticker = {r.get("ticker"): r for _, r in features_df.iterrows()}

    theses = []
    for r in results:
        row = rows_by_ticker.get(r["ticker"])
        if row is None:
            continue
        inst    = r["inst_ownership_pct"]
        insider = r["insider_ownership_pct"]
        thesis  = build_thesis(
            row,
            inst / 100 if inst is not None else None,
            insider / 100 if insider is not None else None,
        )
        band = _confidence_band(r["composite_score"])
        theses.append({
            "ticker":           r["ticker"],
            "name":             r["name"],
            "sector":           r["sector"],
            "market":           r["market"],
            "signal":           r["signal"],
            "composite_score":  r["composite_score"],
            "confidence_label": band,
            "band_hit_rate":    bands.get(band, {}).get("hit_rate"),
            "bull":             thesis["bull"],
            "bear":             thesis["bear"],
        })

    snapshot = {
        "status":         "ok",
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "universe_size":  len(theses),
        "training_source": getattr(model, "training_source", None),
        "track_record":   track_record,
        "bands":          bands,
        "theses":         theses,
    }
    _atomic_write(snapshot)
    return snapshot


# ── Persistence helpers ──────────────────────────────────────────────────────

def _atomic_write(snapshot: dict) -> None:
    INSIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(INSIGHTS_DIR), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(snapshot, f)
        os.replace(tmp, SNAPSHOT_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def load_snapshot() -> dict | None:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        with open(SNAPSHOT_PATH) as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"insights: failed to read snapshot ({exc})")
        return None


def snapshot_age_hours() -> float | None:
    if not SNAPSHOT_PATH.exists():
        return None
    age = datetime.now(timezone.utc).timestamp() - SNAPSHOT_PATH.stat().st_mtime
    return age / 3600


def is_stale() -> bool:
    age = snapshot_age_hours()
    return age is None or age > SNAPSHOT_MAX_AGE_HOURS
