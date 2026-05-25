"""
NUMKT ML Backend  (v2 — self-improving)
────────────────────────────────────────
Changes from v1:
  - Data cached to disk (cache/ folder) — Yahoo only hit once per ticker per day
  - Model trained on real historical returns, not synthetic quality scores
  - Trained model saved to disk — survives server restarts
  - /analyse loads saved model on startup if available
  - /backtest trains AND saves model on real return data
  - New /cache and /model endpoints for visibility
"""

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import logging
import asyncio
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlmodel import select, or_

# Absolute path to index.html — works regardless of working directory.
# main.py lives in backend/, so parent.parent is the project root.
_FRONTEND_HTML = Path(__file__).parent.parent / "index.html"

from ml.data_fetcher import (fetch_stock_data, fetch_multiple_stocks,
                              get_cache_status, clear_cache as clear_data_cache)
from ml.feature_engineering import build_features
from ml.model import NUMKTEnsemble
from ml.backtest import run_backtest
from ml.scoring import score_universe
from ml.insights import build_snapshot, load_snapshot, is_stale
from auth.router import router as auth_router
from auth.dependencies import get_current_user
from auth.models import User, UserSession
from user.router import router as user_router
from db.session import get_db
from db.base import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Rate limiter ───────────────────────────────────────────────────────────
# Key function uses X-Forwarded-For when behind Render's proxy, else direct IP
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="NUMKT ML Backend v2",
    description="Self-improving ML backend — trains on real historical returns",
    version="2.0.0",
    docs_url=None,
    redoc_url=None,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Security headers ───────────────────────────────────────────────────────
# Applied to every response. CSP is permissive enough for:
#   - inline <script> (the entire frontend JS lives in index.html)
#   - Chart.js from cdnjs
#   - Google Fonts (stylesheet + gstatic fonts)
# HSTS is only emitted over HTTPS (Render sets X-Forwarded-Proto=https).
_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self' https://cdnjs.cloudflare.com; "
    "frame-ancestors 'none'; "
    "base-uri 'self'; "
    "form-action 'self'"
)

@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]          = "DENY"
    response.headers["Referrer-Policy"]          = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"]       = "geolocation=(), microphone=(), camera=()"
    response.headers["Content-Security-Policy"]  = _CSP
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    if proto == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# CORS — because the frontend is served by this same FastAPI app, all browser
# fetch() calls are same-origin and CORS headers are not consulted by the browser.
# The origin list here only matters if a *separate* frontend domain calls this API.
# ALLOWED_ORIGINS env var lets you add those domains without touching code.
# Example: ALLOWED_ORIGINS=https://app.example.com,https://staging.example.com
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "")
_extra_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_extra_origins or ["*"],   # lock down if env var is set
    allow_credentials=bool(_extra_origins),  # credentials only when origins are explicit
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)

# ── Auth + user routes ────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(user_router)

# ── In-memory caches ───────────────────────────────────────────────────────
_model_cache: dict = {}   # profile_risk_lookback -> NUMKTEnsemble
_data_cache:  dict = {}   # "latest" -> raw_data, features_df

# ── Background retraining ──────────────────────────────────────────────────
_last_retrain: float = 0.0          # Unix timestamp of last background retrain
_RETRAIN_COOLDOWN    = 3600         # seconds — don't retrain more than once per hour

# ── Public insights snapshot ───────────────────────────────────────────────
_last_insights_build: float = 0.0   # Unix timestamp of last snapshot build
_INSIGHTS_COOLDOWN          = 3600  # seconds — rebuild at most once per hour


async def _build_insights_snapshot() -> None:
    """Build the public /api/insights snapshot in the background.

    Runs in an asyncio task so it never blocks a request. The cooldown stamp is
    set up-front so concurrent requests don't trigger overlapping builds. A true
    daily Render cron / APScheduler job is the more robust alternative; this lazy
    refresh mirrors the existing _background_retrain pattern and needs no new
    dependency on Render's single web service.
    """
    global _last_insights_build
    _last_insights_build = datetime.now(timezone.utc).timestamp()
    try:
        logger.info("Insights snapshot: building")
        snap = await build_snapshot()
        logger.info(f"Insights snapshot: built ({snap.get('universe_size')} stocks)")
    except Exception as exc:
        logger.error(f"Insights snapshot build failed (non-fatal): {exc}", exc_info=True)


async def _background_retrain(raw_data: dict, features_df) -> None:
    """Retrain and save the model in the background after new data is fetched.

    Runs in an asyncio task so it never delays the /analyse response.
    Failures are logged but swallowed — the caller's result is unaffected.
    """
    global _last_retrain
    try:
        logger.info("Background retrain: starting model update with newly fetched data")
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(
            None,
            lambda: run_backtest(
                features_df=features_df,
                raw_data=raw_data,
                profile="quality",
                forward_months=12,
                rebalance_months=3,
                train_model=True,
                model_name="default",
            ),
        )
        if metrics.get("training", {}).get("trained"):
            trained_model = NUMKTEnsemble.load("default")
            if trained_model:
                _model_cache["__trained__"] = trained_model
                _last_retrain = datetime.now(timezone.utc).timestamp()
                logger.info(
                    f"Background retrain complete — model updated "
                    f"({trained_model.n_training_rows} rows, "
                    f"IC={trained_model.cv_ic:.4f}, acc={trained_model.cv_accuracy:.3f})"
                )
        else:
            logger.warning("Background retrain: run_backtest returned no trained model")
    except Exception as exc:
        logger.error(f"Background retrain failed (non-fatal): {exc}", exc_info=True)


# ── Load any previously trained models on startup ─────────────────────────
def _run_migrations() -> None:
    """Run alembic upgrade head programmatically.

    Handles the case where tables were previously created via SQLModel
    create_all (no alembic_version table) by stamping the DB at the
    current head before attempting to apply migrations.
    """
    if engine is None:
        return

    from alembic.config import Config as AlembicConfig
    from alembic import command as alembic_command
    from alembic.runtime.migration import MigrationContext
    import sqlalchemy as _sa

    _base = Path(__file__).parent
    cfg = AlembicConfig(str(_base / "alembic.ini"))
    cfg.set_main_option("script_location", str(_base / "alembic"))

    with engine.connect() as conn:
        # If tables exist but alembic_version does not, the DB was bootstrapped
        # via SQLModel create_all and has never been under Alembic control.
        # Stamp it at head so upgrade head becomes a no-op instead of crashing
        # with "relation already exists".
        has_version_table = _sa.inspect(conn).has_table("alembic_version")
        if not has_version_table:
            has_app_tables = _sa.inspect(conn).has_table("users")
            if has_app_tables:
                logger.warning(
                    "Database has app tables but no alembic_version — "
                    "stamping at head revision without re-running migrations"
                )
                alembic_command.stamp(cfg, "head")
                return

        ctx = MigrationContext.configure(conn)
        current = ctx.get_current_revision()

    alembic_command.upgrade(cfg, "head")
    logger.info(f"Migrations applied (was at revision: {current or 'none'})")


@app.on_event("startup")
async def startup():
    # ── Run database migrations ────────────────────────────────────────────
    if engine is not None:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _run_migrations)
        except Exception as exc:
            logger.error(f"Database migration failed: {exc}", exc_info=True)
            raise RuntimeError("Startup aborted — database migration failed") from exc

    # ── Prune stale sessions ───────────────────────────────────────────────
    try:
        for db in get_db():
            now = datetime.now(timezone.utc)
            stmt = select(UserSession).where(
                or_(UserSession.expires_at < now, UserSession.revoked == True)  # noqa: E712
            )
            stale = db.exec(stmt).all()
            for s in stale:
                db.delete(s)
            db.commit()
            if stale:
                logger.info(f"Pruned {len(stale)} expired/revoked session(s)")
    except Exception as exc:
        logger.warning(f"Session pruning skipped (DB may not be ready): {exc}")

    # ── Warn when cache directory is not persistent ────────────────────────
    if not os.environ.get("CACHE_DIR"):
        logger.warning(
            "CACHE_DIR is not set — trained models and price cache are stored in the "
            "container's local filesystem and will be lost on every restart or redeploy. "
            "Mount a persistent disk on Render and set CACHE_DIR to its path to retain "
            "learning between deploys."
        )

    # ── Load saved ML model ────────────────────────────────────────────────
    saved = NUMKTEnsemble.list_saved()
    if saved:
        logger.info(f"Found {len(saved)} saved model(s):")
        for m in saved:
            logger.info(f"  {m['name']} — trained {m['trained_at']} "
                        f"on {m['n_training_rows']} rows ({m['training_source']}) "
                        f"IC={m.get('cv_ic', 0):.4f}, acc={m['cv_accuracy']:.3f}")
        model = NUMKTEnsemble.load("default")
        if model:
            _model_cache["__trained__"] = model
            logger.info("Loaded saved 'default' model into memory")
    else:
        logger.info("No saved models found — will use synthetic labels until backtest runs")

    # ── Warm the public insights snapshot ──────────────────────────────────
    try:
        if is_stale():
            asyncio.create_task(_build_insights_snapshot())
            logger.info("Insights snapshot warm-up queued on startup")
    except Exception as exc:
        logger.warning(f"Insights warm-up skipped: {exc}")


# ── Schemas ────────────────────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    tickers:        List[str]
    profile:        str   = "quality"
    risk:           str   = "medium"
    lookback_years: int   = 5         # more history = better backtest later
    cv_folds:       int   = 5
    lambda_reg:     float = 0.10
    use_macro:      bool  = True
    use_hf_signals: bool  = True

class BacktestRequest(BaseModel):
    tickers:          List[str]
    profile:          str = "quality"
    lookback_years:   int = 5
    forward_months:   int = 12
    rebalance_months: int = 3
    train_model:      bool = True    # train on real returns after backtest


# ── Frontend ───────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_frontend():
    """
    Serve the single-page frontend. Hosting the HTML here (same origin as the
    API) is what makes HttpOnly session cookies work — they cannot be set or
    sent across origins on file:// or a separate domain without CORS
    credential sharing, which itself breaks SameSite cookie semantics.
    """
    if not _FRONTEND_HTML.exists():
        raise HTTPException(
            404,
            "index.html not found. Expected it at: " + str(_FRONTEND_HTML)
        )
    return FileResponse(_FRONTEND_HTML, media_type="text/html")


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def api_status():
    """Machine-readable service status — the old GET / payload, now at /api/status."""
    saved = NUMKTEnsemble.list_saved()
    return {
        "service":      "NUMKT ML Backend v2",
        "status":       "running",
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "saved_models": len(saved),
        "best_model":   saved[0] if saved else None,
    }


# ── Public insights (no auth) ──────────────────────────────────────────────

@app.get("/api/insights")
@limiter.limit("30/minute")
async def get_insights(request: Request):
    """Public, login-free snapshot: verified track record + today's theses.

    Intentionally has no get_current_user dependency — this is the one public
    data endpoint and the acquisition surface. Serves the last snapshot
    immediately; if it's stale, kicks a non-blocking background rebuild guarded
    by a cooldown so repeated hits never trigger a Yahoo fetch storm.
    """
    snap   = load_snapshot()
    now_ts = datetime.now(timezone.utc).timestamp()
    if is_stale() and (now_ts - _last_insights_build) > _INSIGHTS_COOLDOWN:
        asyncio.create_task(_build_insights_snapshot())
    if snap is None:
        return {"status": "warming",
                "message": "Insights are being generated. Check back shortly."}
    return snap


# ── Main analysis ──────────────────────────────────────────────────────────

@app.post("/analyse")
@limiter.limit("10/minute")
async def analyse(request: Request, req: AnalyseRequest, current_user: User = Depends(get_current_user)):
    if len(req.tickers) < 3:
        raise HTTPException(400, "Please provide at least 3 tickers.")
    if len(req.tickers) > 40:
        raise HTTPException(400, "Maximum 40 tickers per request.")

    logger.info(f"Analyse: {req.tickers} | profile={req.profile}")

    try:
        raw_data, uncached_count = await fetch_multiple_stocks(req.tickers, req.lookback_years+1)
        if not raw_data:
            raise HTTPException(502, "Could not fetch data for any tickers.")

        features_df = build_features(raw_data, profile=req.profile)
        if features_df.empty:
            raise HTTPException(422, "Not enough data to build features.")

        # Use the model trained on real returns if available
        # Otherwise fall back to per-profile model with synthetic labels
        if "__trained__" in _model_cache:
            model = _model_cache["__trained__"]
            logger.info(f"Using pre-trained model "
                        f"({model.training_source}, {model.n_training_rows} rows)")
        else:
            cache_key = f"{req.profile}_{req.risk}_{req.lookback_years}"
            if cache_key not in _model_cache:
                model = NUMKTEnsemble(cv_folds=req.cv_folds,
                                      lambda_reg=req.lambda_reg, max_depth=5)
                model.fit(features_df)
                _model_cache[cache_key] = model
                logger.info(f"Trained new synthetic model: {cache_key}")
            else:
                model = _model_cache[cache_key]
                logger.info(f"Using cached synthetic model: {cache_key}")

        results = score_universe(
            model=model, features_df=features_df, raw_data=raw_data,
            profile=req.profile, risk=req.risk,
        )

        # Cache for backtest
        _data_cache["latest"]           = raw_data
        _data_cache["latest_features"]  = features_df

        # Trigger background retraining when new data was fetched from Yahoo
        # and the cooldown has elapsed — runs after the response is returned
        now_ts = datetime.now(timezone.utc).timestamp()
        if uncached_count > 0 and (now_ts - _last_retrain) > _RETRAIN_COOLDOWN:
            asyncio.create_task(_background_retrain(raw_data, features_df))
            logger.info(
                f"Background retrain queued ({uncached_count} new ticker(s) fetched)"
            )

        return {
            "status":             "ok",
            "timestamp":          datetime.now(timezone.utc).isoformat(),
            "profile":            req.profile,
            "risk":               req.risk,
            "cv_folds":           req.cv_folds,
            "cv_ic":              round(model.cv_ic, 4),
            "cv_accuracy":        round(model.cv_accuracy, 4),
            "training_source":    model.training_source,
            "n_training_rows":    model.n_training_rows,
            "feature_importance": model.feature_importance(),
            "results":            results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(500, "Analysis failed. Check server logs for details.")


# ── Backtest + train ───────────────────────────────────────────────────────

@app.post("/backtest")
@limiter.limit("3/minute")
async def backtest(request: Request, req: BacktestRequest):
    """
    Runs walk-forward backtest AND trains the model on real returns.
    After this runs, /analyse will use the trained model automatically.
    """
    if len(req.tickers) < 3:
        raise HTTPException(400, "Backtest requires at least 3 tickers.")

    logger.info(f"Backtest: {req.tickers} | train={req.train_model}")

    try:
        # Use cached data if available to avoid hitting Yahoo again
        if "latest" in _data_cache:
            logger.info("Using cached data from last /analyse call")
            raw_data    = _data_cache["latest"]
            features_df = _data_cache["latest_features"]
        else:
            logger.info("No cache — fetching fresh data")
            raw_data, _ = await fetch_multiple_stocks(req.tickers, req.lookback_years+1)
            if not raw_data:
                raise HTTPException(502, "Could not fetch backtest data.")
            features_df = build_features(raw_data, profile=req.profile)

        metrics = run_backtest(
            features_df      = features_df,
            raw_data         = raw_data,
            profile          = req.profile,
            forward_months   = req.forward_months,
            rebalance_months = req.rebalance_months,
            train_model      = req.train_model,
            model_name       = "default",
        )

        # If model was trained, load it into memory so next /analyse uses it
        if req.train_model and metrics.get("training", {}).get("trained"):
            trained_model = NUMKTEnsemble.load("default")
            if trained_model:
                _model_cache["__trained__"] = trained_model
                logger.info("Loaded newly trained model into memory — "
                            "next /analyse will use real-return model")

        return {
            "status":           "ok",
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "tickers":          req.tickers,
            "backtest_period":  f"{req.lookback_years} years",
            "forward_months":   req.forward_months,
            "metrics":          metrics,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        raise HTTPException(500, "Backtest failed. Check server logs for details.")


# ── Cache endpoints ────────────────────────────────────────────────────────

@app.get("/cache/status")
def cache_status(current_user: User = Depends(get_current_user)):
    """Show what price data is cached to disk."""
    return {
        "status":   "ok",
        "tickers":  get_cache_status(),
        "memory":   {
            "model_keys": list(_model_cache.keys()),
            "data_cached": list(_data_cache.keys()),
        }
    }

@app.delete("/cache/data")
def clear_data(ticker: str | None = None, current_user: User = Depends(get_current_user)):
    """Clear disk cache for one ticker or all."""
    clear_data_cache(ticker)
    return {"status": f"cleared {'all' if not ticker else ticker}"}

@app.delete("/cache/models")
def clear_model_cache(current_user: User = Depends(get_current_user)):
    """Clear in-memory model cache (forces retrain on next /analyse)."""
    _model_cache.clear()
    return {"status": "model cache cleared"}


# ── Model endpoints ────────────────────────────────────────────────────────

@app.get("/model/info")
def model_info(current_user: User = Depends(get_current_user)):
    """Info about saved and in-memory models."""
    saved  = NUMKTEnsemble.list_saved()
    active = _model_cache.get("__trained__")
    return {
        "saved_models": saved,
        "active_model": {
            "training_source": active.training_source if active else None,
            "n_training_rows": active.n_training_rows if active else 0,
            "cv_accuracy":     active.cv_accuracy     if active else None,
            "cv_ic":           active.cv_ic            if active else None,
            "trained_at":      active.trained_at       if active else None,
        } if active else None,
        "using_real_returns": "__trained__" in _model_cache,
    }

@app.get("/stock/{ticker}")
async def get_stock(ticker: str):
    try:
        data = await fetch_stock_data(ticker.upper(), lookback_years=5)
        if not data:
            raise HTTPException(404, f"No data for {ticker}")
        safe = {k: v for k, v in data.items() if k != "_raw"}
        return {"status": "ok", "ticker": ticker.upper(), "data": safe}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock fetch error ({ticker}): {e}", exc_info=True)
        raise HTTPException(500, "Failed to fetch stock data. Check server logs for details.")