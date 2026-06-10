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
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from typing import List, Optional
import re
import requests as _requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import logging
import asyncio
import os
import warnings
warnings.filterwarnings("ignore")

# ── Ticker validation ──────────────────────────────────────────────────────
# Lookahead forces at least one letter — rejects pure-digit/symbol inputs like "123".
# Allows uppercase letters, digits, dots and hyphens (covers BRK-B, AZN.L, A5G.IR etc.).
_TICKER_RE = re.compile(r"^(?=[A-Z0-9.\-]*[A-Z])[A-Z0-9.\-]{1,15}$")

def _validate_tickers(tickers: list[str]) -> list[str]:
    """Uppercase, strip whitespace, and validate ticker format.
    Raises HTTPException(400) for any ticker that fails validation."""
    cleaned = []
    for raw in tickers:
        t = raw.strip().upper()
        if not _TICKER_RE.match(t):
            raise HTTPException(
                400,
                f"Invalid ticker '{raw}': must be 1–15 uppercase letters, digits, dots, or hyphens."
            )
        cleaned.append(t)
    return cleaned

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from sqlmodel import select, or_

# Absolute path to index.html — works regardless of working directory.
# main.py lives in backend/, so parent.parent is the project root.
_FRONTEND_HTML = Path(__file__).parent.parent / "index.html"

from ml.data_fetcher import (fetch_stock_data, fetch_multiple_stocks,
                              get_cache_status, clear_cache as clear_data_cache,
                              _get_finnhub_client, _is_us_ticker)
from ml.feature_engineering import build_features
from ml.model import NUMKTEnsemble
from ml.backtest import run_backtest
from ml.scoring import score_universe
from ml.insights import build_snapshot, load_snapshot, is_stale
from auth.router import router as auth_router
from auth.dependencies import get_current_user
from auth.models import User, UserSession, ActivityEvent
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

# ── Static assets (CSS + JS extracted from index.html) ────────────────────
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# ── In-memory caches ───────────────────────────────────────────────────────
_model_cache: dict = {}   # profile_risk_lookback -> NUMKTEnsemble
_data_cache:  dict = {}   # "latest" -> raw_data, features_df

# ── Search result cache ────────────────────────────────────────────────────
# Keyed by "normalised_query|types". TTL prevents stale results; max cap
# prevents unbounded memory growth on a long-running server.
_search_cache: dict = {}
_SEARCH_CACHE_TTL = 300   # 5 minutes — symbol lists rarely change intraday
_SEARCH_CACHE_MAX = 500   # evict oldest when this is exceeded

# ── Background retraining ──────────────────────────────────────────────────
_last_retrain: float = 0.0          # Unix timestamp of last background retrain
_RETRAIN_COOLDOWN    = 3600         # seconds — don't retrain more than once per hour
_head_train_running: bool = False   # guard against concurrent startup head-training
                                    # ("pre-training" now means the offline encoder)

# ── Public insights snapshot ───────────────────────────────────────────────
_last_insights_build: float = 0.0   # Unix timestamp of last snapshot build
_INSIGHTS_COOLDOWN          = 3600  # seconds — rebuild at most once per hour

# ── Background task registry ───────────────────────────────────────────────
# asyncio holds only weak references to tasks; without a strong reference the
# GC can collect a task before it finishes.  _background_tasks holds that
# strong reference for the lifetime of each task.
_background_tasks: set[asyncio.Task] = set()

def _fire_task(coro) -> asyncio.Task:
    """Schedule a background coroutine, keeping a strong reference until done."""
    t = asyncio.create_task(coro)
    _background_tasks.add(t)
    t.add_done_callback(_background_tasks.discard)
    return t


async def _build_insights_snapshot() -> None:
    """Build the public /api/insights snapshot in the background.

    Callers must stamp _last_insights_build = now_ts BEFORE firing this task
    so that concurrent requests see the cooldown immediately and don't queue
    duplicate builds.
    """
    try:
        logger.info("Insights snapshot: building")
        snap = await build_snapshot()
        logger.info(f"Insights snapshot: built ({snap.get('universe_size')} stocks)")
    except Exception as exc:
        logger.error(f"Insights snapshot build failed (non-fatal): {exc}", exc_info=True)


async def _background_retrain(raw_data: dict, features_df) -> None:
    """Retrain and save the model in the background after new data is fetched.

    Runs in an asyncio task so it never delays the /analyse response.
    Callers must stamp _last_retrain = now_ts BEFORE firing this task.
    Failures are logged but swallowed — the caller's result is unaffected.
    """
    try:
        logger.info("Background retrain: starting model update with newly fetched data")
        loop = asyncio.get_running_loop()
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
                logger.info(
                    f"Background retrain complete — model updated "
                    f"({trained_model.n_training_rows} rows, "
                    f"IC={trained_model.cv_ic:.4f}, acc={trained_model.cv_accuracy:.3f})"
                )
        else:
            logger.warning("Background retrain: run_backtest returned no trained model")
    except Exception as exc:
        logger.error(f"Background retrain failed (non-fatal): {exc}", exc_info=True)


def _is_model_stale(max_age_days: int = 7) -> bool:
    """Return True if the default model is missing or older than max_age_days."""
    saved = NUMKTEnsemble.list_saved()
    if not saved:
        return True
    try:
        trained_at = datetime.fromisoformat(
            saved[0]["trained_at"].replace("Z", "+00:00")
        )
        return (datetime.now(timezone.utc) - trained_at).days >= max_age_days
    except Exception:
        return True


async def _startup_train_head() -> None:
    """Fetch the anchor universe and train a fresh 'default' head model.

    This trains the RF+GB *head* (which consumes encoder embeddings when one is
    loaded). It is NOT the self-supervised pre-training — that happens offline in
    scripts/pretrain_encoder.py. Fires at startup when no head model exists or it
    is stale. Non-blocking — runs in executor so uvicorn workers are never held.
    """
    global _head_train_running, _last_retrain
    if _head_train_running:
        return
    _head_train_running = True
    # Stamp the retrain cooldown immediately so a concurrent /analyse request
    # doesn't also fire _background_retrain while we're in flight.
    _last_retrain = datetime.now(timezone.utc).timestamp()
    try:
        from ml.model import _ANCHOR_TICKERS
        from ml.data_fetcher import _price_cache_path, _cache_is_fresh

        # Only train on tickers already in the price cache — never fetch fresh
        # data at startup. Fetching all 58 anchor tickers would fire ~580 Finnhub
        # API calls in the first minute, exhausting the free-tier rate limit (60/min)
        # before any user request can succeed. The _background_retrain triggered by
        # user requests grows the training universe over time instead.
        cached_tickers = [t for t in _ANCHOR_TICKERS if _cache_is_fresh(_price_cache_path(t))]
        if len(cached_tickers) < 3:
            logger.info(
                f"Startup head-train: only {len(cached_tickers)} anchor tickers cached "
                "— skipping until user requests populate the cache"
            )
            return
        logger.info(f"Startup head-train: training on {len(cached_tickers)} cached anchor tickers")
        raw_data, _ = await fetch_multiple_stocks(cached_tickers, lookback_years=5)
        if len(raw_data) < 3:
            logger.warning(
                f"Startup head-train: only {len(raw_data)} tickers loaded, need 3+ — skipping"
            )
            return
        features_df = build_features(raw_data)
        loop = asyncio.get_running_loop()
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
                    f"Startup head-train complete — "
                    f"{trained_model.n_training_rows} rows, "
                    f"IC={trained_model.cv_ic:.4f}, acc={trained_model.cv_accuracy:.3f}"
                )
        else:
            logger.warning("Startup head-train: backtest returned no trained model")
    except Exception as exc:
        logger.error(f"Startup head-train failed (non-fatal): {exc}", exc_info=True)
    finally:
        _head_train_running = False


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


def _bootstrap_encoder() -> None:
    """Download (if needed) and load the offline-pretrained price encoder.

    Scenario-1 production half: the heavy training happened offline on a GPU
    box; here we just fetch the frozen checkpoint from object storage and load
    it for CPU inference. Fully non-fatal — any failure (torch absent, no URL,
    download/parse error, USE_ENCODER off) leaves the app in price-ranks-only
    mode, identical to its pre-encoder behaviour.

    Env vars:
      USE_ENCODER       "0"/"false" to disable (default on)
      ENCODER_URL       object-storage URL of encoder.pt (downloaded if missing)
      ENCODER_VERSION   optional expected version hash (logged on mismatch)
    """
    from ml.embedding_features import set_encoder
    from ml.encoder import load_encoder

    if os.environ.get("USE_ENCODER", "1").strip().lower() in ("0", "false", "no", "off"):
        logger.info("USE_ENCODER disabled — price-ranks-only mode.")
        set_encoder(None)
        return

    cache_root = os.environ.get("CACHE_DIR")
    base = Path(cache_root) if cache_root else Path(__file__).parent / "cache"
    model_dir = base / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = model_dir / "encoder.pt"

    url = os.environ.get("ENCODER_URL")
    if url and not ckpt.exists():
        try:
            logger.info(f"Downloading encoder checkpoint from {url} …")
            with _requests.get(url, stream=True, timeout=180) as resp:
                resp.raise_for_status()
                with open(ckpt, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            logger.info(f"Encoder checkpoint saved to {ckpt}")
        except Exception as exc:
            logger.warning(f"Encoder download failed (non-fatal): {exc}")

    enc = load_encoder(ckpt)
    if enc is None:
        logger.info("No usable encoder checkpoint — price-ranks-only mode.")
        set_encoder(None)
        return

    want = os.environ.get("ENCODER_VERSION")
    if want and enc.version != want:
        logger.warning(
            f"Loaded encoder version {enc.version} != ENCODER_VERSION={want}. "
            "Head model will be re-fit on next retrain to match."
        )
    set_encoder(enc)


async def _warm_reference_etfs(tickers: list[str]) -> None:
    """Cache SPY + sector-SPDR price series for the encoder's market/sector
    channels. Background, non-fatal — only fetches what isn't already fresh."""
    try:
        from ml.data_fetcher import _price_cache_path, _cache_is_fresh
        missing = [t for t in tickers if not _cache_is_fresh(_price_cache_path(t))]
        if not missing:
            return
        logger.info(f"Warming {len(missing)} reference ETF(s) for encoder channels: {missing}")
        await fetch_multiple_stocks(missing, lookback_years=20)
    except Exception as exc:
        logger.warning(f"Reference-ETF warm failed (non-fatal): {exc}")


@app.on_event("startup")
async def startup():
    global _last_insights_build
    loop = asyncio.get_running_loop()

    # ── Run database migrations (with exponential-backoff retry) ──────────
    if engine is not None:
        _MAX_MIGRATION_ATTEMPTS = 5
        for attempt in range(1, _MAX_MIGRATION_ATTEMPTS + 1):
            try:
                await loop.run_in_executor(None, _run_migrations)
                break
            except Exception as exc:
                if attempt == _MAX_MIGRATION_ATTEMPTS:
                    logger.error(
                        f"Database migration failed after {_MAX_MIGRATION_ATTEMPTS} attempts: {exc}",
                        exc_info=True,
                    )
                    raise RuntimeError("Startup aborted — database migration failed") from exc
                wait = 2 ** attempt  # 2 s, 4 s, 8 s, 16 s
                logger.warning(
                    f"Migration attempt {attempt}/{_MAX_MIGRATION_ATTEMPTS} failed: {exc}. "
                    f"Retrying in {wait}s…"
                )
                await asyncio.sleep(wait)

    # ── Prune stale sessions ───────────────────────────────────────────────
    try:
        db = next(get_db())
        try:
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
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f"Session pruning skipped (DB may not be ready): {exc}")

    # ── Purge old ActivityEvent rows (retain 90 days) ─────────────────────
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=90)
        db = next(get_db())
        try:
            old_events = db.exec(
                select(ActivityEvent).where(ActivityEvent.created_at < cutoff)
            ).all()
            for ev in old_events:
                db.delete(ev)
            db.commit()
            if old_events:
                logger.info(f"Purged {len(old_events)} ActivityEvent row(s) older than 90 days")
        finally:
            db.close()
    except Exception as exc:
        logger.warning(f"ActivityEvent purge skipped (non-fatal): {exc}")

    # ── Warn when cache directory is not persistent ────────────────────────
    if not os.environ.get("CACHE_DIR"):
        logger.warning(
            "CACHE_DIR is not set — trained models and price cache are stored in the "
            "container's local filesystem and will be lost on every restart or redeploy. "
            "Mount a persistent disk on Render and set CACHE_DIR to its path to retain "
            "learning between deploys."
        )

    # ── Load the offline-pretrained encoder (non-fatal, opt-in) ────────────
    # Done before model load / head training so feature building uses embeddings.
    try:
        await loop.run_in_executor(None, _bootstrap_encoder)
    except Exception as exc:
        logger.warning(f"Encoder bootstrap skipped (non-fatal): {exc}")

    # ── Warm reference ETFs for the market/sector channels ─────────────────
    # The encoder's excess_ret_vs_market / rel_strength_vs_sector channels need
    # SPY + sector-SPDR price series cached. Fetch them in the background so the
    # channels are populated; missing references just yield neutral channels.
    try:
        from ml.embedding_features import encoder_enabled
        if encoder_enabled():
            from ml.channels import REFERENCE_TICKERS
            _fire_task(_warm_reference_etfs(list(REFERENCE_TICKERS)))
    except Exception as exc:
        logger.warning(f"Reference-ETF warm skipped (non-fatal): {exc}")

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
        if _is_model_stale():
            logger.info("Saved model is stale (>7 days) — queuing background head-train")
            _fire_task(_startup_train_head())
    else:
        logger.info("No saved model — queuing startup head-train on anchor universe")
        _fire_task(_startup_train_head())

    # ── Warm the public insights snapshot ──────────────────────────────────
    try:
        if is_stale():
            _last_insights_build = datetime.now(timezone.utc).timestamp()
            _fire_task(_build_insights_snapshot())
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


# ── Fetch diagnostic (no auth) ────────────────────────────────────────────

@app.get("/api/debug/fetch-test")
@limiter.limit("5/minute")
async def fetch_test(request: Request, ticker: str = "AAPL"):
    """Diagnostic: test the data fetch pipeline for one ticker. No auth required."""
    ticker = ticker.strip().upper()
    try:
        result = await fetch_stock_data(ticker, lookback_years=1)
        if result is None:
            return {"status": "fail", "ticker": ticker,
                    "detail": "fetch_stock_data returned None — check server logs for error"}
        prices = result.get("_raw", {}).get("prices")
        return {
            "status":         "ok",
            "ticker":         ticker,
            "bars":           len(prices) if prices is not None else 0,
            "last_price":     result.get("last_price"),
            "finnhub_active": _get_finnhub_client() is not None,
        }
    except Exception as exc:
        return {"status": "error", "ticker": ticker, "detail": str(exc)}


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
    global _last_insights_build
    snap   = load_snapshot()
    now_ts = datetime.now(timezone.utc).timestamp()
    if is_stale() and (now_ts - _last_insights_build) > _INSIGHTS_COOLDOWN:
        _last_insights_build = now_ts
        _fire_task(_build_insights_snapshot())
    if snap is None:
        return {"status": "warming",
                "message": "Insights are being generated. Check back shortly."}
    return snap


# ── Main analysis ──────────────────────────────────────────────────────────

@app.post("/analyse")
@limiter.limit("10/minute")
async def analyse(request: Request, req: AnalyseRequest, current_user: User = Depends(get_current_user)):
    global _last_retrain
    if len(req.tickers) < 3:
        raise HTTPException(400, "Please provide at least 3 tickers.")
    if len(req.tickers) > 40:
        raise HTTPException(400, "Maximum 40 tickers per request.")
    tickers = _validate_tickers(req.tickers)

    logger.info(f"Analyse: {tickers} | profile={req.profile}")

    try:
        raw_data, uncached_count = await fetch_multiple_stocks(tickers, req.lookback_years+1)
        if not raw_data:
            raise HTTPException(502, "Could not fetch data for any tickers.")

        features_df = build_features(raw_data, profile=req.profile)
        if features_df.empty:
            raise HTTPException(422, "Not enough data to build features.")

        # Use the model trained on real returns if available
        # Otherwise fall back to per-profile model with synthetic labels
        if "__trained__" in _model_cache:
            model = _model_cache["__trained__"]
            logger.info(f"Using trained head model "
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
            _last_retrain = now_ts
            _fire_task(_background_retrain(raw_data, features_df))
            logger.info(
                f"Background retrain queued ({uncached_count} new ticker(s) fetched)"
            )

        us_tickers    = [t for t in tickers if _is_us_ticker(t)]
        uk_ie_tickers = [t for t in tickers if not _is_us_ticker(t)]
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
            "finnhub_active":     _get_finnhub_client() is not None,
            "fresh_fetched":      uncached_count,
            "us_tickers":         us_tickers,
            "uk_ie_tickers":      uk_ie_tickers,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(500, "Analysis failed. Check server logs for details.")


# ── Backtest + train ───────────────────────────────────────────────────────

@app.post("/backtest")
@limiter.limit("3/minute")
async def backtest(request: Request, req: BacktestRequest, current_user: User = Depends(get_current_user)):
    """
    Runs walk-forward backtest AND trains the model on real returns.
    After this runs, /analyse will use the trained model automatically.
    """
    if len(req.tickers) < 3:
        raise HTTPException(400, "Backtest requires at least 3 tickers.")
    tickers = _validate_tickers(req.tickers)

    logger.info(f"Backtest: {tickers} | train={req.train_model}")

    try:
        # Use cached data only when every requested ticker is present in the cache.
        # A stale cache from a different user's /analyse call must not contaminate
        # this user's backtest — verify the ticker sets match before reusing.
        cached_raw = _data_cache.get("latest")
        if (cached_raw is not None
                and set(tickers).issubset(set(cached_raw.keys()))):
            logger.info("Using cached data from last /analyse call")
            raw_data    = cached_raw
            features_df = _data_cache["latest_features"]
        else:
            if cached_raw is not None:
                logger.info(
                    "Cache ticker mismatch — fetching fresh data "
                    f"(requested: {tickers}, cached: {sorted(cached_raw.keys())})"
                )
            else:
                logger.info("No cache — fetching fresh data")
            raw_data, _ = await fetch_multiple_stocks(tickers, req.lookback_years+1)
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
            "finnhub_active":   _get_finnhub_client() is not None,
            "data_from_cache":  "latest" in _data_cache,
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


@app.get("/api/search")
@limiter.limit("20/minute")
async def search_stocks(q: str, request: Request, types: str = "stock"):
    """
    Search for stocks/ETFs by ticker or company name via Finnhub symbol search.

    Query params:
      q     — ticker symbol or company name fragment (required)
      types — "stock" (default) | "etf" | "all"

    Returns up to 15 normalised results. Results are cached per-query for
    _SEARCH_CACHE_TTL seconds so repeated keystrokes never hit Finnhub twice.
    No auth required — this is the public acquisition surface.
    """
    q = q.strip()
    if not q:
        return {"results": []}

    # Normalise cache key so 'aapl' and 'AAPL' share the same entry.
    cache_key = f"{q.lower()}|{types}"
    now_ts    = datetime.now(timezone.utc).timestamp()

    cached = _search_cache.get(cache_key)
    if cached and (now_ts - cached["ts"]) < _SEARCH_CACHE_TTL:
        return {"results": cached["results"], "cached": True}

    api_key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not api_key:
        return {"results": [], "finnhub_active": False,
                "note": "Finnhub key not configured — search unavailable"}

    # Map frontend types param to the Finnhub type strings we accept.
    _accepted: dict[str, set] = {
        "stock": {"Common Stock"},
        "etf":   {"ETP", "ETF"},
        "all":   {"Common Stock", "ETP", "ETF", "ADR"},
    }
    accepted_types = _accepted.get(types, {"Common Stock"})

    def _do_search() -> dict:
        r = _requests.get(
            "https://finnhub.io/api/v1/search",
            params={"q": q, "token": api_key},
            timeout=8,
        )
        r.raise_for_status()
        return r.json()

    try:
        loop = asyncio.get_running_loop()
        resp = await asyncio.wait_for(
            loop.run_in_executor(None, _do_search),
            timeout=10.0,
        )
        raw = resp.get("result", []) if isinstance(resp, dict) else []
        logger.info(f"Finnhub /search '{q}': {len(raw)} raw results")
        results = []

        for r in raw:
            symbol = r.get("symbol", "")
            rtype  = r.get("type", "")

            # Skip exchange-qualified duplicates like "AAPL:NASDAQ" — the plain
            # symbol "AAPL" will appear separately and is what downstream code expects.
            if ":" in symbol:
                continue

            # Use the same validation regex as /analyse so any ticker that passes
            # search can also be passed to the ML endpoint without 400 errors.
            # This naturally allows BRK-B, BRK.B, AZN.L, A5G.IR etc.
            if not symbol or not _TICKER_RE.match(symbol.upper()):
                continue

            if rtype not in accepted_types:
                continue

            results.append({
                "ticker":        symbol.upper(),
                "displaySymbol": r.get("displaySymbol", symbol).upper(),
                "name":          r.get("description", symbol),
                "exchange":      r.get("primaryExchange", ""),
                "type":          rtype,
            })
            if len(results) >= 15:
                break

        # Cache the result; evict the oldest entry when the cap is hit.
        _search_cache[cache_key] = {"ts": now_ts, "results": results}
        if len(_search_cache) > _SEARCH_CACHE_MAX:
            oldest = min(_search_cache, key=lambda k: _search_cache[k]["ts"])
            del _search_cache[oldest]

        return {"results": results, "finnhub_active": True, "raw_count": len(raw)}

    except asyncio.TimeoutError:
        logger.warning(f"Search timed out for query '{q}'")
        return {"results": [], "finnhub_active": True, "error": "Search timed out — please try again"}
    except Exception as exc:
        logger.warning(f"Stock search failed: {exc}", exc_info=True)
        return {"results": [], "finnhub_active": True, "error": f"Search error: {exc}"}


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
async def get_stock(ticker: str, current_user: User = Depends(get_current_user)):
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