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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import asyncio
import warnings
warnings.filterwarnings("ignore")

from ml.data_fetcher import (fetch_stock_data, fetch_multiple_stocks,
                              get_cache_status, clear_cache as clear_data_cache)
from ml.feature_engineering import build_features
from ml.model import NUMKTEnsemble
from ml.backtest import run_backtest
from ml.scoring import score_universe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NUMKT ML Backend v2",
    description="Self-improving ML backend — trains on real historical returns",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

# ── In-memory caches ───────────────────────────────────────────────────────
_model_cache: dict = {}   # profile_risk_lookback -> NUMKTEnsemble
_data_cache:  dict = {}   # "latest" -> raw_data, features_df


# ── Load any previously trained models on startup ─────────────────────────
@app.on_event("startup")
async def startup():
    saved = NUMKTEnsemble.list_saved()
    if saved:
        logger.info(f"Found {len(saved)} saved model(s):")
        for m in saved:
            logger.info(f"  {m['name']} — trained {m['trained_at']} "
                        f"on {m['n_training_rows']} rows ({m['training_source']}) "
                        f"CV={m['cv_accuracy']:.3f}")
        # Load the default model into cache
        model = NUMKTEnsemble.load("default")
        if model:
            _model_cache["__trained__"] = model
            logger.info("Loaded saved 'default' model into memory")
    else:
        logger.info("No saved models found — will use synthetic labels until backtest runs")


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


# ── Health ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    saved = NUMKTEnsemble.list_saved()
    return {
        "service":      "NUMKT ML Backend v2",
        "status":       "running",
        "timestamp":    datetime.utcnow().isoformat(),
        "saved_models": len(saved),
        "best_model":   saved[0] if saved else None,
    }

@app.get("/health")
def health():
    return {"status": "ok"}


# ── Main analysis ──────────────────────────────────────────────────────────

@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    if len(req.tickers) < 3:
        raise HTTPException(400, "Please provide at least 3 tickers.")
    if len(req.tickers) > 40:
        raise HTTPException(400, "Maximum 40 tickers per request.")

    logger.info(f"Analyse: {req.tickers} | profile={req.profile}")

    try:
        raw_data = await fetch_multiple_stocks(req.tickers, req.lookback_years+1)
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
            profile=req.profile, risk=req.risk, use_macro=req.use_macro,
        )

        # Cache for backtest
        _data_cache["latest"]           = raw_data
        _data_cache["latest_features"]  = features_df

        return {
            "status":             "ok",
            "timestamp":          datetime.utcnow().isoformat(),
            "profile":            req.profile,
            "risk":               req.risk,
            "cv_folds":           req.cv_folds,
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
        raise HTTPException(500, f"Model error: {str(e)}")


# ── Backtest + train ───────────────────────────────────────────────────────

@app.post("/backtest")
async def backtest(req: BacktestRequest):
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
            raw_data = await fetch_multiple_stocks(req.tickers, req.lookback_years+1)
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
            "timestamp":        datetime.utcnow().isoformat(),
            "tickers":          req.tickers,
            "backtest_period":  f"{req.lookback_years} years",
            "forward_months":   req.forward_months,
            "metrics":          metrics,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        raise HTTPException(500, f"Backtest error: {str(e)}")


# ── Cache endpoints ────────────────────────────────────────────────────────

@app.get("/cache/status")
def cache_status():
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
def clear_data(ticker: str | None = None):
    """Clear disk cache for one ticker or all."""
    clear_data_cache(ticker)
    return {"status": f"cleared {'all' if not ticker else ticker}"}

@app.delete("/cache/models")
def clear_model_cache():
    """Clear in-memory model cache (forces retrain on next /analyse)."""
    _model_cache.clear()
    return {"status": "model cache cleared"}


# ── Model endpoints ────────────────────────────────────────────────────────

@app.get("/model/info")
def model_info():
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
        raise HTTPException(500, str(e))