"""
NUMKT ML Backend
────────────────
FastAPI server that:
  1. Fetches real fundamental + price data from Yahoo Finance
  2. Trains a Random Forest + Gradient Boosting ensemble on historical data
  3. Uses proper time-series cross-validation (no data leakage)
  4. Returns scored, ranked signals with factor breakdowns
  5. Exposes a /backtest endpoint so you can measure model quality

Run locally:  uvicorn main:app --reload --port 8000
Deploy:       Railway auto-detects Procfile and requirements.txt
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import asyncio
import warnings
warnings.filterwarnings("ignore")

from ml.data_fetcher import fetch_stock_data, fetch_multiple_stocks
from ml.feature_engineering import build_features
from ml.model import NUMKTEnsemble
from ml.backtest import run_backtest
from ml.scoring import score_universe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NUMKT ML Backend",
    description="Real ML backend for NUMKT multi-market stock analyser",
    version="1.0.0"
)

# Allow requests from GitHub Pages domain and localhost for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://*.github.io",   # your GitHub Pages domain
        "http://localhost:*",
        "http://127.0.0.1:*",
    ],
    allow_origin_regex=r"https://.*\.github\.io",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── In-memory model cache ──────────────────────────────────────────────────
# Keeps the last trained model so repeated /analyse calls are fast
_model_cache: dict = {}


# ── Request / Response schemas ─────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    tickers: List[str]
    profile: str = "quality"       # value | growth | dividend | momentum | quality | macro
    risk: str = "medium"           # low | medium | high
    lookback_years: int = 3        # how many years of history to train on
    cv_folds: int = 5
    lambda_reg: float = 0.10       # L2 regularisation strength
    use_macro: bool = True
    use_hf_signals: bool = True


class BacktestRequest(BaseModel):
    tickers: List[str]
    profile: str = "quality"
    lookback_years: int = 5
    forward_months: int = 12       # how far ahead to measure signal quality


# ── Health check ───────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "NUMKT ML Backend",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Main analysis endpoint ─────────────────────────────────────────────────

@app.post("/analyse")
async def analyse(req: AnalyseRequest):
    """
    Core endpoint. Takes a list of tickers and returns ML-scored signals.

    Steps:
      1. Fetch real price + fundamental data via yfinance
      2. Engineer features (factor ratios, momentum, volatility)
      3. Train RF + GB ensemble with TimeSeriesSplit CV
      4. Score each stock, return ranked results with explanation
    """
    if len(req.tickers) < 3:
        raise HTTPException(400, "Please provide at least 3 tickers.")
    if len(req.tickers) > 40:
        raise HTTPException(400, "Maximum 40 tickers per request.")

    logger.info(f"Analyse request: {req.tickers} | profile={req.profile}")

    try:
        # 1. Fetch data (parallel async calls)
        raw_data = await fetch_multiple_stocks(
            req.tickers,
            lookback_years=req.lookback_years + 1  # extra year for feature lags
        )

        if not raw_data:
            raise HTTPException(502, "Could not fetch data for any of the provided tickers.")

        # 2. Feature engineering
        features_df = build_features(raw_data, profile=req.profile)

        if features_df.empty:
            raise HTTPException(422, "Not enough historical data to build features.")

        # 3. Train / retrieve cached model
        cache_key = f"{req.profile}_{req.risk}_{req.lookback_years}"
        if cache_key not in _model_cache:
            model = NUMKTEnsemble(
                cv_folds=req.cv_folds,
                lambda_reg=req.lambda_reg,
                max_depth=5,
            )
            model.fit(features_df)
            _model_cache[cache_key] = model
            logger.info(f"Trained new model for cache key: {cache_key}")
        else:
            model = _model_cache[cache_key]
            logger.info(f"Using cached model: {cache_key}")

        # 4. Score current snapshot
        results = score_universe(
            model=model,
            features_df=features_df,
            raw_data=raw_data,
            profile=req.profile,
            risk=req.risk,
            use_macro=req.use_macro,
        )

        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "profile": req.profile,
            "risk": req.risk,
            "cv_folds": req.cv_folds,
            "cv_accuracy": round(model.cv_accuracy, 4),
            "feature_importance": model.feature_importance(),
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        raise HTTPException(500, f"Model error: {str(e)}")


# ── Backtest endpoint ──────────────────────────────────────────────────────

@app.post("/backtest")
async def backtest(req: BacktestRequest):
    """
    Runs a walk-forward backtest and returns industry-standard metrics:
      - Hit rate (% of BUY signals that outperformed benchmark)
      - Information Coefficient (Spearman rank correlation)
      - Sharpe Ratio of signal-based portfolio
      - Max Drawdown
      - Alpha vs benchmark
      - Calmar Ratio
    """
    if len(req.tickers) < 5:
        raise HTTPException(400, "Backtest requires at least 5 tickers.")

    logger.info(f"Backtest request: {req.tickers}")

    try:
        raw_data = await fetch_multiple_stocks(
            req.tickers,
            lookback_years=req.lookback_years + 1
        )

        if not raw_data:
            raise HTTPException(502, "Could not fetch backtest data.")

        features_df = build_features(raw_data, profile=req.profile)
        metrics = run_backtest(
            features_df=features_df,
            raw_data=raw_data,
            profile=req.profile,
            forward_months=req.forward_months,
        )

        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "tickers": req.tickers,
            "backtest_period": f"{req.lookback_years} years",
            "forward_months": req.forward_months,
            "metrics": metrics,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        raise HTTPException(500, f"Backtest error: {str(e)}")


# ── Single ticker data endpoint ────────────────────────────────────────────

@app.get("/stock/{ticker}")
async def get_stock(ticker: str):
    """Returns raw fundamentals + price history for a single ticker."""
    try:
        data = await fetch_stock_data(ticker.upper(), lookback_years=2)
        if not data:
            raise HTTPException(404, f"No data found for {ticker}")
        return {"status": "ok", "ticker": ticker.upper(), "data": data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Model info endpoint ────────────────────────────────────────────────────

@app.get("/model/info")
def model_info():
    """Returns info about currently cached models."""
    return {
        "cached_models": list(_model_cache.keys()),
        "count": len(_model_cache),
    }


@app.delete("/model/cache")
def clear_cache():
    """Clears model cache — forces retrain on next request."""
    _model_cache.clear()
    return {"status": "cache cleared"}
