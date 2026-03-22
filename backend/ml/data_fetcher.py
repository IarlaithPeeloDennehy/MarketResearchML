"""
data_fetcher.py  (v2 — with disk cache)
────────────────────────────────────────
Fetches price + fundamental data from Yahoo Finance and caches it to disk.

Cache strategy:
  - Price history saved as parquet files in ./cache/prices/
  - Fundamentals saved as JSON in ./cache/info/
  - Cache is considered fresh for 24 hours
  - On cache hit: loads from disk instantly, no Yahoo request
  - On cache miss: fetches from Yahoo, saves to disk, returns data

This means:
  - First run per ticker: ~3s to fetch from Yahoo
  - All subsequent runs that day: ~0.05s from disk
  - Backtest can use years of cached history without any rate limits
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import time
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

# Cache directories — created automatically
CACHE_DIR   = Path(__file__).parent.parent / "cache"
PRICE_DIR   = CACHE_DIR / "prices"
INFO_DIR    = CACHE_DIR / "info"
CACHE_DIR.mkdir(exist_ok=True)
PRICE_DIR.mkdir(exist_ok=True)
INFO_DIR.mkdir(exist_ok=True)

CACHE_MAX_AGE_HOURS = 24   # refresh stale cache after this many hours
MIN_BARS            = 60   # minimum price bars to consider valid

_executor = ThreadPoolExecutor(max_workers=1)  # sequential = no rate limits


# ── Cache helpers ──────────────────────────────────────────────────────────

def _price_cache_path(ticker: str) -> Path:
    safe = ticker.replace(".", "_").replace("/", "_")
    return PRICE_DIR / f"{safe}.parquet"

def _info_cache_path(ticker: str) -> Path:
    safe = ticker.replace(".", "_").replace("/", "_")
    return INFO_DIR / f"{safe}.json"

def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age.total_seconds() < CACHE_MAX_AGE_HOURS * 3600

def _load_cached_prices(ticker: str) -> pd.DataFrame | None:
    path = _price_cache_path(ticker)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        logger.info(f"{ticker}: loaded {len(df)} bars from disk cache")
        return df
    except Exception as e:
        logger.warning(f"{ticker}: cache read failed ({e}), will re-fetch")
        return None

def _save_prices(ticker: str, df: pd.DataFrame):
    try:
        df.to_parquet(_price_cache_path(ticker))
    except Exception as e:
        logger.warning(f"{ticker}: could not save price cache: {e}")

def _load_cached_info(ticker: str) -> dict | None:
    path = _info_cache_path(ticker)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def _save_info(ticker: str, info: dict):
    try:
        # Strip non-serialisable values
        clean = {k: v for k, v in info.items()
                 if isinstance(v, (str, int, float, bool, type(None)))}
        with open(_info_cache_path(ticker), "w") as f:
            json.dump(clean, f)
    except Exception as e:
        logger.warning(f"{ticker}: could not save info cache: {e}")


# ── Core fetch ─────────────────────────────────────────────────────────────

def _fetch_one(ticker: str, start: str, end: str) -> dict | None:
    """
    Fetch one ticker. Returns cached data if fresh, otherwise hits Yahoo.
    Retries up to 3 times with backoff on failure.
    """
    # Check price cache first
    price_path = _price_cache_path(ticker)
    info_path  = _info_cache_path(ticker)

    cached_prices = _load_cached_prices(ticker) if price_path.exists() else None
    cached_info   = _load_cached_info(ticker)   if info_path.exists()  else None

    # If we have fresh cache that covers enough history, use it
    if cached_prices is not None and len(cached_prices) >= MIN_BARS and _cache_is_fresh(price_path):
        info = cached_info or {}
        logger.info(f"{ticker}: using fresh disk cache ({len(cached_prices)} bars)")
        return {"ticker": ticker, "prices": cached_prices, "info": info,
                "financials": pd.DataFrame(), "balance": pd.DataFrame()}

    # Otherwise fetch from Yahoo
    logger.info(f"{ticker}: fetching from Yahoo Finance (start={start})")

    for attempt in range(3):
        try:
            if attempt > 0:
                wait = attempt * 5
                logger.info(f"{ticker}: retry {attempt}/2 — waiting {wait}s")
                time.sleep(wait)

            t    = yf.Ticker(ticker)
            hist = t.history(start=start, end=end, auto_adjust=True)

            if hist.empty or len(hist) < MIN_BARS:
                logger.warning(f"{ticker}: only {len(hist)} bars returned")
                if attempt < 2:
                    continue
                # Return cached data even if stale rather than nothing
                if cached_prices is not None and len(cached_prices) >= MIN_BARS:
                    logger.info(f"{ticker}: using stale cache as fallback")
                    return {"ticker": ticker, "prices": cached_prices,
                            "info": cached_info or {},
                            "financials": pd.DataFrame(), "balance": pd.DataFrame()}
                return None

            prices = hist[["Close", "Volume"]].copy()
            prices.index = pd.to_datetime(prices.index).tz_localize(None)

            try:
                info = t.info or {}
            except Exception:
                info = cached_info or {}

            try:
                fin = t.financials
            except Exception:
                fin = pd.DataFrame()

            try:
                bal = t.balance_sheet
            except Exception:
                bal = pd.DataFrame()

            # Save to cache
            _save_prices(ticker, prices)
            _save_info(ticker, info)
            logger.info(f"{ticker}: fetched {len(prices)} bars, saved to cache")

            return {"ticker": ticker, "prices": prices, "info": info,
                    "financials": fin, "balance": bal}

        except Exception as e:
            logger.error(f"{ticker} attempt {attempt+1}/3 failed: {e}")
            if attempt == 2:
                # Last resort: return stale cache
                if cached_prices is not None and len(cached_prices) >= MIN_BARS:
                    logger.info(f"{ticker}: all retries failed, using stale cache")
                    return {"ticker": ticker, "prices": cached_prices,
                            "info": cached_info or {},
                            "financials": pd.DataFrame(), "balance": pd.DataFrame()}
                return None

    return None


# ── Public API ─────────────────────────────────────────────────────────────

async def fetch_stock_data(ticker: str, lookback_years: int = 5) -> dict | None:
    """Async wrapper. Returns flat dict with _raw for feature engineering."""
    end   = datetime.today()
    start = end - timedelta(days=365 * lookback_years + 60)

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, _fetch_one,
        ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )

    if result is None:
        return None

    info   = result["info"]
    prices = result["prices"]

    last_price   = float(prices["Close"].iloc[-1])   if not prices.empty   else None
    price_1y_ago = float(prices["Close"].iloc[-252]) if len(prices) >= 252 else None
    price_3m_ago = float(prices["Close"].iloc[-63])  if len(prices) >= 63  else None

    return {
        "ticker":         ticker,
        "name":           info.get("longName", ticker),
        "sector":         info.get("sector", "Unknown"),
        "market":         _detect_market(ticker),
        "currency":       info.get("currency", "USD"),
        "last_price":     last_price,
        "market_cap_bn":  round(info.get("marketCap", 0) / 1e9, 1),
        "pe":             info.get("trailingPE"),
        "forward_pe":     info.get("forwardPE"),
        "pb":             info.get("priceToBook"),
        "roe":            info.get("returnOnEquity"),
        "net_margin":     info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "debt_equity":    info.get("debtToEquity"),
        "dividend_yield": info.get("dividendYield"),
        "beta":           info.get("beta"),
        "momentum_12m":   round(last_price / price_1y_ago - 1, 4)
                          if last_price and price_1y_ago else None,
        "momentum_3m":    round(last_price / price_3m_ago - 1, 4)
                          if last_price and price_3m_ago else None,
        "_raw": result,
    }


async def fetch_multiple_stocks(tickers: list[str], lookback_years: int = 5) -> dict:
    """
    Fetches tickers sequentially with 1s gaps between uncached tickers.
    Cached tickers load instantly from disk with no delay needed.
    """
    out = {}
    uncached_count = 0

    for ticker in tickers:
        # Check if this ticker is already cached and fresh
        price_path = _price_cache_path(ticker)
        is_cached  = _cache_is_fresh(price_path)

        if not is_cached and uncached_count > 0:
            # Only delay when actually hitting Yahoo
            logger.info(f"Waiting 2s before fetching {ticker} from Yahoo")
            await asyncio.sleep(2)

        result = await fetch_stock_data(ticker, lookback_years)

        if result is not None:
            out[ticker] = result
            if not is_cached:
                uncached_count += 1
        else:
            logger.warning(f"Dropping {ticker} — no data available")

    logger.info(f"Fetched {len(out)}/{len(tickers)} tickers "
                f"({uncached_count} from Yahoo, {len(out)-uncached_count} from cache)")
    return out


def get_cache_status() -> dict:
    """Returns info about what's currently cached."""
    cached = list(PRICE_DIR.glob("*.parquet"))
    result = {}
    for f in cached:
        ticker = f.stem.replace("_", ".").replace("IR", ".IR").replace("L", ".L")
        try:
            df  = pd.read_parquet(f)
            age = datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)
            result[f.stem] = {
                "bars":     len(df),
                "from":     str(df.index[0].date()),
                "to":       str(df.index[-1].date()),
                "age_hours": round(age.total_seconds() / 3600, 1),
                "fresh":    age.total_seconds() < CACHE_MAX_AGE_HOURS * 3600,
            }
        except Exception:
            pass
    return result


def clear_cache(ticker: str | None = None):
    """Clear cache for one ticker or all tickers."""
    if ticker:
        _price_cache_path(ticker).unlink(missing_ok=True)
        _info_cache_path(ticker).unlink(missing_ok=True)
        logger.info(f"Cleared cache for {ticker}")
    else:
        for f in PRICE_DIR.glob("*.parquet"):
            f.unlink()
        for f in INFO_DIR.glob("*.json"):
            f.unlink()
        logger.info("Cleared all cache")


def _detect_market(ticker: str) -> str:
    t = ticker.upper()
    if t.endswith(".L"):  return "UK"
    if t.endswith(".IR"): return "IE"
    return "US"