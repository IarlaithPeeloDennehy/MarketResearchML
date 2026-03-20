"""
data_fetcher.py  (fixed)
────────────────────────
Fetches real price + fundamental data from Yahoo Finance.

Key fixes vs original:
  - Sequential fetching with 2s delay between tickers (was parallel/concurrent)
  - Retry logic: up to 3 attempts per ticker with exponential backoff
  - Reduced ThreadPoolExecutor workers from 10 to 1
  - These changes prevent Yahoo Finance rate limiting (the "Expecting value:
    line 1 column 1" error that killed all parallel requests)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

# Single worker — sequential fetching avoids Yahoo rate limits
_executor = ThreadPoolExecutor(max_workers=1)


def _fetch_one(ticker: str, start: str, end: str) -> dict | None:
    """
    Synchronous fetch of one ticker with retry logic.
    Retries up to 3 times with increasing delays on failure.
    """
    for attempt in range(3):
        try:
            if attempt > 0:
                wait = attempt * 5  # 5s then 10s
                logger.info(f"{ticker}: retry {attempt}/2 — waiting {wait}s")
                time.sleep(wait)

            t = yf.Ticker(ticker)

            # Price history
            hist = t.history(start=start, end=end, auto_adjust=True)

            if hist.empty or len(hist) < 60:
                logger.warning(f"{ticker}: insufficient price history ({len(hist)} bars)")
                if attempt < 2:
                    continue  # retry
                return None

            prices = hist[["Close", "Volume"]].copy()
            prices.index = pd.to_datetime(prices.index).tz_localize(None)

            # Fundamentals — wrap in try/except as info can fail independently
            try:
                info = t.info or {}
            except Exception as e:
                logger.warning(f"{ticker}: could not fetch info ({e}), using empty dict")
                info = {}

            # Income statement
            try:
                fin = t.financials
            except Exception:
                fin = pd.DataFrame()

            # Balance sheet
            try:
                bal = t.balance_sheet
            except Exception:
                bal = pd.DataFrame()

            logger.info(f"{ticker}: fetched {len(prices)} bars successfully")
            return {
                "ticker":     ticker,
                "prices":     prices,
                "info":       info,
                "financials": fin,
                "balance":    bal,
            }

        except Exception as e:
            logger.error(f"{ticker} attempt {attempt + 1}/3 failed: {e}")
            if attempt == 2:
                return None

    return None


async def fetch_stock_data(ticker: str, lookback_years: int = 4) -> dict | None:
    """Async wrapper — runs the blocking yfinance call in a thread."""
    end   = datetime.today()
    start = end - timedelta(days=365 * lookback_years + 30)

    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        _fetch_one,
        ticker,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )

    if result is None:
        return None

    # Build a flat snapshot for the /stock endpoint and for feature engineering
    info   = result["info"]
    prices = result["prices"]

    last_price   = float(prices["Close"].iloc[-1])   if not prices.empty       else None
    price_1y_ago = float(prices["Close"].iloc[-252]) if len(prices) >= 252     else None
    price_3m_ago = float(prices["Close"].iloc[-63])  if len(prices) >= 63      else None

    return {
        "ticker":        ticker,
        "name":          info.get("longName", ticker),
        "sector":        info.get("sector", "Unknown"),
        "market":        _detect_market(ticker),
        "currency":      info.get("currency", "USD"),
        "last_price":    last_price,
        "market_cap_bn": round(info.get("marketCap", 0) / 1e9, 1),
        "pe":            info.get("trailingPE"),
        "forward_pe":    info.get("forwardPE"),
        "pb":            info.get("priceToBook"),
        "roe":           info.get("returnOnEquity"),
        "net_margin":    info.get("profitMargins"),
        "revenue_growth":info.get("revenueGrowth"),
        "debt_equity":   info.get("debtToEquity"),
        "dividend_yield":info.get("dividendYield"),
        "beta":          info.get("beta"),
        "momentum_12m":  round(last_price / price_1y_ago - 1, 4) if last_price and price_1y_ago else None,
        "momentum_3m":   round(last_price / price_3m_ago  - 1, 4) if last_price and price_3m_ago  else None,
        "_raw": result,  # kept for feature engineering, not sent in JSON responses
    }


async def fetch_multiple_stocks(tickers: list[str], lookback_years: int = 4) -> dict:
    """
    Fetches tickers ONE AT A TIME with a 2-second gap between each.
    Sequential fetching is slower than parallel but avoids Yahoo rate limits.
    For 10 tickers this takes ~20-30 seconds — a worthwhile trade-off.
    """
    out = {}

    for i, ticker in enumerate(tickers):
        # Pause between requests — critical to avoid rate limiting
        if i > 0:
            logger.info(f"Waiting 2s before fetching {ticker} ({i+1}/{len(tickers)})")
            await asyncio.sleep(2)

        result = await fetch_stock_data(ticker, lookback_years)

        if result is not None:
            out[ticker] = result
        else:
            logger.warning(f"Dropping {ticker} — no data returned")

    logger.info(f"Fetched {len(out)}/{len(tickers)} tickers successfully")
    return out


def _detect_market(ticker: str) -> str:
    """Infer which market a ticker belongs to from its suffix."""
    t = ticker.upper()
    if t.endswith(".L"):
        return "UK"
    if t.endswith(".IR") or t.endswith(".I"):
        return "IE"
    return "US"