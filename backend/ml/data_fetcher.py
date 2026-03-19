"""
data_fetcher.py
───────────────
Fetches real price history and fundamental data from Yahoo Finance (yfinance).

Yahoo Finance gives us:
  - Daily OHLCV price history (free, no API key needed)
  - Key fundamental ratios: P/E, P/B, ROE, margins, revenue growth, debt/equity,
    dividend yield, beta, market cap

We fetch async using asyncio + ThreadPoolExecutor so multiple tickers
download in parallel rather than sequentially.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger(__name__)

# Thread pool for blocking yfinance calls
_executor = ThreadPoolExecutor(max_workers=10)


def _fetch_one(ticker: str, start: str, end: str) -> dict | None:
    """
    Synchronous fetch of one ticker. Runs inside a thread.
    Returns a dict with:
      - 'prices':      pd.DataFrame  (daily Close prices, indexed by date)
      - 'info':        dict          (fundamentals from yf.Ticker.info)
      - 'financials':  pd.DataFrame  (income statement)
      - 'balance':     pd.DataFrame  (balance sheet)
    """
    try:
        t = yf.Ticker(ticker)

        # Price history
        hist = t.history(start=start, end=end, auto_adjust=True)
        if hist.empty or len(hist) < 60:
            logger.warning(f"{ticker}: insufficient price history")
            return None

        prices = hist[["Close", "Volume"]].copy()
        prices.index = pd.to_datetime(prices.index).tz_localize(None)

        # Fundamentals
        info = t.info or {}

        # Income statement (annual)
        try:
            fin = t.financials  # columns = dates, rows = line items
        except Exception:
            fin = pd.DataFrame()

        # Balance sheet
        try:
            bal = t.balance_sheet
        except Exception:
            bal = pd.DataFrame()

        return {
            "ticker": ticker,
            "prices": prices,
            "info": info,
            "financials": fin,
            "balance": bal,
        }

    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return None


async def fetch_stock_data(ticker: str, lookback_years: int = 4) -> dict | None:
    """Async wrapper around the synchronous yfinance fetch."""
    end = datetime.today()
    start = end - timedelta(days=365 * lookback_years + 30)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor, _fetch_one, ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )

    if result is None:
        return None

    # Flatten into a JSON-serialisable snapshot for the /stock endpoint
    info = result["info"]
    prices = result["prices"]
    last_price = float(prices["Close"].iloc[-1]) if not prices.empty else None
    price_1y_ago = float(prices["Close"].iloc[-252]) if len(prices) >= 252 else None
    price_3m_ago = float(prices["Close"].iloc[-63])  if len(prices) >= 63  else None

    return {
        "ticker": ticker,
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "Unknown"),
        "market": _detect_market(ticker),
        "currency": info.get("currency", "USD"),
        "last_price": last_price,
        "market_cap_bn": round(info.get("marketCap", 0) / 1e9, 1),
        "pe": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "pb": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "net_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "debt_equity": info.get("debtToEquity"),
        "dividend_yield": info.get("dividendYield"),
        "beta": info.get("beta"),
        "momentum_12m": round((last_price / price_1y_ago - 1), 4) if last_price and price_1y_ago else None,
        "momentum_3m":  round((last_price / price_3m_ago  - 1), 4) if last_price and price_3m_ago  else None,
        "_raw": result,  # kept for feature engineering, stripped before JSON response
    }


async def fetch_multiple_stocks(tickers: list[str], lookback_years: int = 4) -> dict:
    """
    Fetches all tickers concurrently. Returns dict keyed by ticker.
    Silently drops any tickers that fail to fetch.
    """
    tasks = [fetch_stock_data(t, lookback_years) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    out = {}
    for ticker, result in zip(tickers, results):
        if result is not None:
            out[ticker] = result
        else:
            logger.warning(f"Dropping {ticker} — no data returned")

    logger.info(f"Fetched {len(out)}/{len(tickers)} tickers successfully")
    return out


def _detect_market(ticker: str) -> str:
    """Infer market from ticker suffix."""
    t = ticker.upper()
    if t.endswith(".L"):
        return "UK"
    if t.endswith(".IR") or t.endswith(".I"):
        return "IE"
    return "US"
