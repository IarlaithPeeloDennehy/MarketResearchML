"""
data_fetcher.py  (v3 — Finnhub primary, yfinance fallback)
────────────────────────────────────────────────────────────
US tickers   → Finnhub official REST API (requires FINNHUB_API_KEY env var)
UK/IE tickers → yfinance (unchanged path)
No key set   → all tickers fall back to yfinance (safe for local dev)

Cache strategy unchanged:
  - Price history  → ./cache/prices/{TICKER}.parquet  (7-day TTL)
  - Fundamentals   → ./cache/info/{TICKER}.json       (7-day TTL)
  - Cache hit:  ~0.05 s from disk, zero API calls
  - Cache miss: fetch from source, save, return

Finnhub notes:
  - Free tier: 60 calls/minute
  - Daily candles: 1 year max per call → stitched in 1-year chunks
  - New info fields added: fcf_yield, ev_ebitda, earnings_surprise_avg,
    next_earnings_date, analyst_buy, analyst_hold, analyst_sell
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

try:
    import finnhub as _finnhub_lib
    _FINNHUB_AVAILABLE = True
except ImportError:
    _FINNHUB_AVAILABLE = False

from .feature_engineering import safe_ticker_filename

logger = logging.getLogger(__name__)

# ── Cache setup ────────────────────────────────────────────────────────────

_cache_root = os.environ.get("CACHE_DIR")
CACHE_DIR   = Path(_cache_root) if _cache_root else Path(__file__).parent.parent / "cache"
PRICE_DIR   = CACHE_DIR / "prices"
INFO_DIR    = CACHE_DIR / "info"
CACHE_DIR.mkdir(exist_ok=True)
PRICE_DIR.mkdir(exist_ok=True)
INFO_DIR.mkdir(exist_ok=True)

CACHE_MAX_AGE_HOURS = 168  # 7 days
MIN_BARS            = 60

_executor = ThreadPoolExecutor(max_workers=1)

# ── Cache helpers ──────────────────────────────────────────────────────────

def _price_cache_path(ticker: str) -> Path:
    return PRICE_DIR / f"{safe_ticker_filename(ticker)}.parquet"

def _info_cache_path(ticker: str) -> Path:
    return INFO_DIR / f"{safe_ticker_filename(ticker)}.json"

def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        age = datetime.now().timestamp() - path.stat().st_mtime
        return age < CACHE_MAX_AGE_HOURS * 3600
    except Exception:
        return False


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
        clean = {k: v for k, v in info.items()
                 if isinstance(v, (str, int, float, bool, type(None)))}
        with open(_info_cache_path(ticker), "w") as f:
            json.dump(clean, f)
    except Exception as e:
        logger.warning(f"{ticker}: could not save info cache: {e}")


# ── Finnhub helpers ────────────────────────────────────────────────────────

def _is_us_ticker(ticker: str) -> bool:
    """UK (.L) and Irish exchange (.IR) tickers stay on yfinance."""
    t = ticker.upper()
    return not (t.endswith(".L") or t.endswith(".IR"))

_fh_client = None
_fh_init_logged = False   # only log the config state once per process

def _get_finnhub_client():
    global _fh_client, _fh_init_logged
    if _fh_client is not None:
        return _fh_client
    key = os.environ.get("FINNHUB_API_KEY", "").strip()
    if not key:
        if not _fh_init_logged:
            logger.warning(
                "FINNHUB_API_KEY is not set — all tickers will use Yahoo Finance. "
                "Set the key in your Render environment to enable Finnhub."
            )
            _fh_init_logged = True
        return None
    if not _FINNHUB_AVAILABLE:
        if not _fh_init_logged:
            logger.warning("finnhub-python package not installed — falling back to Yahoo Finance.")
            _fh_init_logged = True
        return None
    try:
        _fh_client = _finnhub_lib.Client(api_key=key)
        if not _fh_init_logged:
            logger.info(
                "Finnhub client initialised — US tickers will use the official Finnhub API. "
                "UK/IE tickers (.L / .IR) remain on Yahoo Finance."
            )
            _fh_init_logged = True
        return _fh_client
    except Exception as e:
        logger.warning(f"Finnhub client init failed: {e}")
        return None


def _fetch_candles_finnhub(client, ticker: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame | None:
    """Fetch daily OHLCV from Finnhub in 1-year chunks and stitch into one DataFrame."""
    all_frames = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=365), end_dt)
        ts_from = int(chunk_start.timestamp())
        ts_to   = int(chunk_end.timestamp())
        try:
            data = client.stock_candles(ticker, "D", ts_from, ts_to)
        except Exception as e:
            logger.warning(f"{ticker}: Finnhub candle chunk failed: {e}")
            break
        if data and data.get("s") == "ok" and data.get("c"):
            df = pd.DataFrame(
                {"Close": data["c"], "Volume": data["v"]},
                index=pd.to_datetime(data["t"], unit="s"),
            )
            df.index = df.index.tz_localize(None)
            all_frames.append(df)
        chunk_start = chunk_end + timedelta(days=1)

    if not all_frames:
        return None
    combined = pd.concat(all_frames).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def _fetch_one_finnhub(ticker: str, client, start_dt: datetime, end_dt: datetime) -> dict | None:
    """Fetch all data for one US ticker from Finnhub. Returns the same dict
    shape as _fetch_one so all downstream code is unchanged."""
    try:
        prices = _fetch_candles_finnhub(client, ticker, start_dt, end_dt)
        if prices is None or len(prices) < MIN_BARS:
            logger.warning(f"{ticker}: Finnhub returned <{MIN_BARS} bars")
            return None

        # Company profile (name, sector, currency, market cap)
        profile = {}
        try:
            profile = client.company_profile2(symbol=ticker) or {}
        except Exception as e:
            logger.warning(f"{ticker}: Finnhub profile failed: {e}")

        # 117 fundamental metrics
        m = {}
        try:
            resp = client.company_basic_financials(ticker, "all") or {}
            m = resp.get("metric", {})
        except Exception as e:
            logger.warning(f"{ticker}: Finnhub metrics failed: {e}")

        # Earnings surprises (last 4 quarters) → PEAD signal
        earnings_surprise_avg = None
        try:
            earnings = client.company_earnings(ticker, limit=4) or []
            surprises = [
                e["surprisePercent"] for e in earnings
                if e.get("surprisePercent") is not None
            ]
            if surprises:
                avg_pct = sum(surprises) / len(surprises)
                # 50% avg beat → 1.0; -50% → -1.0; clipped to [-1, 1]
                earnings_surprise_avg = max(-1.0, min(1.0, avg_pct / 50.0))
        except Exception as e:
            logger.warning(f"{ticker}: Finnhub earnings failed: {e}")

        # Analyst consensus (most recent month)
        analyst_buy = analyst_hold = analyst_sell = None
        try:
            rec = client.recommendation_trends(ticker) or []
            if rec:
                r = rec[0]
                analyst_buy  = (r.get("buy") or 0) + (r.get("strongBuy") or 0)
                analyst_hold = r.get("hold") or 0
                analyst_sell = (r.get("sell") or 0) + (r.get("strongSell") or 0)
        except Exception as e:
            logger.warning(f"{ticker}: Finnhub recommendations failed: {e}")

        # Next earnings date (within 90 days)
        next_earnings_date = None
        try:
            today  = datetime.today()
            future = today + timedelta(days=90)
            cal = client.earnings_calendar(
                _from=today.strftime("%Y-%m-%d"),
                to=future.strftime("%Y-%m-%d"),
                symbol=ticker,
            ) or {}
            upcoming = cal.get("earningsCalendar", [])
            if upcoming:
                next_earnings_date = upcoming[0].get("date")
        except Exception as e:
            logger.warning(f"{ticker}: Finnhub earnings calendar failed: {e}")

        # Institutional ownership %
        held_pct_institutions = None
        try:
            own = client.fund_ownership(ticker, limit=20) or {}
            holdings = own.get("ownership", [])
            if holdings:
                pcts = [h.get("percentageHeld") for h in holdings if h.get("percentageHeld")]
                if pcts:
                    held_pct_institutions = min(1.0, sum(pcts) / 100.0)
        except Exception as e:
            logger.debug(f"{ticker}: Finnhub fund ownership unavailable: {e}")

        # Dividend yield: Finnhub currentDividendYieldTTM is in %, Yahoo uses decimal
        raw_yield = m.get("currentDividendYieldTTM")
        dividend_yield = (raw_yield / 100.0) if raw_yield is not None else None

        # FCF yield: Finnhub freeCashFlowYieldTTM is in %, convert to decimal
        raw_fcf = m.get("freeCashFlowYieldTTM")
        fcf_yield = (raw_fcf / 100.0) if raw_fcf is not None else None

        # Build info dict using Yahoo-compatible key names for existing fields
        info = {
            # Metadata
            "longName":   profile.get("name", ticker),
            "sector":     profile.get("finnhubIndustry", "Unknown"),
            "currency":   profile.get("currency", "USD"),
            "marketCap":  (profile.get("marketCapitalization") or 0) * 1_000_000,
            # Valuation ratios
            "trailingPE":  m.get("peNormalizedAnnual"),
            "forwardPE":   m.get("forwardPE"),
            "priceToBook": m.get("pbAnnual"),
            # Quality
            "returnOnEquity": m.get("roeAnnual"),
            "profitMargins":  m.get("netProfitMarginAnnual"),
            "revenueGrowth":  m.get("revenueGrowthTTMYoy"),
            # Safety
            "debtToEquity":  m.get("totalDebt/totalEquityAnnual"),
            "dividendYield": dividend_yield,
            "beta":          m.get("beta"),
            # Ownership
            "heldPercentInstitutions": held_pct_institutions,
            "heldPercentInsiders":     None,
            # ── New Phase 2 ML features ──────────────────────────────────
            "fcf_yield":             fcf_yield,
            "ev_ebitda":             m.get("evToEbitdaAnnual") or m.get("evToEbitda"),
            "earnings_surprise_avg": earnings_surprise_avg,
            # ── New Phase 3 UX fields ────────────────────────────────────
            "next_earnings_date": next_earnings_date,
            "analyst_buy":        analyst_buy,
            "analyst_hold":       analyst_hold,
            "analyst_sell":       analyst_sell,
        }

        n_chunks = max(1, (end_dt - start_dt).days // 365)
        logger.info(f"{ticker}: fetched from Finnhub ({n_chunks} candle chunks, {len(prices)} bars)")
        return {
            "ticker": ticker, "prices": prices, "info": info,
            "financials": pd.DataFrame(), "balance": pd.DataFrame(),
        }

    except Exception as e:
        logger.warning(f"{ticker}: Finnhub fetch failed: {e}")
        return None


# ── Core fetch ─────────────────────────────────────────────────────────────

def _fetch_one(ticker: str, start: str, end: str) -> dict | None:
    """Fetch one ticker. Uses Finnhub for US tickers (if key present),
    yfinance for UK/IE tickers or as fallback. Caches both paths identically."""
    price_path = _price_cache_path(ticker)
    info_path  = _info_cache_path(ticker)

    cached_prices = _load_cached_prices(ticker) if price_path.exists() else None
    cached_info   = _load_cached_info(ticker)   if info_path.exists()  else None

    if cached_prices is not None and len(cached_prices) >= MIN_BARS and _cache_is_fresh(price_path):
        info = cached_info or {}
        logger.info(f"{ticker}: using fresh disk cache ({len(cached_prices)} bars)")
        return {"ticker": ticker, "prices": cached_prices, "info": info,
                "financials": pd.DataFrame(), "balance": pd.DataFrame()}

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    # ── Finnhub path (US tickers only) ─────────────────────────────────────
    if _is_us_ticker(ticker):
        client = _get_finnhub_client()
        if client is not None:
            result = _fetch_one_finnhub(ticker, client, start_dt, end_dt)
            if result is not None:
                _save_prices(ticker, result["prices"])
                _save_info(ticker, result["info"])
                return result
            logger.warning(f"{ticker}: Finnhub fetch failed — falling back to Yahoo Finance")
        else:
            logger.info(f"{ticker}: no Finnhub client — fetching from Yahoo Finance")

    # ── yfinance path (UK/IE tickers, or Finnhub fallback) ─────────────────
    source = "Yahoo Finance (UK/IE)" if not _is_us_ticker(ticker) else "Yahoo Finance (fallback)"
    logger.info(f"{ticker}: fetching from {source} (start={start})")

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

            _save_prices(ticker, prices)
            _save_info(ticker, info)
            logger.info(f"{ticker}: fetched {len(prices)} bars from Yahoo, saved to cache")

            return {"ticker": ticker, "prices": prices, "info": info,
                    "financials": fin, "balance": bal}

        except Exception as e:
            logger.error(f"{ticker} attempt {attempt+1}/3 failed: {e}")
            if attempt == 2:
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
        "inst_ownership":    info.get("heldPercentInstitutions"),
        "insider_ownership": info.get("heldPercentInsiders"),
        # New fields
        "fcf_yield":             info.get("fcf_yield"),
        "ev_ebitda":             info.get("ev_ebitda"),
        "earnings_surprise_avg": info.get("earnings_surprise_avg"),
        "next_earnings_date":    info.get("next_earnings_date"),
        "analyst_buy":           info.get("analyst_buy"),
        "analyst_hold":          info.get("analyst_hold"),
        "analyst_sell":          info.get("analyst_sell"),
        "_raw": result,
    }


async def fetch_multiple_stocks(
    tickers: list[str], lookback_years: int = 5
) -> tuple[dict, int]:
    """Return (data_dict, uncached_count)."""
    out = {}
    uncached_count = 0

    for ticker in tickers:
        price_path = _price_cache_path(ticker)

        if _cache_is_fresh(price_path):
            cached = _load_cached_prices(ticker)
            if cached is not None and len(cached) >= MIN_BARS:
                info = _load_cached_info(ticker) or {}
                result = {
                    "ticker":     ticker,
                    "prices":     cached,
                    "info":       info,
                    "financials": pd.DataFrame(),
                    "balance":    pd.DataFrame(),
                }
                flat = await _build_flat(ticker, result)
                if flat:
                    out[ticker] = flat
                    source = "Finnhub cache" if _is_us_ticker(ticker) else "Yahoo cache"
                    logger.info(f"{ticker}: served from disk cache ({source})")
                    continue

        if uncached_count > 0:
            delay = 0.5 if _is_us_ticker(ticker) else 2
            logger.info(f"Waiting {delay}s before fetching {ticker}")
            await asyncio.sleep(delay)

        result = await fetch_stock_data(ticker, lookback_years)
        if result is not None:
            out[ticker] = result
            uncached_count += 1
        else:
            logger.warning(f"Dropping {ticker} — no data available")

    fh_count  = sum(1 for t in out if _is_us_ticker(t))
    yf_count  = sum(1 for t in out if not _is_us_ticker(t))
    fh_label  = "Finnhub" if _get_finnhub_client() is not None else "Yahoo Finance (no Finnhub key)"
    logger.info(
        f"Data fetch complete: {len(out)}/{len(tickers)} tickers loaded — "
        f"{uncached_count} fresh, {len(out)-uncached_count} from cache | "
        f"US ({fh_count}): {fh_label} | UK/IE ({yf_count}): Yahoo Finance"
    )
    return out, uncached_count


def get_cache_status() -> dict:
    cached = list(PRICE_DIR.glob("*.parquet"))
    result = {}
    for f in cached:
        try:
            df  = pd.read_parquet(f)
            age = datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)
            result[f.stem] = {
                "bars":      len(df),
                "from":      str(df.index[0].date()),
                "to":        str(df.index[-1].date()),
                "age_hours": round(age.total_seconds() / 3600, 1),
                "fresh":     age.total_seconds() < CACHE_MAX_AGE_HOURS * 3600,
            }
        except Exception:
            pass
    return result


async def _build_flat(ticker: str, result: dict) -> dict | None:
    """Build the flat response dict from a cached result without hitting any API."""
    try:
        info   = result["info"]
        prices = result["prices"]
        last_price   = float(prices["Close"].iloc[-1])
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
            "momentum_12m":      round(last_price/price_1y_ago-1,4) if price_1y_ago else None,
            "momentum_3m":       round(last_price/price_3m_ago-1,4) if price_3m_ago else None,
            "inst_ownership":    info.get("heldPercentInstitutions"),
            "insider_ownership": info.get("heldPercentInsiders"),
            "fcf_yield":             info.get("fcf_yield"),
            "ev_ebitda":             info.get("ev_ebitda"),
            "earnings_surprise_avg": info.get("earnings_surprise_avg"),
            "next_earnings_date":    info.get("next_earnings_date"),
            "analyst_buy":           info.get("analyst_buy"),
            "analyst_hold":          info.get("analyst_hold"),
            "analyst_sell":          info.get("analyst_sell"),
            "_raw": result,
        }
    except Exception as e:
        logger.warning(f"{ticker}: could not build flat from cache: {e}")
        return None


def clear_cache(ticker: str | None = None):
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
