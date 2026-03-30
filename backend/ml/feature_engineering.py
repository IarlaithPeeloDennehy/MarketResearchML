"""
feature_engineering.py
───────────────────────
Turns raw Yahoo Finance data into a feature matrix suitable for sklearn.

Features engineered:
  Valuation:   pe_ratio, pb_ratio, ev_ebitda (if available)
  Profitability: roe, net_margin, gross_margin
  Growth:      revenue_growth, earnings_growth
  Safety:      debt_equity, current_ratio
  Momentum:    mom_1m, mom_3m, mom_6m, mom_12m (price return)
  Volatility:  realised_vol_60d, beta
  Size:        log_market_cap
  Technical:   rsi_14, price_vs_52w_high

The target variable (for training) is:
  forward_return_12m — total return over the next 12 months
  We convert this to a binary label: 1 if stock beat the equal-weight
  universe median return (i.e., top half), 0 otherwise.
  This avoids label leakage from absolute return levels.
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Features used as model inputs (must match what model.py expects)
FEATURE_COLS = [
    "pe_ratio",
    "pb_ratio",
    "roe",
    "net_margin",
    "revenue_growth",
    "debt_equity",
    "dividend_yield",
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "mom_12m",
    "vol_60d",
    "beta",
    "log_mktcap",
    "rsi_14",
    "price_vs_52w_high",
]


def _safe(val, default=np.nan):
    """Return val if it's a finite float, else default."""
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _compute_rsi(prices: pd.Series, period: int = 14) -> float:
    """Compute RSI for a price series. Returns latest RSI value."""
    delta = prices.diff().dropna()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean().iloc[-1]
    avg_loss = loss.ewm(span=period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - 100 / (1 + rs), 2)


def _momentum(prices: pd.Series, days: int) -> Optional[float]:
    """Simple price momentum over `days` trading days."""
    if len(prices) < days + 1:
        return np.nan
    current = prices.iloc[-1]
    past = prices.iloc[-(days + 1)]
    if past == 0:
        return np.nan
    return round(current / past - 1, 6)


def _realised_vol(prices: pd.Series, days: int = 60) -> float:
    """Annualised realised volatility from daily log returns."""
    if len(prices) < days + 1:
        return np.nan
    returns = np.log(prices.iloc[-days:] / prices.iloc[-days:].shift(1)).dropna()
    return round(float(returns.std() * np.sqrt(252)), 6)


def extract_snapshot_features(stock_data: dict) -> dict:
    """
    Extract a single feature row (today's snapshot) for one stock.
    Used for scoring current positions.
    """
    info = stock_data.get("info", {})
    prices = stock_data.get("prices")
    raw = stock_data.get("_raw", {})

    # If _raw not available (JSON response path), use flat fields
    if prices is None and "_raw" in stock_data:
        prices = stock_data["_raw"].get("prices")

    close = prices["Close"] if prices is not None and not prices.empty else None

    row = {
        "ticker":          stock_data.get("ticker", ""),
        "name":            stock_data.get("name", ""),
        "sector":          stock_data.get("sector", "Unknown"),
        "market":          stock_data.get("market", "US"),
        "last_price":      _safe(stock_data.get("last_price")),
        # --- Valuation ---
        "pe_ratio":        _safe(stock_data.get("pe") or (info.get("trailingPE"))),
        "pb_ratio":        _safe(stock_data.get("pb") or (info.get("priceToBook"))),
        # --- Profitability ---
        "roe":             _safe(stock_data.get("roe") or (info.get("returnOnEquity"))),
        "net_margin":      _safe(stock_data.get("net_margin") or (info.get("profitMargins"))),
        # --- Growth ---
        "revenue_growth":  _safe(stock_data.get("revenue_growth") or (info.get("revenueGrowth"))),
        # --- Safety ---
        "debt_equity":     _safe(stock_data.get("debt_equity") or (info.get("debtToEquity"))),
        "dividend_yield":  _safe(stock_data.get("dividend_yield") or (info.get("dividendYield"))),
        # --- Risk ---
        "beta":            _safe(stock_data.get("beta") or (info.get("beta"))),
        # --- Size ---
        "log_mktcap":      np.log(max(_safe(info.get("marketCap", 0)), 1e6)),
    }

    # Momentum (requires price history)
    if close is not None and len(close) > 0:
        row["mom_1m"]  = _safe(_momentum(close, 21))
        row["mom_3m"]  = _safe(_momentum(close, 63))
        row["mom_6m"]  = _safe(_momentum(close, 126))
        row["mom_12m"] = _safe(_momentum(close, 252))
        row["vol_60d"] = _safe(_realised_vol(close, 60))
        row["rsi_14"]  = _safe(_compute_rsi(close, 14))
        row["price_vs_52w_high"] = _safe(
            close.iloc[-1] / close.iloc[-252:].max() - 1 if len(close) >= 252 else np.nan
        )
    else:
        for col in ["mom_1m","mom_3m","mom_6m","mom_12m","vol_60d","rsi_14","price_vs_52w_high"]:
            row[col] = np.nan

    return row


def build_features(raw_data: dict, profile: str = "quality") -> pd.DataFrame:
    """
    Builds a feature DataFrame from fetched stock data.
    Each row = one stock's current snapshot features.

    Also generates a synthetic historical panel (rolling windows) so the
    model has enough rows to train on even with a small universe.
    Returns a DataFrame with FEATURE_COLS plus metadata columns.
    """
    rows = []
    for ticker, stock in raw_data.items():
        row = extract_snapshot_features(stock)
        row["ticker"] = ticker
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Winsorise extreme outliers (1st / 99th percentile) ─────────────────
    for col in FEATURE_COLS:
        if col in df.columns:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    # ── Cross-sectional rank normalisation (0–1) ────────────────────────────
    # Each feature is replaced by its percentile rank within the universe.
    # This makes the model robust to outliers and scale differences.
    for col in FEATURE_COLS:
        if col in df.columns and df[col].notna().sum() > 1:
            df[col + "_rank"] = df[col].rank(pct=True)

    # Fill remaining NaN with cross-sectional median
    for col in FEATURE_COLS:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median if pd.notna(median) else 0.5)

    logger.info(f"Built feature matrix: {df.shape[0]} stocks × {df.shape[1]} columns")
    return df
