"""
scoring.py
──────────
Takes fitted model + feature data → produces ranked BUY/HOLD/SELL signals.

Signal thresholds (tunable):
  score > 0.68  →  STRONG BUY
  score > 0.54  →  BUY
  score > 0.38  →  HOLD
  else          →  SELL

Each result includes:
  - composite_score    (0–100, from model probability minus risk penalty)
  - fundamental_score  (raw ML model probability)
  - ff5_scores         (Fama-French 5-factor decomposition)
  - signal             (STRONG BUY / BUY / HOLD / SELL)
  - buy_reasons        (list of plain-English reasons, derived from factors)
  - key_fundamentals   (subset of data for display)
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

from .feature_engineering import FEATURE_COLS, extract_snapshot_features
from .model import NUMKTEnsemble

logger = logging.getLogger(__name__)


def _ff5_decomposition(row: pd.Series) -> dict:
    """
    Fama-French 5-factor decomposition for a single stock.
    Returns factor contributions as percentages.
    """
    pe  = row.get("pe_ratio",    20)
    mg  = row.get("net_margin",  0.1)
    rg  = row.get("revenue_growth", 0.05)
    mc  = row.get("log_mktcap", np.log(10e9))
    m12 = row.get("mom_12m",    0.05)

    hml = 0.03 if (pe > 0 and 1/pe > 0.06) else (-0.02 if pe > 0 and 1/pe < 0.02 else 0)
    rmw = 0.025 if mg > 0.20 else (-0.015 if mg < 0.08 else 0.005)
    cma = 0.015 if rg < 0.05 else (-0.020 if rg > 0.20 else 0)
    smb = 0.04  if mc < np.log(1e10) else (0.01 if mc < np.log(5e10) else -0.01)
    mom = 0.03  if m12 > 0.20 else (-0.025 if m12 < -0.05 else 0.005)
    alpha = round(hml + rmw + cma + mom + 0.005, 4)

    return {
        "hml": round(hml * 100, 2),
        "rmw": round(rmw * 100, 2),
        "cma": round(cma * 100, 2),
        "smb": round(smb * 100, 2),
        "mom": round(mom * 100, 2),
        "alpha_est": round(alpha * 100, 2),
    }


def _get_signal(score: float) -> str:
    if score > 0.68: return "STRONG BUY"
    if score > 0.54: return "BUY"
    if score > 0.38: return "HOLD"
    return "SELL"


def _build_reasons(row: pd.Series, signal: str, inst_pct: float | None, insider_pct: float | None) -> list[str]:
    """Generate plain-English factor reasons for the signal."""
    reasons = []
    pe  = row.get("pe_ratio")
    roe = row.get("roe")
    mg  = row.get("net_margin")
    de  = row.get("debt_equity")
    rg  = row.get("revenue_growth")
    m12 = row.get("mom_12m")
    dy  = row.get("dividend_yield")

    is_buy = signal in ("STRONG BUY", "BUY")

    if is_buy:
        if pe and pe < 18:
            reasons.append(f"Attractive valuation at P/E {pe:.1f}x — below market average")
        if roe and roe > 0.20:
            reasons.append(f"Strong ROE of {roe*100:.0f}% — durable competitive advantage")
        if mg and mg > 0.15:
            reasons.append(f"High net margin of {mg*100:.1f}% — pricing power and scalability")
        if de and de < 0.5:
            reasons.append(f"Conservative balance sheet at {de:.2f}x D/E — financial flexibility")
        if rg and rg > 0.12:
            reasons.append(f"Revenue growing at {rg*100:.0f}% — compounding top-line momentum")
        if m12 and m12 > 0.15:
            reasons.append(f"Strong 12M momentum (+{m12*100:.0f}%)")
        if dy and dy > 0.03:
            reasons.append(f"Dividend yield of {dy*100:.1f}% — income with quality backing")
        if inst_pct and inst_pct > 0.70:
            reasons.append(f"{inst_pct*100:.0f}% institutional ownership — strong professional conviction")
        if insider_pct and insider_pct > 0.10:
            reasons.append(f"Insiders hold {insider_pct*100:.0f}% — management aligned with shareholders")
    else:
        if pe and pe > 50:
            reasons.append(f"Stretched P/E of {pe:.0f}x requires perfect execution")
        if m12 and m12 < -0.08:
            reasons.append(f"Negative 12M trend ({m12*100:.0f}%) — market expressing concern")
        if de and de > 2.0:
            reasons.append(f"High leverage at {de:.2f}x D/E — elevated refinancing risk")
        if rg and rg < 0:
            reasons.append(f"Declining revenues ({rg*100:.0f}%) — structural headwinds")
        if mg and mg < 0.05:
            reasons.append(f"Very thin margins ({mg*100:.1f}%) — limited operational resilience")
        if inst_pct and inst_pct < 0.30:
            reasons.append(f"Low institutional interest ({inst_pct*100:.0f}%) — limited professional coverage")

    return reasons[:4]


def score_universe(
    model: NUMKTEnsemble,
    features_df: pd.DataFrame,
    raw_data: dict,
    profile: str,
    risk: str,
) -> list[dict]:
    """
    Scores every stock in the universe and returns a ranked list.
    composite_score = ML model probability - risk penalty (clipped to [0.05, 0.97]).
    """
    # Build feature matrix for prediction
    feat_cols = model._feature_names
    if not feat_cols or features_df.empty:
        return []

    # Ensure all columns exist
    for col in feat_cols:
        if col not in features_df.columns:
            features_df[col] = 0.5

    X = features_df[feat_cols].values

    # Get model probabilities
    proba = model.predict_proba(X)

    results = []
    for i, (_, row) in enumerate(features_df.iterrows()):
        ticker  = row.get("ticker", "")
        sector  = row.get("sector", "Unknown")
        market  = row.get("market", "US")

        fund_score = float(proba[i])

        # Risk penalty for high-beta stocks
        beta = row.get("beta", 1.0) or 1.0
        risk_penalty = 0.0
        if risk == "low"    and beta > 0.8:  risk_penalty = (beta - 0.8) * 0.15
        if risk == "medium" and beta > 1.5:  risk_penalty = (beta - 1.5) * 0.10

        # Composite score: ML probability minus risk penalty only
        composite = float(np.clip(fund_score - risk_penalty, 0.05, 0.97))

        # Real ownership data from Yahoo Finance
        stock = raw_data.get(ticker, {})
        inst_pct    = _fmt(stock.get("inst_ownership"))
        insider_pct = _fmt(stock.get("insider_ownership"))

        signal   = _get_signal(composite)
        ff5      = _ff5_decomposition(row)
        reasons  = _build_reasons(row, signal, inst_pct, insider_pct)

        result = {
            "ticker":           ticker,
            "name":             row.get("name", ticker),
            "sector":           sector,
            "market":           market,
            "last_price":       round(float(row.get("last_price", 0) or 0), 2),
            "currency":         stock.get("currency", "USD"),
            "market_cap_bn":    stock.get("market_cap_bn"),
            "signal":              signal,
            "is_buy":              signal in ("STRONG BUY", "BUY"),
            "composite_score":     round(composite * 100, 1),
            "fundamental_score":   round(fund_score * 100, 1),
            "inst_ownership_pct":  round(inst_pct * 100, 1) if inst_pct is not None else None,
            "insider_ownership_pct": round(insider_pct * 100, 1) if insider_pct is not None else None,
            "ff5":                 ff5,
            "buy_reasons":         reasons,
            "fundamentals": {
                "pe_ratio":       _fmt(row.get("pe_ratio")),
                "pb_ratio":       _fmt(row.get("pb_ratio")),
                "roe":            _pct(row.get("roe")),
                "net_margin":     _pct(row.get("net_margin")),
                "revenue_growth": _pct(row.get("revenue_growth")),
                "debt_equity":    _fmt(row.get("debt_equity")),
                "dividend_yield": _pct(row.get("dividend_yield")),
                "beta":           _fmt(row.get("beta")),
                "mom_12m":        _pct(row.get("mom_12m")),
                "mom_3m":         _pct(row.get("mom_3m")),
                "rsi_14":         _fmt(row.get("rsi_14")),
                "vol_60d":        _pct(row.get("vol_60d")),
            },
        }
        results.append(result)

    # Sort by composite score descending
    results.sort(key=lambda x: x["composite_score"], reverse=True)
    return results


# ── Formatting helpers ─────────────────────────────────────────────────────

def _fmt(val, decimals=2) -> str | None:
    try:
        f = float(val)
        return round(f, decimals) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None

def _pct(val) -> str | None:
    try:
        f = float(val)
        return round(f * 100, 1) if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
