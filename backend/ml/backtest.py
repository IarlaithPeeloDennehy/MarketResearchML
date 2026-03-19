"""
backtest.py
───────────
Walk-forward backtest engine.

Industry-standard metrics produced:
  ┌─────────────────────────────────────────────────────────────┐
  │  Hit Rate         % of BUY signals that beat the benchmark  │
  │  IC               Information Coefficient (Spearman ρ)       │
  │  ICIR             IC / std(IC) — measures IC consistency     │
  │  Sharpe Ratio     annualised return / annualised volatility  │
  │  Max Drawdown     worst peak-to-trough loss                  │
  │  Calmar Ratio     annualised return / |max drawdown|         │
  │  Alpha            excess return vs equal-weight benchmark    │
  │  Beta             sensitivity to benchmark                   │
  │  Turnover         avg monthly portfolio turnover %           │
  └─────────────────────────────────────────────────────────────┘

Walk-forward methodology:
  - Expand training window by 3 months each period
  - Score universe at each period-end
  - Measure return of top-quintile portfolio over next month
  - Compare to equal-weight universe benchmark
  - Report aggregate statistics across all periods
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import logging
from typing import Optional

from .feature_engineering import FEATURE_COLS, FEATURE_COLS

logger = logging.getLogger(__name__)


def _compute_returns(prices_dict: dict, start_idx: int, end_idx: int) -> dict[str, float]:
    """Compute period return for each ticker between start_idx and end_idx."""
    returns = {}
    for ticker, data in prices_dict.items():
        prices = data.get("prices") or (data.get("_raw") or {}).get("prices")
        if prices is None or len(prices) < end_idx:
            continue
        try:
            p0 = float(prices["Close"].iloc[start_idx])
            p1 = float(prices["Close"].iloc[end_idx])
            if p0 > 0:
                returns[ticker] = (p1 / p0) - 1
        except (IndexError, KeyError):
            continue
    return returns


def run_backtest(
    features_df: pd.DataFrame,
    raw_data: dict,
    profile: str = "quality",
    forward_months: int = 12,
    rebalance_months: int = 3,
) -> dict:
    """
    Runs walk-forward backtest and returns a dict of quality metrics.
    
    Note: with a small universe (<30 stocks) and limited history,
    some metrics will have wide confidence intervals. The IC and
    hit rate are the most reliable statistics for small samples.
    """
    trading_days_per_month = 21
    forward_days  = forward_months  * trading_days_per_month
    rebalance_days = rebalance_months * trading_days_per_month

    # Build list of all price series
    price_series = {}
    for ticker, data in raw_data.items():
        raw = data.get("_raw") or data
        prices = raw.get("prices") if isinstance(raw, dict) else None
        if prices is not None and len(prices) > forward_days + rebalance_days:
            price_series[ticker] = prices

    if len(price_series) < 3:
        return _insufficient_data_response()

    tickers     = list(price_series.keys())
    min_len     = min(len(price_series[t]) for t in tickers)
    n_periods   = max(1, (min_len - forward_days) // rebalance_days)

    period_ics      = []
    period_hit_rates = []
    portfolio_returns = []   # top-quintile portfolio return each period
    benchmark_returns = []   # equal-weight benchmark return each period
    prev_positions  = set()
    turnovers       = []

    for period in range(n_periods):
        start_idx = period * rebalance_days
        end_idx   = start_idx + forward_days

        if end_idx >= min_len:
            break

        # ── Compute features at this snapshot ─────────────────────────────
        # We use momentum calculated up to start_idx as features
        snap_features = {}
        for ticker in tickers:
            prices = price_series[ticker]["Close"]
            if len(prices) < start_idx + 252:
                continue

            snap = {}
            p = prices.iloc[:start_idx] if start_idx > 0 else prices

            # Momentum features from historical prices
            for days, key in [(21,"mom_1m"), (63,"mom_3m"), (126,"mom_6m"), (252,"mom_12m")]:
                if len(p) > days:
                    snap[key] = float(p.iloc[-1] / p.iloc[-(days+1)] - 1)
                else:
                    snap[key] = 0.0

            # Volatility
            if len(p) > 60:
                log_ret = np.log(p / p.shift(1)).dropna().iloc[-60:]
                snap["vol_60d"] = float(log_ret.std() * np.sqrt(252))
            else:
                snap["vol_60d"] = 0.15

            # Fundamental features from raw_data (static — same for all periods)
            fd = raw_data.get(ticker, {})
            info = (fd.get("_raw") or {}).get("info", {}) if "_raw" in fd else {}
            snap["pe_ratio"]       = _safe(info.get("trailingPE", 20))
            snap["pb_ratio"]       = _safe(info.get("priceToBook", 2))
            snap["roe"]            = _safe(info.get("returnOnEquity", 0.1))
            snap["net_margin"]     = _safe(info.get("profitMargins", 0.1))
            snap["revenue_growth"] = _safe(info.get("revenueGrowth", 0.05))
            snap["debt_equity"]    = _safe(info.get("debtToEquity", 0.5))
            snap["dividend_yield"] = _safe(info.get("dividendYield", 0.02))
            snap["beta"]           = _safe(info.get("beta", 1.0))
            snap["log_mktcap"]     = np.log(max(_safe(info.get("marketCap", 1e10)), 1e6))
            snap["rsi_14"]         = 50.0  # neutral default
            snap["price_vs_52w_high"] = float(
                p.iloc[-1] / p.iloc[-252:].max() - 1 if len(p) >= 252 else 0
            )

            snap_features[ticker] = snap

        if len(snap_features) < 3:
            continue

        snap_df = pd.DataFrame(snap_features).T
        tickers_here = list(snap_df.index)

        # ── Rank-normalise features ────────────────────────────────────────
        for col in FEATURE_COLS:
            if col in snap_df.columns and snap_df[col].notna().sum() > 1:
                snap_df[col + "_rank"] = snap_df[col].rank(pct=True)

        # ── Compute forward returns ────────────────────────────────────────
        fwd_rets = {}
        for ticker in tickers_here:
            prices = price_series[ticker]["Close"]
            if end_idx < len(prices) and start_idx < len(prices):
                p0 = float(prices.iloc[start_idx])
                p1 = float(prices.iloc[min(end_idx, len(prices)-1)])
                if p0 > 0:
                    fwd_rets[ticker] = (p1 / p0) - 1

        if len(fwd_rets) < 3:
            continue

        # ── Build a simple factor score for IC calculation ─────────────────
        scores = {}
        for ticker in tickers_here:
            if ticker not in snap_df.index:
                continue
            row = snap_df.loc[ticker]
            score = 0.0
            for col, weight in [
                ("roe_rank", 0.25), ("net_margin_rank", 0.20),
                ("mom_12m_rank", 0.20), ("pe_ratio_rank", -0.15),
                ("debt_equity_rank", -0.10), ("revenue_growth_rank", 0.10),
            ]:
                if col in row:
                    score += weight * float(row[col] if col != "pe_ratio_rank" else 1 - row[col])
            scores[ticker] = score

        common = [t for t in tickers_here if t in scores and t in fwd_rets]
        if len(common) < 3:
            continue

        score_vals = [scores[t] for t in common]
        ret_vals   = [fwd_rets[t] for t in common]

        # Information Coefficient (Spearman rank correlation)
        ic, _ = spearmanr(score_vals, ret_vals)
        if np.isfinite(ic):
            period_ics.append(ic)

        # Hit rate — did top-half outperform bottom-half?
        median_score = np.median(score_vals)
        top_half  = [fwd_rets[t] for t, s in zip(common, score_vals) if s >= median_score]
        bot_half  = [fwd_rets[t] for t, s in zip(common, score_vals) if s < median_score]
        if top_half and bot_half:
            hit = 1 if np.mean(top_half) > np.mean(bot_half) else 0
            period_hit_rates.append(hit)

        # Portfolio return (top quintile)
        n_top = max(1, len(common) // 5)
        sorted_by_score = sorted(common, key=lambda t: scores[t], reverse=True)
        top_tickers = sorted_by_score[:n_top]
        port_ret = np.mean([fwd_rets[t] for t in top_tickers])
        bench_ret = np.mean(ret_vals)
        portfolio_returns.append(port_ret)
        benchmark_returns.append(bench_ret)

        # Turnover
        current_positions = set(top_tickers)
        if prev_positions:
            overlap = len(current_positions & prev_positions) / max(len(current_positions), 1)
            turnovers.append(1 - overlap)
        prev_positions = current_positions

    # ── Aggregate metrics ──────────────────────────────────────────────────
    if not portfolio_returns:
        return _insufficient_data_response()

    port_arr  = np.array(portfolio_returns)
    bench_arr = np.array(benchmark_returns)
    excess    = port_arr - bench_arr

    # Annualise (periods are rebalance_months months each)
    periods_per_year = 12 / rebalance_months
    ann_port_ret  = float(np.mean(port_arr)  * periods_per_year)
    ann_bench_ret = float(np.mean(bench_arr) * periods_per_year)
    ann_alpha     = ann_port_ret - ann_bench_ret

    # Sharpe (risk-free rate = 4.5%)
    rf_period = 0.045 / periods_per_year
    excess_rf = port_arr - rf_period
    sharpe = float(np.mean(excess_rf) / (np.std(excess_rf) + 1e-9) * np.sqrt(periods_per_year))

    # Max drawdown
    cumulative = np.cumprod(1 + port_arr)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - rolling_max) / (rolling_max + 1e-9)
    max_dd = float(np.min(drawdowns))

    # Calmar
    calmar = ann_port_ret / (abs(max_dd) + 1e-9)

    # IC stats
    ic_mean  = float(np.mean(period_ics))  if period_ics else 0.0
    ic_std   = float(np.std(period_ics))   if len(period_ics) > 1 else 0.0
    icir     = ic_mean / (ic_std + 1e-9)

    # Hit rate
    hit_rate = float(np.mean(period_hit_rates)) if period_hit_rates else 0.5

    # Beta vs benchmark
    if len(bench_arr) > 1 and np.std(bench_arr) > 0:
        beta = float(np.cov(port_arr, bench_arr)[0, 1] / np.var(bench_arr))
    else:
        beta = 1.0

    # Turnover
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.5

    return {
        "n_periods":           len(portfolio_returns),
        "forward_months":      forward_months,
        "ann_portfolio_return":round(ann_port_ret * 100, 2),
        "ann_benchmark_return":round(ann_bench_ret * 100, 2),
        "ann_alpha":           round(ann_alpha * 100, 2),
        "sharpe_ratio":        round(sharpe, 3),
        "max_drawdown":        round(max_dd * 100, 2),
        "calmar_ratio":        round(calmar, 3),
        "hit_rate":            round(hit_rate * 100, 1),
        "ic_mean":             round(ic_mean, 4),
        "ic_std":              round(ic_std, 4),
        "icir":                round(icir, 3),
        "beta_vs_benchmark":   round(beta, 3),
        "avg_monthly_turnover":round(avg_turnover * 100, 1),
        "interpretation": _interpret_metrics(
            hit_rate, ic_mean, sharpe, max_dd, ann_alpha
        ),
    }


def _interpret_metrics(hit_rate, ic, sharpe, max_dd, alpha) -> dict:
    """Plain-English interpretation of backtest quality."""
    return {
        "hit_rate":   "Strong" if hit_rate > 0.55 else ("Moderate" if hit_rate > 0.45 else "Weak"),
        "ic":         "Strong (>0.10)" if ic > 0.10 else ("Good (0.05–0.10)" if ic > 0.05 else ("Weak (<0.05)" if ic > 0 else "Negative — review model")),
        "sharpe":     "Excellent (>2)" if sharpe > 2 else ("Good (1–2)" if sharpe > 1 else ("Acceptable (0–1)" if sharpe > 0 else "Poor (<0)")),
        "drawdown":   "Low (<10%)" if max_dd > -0.10 else ("Moderate (10–20%)" if max_dd > -0.20 else "High (>20%)"),
        "alpha":      "Positive alpha" if alpha > 0 else "Negative alpha — model underperforming benchmark",
    }


def _insufficient_data_response() -> dict:
    return {
        "error": "Insufficient price history for reliable backtest. Add more tickers or increase lookback_years.",
        "n_periods": 0,
        "sharpe_ratio": None,
        "hit_rate": None,
        "ic_mean": None,
        "max_drawdown": None,
    }


def _safe(val, default=0.0) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default
