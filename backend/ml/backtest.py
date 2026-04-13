"""
backtest.py  (v2 — trains model on real returns)
─────────────────────────────────────────────────
Walk-forward backtest that does two things simultaneously:

  1. MEASURES model quality (IC, Sharpe, hit rate etc.)
  2. TRAINS the model on real return data so future analyses
     use what was actually learned from historical outcomes

This is the self-improvement loop:

  Period 1 (e.g. Jan 2021):
    - Compute factor scores for each stock
    - Record scores + actual 12M forward returns
    - Label: did this stock beat the median return? (1/0)

  Period 2 (Apr 2021):
    - Same as above, now have 2 periods of real outcomes

  ...after all periods...

  Final step:
    - Train RF+GB on ALL collected (features, real_return_label) pairs
    - Save model to disk
    - Next time /analyse runs, it loads THIS model
    - The model now knows what factor combinations actually
      predicted outperformance IN YOUR specific universe

The more tickers and history you have, the better the model gets.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import logging
from pathlib import Path

from .feature_engineering import FEATURE_COLS, PRICE_FEATURE_COLS
from .model import NUMKTEnsemble

logger = logging.getLogger(__name__)
TRADING_DAYS_PER_MONTH = 21


def _extract_prices(data: dict) -> pd.Series | None:
    raw = data.get("_raw")
    if raw and isinstance(raw, dict):
        df = raw.get("prices")
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            return df["Close"].dropna()
    df = data.get("prices")
    if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
        return df["Close"].dropna()
    if isinstance(df, pd.Series):
        return df.dropna()
    return None


def _extract_info(data: dict) -> dict:
    raw = data.get("_raw")
    if raw and isinstance(raw, dict):
        return raw.get("info") or {}
    return data.get("info") or {}


def _safe(val, default=0.0) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def run_backtest(
    features_df:      pd.DataFrame,
    raw_data:         dict,
    profile:          str  = "quality",
    forward_months:   int  = 12,
    rebalance_months: int  = 3,
    train_model:      bool = True,   # if True, trains model on real returns
    model_name:       str  = "default",
) -> dict:
    """
    Walk-forward backtest.

    When train_model=True (default):
      - Collects (features, real_return_label) pairs across all periods
      - Trains the ensemble on these real outcomes
      - Saves the trained model to disk
      - Future /analyse calls will use this model

    Returns standard quality metrics plus training info.
    """
    forward_days   = forward_months  * TRADING_DAYS_PER_MONTH
    rebalance_days = rebalance_months * TRADING_DAYS_PER_MONTH
    min_bars       = forward_days + rebalance_days + 30

    # ── Collect price series ───────────────────────────────────────────────
    price_series = {}
    for ticker, data in raw_data.items():
        close = _extract_prices(data)
        if close is not None and len(close) >= min_bars:
            price_series[ticker] = close
            logger.info(f"{ticker}: {len(close)} bars available for backtest")
        else:
            bars = len(close) if close is not None else 0
            logger.warning(f"{ticker}: {bars} bars, need {min_bars} — skipped")

    if len(price_series) < 2:
        return _fail(
            f"Only {len(price_series)} ticker(s) had enough price history. "
            f"Need at least 2 tickers with {min_bars}+ trading days (~"
            f"{min_bars//21} months). Try increasing lookback_years."
        )

    tickers   = list(price_series.keys())
    min_len   = min(len(price_series[t]) for t in tickers)
    n_periods = max(1, (min_len - forward_days) // rebalance_days)
    logger.info(f"Backtest: {len(tickers)} tickers, {n_periods} periods, "
                f"{forward_months}M forward window")

    # ── Train / holdout split ──────────────────────────────────────────────
    # Hold out the last ~20% of periods so the model can be evaluated on
    # data it never saw during training. Requires at least 4 periods to be
    # worth splitting; below that threshold all data goes to training and
    # holdout_ic is skipped.
    if n_periods >= 4:
        n_holdout    = max(1, n_periods // 5)
        train_cutoff = n_periods - n_holdout
    else:
        n_holdout    = 0
        train_cutoff = n_periods

    logger.info(
        f"Train/holdout split: {train_cutoff} training periods, "
        f"{n_holdout} holdout periods"
    )

    # ── Accumulators ──────────────────────────────────────────────────────
    period_ics        = []
    period_hit_rates  = []
    portfolio_returns = []
    benchmark_returns = []
    prev_positions    = set()
    turnovers         = []

    # Training periods (period < train_cutoff)
    all_feature_rows    = []
    all_return_labels   = []
    all_forward_returns = []

    # Holdout periods (period >= train_cutoff) — never seen by the model
    holdout_feature_rows    = []
    holdout_return_labels   = []
    holdout_forward_returns = []

    for period in range(n_periods):
        start_idx = period * rebalance_days
        end_idx   = start_idx + forward_days

        if end_idx >= min_len:
            break

        # ── Build feature snapshot at start_idx ───────────────────────────
        snap_features = {}
        for ticker in tickers:
            close = price_series[ticker]
            p     = close.iloc[:start_idx] if start_idx > 0 else close
            if len(p) < 5:
                continue

            snap = {}

            # Momentum at this point in time
            for days, key in [(21,"mom_1m"),(63,"mom_3m"),(126,"mom_6m"),(252,"mom_12m")]:
                snap[key] = float(p.iloc[-1]/p.iloc[-(days+1)]-1) if len(p)>days else 0.0

            # Volatility
            if len(p) > 10:
                lr = np.log(p/p.shift(1)).dropna().iloc[-min(60,len(p)-1):]
                snap["vol_60d"] = float(lr.std()*np.sqrt(252)) if len(lr)>1 else 0.20
            else:
                snap["vol_60d"] = 0.20

            # RSI-14 (Wilder's EMA)
            if len(p) > 15:
                d    = p.diff().dropna()
                gain = d.clip(lower=0).ewm(span=14, adjust=False).mean().iloc[-1]
                loss = (-d.clip(upper=0)).ewm(span=14, adjust=False).mean().iloc[-1]
                snap["rsi_14"] = float(100-100/(1+gain/(loss+1e-9)))
            else:
                snap["rsi_14"] = 50.0

            hi = p.iloc[-min(252,len(p)):].max()
            snap["price_vs_52w_high"] = float(p.iloc[-1]/hi-1) if hi>0 else 0.0

            # Fundamentals from Yahoo info (static across periods)
            info = _extract_info(raw_data[ticker])
            snap["pe_ratio"]       = _safe(info.get("trailingPE"),    20.0)
            snap["pb_ratio"]       = _safe(info.get("priceToBook"),    2.0)
            snap["roe"]            = _safe(info.get("returnOnEquity"), 0.10)
            snap["net_margin"]     = _safe(info.get("profitMargins"),  0.10)
            snap["revenue_growth"] = _safe(info.get("revenueGrowth"),  0.05)
            snap["debt_equity"]    = _safe(info.get("debtToEquity"),   0.50)
            snap["dividend_yield"] = _safe(info.get("dividendYield"),  0.02)
            snap["beta"]           = _safe(info.get("beta"),           1.00)
            snap["log_mktcap"]     = np.log(max(_safe(info.get("marketCap"),1e10),1e6))

            snap_features[ticker] = snap

        if len(snap_features) < 2:
            logger.info(f"Period {period}: only {len(snap_features)} snapshots, skip")
            continue

        snap_df      = pd.DataFrame(snap_features).T
        tickers_here = list(snap_df.index)

        # Rank-normalise cross-sectionally
        for col in FEATURE_COLS:
            if col in snap_df.columns and snap_df[col].notna().sum() > 1:
                snap_df[col+"_rank"] = snap_df[col].rank(pct=True)
            elif col+"_rank" not in snap_df.columns:
                snap_df[col+"_rank"] = 0.5

        # ── Compute ACTUAL forward returns ─────────────────────────────────
        fwd_rets = {}
        for ticker in tickers_here:
            close = price_series[ticker]
            si = min(start_idx, len(close)-1)
            ei = min(end_idx,   len(close)-1)
            if ei > si:
                p0 = float(close.iloc[si])
                p1 = float(close.iloc[ei])
                if p0 > 0:
                    fwd_rets[ticker] = (p1/p0)-1

        if len(fwd_rets) < 2:
            continue

        # ── Factor score for ranking ───────────────────────────────────────
        scores = {}
        for ticker in tickers_here:
            if ticker not in snap_df.index:
                continue
            row = snap_df.loc[ticker]
            s   = 0.0
            for col, wt in [
                ("roe_rank",            0.25),
                ("net_margin_rank",     0.20),
                ("mom_12m_rank",        0.20),
                ("pe_ratio_rank",      -0.15),
                ("debt_equity_rank",   -0.10),
                ("revenue_growth_rank", 0.10),
            ]:
                if col in row.index and pd.notna(row[col]):
                    s += abs(wt)*(1-float(row[col]) if wt<0 else float(row[col]))
            scores[ticker] = s

        common = [t for t in tickers_here if t in scores and t in fwd_rets]
        if len(common) < 2:
            continue

        sv = [scores[t]   for t in common]
        rv = [fwd_rets[t] for t in common]

        # ── Collect training / holdout data for this period ───────────────
        # Only price-derived rank features are included (no fundamentals).
        # Periods before train_cutoff go into the training set; periods from
        # train_cutoff onward go into the holdout set and are never seen by
        # the model during fitting.
        if train_model:
            median_ret = np.median(rv)
            for ticker, ret in zip(common, rv):
                if ticker in snap_df.index:
                    row_dict = {
                        f"{col}_rank": float(snap_df.loc[ticker].get(f"{col}_rank", 0.5))
                        for col in PRICE_FEATURE_COLS
                        if f"{col}_rank" in snap_df.columns
                    }
                    row_dict["ticker"] = ticker
                    label = 1 if ret >= median_ret else 0
                    if period < train_cutoff:
                        all_feature_rows.append(row_dict)
                        all_return_labels.append(label)
                        all_forward_returns.append(ret)
                    else:
                        row_dict["_period_idx"] = period
                        holdout_feature_rows.append(row_dict)
                        holdout_return_labels.append(label)
                        holdout_forward_returns.append(ret)

        # ── IC ─────────────────────────────────────────────────────────────
        if len(set(sv)) > 1:
            ic, _ = spearmanr(sv, rv)
            if np.isfinite(ic):
                period_ics.append(float(ic))

        # ── Hit rate ───────────────────────────────────────────────────────
        med  = np.median(sv)
        top  = [fwd_rets[t] for t,s in zip(common,sv) if s >= med]
        bot  = [fwd_rets[t] for t,s in zip(common,sv) if s <  med]
        if top and bot:
            period_hit_rates.append(1 if np.mean(top) > np.mean(bot) else 0)

        # ── Portfolio returns ──────────────────────────────────────────────
        n_top = max(1, len(common)//5)
        top_tickers = sorted(common, key=lambda t: scores[t], reverse=True)[:n_top]
        port_ret  = float(np.mean([fwd_rets[t] for t in top_tickers]))
        bench_ret = float(np.mean(rv))
        portfolio_returns.append(port_ret)
        benchmark_returns.append(bench_ret)

        curr = set(top_tickers)
        if prev_positions:
            turnovers.append(1 - len(curr & prev_positions)/max(len(curr),1))
        prev_positions = curr

        ic_str = f"{period_ics[-1]:.3f}" if period_ics else "N/A"
        logger.info(
            f"Period {period}: {len(common)} stocks | IC={ic_str} | "
            f"port={port_ret:.2%} | bench={bench_ret:.2%}"
        )

    # ── Train model on collected real-return data ──────────────────────────
    training_info = {}
    if train_model and all_feature_rows:
        logger.info(
            f"Training model on {len(all_feature_rows)} real-return observations "
            f"from {n_periods} historical periods..."
        )
        model = NUMKTEnsemble(cv_folds=5, lambda_reg=0.10, max_depth=5)
        model.fit_on_real_returns(
            feature_rows    = all_feature_rows,
            return_labels   = all_return_labels,
            forward_returns = all_forward_returns,
        )
        model.save(model_name)

        # ── Holdout IC + period examples — genuine out-of-sample ──────────
        # Score all holdout rows in one pass, then compute IC and build
        # per-period examples showing correct vs incorrect predictions.
        holdout_ic      = None
        period_examples = []
        if model.is_fitted and holdout_feature_rows:
            try:
                X_h = pd.DataFrame(holdout_feature_rows).fillna(0.5)

                # Pull non-feature columns out before scoring
                period_idxs = (X_h.pop("_period_idx")
                               if "_period_idx" in X_h.columns
                               else pd.Series([0] * len(X_h)))
                tickers_h   = (X_h.pop("ticker")
                               if "ticker" in X_h.columns
                               else pd.Series([""] * len(X_h)))

                for c in model._feature_names:
                    if c not in X_h.columns:
                        X_h[c] = 0.5

                h_preds = model.predict_proba(X_h[model._feature_names].values)

                # Holdout IC
                ic_val, _ = spearmanr(h_preds, holdout_forward_returns)
                holdout_ic = round(float(ic_val), 4) if np.isfinite(ic_val) else None
                logger.info(
                    f"Holdout IC (out-of-sample, {len(holdout_feature_rows)} rows): "
                    f"{holdout_ic}"
                )

                # Date lookup: map period index → "MMM YYYY → MMM YYYY"
                period_dates: dict[int, str] = {}
                ref_prices = next(iter(price_series.values()))
                for p_idx in range(n_periods):
                    si = p_idx * rebalance_days
                    ei = si + forward_days
                    if ei < len(ref_prices):
                        sd = ref_prices.index[si]
                        ed = ref_prices.index[ei]
                        period_dates[p_idx] = (
                            f"{sd.strftime('%b %Y')} → {ed.strftime('%b %Y')}"
                        )

                # Per-period breakdown
                df_ex = pd.DataFrame({
                    "period":        period_idxs.values,
                    "ticker":        tickers_h.values,
                    "model_score":   h_preds,
                    "pred_top":      (h_preds >= 0.5),
                    "actual_return": holdout_forward_returns,
                    "actual_top":    np.array(holdout_return_labels, dtype=bool),
                })
                df_ex["correct"] = df_ex["pred_top"] == df_ex["actual_top"]

                for p_idx, grp in df_ex.groupby("period"):
                    grp = grp.sort_values("model_score", ascending=False)
                    period_examples.append({
                        "period":       int(p_idx),
                        "date_range":   period_dates.get(int(p_idx), f"Period {p_idx}"),
                        "n_stocks":     len(grp),
                        "hit_rate_pct": round(float(grp["correct"].mean()) * 100, 1),
                        "stocks": [
                            {
                                "ticker":            str(row["ticker"]),
                                "model_score":       round(float(row["model_score"]) * 100, 1),
                                "predicted_top":     bool(row["pred_top"]),
                                "actual_return_pct": round(float(row["actual_return"]) * 100, 1),
                                "actual_top":        bool(row["actual_top"]),
                                "correct":           bool(row["correct"]),
                            }
                            for _, row in grp.iterrows()
                        ],
                    })

            except Exception as e:
                logger.warning(f"Could not compute holdout analysis: {e}")

        holdout_note = (
            f"Holdout IC (out-of-sample): {holdout_ic:.4f}. "
            if holdout_ic is not None else
            "Holdout IC: n/a (too few periods to split). "
        )

        training_info = {
            "trained":         True,
            "training_rows":   len(all_feature_rows),
            "holdout_rows":    len(holdout_feature_rows),
            "training_source": "real_returns",
            "cv_accuracy":     round(model.cv_accuracy, 4),
            "oof_ic":          round(model.cv_ic, 4),
            "holdout_ic":      holdout_ic,
            "period_examples": period_examples,
            "model_name":      model_name,
            "message": (
                f"Model trained on {len(all_feature_rows)} observations "
                f"({len(holdout_feature_rows)} held out). "
                f"OOF CV accuracy: {model.cv_accuracy:.1%}. "
                f"OOF IC: {model.cv_ic:.4f}. "
                f"{holdout_note}"
                f"Next analysis will use this model."
            )
        }
        logger.info(f"Model training complete. {training_info['message']}")
    else:
        training_info = {"trained": False, "reason": "No training data collected"}

    # ── Aggregate metrics ──────────────────────────────────────────────────
    if not portfolio_returns:
        result = _fail(
            "No valid test periods. Increase lookback_years or add more tickers."
        )
        result["training"] = training_info
        return result

    pa = np.array(portfolio_returns)
    ba = np.array(benchmark_returns)
    ppy = 12 / rebalance_months

    ann_port  = float((np.prod(1 + pa) ** (ppy / len(pa))) - 1)
    ann_bench = float((np.prod(1 + ba) ** (ppy / len(ba))) - 1)
    ann_alpha = ann_port - ann_bench

    rf     = 0.045 / ppy
    exc_rf = pa - rf
    sharpe = float(np.mean(exc_rf)/(np.std(exc_rf)+1e-9)*np.sqrt(ppy))

    cum    = np.cumprod(1+pa)
    rm     = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum-rm)/(rm+1e-9)))
    calmar = ann_port/(abs(max_dd)+1e-9)

    ic_mean  = float(np.mean(period_ics))       if period_ics       else 0.0
    ic_std   = float(np.std(period_ics))        if len(period_ics)>1 else 0.0
    icir     = ic_mean/(ic_std+1e-9)
    hit_rate = float(np.mean(period_hit_rates)) if period_hit_rates  else 0.5

    beta = 1.0
    if len(ba) > 1 and np.std(ba) > 0:
        beta = float(np.cov(pa,ba)[0,1]/(np.var(ba)+1e-9))

    avg_turn = float(np.mean(turnovers)) if turnovers else 0.5

    logger.info(
        f"BACKTEST COMPLETE | periods={len(pa)} | IC={ic_mean:.4f} | "
        f"Sharpe={sharpe:.3f} | Hit={hit_rate:.1%} | Alpha={ann_alpha:.2%}"
    )

    return {
        "n_periods":            len(pa),
        "forward_months":       forward_months,
        "ann_portfolio_return": round(ann_port*100, 2),
        "ann_benchmark_return": round(ann_bench*100, 2),
        "ann_alpha":            round(ann_alpha*100, 2),
        "sharpe_ratio":         round(sharpe, 3),
        "max_drawdown":         round(max_dd*100, 2),
        "calmar_ratio":         round(calmar, 3),
        "hit_rate":             round(hit_rate*100, 1),
        "ic_mean":              round(ic_mean, 4),
        "ic_std":               round(ic_std, 4),
        "icir":                 round(icir, 3),
        "beta_vs_benchmark":    round(beta, 3),
        "avg_monthly_turnover": round(avg_turn*100, 1),
        "interpretation":       _interpret(hit_rate, ic_mean, sharpe, max_dd, ann_alpha),
        "training":             training_info,
    }


def _interpret(hit_rate, ic, sharpe, max_dd, alpha) -> dict:
    return {
        "hit_rate": "Strong"   if hit_rate>0.55 else ("Moderate" if hit_rate>0.45 else "Weak"),
        "ic":       "Strong (>0.10)" if ic>0.10 else (
                    "Good (0.05–0.10)" if ic>0.05 else (
                    "Weak (<0.05)"     if ic>0    else "Negative")),
        "sharpe":   "Excellent (>2)" if sharpe>2 else (
                    "Good (1–2)"     if sharpe>1 else (
                    "Acceptable"     if sharpe>0 else "Poor (<0)")),
        "drawdown": "Low (<10%)" if max_dd>-0.10 else (
                    "Moderate"   if max_dd>-0.20 else "High (>20%)"),
        "alpha":    "Positive — model adding value" if alpha>0
                    else "Negative — model underperforming equal-weight",
    }


def _fail(reason: str) -> dict:
    return {
        "error": reason, "n_periods": 0,
        "sharpe_ratio": None, "hit_rate": None,
        "ic_mean": None, "max_drawdown": None,
    }