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

from .feature_engineering import PRICE_FEATURE_COLS
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



def run_backtest(
    features_df:      pd.DataFrame,
    raw_data:         dict,
    profile:          str  = "quality",   # reserved for future per-profile feature selection
    forward_months:   int  = 12,
    rebalance_months: int  = 3,
    train_model:      bool = True,   # if True, trains model on real returns
    model_name:       str  = "default",
    cost_bps:         int  = 30,     # round-trip transaction cost in basis points
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

    tickers               = list(price_series.keys())
    min_len               = min(len(price_series[t]) for t in tickers)
    effective_lookback_yrs = round(min_len / 252, 1)
    n_periods = max(1, (min_len - forward_days) // rebalance_days)
    if effective_lookback_yrs < 2.0:
        logger.warning(
            f"Effective lookback is only {effective_lookback_yrs}y ({min_len} bars). "
            "Results may be unreliable — consider increasing lookback_years."
        )
    logger.info(f"Backtest: {len(tickers)} tickers, {n_periods} periods, "
                f"{forward_months}M forward window, {effective_lookback_yrs}y effective")

    # ── Embargoed expanding-window walk-forward ────────────────────────────
    # The evaluation ruler. For every period t we train a FRESH model on all
    # periods ending at least `embargo_periods` before t, then score period t
    # — which the model has never seen. Every scored period is therefore a
    # genuine out-of-sample observation, so the whole walk-forward IS the
    # holdout (no separate 20% tail needed). This uses ~all periods for
    # evaluation instead of 4, roughly halving the standard error on mean IC.
    #
    # EMBARGO: forward windows are `forward_days` long but periods are only
    # `rebalance_days` apart, so period t's label window overlaps the periods
    # within forward_days/rebalance_days of it. Dropping that many periods
    # before t prevents overlapping-label leakage (López de Prado).
    embargo_periods   = forward_days // rebalance_days   # e.g. 252//63 = 4
    min_train_periods = 3
    rank_cols         = [f"{c}_rank" for c in PRICE_FEATURE_COLS]

    logger.info(
        f"Walk-forward eval: embargo={embargo_periods} periods, "
        f"min_train={min_train_periods}, forward={forward_months}M, "
        f"rebalance={rebalance_months}M"
    )

    # ── Accumulators ──────────────────────────────────────────────────────
    bench_ret_by_period: dict[int, float] = {}
    period_frames:       dict[int, pd.DataFrame] = {}   # p -> [ticker, *rank, fwd_ret]

    period_ics        = []
    period_hit_rates  = []
    portfolio_returns       = []   # net of transaction costs
    portfolio_returns_gross = []   # before transaction costs
    benchmark_returns       = []
    turnovers               = []

    # Rows for the FINAL production model (all periods, every label its own
    # period median). Kept separate from the walk-forward, which trains its
    # own per-step throwaway models.
    all_feature_rows    = []
    all_return_labels   = []
    all_forward_returns = []

    # ── Phase 1: build per-period feature/return frames ────────────────────
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
                gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
                loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
                snap["rsi_14"] = float(100-100/(1+gain/(loss+1e-9)))
            else:
                snap["rsi_14"] = 50.0

            hi = p.iloc[-min(252,len(p)):].max()
            snap["price_vs_52w_high"] = float(p.iloc[-1]/hi-1) if hi>0 else 0.0

            snap_features[ticker] = snap

        if len(snap_features) < 2:
            logger.info(f"Period {period}: only {len(snap_features)} snapshots, skip")
            continue

        snap_df      = pd.DataFrame(snap_features).T
        tickers_here = list(snap_df.index)

        # Rank-normalise cross-sectionally (price features only — no fundamentals)
        for col in PRICE_FEATURE_COLS:
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

        common = [t for t in tickers_here if t in fwd_rets]
        if len(common) < 2:
            continue

        rv         = [fwd_rets[t] for t in common]
        median_ret = float(np.median(rv))
        bench_ret_by_period[period] = float(np.mean(rv))

        frame_rows = []
        for ticker in common:
            rec = {f"{c}_rank": float(snap_df.loc[ticker].get(f"{c}_rank", 0.5))
                   for c in PRICE_FEATURE_COLS}
            rec["ticker"]  = ticker
            rec["fwd_ret"] = fwd_rets[ticker]
            # Raw point-in-time volatility (window start) for risk-based sizing.
            rec["vol"]     = float(snap_df.loc[ticker].get("vol_60d", 0.20) or 0.20)
            frame_rows.append(rec)

            # Accumulate rows for the final production model (all periods).
            frow = {k: v for k, v in rec.items() if k.endswith("_rank")}
            frow["ticker"]      = ticker
            frow["_period_idx"] = period
            all_feature_rows.append(frow)
            all_return_labels.append(1 if fwd_rets[ticker] >= median_ret else 0)
            all_forward_returns.append(fwd_rets[ticker])

        period_frames[period] = pd.DataFrame(frame_rows)
        logger.info(f"Period {period}: {len(common)} stocks | bench={np.mean(rv):.2%}")

    # ── Phase 2: walk-forward scoring ──────────────────────────────────────
    # wf_scored[t] = DataFrame(ticker, model_score, fwd_ret) — genuine OOS.
    wf_scored: dict[int, pd.DataFrame] = {}
    if train_model and period_frames:
        periods_sorted = sorted(period_frames)
        for t in periods_sorted:
            train_ps = [p for p in periods_sorted if p <= t - embargo_periods - 1]
            if len(train_ps) < min_train_periods:
                continue

            X_parts, y_parts = [], []
            for p in train_ps:
                fr  = period_frames[p]
                med = fr["fwd_ret"].median()
                X_parts.append(fr[rank_cols].values)
                y_parts.append((fr["fwd_ret"].values >= med).astype(int))
            X_tr = np.vstack(X_parts)
            y_tr = np.concatenate(y_parts)
            if y_tr.sum() < 2 or (len(y_tr) - y_tr.sum()) < 2:
                continue

            step = NUMKTEnsemble(cv_folds=5, lambda_reg=0.10, max_depth=5)
            step.fit_pipelines(X_tr, y_tr, feature_names=rank_cols)

            ft     = period_frames[t]
            scores = step.predict_proba(ft[rank_cols].values)
            wf_scored[t] = pd.DataFrame({
                "ticker":     ft["ticker"].values,
                "model_score": scores,
                "fwd_ret":    ft["fwd_ret"].values,
                "vol":        ft["vol"].values if "vol" in ft.columns else 0.20,
            })

    # ── Phase 3: metrics from walk-forward scores ──────────────────────────
    holdout_ic      = None   # pooled Spearman over ALL walk-forward (OOS) rows
    holdout_ic_mean = None   # mean of per-period walk-forward ICs == ic_mean
    holdout_icir    = None
    period_examples = []
    prev_top_set    = set()

    # ── Behavioural (co-equal) portfolios ──────────────────────────────────
    # Same top-quintile selection as the equal-weight book, but:
    #   sized      — conviction × inverse-volatility weights (size correctly:
    #                more into high-conviction low-vol names, less into churny
    #                high-vol ones), capped per name.
    #   hysteresis — hold a name until its score falls below the period median,
    #                instead of dropping it the instant it leaves the top
    #                quintile (reduces overtrading / turnover).
    sized_returns:    list[float] = []   # net of cost
    sized_turnovers:  list[float] = []
    prev_weights:     dict[str, float] = {}
    hyst_returns:     list[float] = []   # net of cost
    hyst_turnovers:   list[float] = []
    hyst_hold:        set[str] = set()

    wf_periods = sorted(wf_scored)
    pooled_scores, pooled_rets = [], []
    for t in wf_periods:
        grp = wf_scored[t]
        if len(grp) < 2 or grp["model_score"].nunique() < 2:
            continue

        benchmark_returns.append(bench_ret_by_period[t])
        pooled_scores.extend(grp["model_score"].tolist())
        pooled_rets.extend(grp["fwd_ret"].tolist())

        ic, _ = spearmanr(grp["model_score"], grp["fwd_ret"])
        if np.isfinite(ic):
            period_ics.append(float(ic))

        med_score = grp["model_score"].median()
        top_ret   = grp.loc[grp["model_score"] >= med_score, "fwd_ret"].tolist()
        bot_ret   = grp.loc[grp["model_score"] <  med_score, "fwd_ret"].tolist()
        if top_ret and bot_ret:
            period_hit_rates.append(1 if np.mean(top_ret) > np.mean(bot_ret) else 0)

        n_top     = max(1, len(grp) // 5)
        top_grp   = grp.nlargest(n_top, "model_score")
        curr_set  = set(top_grp["ticker"].tolist())
        gross_ret = float(top_grp["fwd_ret"].mean())

        new_entries   = curr_set - prev_top_set if prev_top_set else curr_set
        turnover_frac = len(new_entries) / max(len(curr_set), 1)
        cost_drag     = turnover_frac * cost_bps / 10_000
        portfolio_returns_gross.append(gross_ret)
        portfolio_returns.append(gross_ret - cost_drag)

        if prev_top_set:
            turnovers.append(1 - len(curr_set & prev_top_set) / max(len(curr_set), 1))
        prev_top_set = curr_set

        # ── Sized book: conviction × inverse-vol over the top quintile ─────
        tg = top_grp.copy()
        conv = (tg["model_score"] - 0.5).clip(lower=0.0)
        ivol = 1.0 / tg["vol"].clip(lower=0.05)
        w    = (conv * ivol).to_numpy(dtype=float)
        if w.sum() <= 0:                     # no conviction → equal weight
            w = np.ones(len(tg))
        w = w / w.sum()
        w = np.minimum(w, 0.25)              # per-name cap
        w = w / w.sum()
        weights = dict(zip(tg["ticker"], w))
        sized_gross = float((w * tg["fwd_ret"].to_numpy(dtype=float)).sum())
        # Cost on traded notional vs previous weights (round-trip cost_bps).
        names = set(weights) | set(prev_weights)
        traded = sum(abs(weights.get(n, 0.0) - prev_weights.get(n, 0.0)) for n in names)
        sized_returns.append(sized_gross - traded * cost_bps / 10_000)
        sized_turnovers.append(traded / 2.0)
        prev_weights = weights

        # ── Hysteresis book: hold above the period median, enter on top ────
        med_all   = grp["model_score"].median()
        score_map = dict(zip(grp["ticker"], grp["model_score"]))
        ret_map   = dict(zip(grp["ticker"], grp["fwd_ret"]))
        kept      = {n for n in hyst_hold if score_map.get(n, 0.0) >= med_all}
        new_hold  = kept | curr_set
        new_hold  = {n for n in new_hold if n in ret_map}
        if new_hold:
            hg = float(np.mean([ret_map[n] for n in new_hold]))
            h_new = new_hold - hyst_hold if hyst_hold else new_hold
            h_turn = len(h_new) / max(len(new_hold), 1)
            hyst_returns.append(hg - h_turn * cost_bps / 10_000)
            if hyst_hold:
                hyst_turnovers.append(
                    1 - len(new_hold & hyst_hold) / max(len(new_hold), 1)
                )
        hyst_hold = new_hold

    if period_ics:
        holdout_ic_mean = round(float(np.mean(period_ics)), 4)
    if len(period_ics) > 1:
        holdout_icir = round(float(np.mean(period_ics) / (np.std(period_ics) + 1e-9)), 3)
    if len(pooled_scores) > 2:
        ic_val, _ = spearmanr(pooled_scores, pooled_rets)
        holdout_ic = round(float(ic_val), 4) if np.isfinite(ic_val) else None

    # ── Per-period examples: the most recent walk-forward periods ──────────
    period_dates: dict[int, str] = {}
    if price_series:
        ref_prices = next(iter(price_series.values()))
        for p_idx in range(n_periods):
            si = p_idx * rebalance_days
            ei = si + forward_days
            if ei < len(ref_prices):
                sd = ref_prices.index[si]
                ed = ref_prices.index[ei]
                period_dates[p_idx] = f"{sd.strftime('%b %Y')} → {ed.strftime('%b %Y')}"

    for t in wf_periods[-6:]:
        grp = wf_scored[t]
        med = grp["fwd_ret"].median()
        ex  = grp.assign(
            pred_top=grp["model_score"] >= 0.5,
            actual_top=grp["fwd_ret"] >= med,
        )
        ex["correct"] = ex["pred_top"] == ex["actual_top"]
        ex = ex.sort_values("model_score", ascending=False)
        period_examples.append({
            "period":       int(t),
            "date_range":   period_dates.get(int(t), f"Period {t}"),
            "n_stocks":     len(ex),
            "hit_rate_pct": round(float(ex["correct"].mean()) * 100, 1),
            "stocks": [
                {
                    "ticker":            str(r["ticker"]),
                    "model_score":       round(float(r["model_score"]) * 100, 1),
                    "predicted_top":     bool(r["pred_top"]),
                    "actual_return_pct": round(float(r["fwd_ret"]) * 100, 1),
                    "actual_top":        bool(r["actual_top"]),
                    "correct":           bool(r["correct"]),
                }
                for _, r in ex.iterrows()
            ],
        })

    # ── Phase 4: train + save the final production model on ALL periods ────
    # Independent of the walk-forward above, so it may safely include the
    # current-snapshot rows (there is no holdout left for them to leak into).
    # oof_ic / cv_accuracy below come from this model's own internal CV and
    # are diagnostics only — the walk-forward IC is the ruler.
    training_info = {}
    if train_model and all_feature_rows:
        logger.info(
            f"Training final model on {len(all_feature_rows)} observations "
            f"from {len(period_frames)} periods "
            f"({len(wf_periods)} walk-forward OOS periods scored)..."
        )
        model = NUMKTEnsemble(cv_folds=5, lambda_reg=0.10, max_depth=5)
        model.fit_on_real_returns(
            feature_rows    = all_feature_rows,
            return_labels   = all_return_labels,
            forward_returns = all_forward_returns,
            features_df     = features_df,
            embargo         = 0,
            include_snapshot = True,
        )
        model.save(model_name)

        # ── Monitoring: capture training baseline (additive, fully wrapped) ──
        try:
            from monitoring import capture_training_baseline
            capture_training_baseline(model, features_df)
        except Exception:
            pass

        wf_note = (
            f"Walk-forward IC (out-of-sample, {len(wf_periods)} periods): "
            f"{holdout_ic_mean:.4f}. " if holdout_ic_mean is not None else
            "Walk-forward IC: n/a (too few periods). "
        )
        training_info = {
            "trained":         True,
            "training_rows":   len(all_feature_rows),
            "holdout_rows":    len(pooled_scores),
            "training_source": "real_returns",
            "cv_accuracy":     round(model.cv_accuracy, 4),
            "oof_ic":          round(model.cv_ic, 4),
            "holdout_ic":      holdout_ic,
            "holdout_ic_mean": holdout_ic_mean,
            "holdout_icir":    holdout_icir,
            "wf_periods":      len(wf_periods),
            "period_examples": period_examples,
            "model_name":      model_name,
            "message": (
                f"Model trained on {len(all_feature_rows)} observations. "
                f"OOF CV accuracy: {model.cv_accuracy:.1%}. OOF IC (diag): "
                f"{model.cv_ic:.4f}. {wf_note}Next analysis will use this model."
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
        result["effective_lookback_years"] = effective_lookback_yrs
        return result

    pa       = np.array(portfolio_returns)        # net of costs
    pa_gross = np.array(portfolio_returns_gross)   # before costs
    ba       = np.array(benchmark_returns)
    ppy      = 12 / rebalance_months

    ann_port       = float((np.prod(1 + pa_gross) ** (ppy / len(pa_gross))) - 1)
    ann_port_net   = float((np.prod(1 + pa)       ** (ppy / len(pa)))       - 1)
    ann_bench      = float((np.prod(1 + ba)       ** (ppy / len(ba)))       - 1)
    ann_alpha      = ann_port     - ann_bench
    ann_alpha_net  = ann_port_net - ann_bench

    rf     = 0.045 / ppy
    exc_rf = pa - rf   # Sharpe on net returns (after costs)
    sharpe = float(np.mean(exc_rf)/(np.std(exc_rf)+1e-9)*np.sqrt(ppy))

    cum    = np.cumprod(1+pa)
    rm     = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum-rm)/(rm+1e-9)))
    # Calmar is undefined when there is no drawdown — return None rather than
    # dividing by epsilon and producing a meaningless astronomical value.
    calmar = round(ann_port / abs(max_dd), 3) if max_dd < -1e-6 else None

    ic_mean  = float(np.mean(period_ics))       if period_ics       else 0.0
    ic_std   = float(np.std(period_ics))        if len(period_ics)>1 else 0.0
    icir     = ic_mean/(ic_std+1e-9)
    hit_rate = float(np.mean(period_hit_rates)) if period_hit_rates  else 0.5

    beta = 1.0
    if len(ba) > 1 and np.std(ba) > 0:
        beta = float(np.cov(pa,ba)[0,1]/(np.var(ba)+1e-9))

    avg_turn = float(np.mean(turnovers)) if turnovers else 0.5

    # ── Behavioural portfolios (co-equal ruler) ────────────────────────────
    # Risk-adjusted outcome of the sized + hysteresis books vs the naive
    # equal-weight top quintile. These measure the "behavioural edge" (size
    # correctly, trade less) the equal-weight IC book can't.
    behavioural = {
        "equal_weight": _port_stats(pa, ppy, rf, avg_turn),
        "sized":        _port_stats(np.array(sized_returns), ppy, rf,
                                    float(np.mean(sized_turnovers)) if sized_turnovers else None),
        "hysteresis":   _port_stats(np.array(hyst_returns), ppy, rf,
                                    float(np.mean(hyst_turnovers)) if hyst_turnovers else None),
        "note": ("Sized = conviction × inverse-vol weighting; hysteresis = hold "
                 "until score < median (fewer trades). Net of "
                 f"{cost_bps}bps round-trip costs."),
    }

    logger.info(
        f"BACKTEST COMPLETE | periods={len(pa)} | IC={ic_mean:.4f} | "
        f"Sharpe eq={sharpe:.2f}/sized={behavioural['sized'].get('sharpe')}/"
        f"hyst={behavioural['hysteresis'].get('sharpe')} | "
        f"turn eq={avg_turn:.0%}/hyst={behavioural['hysteresis'].get('avg_turnover_pct')} | "
        f"Alpha net={ann_alpha_net:.2%} ({cost_bps}bps)"
    )

    return {
        "n_periods":            len(pa),
        "forward_months":       forward_months,
        "ann_portfolio_return": round(ann_port*100, 2),
        "ann_portfolio_net":    round(ann_port_net*100, 2),
        "ann_benchmark_return": round(ann_bench*100, 2),
        "ann_alpha":            round(ann_alpha*100, 2),
        "ann_alpha_net":        round(ann_alpha_net*100, 2),
        "cost_bps":             cost_bps,
        "sharpe_ratio":         round(sharpe, 3),
        "max_drawdown":         round(max_dd*100, 2),
        "calmar_ratio":         calmar,
        "hit_rate":             round(hit_rate*100, 1),
        "ic_mean":              round(ic_mean, 4),
        "ic_std":               round(ic_std, 4),
        "icir":                 round(icir, 3),
        "beta_vs_benchmark":    round(beta, 3),
        "avg_monthly_turnover": round(avg_turn*100, 1),
        "effective_lookback_years": effective_lookback_yrs,
        "interpretation":       _interpret(hit_rate, ic_mean, sharpe, max_dd, ann_alpha_net),
        "behavioural":          behavioural,
        "training":             training_info,
    }


def _port_stats(returns: np.ndarray, ppy: float, rf: float,
                avg_turnover: float | None) -> dict:
    """Annualised return / Sharpe / max-drawdown / Calmar / turnover for a
    per-period net-return series. Used to compare the behavioural books."""
    if returns is None or len(returns) == 0:
        return {"n_periods": 0, "ann_return_pct": None, "sharpe": None,
                "max_drawdown_pct": None, "calmar": None, "avg_turnover_pct": None}
    r      = np.asarray(returns, dtype=float)
    ann    = float(np.prod(1 + r) ** (ppy / len(r)) - 1)
    exc    = r - rf
    sharpe = float(np.mean(exc) / (np.std(exc) + 1e-9) * np.sqrt(ppy))
    cum    = np.cumprod(1 + r)
    rm     = np.maximum.accumulate(cum)
    max_dd = float(np.min((cum - rm) / (rm + 1e-9)))
    calmar = round(ann / abs(max_dd), 3) if max_dd < -1e-6 else None
    return {
        "n_periods":        len(r),
        "ann_return_pct":   round(ann * 100, 2),
        "sharpe":           round(sharpe, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar":           calmar,
        "avg_turnover_pct": round(avg_turnover * 100, 1) if avg_turnover is not None else None,
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