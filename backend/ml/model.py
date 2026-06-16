"""
model.py  (v3 — data-driven synthetic labels)
──────────────────────────────────────────────
Key fix over v2:
  _fit_synthetic() no longer uses hand-weighted factor scores as training
  labels. Instead it mines the price history already cached on disk to
  compute real historical forward returns for each stock, then labels each
  observation 1 if the stock beat the equal-weight universe median that
  period, 0 if not.

  This is the same labelling logic used by fit_on_real_returns(), but it
  runs automatically from cached price data without needing a full backtest
  trigger. The result is that even the "fallback" model trains on something
  grounded in actual market outcomes, not a developer's intuition about
  which factor weights feel right.

Why the old approach was wrong:
  The original _fit_synthetic() built a quality score like:
      quality += 0.25 * roe_rank + 0.20 * mom_12m_rank - 0.15 * pe_ratio_rank ...
  Then trained the model to predict which stocks scored above that median.
  The model learned to replicate the hand-coded formula — not to discover
  what actually predicts returns. Training accuracy looked decent because
  the model was just learning to reverse-engineer the formula it was given.
  That is not machine learning; it is a lookup table with extra steps.

New approach:
  1. Load cached price parquet files from cache/prices/
  2. For each stock compute forward returns at multiple horizons
     (3m, 6m, 12m) using non-overlapping windows
  3. For each time window label = 1 if stock beat universe median, else 0
  4. Stack all (features, label) pairs and train on real outcomes
  5. Fall back to pure cross-sectional momentum rank only if fewer than
     MIN_REAL_ROWS valid observations can be assembled — and log clearly
     that this happened so the developer knows data is thin

NUMKTEnsemble public API is unchanged — fit(), fit_on_real_returns(),
predict_proba(), save(), load(), list_saved() all behave identically.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime

from .feature_engineering import FEATURE_COLS, PRICE_FEATURE_COLS, safe_ticker_filename
from .embedding_features import (
    point_in_time_embedding,
    encoder_enabled,
    standardize_emb_columns,
)

logger = logging.getLogger(__name__)


def _fill_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaNs: price/fundamental rank cols → 0.5 (neutral rank),
    embedding cols → 0.0 (neutral on the standardized embedding scale)."""
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if emb_cols:
        df[emb_cols] = df[emb_cols].fillna(0.0)
    other = [c for c in df.columns if c not in emb_cols]
    if other:
        df[other] = df[other].fillna(0.5)
    return df

import os as _os
_cache_root = _os.environ.get("CACHE_DIR")
_base = Path(_cache_root) if _cache_root else Path(__file__).parent.parent / "cache"
MODEL_DIR = _base / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Where data_fetcher stores parquet price files
PRICE_DIR = _base / "prices"

# Diverse anchor tickers included in training when already cached — never fetched.
# Broadens the cross-sectional universe so factor patterns aren't learned from
# a single user's narrow watchlist.
# Broad, liquid, sector-balanced large/mid-cap universe (~180 names). Breadth
# is the cheapest precision lever for a cross-sectional IC: per-period IC noise
# scales ~1/sqrt(N), so widening 48 -> ~180 names roughly halves the standard
# error on the walk-forward IC (measured: mean IC SE ~0.040 -> ~0.021), which
# is what lets a genuine 0.05-0.07 edge clear the noise floor. Startup never
# bulk-fetches these (rate limits); only anchors already in the price cache are
# used, and the cache fills over time from user requests + background retrains.
# Note: all survivors — adds the same survivorship tilt the universe already
# had, just broader; not a new source of bias.
_ANCHOR_TICKERS = [
    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD",  "INTC", "CRM",  "ORCL",
    "CSCO", "IBM",  "TXN",  "QCOM",  "NOW",  "INTU", "AMAT", "MU",   "ADI",  "LRCX",
    "KLAC", "SNPS", "CDNS", "PANW",  "PYPL", "UBER", "AVGO", "ADBE", "NFLX",
    # Healthcare
    "JNJ",  "PFE",  "UNH",  "ABBV",  "MRK",  "TMO",  "ABT",  "BMY",  "AMGN", "GILD",
    "LLY",  "DHR",  "ISRG", "VRTX",  "REGN", "ZTS",  "CI",   "CVS",  "HUM",  "CNC",
    "MDT",  "SYK",  "BSX",
    # Financials
    "JPM",  "BAC",  "WFC",  "GS",    "MS",   "BLK",  "C",    "AXP",  "V",    "MA",
    "SCHW", "USB",  "PNC",  "TFC",   "COF",  "SPGI", "CME",  "ICE",  "AON",  "MMC",
    "PGR",  "TRV",  "CB",   "AIG",   "MET",  "PRU",
    # Consumer Discretionary
    "HD",   "MCD",  "NKE",  "SBUX",  "TGT",  "COST", "LOW",  "TJX",  "BKNG", "MAR",
    "GM",   "F",    "DIS",  "CMCSA", "TMUS", "VZ",   "T",    "CHTR",
    # Consumer Staples
    "WMT",  "PG",   "KO",   "PEP",   "MDLZ", "CL",   "KMB",  "GIS",  "KHC",  "STZ",
    "MO",   "PM",   "MNST", "KDP",
    # Industrials
    "CAT",  "HON",  "UPS",  "GE",    "BA",   "LMT",  "DE",   "MMM",  "EMR",  "ETN",
    "ITW",  "NSC",  "CSX",  "FDX",   "GD",   "NOC",  "RTX",
    # Energy
    "XOM",  "CVX",  "SLB",  "COP",   "EOG",  "MPC",  "PSX",  "VLO",  "OXY",  "KMI",
    "WMB",  "HAL",
    # Utilities
    "NEE",  "DUK",  "SO",   "D",     "AEP",  "EXC",  "SRE",  "XEL",  "PEG",  "ED",
    # Materials
    "LIN",  "APD",  "SHW",  "ECL",   "NEM",  "FCX",  "DOW",  "NUE",
    # Real Estate
    "AMT",  "PLD",  "CCI",  "EQIX",  "SPG",  "O",    "PSA",  "WELL", "DLR",
    # Diversified holding
    "BRK-B",
    # Sector ETFs — diversify beta profiles across macro regimes
    "XLK",  "XLF",  "XLV",  "XLE",  "XLI",
    # Mid/small-cap — different momentum dynamics than mega-caps
    "IWM",  "SMCI", "CELH",
]

# Minimum real labelled rows before we trust the model.
# With 21 features, 1.5 samples/feature is too thin — 60 gives ~3 samples/feature.
MIN_REAL_ROWS = 60

# Forward-return horizons used for synthetic label generation (trading days).
# Multiple horizons increase the number of labelled observations we can
# extract from the same price history.
FORWARD_HORIZONS = [63, 126, 252]   # ~3m, 6m, 12m


class NUMKTEnsemble:
    """
    Random Forest + Gradient Boosting ensemble.
    Trained on real historical forward returns wherever possible.
    Persists to disk between sessions.
    """

    def __init__(
        self,
        cv_folds:     int   = 5,
        lambda_reg:   float = 0.10,
        max_depth:    int   = 5,
        n_estimators: int   = 300,
        rf_weight:    float = 0.60,
    ):
        self.cv_folds     = cv_folds
        self.lambda_reg   = lambda_reg
        self.max_depth    = max_depth
        self.n_estimators = n_estimators
        self.rf_weight    = rf_weight
        self.gb_weight    = 1.0 - rf_weight

        self.is_fitted       = False
        self.cv_accuracy     = 0.0
        self.cv_ic           = 0.0
        self.n_training_rows = 0
        self.training_source = "none"
        self._feature_names  = []
        self._fi_rf              = {}
        self._fi_gb              = {}
        self.trained_at          = None
        self._signal_thresholds  = {}   # calibrated from OOF probability distribution

        lr = max(0.01, 0.1 * (1 - lambda_reg))

        self._rf_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  RobustScaler()),
            ("clf",     RandomForestClassifier(
                n_estimators    = n_estimators,
                max_depth       = max_depth,
                min_samples_leaf= 10,
                max_features    = "sqrt",
                class_weight    = "balanced",
                n_jobs          = -1,
                random_state    = 42,
            )),
        ])

        self._gb_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  RobustScaler()),
            ("clf",     GradientBoostingClassifier(
                n_estimators    = n_estimators,
                max_depth       = min(max_depth, 4),
                learning_rate   = lr,
                subsample       = 0.8,
                min_samples_leaf= 8,
                max_features    = "sqrt",
                random_state    = 42,
            )),
        ])


    # ── Public training entry points ───────────────────────────────────────

    def fit(self, features_df: pd.DataFrame) -> "NUMKTEnsemble":
        """
        Called by /analyse when no explicit backtest has been run yet.

        Attempts to build real labels from cached price history first.
        Only falls back to the last-resort cross-sectional method if
        price data is genuinely unavailable or too thin.
        """
        return self._fit_from_price_cache(features_df)


    def fit_on_real_returns(
        self,
        feature_rows:    list[dict],
        return_labels:   list[int],
        forward_returns: list[float] | None = None,
        features_df:     pd.DataFrame | None = None,
        embargo:         int = 0,
        include_snapshot: bool = True,
    ) -> "NUMKTEnsemble":
        """
        Train on real historical outcomes supplied by the backtest runner.
        This is the highest-quality path and takes priority over everything.

        features_df is optional. When supplied, current-snapshot rows (all 16
        features including fundamentals) are appended to the backtest's
        price-only historical rows, so the saved model learns to use
        fundamentals at inference time.
        """
        if len(feature_rows) < 6:
            logger.warning(
                f"Only {len(feature_rows)} training rows from backtest — "
                "falling back to price-cache labels"
            )
            return self._fit_from_price_cache(pd.DataFrame(feature_rows))

        # Append current-snapshot rows (price features only) if available.
        # Both historical backtest rows and snapshot rows use only price rank
        # features, keeping the entire training pipeline free of look-ahead bias.
        #
        # IMPORTANT: snapshot rows are labelled by the trailing-12M return, which
        # OVERLAPS the most recent (holdout) windows of a walk-forward backtest —
        # training on them leaks the holdout outcome and inflates holdout IC
        # (badly at short horizons, where every holdout window sits inside the
        # trailing year). The backtest runner therefore passes include_snapshot
        # =False so the holdout metric stays honest. The price-cache fit() path,
        # which has no holdout, still uses them to add current-period data.
        if include_snapshot and features_df is not None and not features_df.empty:
            snap_rows, snap_labels, snap_returns = self._current_snapshot_rows(features_df)
            if snap_rows:
                feature_rows    = feature_rows + snap_rows
                return_labels   = return_labels + snap_labels
                if forward_returns is not None:
                    forward_returns = forward_returns + snap_returns
                logger.info(
                    f"Appended {len(snap_rows)} current-snapshot rows (price features)"
                )

        df = _fill_features(pd.DataFrame(feature_rows))
        y  = np.array(return_labels)

        # Extract period groups from the rows (not the fillna'd frame, which
        # would coerce missing _period_idx to 0.5). Current-snapshot rows have
        # no _period_idx and are mapped to the 999_999 sentinel so the
        # period-group CV excludes them from every test fold.
        groups = np.array([float(r.get("_period_idx", 999_999)) for r in feature_rows])

        rank_cols = self._select_rank_cols(df)
        if not rank_cols:
            logger.warning("No feature columns found in backtest data")
            return self

        self._feature_names = rank_cols
        X = df[rank_cols].values

        pos = y.sum()
        neg = len(y) - pos
        if pos < 2 or neg < 2:
            logger.warning(
                f"Poor class balance in backtest data ({pos} pos, {neg} neg) "
                "— falling back to price-cache labels"
            )
            return self._fit_from_price_cache(df)

        self._run_cv_and_fit(X, y, groups=groups, embargo=embargo)

        # IC from OOF predictions — genuinely out-of-sample. Uncovered rows
        # (first period + snapshot sentinel) are NaN and excluded.
        if (forward_returns is not None
                and len(forward_returns) == len(X)
                and self._oof_proba is not None):
            fr      = np.asarray(forward_returns, dtype=float)
            covered = np.isfinite(self._oof_proba)
            if covered.sum() >= 10:
                ic, _ = spearmanr(self._oof_proba[covered], fr[covered])
                self.cv_ic = float(ic) if np.isfinite(ic) else 0.0
            else:
                self.cv_ic = 0.0
            logger.info(f"OOF IC (Spearman): {self.cv_ic:.4f}")
        else:
            self.cv_ic = 0.0

        self.is_fitted        = True
        self.n_training_rows  = len(y)
        self.training_source  = "real_returns"
        self.trained_at       = datetime.utcnow().isoformat()

        logger.info(
            f"Model fitted on {len(y)} real-return rows. "
            f"CV accuracy: {self.cv_accuracy:.3f}"
        )
        return self


    # ── Core fix: price-cache label generation ─────────────────────────────

    def _fit_from_price_cache(self, features_df: pd.DataFrame) -> "NUMKTEnsemble":
        """
        Mine cached price history to build data-driven training labels.

        For each ticker x forward horizon combination:
          - Step through history in non-overlapping windows of `horizon` days
          - At each step T, record the factor snapshot and the actual forward
            return from T to T+horizon
          - Label = 1 if that return beat the cross-sectional median for that
            window, 0 otherwise

        This gives us many more labelled rows than the number of tickers,
        because we get multiple observations per ticker across time.

        Falls back to _fit_last_resort() if fewer than MIN_REAL_ROWS
        observations can be assembled (e.g. all tickers newly added with
        no cached history yet).
        """
        tickers = (
            list(features_df["ticker"].dropna())
            if "ticker" in features_df.columns
            else []
        )

        if not tickers:
            logger.warning("No tickers in features_df — using last-resort fallback")
            return self._fit_last_resort(features_df)

        # Extend universe with any anchor tickers already in the price cache.
        # Never triggers a new Yahoo fetch — only uses what's on disk.
        anchor_cached = [
            t for t in _ANCHOR_TICKERS
            if t not in tickers
            and (PRICE_DIR / f"{safe_ticker_filename(t)}.parquet").exists()
        ]
        if anchor_cached:
            logger.info(
                f"Extending training universe with {len(anchor_cached)} "
                f"cached anchor ticker(s): {anchor_cached}"
            )
            tickers = tickers + anchor_cached

        feature_rows, return_labels, forward_returns_list = [], [], []

        for horizon in FORWARD_HORIZONS:
            rows, labels, fwd = self._labels_for_horizon(tickers, horizon)
            feature_rows.extend(rows)
            return_labels.extend(labels)
            forward_returns_list.extend(fwd)

        # Append current-snapshot rows: one per ticker using price rank features
        # and the actual past-12M return as the label. These add N more price-only
        # observations for the current period to the training set.
        snap_rows, snap_labels, snap_returns = self._current_snapshot_rows(features_df)
        if snap_rows:
            feature_rows.extend(snap_rows)
            return_labels.extend(snap_labels)
            forward_returns_list.extend(snap_returns)
            logger.info(
                f"Appended {len(snap_rows)} current-snapshot rows (price features)"
            )

        n = len(return_labels)
        logger.info(
            f"Price-cache label generation: {n} total observations "
            f"({n - len(snap_rows)} historical windows + {len(snap_rows)} current snapshots)"
        )

        if n < MIN_REAL_ROWS:
            logger.warning(
                f"Only {n} labelled rows from price cache (need {MIN_REAL_ROWS}). "
                "Price history may be missing or too short. "
                "Using last-resort cross-sectional fallback — "
                "run /backtest to build a proper model."
            )
            return self._fit_last_resort(features_df)

        # Sort chronologically so group CV sees periods in order
        combined = sorted(
            zip(feature_rows, return_labels, forward_returns_list),
            key=lambda x: x[0].get("_time_idx", 0)
        )
        feature_rows, return_labels, forward_returns_list = (
            list(t) for t in zip(*combined)
        )

        # Extract period groups before removing _time_idx
        groups = np.array([r.get("_time_idx", 0) for r in feature_rows])
        for row in feature_rows:
            row.pop("_time_idx", None)

        df_train = _fill_features(pd.DataFrame(feature_rows))
        y        = np.array(return_labels)
        fwd_arr  = np.array(forward_returns_list)

        rank_cols = self._select_rank_cols(df_train)
        if not rank_cols:
            logger.warning("No rank columns in price-cache training data")
            return self._fit_last_resort(features_df)

        self._feature_names = rank_cols
        X = df_train[rank_cols].values

        pos = y.sum()
        neg = n - pos
        if pos < 2 or neg < 2:
            logger.warning(
                f"Poor class balance in price-cache data ({pos} pos, {neg} neg). "
                "All stocks may have moved together. Using last-resort fallback."
            )
            return self._fit_last_resort(features_df)

        self._run_cv_and_fit(X, y, groups=groups)

        # IC from OOF predictions — exclude uncovered rows (zeros from t_start=0
        # and current-snapshot sentinel) to avoid diluting the IC with artifacts.
        if self._oof_proba is not None and len(self._oof_proba) == len(fwd_arr):
            covered = np.isfinite(self._oof_proba)
            if covered.sum() >= 10:
                ic, _ = spearmanr(self._oof_proba[covered], fwd_arr[covered])
                self.cv_ic = float(ic) if np.isfinite(ic) else 0.0
            else:
                self.cv_ic = 0.0
        else:
            self.cv_ic = 0.0

        self.is_fitted        = True
        self.n_training_rows  = n
        self.training_source  = "price_cache"
        self.trained_at       = datetime.utcnow().isoformat()

        logger.info(
            f"Model fitted on {n} price-cache rows. "
            f"OOF CV accuracy: {self.cv_accuracy:.3f}  OOF IC: {self.cv_ic:.4f}"
        )
        return self


    def _labels_for_horizon(
        self,
        tickers: list[str],
        horizon: int,
    ) -> tuple[list[dict], list[int], list[float]]:
        """
        For a given forward horizon (in trading days), load each ticker's
        cached price series and generate non-overlapping (features, label)
        pairs stepping through the history.

        Only price-derived features (momentum, vol, RSI, price_vs_52w_high) are
        used here — computed from price history at each window start with no
        look-ahead bias. Fundamental features are intentionally excluded from
        historical windows because we have no point-in-time fundamental data;
        using today's values for windows from 2–5 years ago would inject
        look-ahead bias into every training observation. Fundamental rank columns
        not present in these rows are filled with 0.5 (neutral) downstream.

        Each row is tagged with _time_idx = t_start so that
        _fit_from_price_cache can sort observations chronologically before
        TimeSeriesSplit CV.
        """
        price_series: dict[str, pd.Series] = {}
        for ticker in tickers:
            path = PRICE_DIR / f"{safe_ticker_filename(ticker)}.parquet"
            if not path.exists():
                continue
            try:
                df_p = pd.read_parquet(path)
                if "Close" in df_p.columns and len(df_p) >= horizon * 2:
                    price_series[ticker] = df_p["Close"].sort_index()
            except Exception as e:
                logger.warning(f"Could not load price cache for {ticker}: {e}")

        if len(price_series) < 3:
            return [], [], []

        min_len   = min(len(s) for s in price_series.values())
        n_windows = (min_len - horizon) // horizon
        if n_windows < 1:
            return [], [], []

        all_feature_rows: list[dict]  = []
        all_labels:       list[int]   = []
        all_fwd:          list[float] = []

        for w in range(n_windows):
            t_start = w * horizon
            t_end   = t_start + horizon

            # Build historical feature snapshots for all tickers at t_start.
            # Use only price data available up to (and including) t_start.
            raw_snaps: dict[str, dict] = {}
            emb_by_ticker: dict[str, dict] = {}
            emb_on = encoder_enabled()
            for ticker, prices in price_series.items():
                if len(prices) <= t_end:
                    continue
                p = prices.iloc[:t_start + 1] if t_start > 0 else prices.iloc[:2]
                if len(p) < 5:
                    continue

                snap: dict = {}  # features for this window

                # Momentum at this historical point
                for days, key in [(21, "mom_1m"), (63, "mom_3m"),
                                   (126, "mom_6m"), (252, "mom_12m")]:
                    snap[key] = (float(p.iloc[-1] / p.iloc[-(days + 1)] - 1)
                                 if len(p) > days else np.nan)

                # Realised volatility (annualised)
                if len(p) > 10:
                    lr = np.log(p / p.shift(1)).dropna().iloc[-min(60, len(p) - 1):]
                    snap["vol_60d"] = float(lr.std() * np.sqrt(252)) if len(lr) > 1 else np.nan
                else:
                    snap["vol_60d"] = np.nan

                # RSI-14 using Wilder's EMA (alpha=1/N)
                if len(p) > 15:
                    d    = p.diff().dropna()
                    gain = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
                    loss = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
                    snap["rsi_14"] = float(100 - 100 / (1 + gain / (loss + 1e-9)))
                else:
                    snap["rsi_14"] = np.nan

                hi = p.iloc[-min(252, len(p)):].max()
                snap["price_vs_52w_high"] = float(p.iloc[-1] / hi - 1) if hi > 0 else np.nan

                raw_snaps[ticker] = snap
                if emb_on:
                    # Point-in-time embedding: only prices up to t_start are used,
                    # so future bars cannot leak into this window's embedding.
                    emb_by_ticker[ticker] = point_in_time_embedding(
                        prices, end_idx=t_start, ticker=ticker
                    )

            if len(raw_snaps) < 3:
                continue

            # Cross-sectional rank normalisation at this window's point in time.
            # Only price-derived features are used here — no fundamental look-ahead.
            snap_df = pd.DataFrame(raw_snaps).T
            for col in PRICE_FEATURE_COLS:
                if col in snap_df.columns and snap_df[col].notna().sum() > 1:
                    snap_df[col + "_rank"] = snap_df[col].rank(pct=True)
                elif col + "_rank" not in snap_df.columns:
                    snap_df[col + "_rank"] = 0.5
            snap_df = snap_df.fillna(0.5)

            # Attach standardized encoder embeddings (cross-sectional per window).
            # Added after fillna(0.5) so emb cols use a 0.0 neutral, not 0.5.
            if emb_on and emb_by_ticker:
                emb_df = standardize_emb_columns(pd.DataFrame(emb_by_ticker).T)
                for c in emb_df.columns:
                    snap_df[c] = emb_df[c].reindex(snap_df.index).fillna(0.0)

            # Actual forward returns from t_start to t_end
            window_returns: dict[str, float] = {}
            for ticker in raw_snaps:
                prices_s = price_series[ticker]
                if len(prices_s) <= t_end:
                    continue
                p0 = prices_s.iloc[t_start]
                p1 = prices_s.iloc[t_end]
                if p0 > 0:
                    window_returns[ticker] = float(p1 / p0 - 1)

            if len(window_returns) < 3:
                continue

            median_ret = float(np.median(list(window_returns.values())))

            for ticker, ret in window_returns.items():
                if ticker not in snap_df.index:
                    continue
                label = 1 if ret > median_ret else 0
                frow  = snap_df.loc[ticker].to_dict()
                frow["_time_idx"] = t_start  # for chronological sorting
                all_feature_rows.append(frow)
                all_labels.append(label)
                all_fwd.append(ret)

        return all_feature_rows, all_labels, all_fwd


    # ── Current-snapshot rows (fundamentals + price) ───────────────────────

    def _current_snapshot_rows(
        self,
        features_df: pd.DataFrame,
    ) -> tuple[list[dict], list[int], list[float]]:
        """
        One training observation per ticker using today's price rank features
        with the actual past-12M return as the label.

        Only price-derived rank features are used here (PRICE_FEATURE_COLS),
        consistent with the historical window rows from _labels_for_horizon.
        Using today's fundamental data (PE, ROE, etc.) to explain a past 12M
        return would be a subtler form of look-ahead bias — the fundamentals
        the model sees in training would not have existed at the start of that
        return window. Price-only labels keep the training signal clean and
        genuinely evaluable via walk-forward backtest.
        """
        all_rank_cols = [
            c + "_rank" for c in PRICE_FEATURE_COLS
            if c + "_rank" in features_df.columns
        ]
        if not all_rank_cols or features_df.empty:
            return [], [], []

        tickers = (
            list(features_df["ticker"].dropna())
            if "ticker" in features_df.columns
            else []
        )
        if not tickers:
            return [], [], []

        # Load the past-12M return for each ticker from cached price files.
        # Using 253 bars (252 + current day) to match the mom_12m definition.
        emb_on = encoder_enabled()
        emb_by_ticker: dict[str, dict] = {}
        fwd_returns: dict[str, float] = {}
        for ticker in tickers:
            path = PRICE_DIR / f"{safe_ticker_filename(ticker)}.parquet"
            if not path.exists():
                continue
            try:
                df_p = pd.read_parquet(path)
                if "Close" not in df_p.columns or len(df_p) < 253:
                    continue
                prices = df_p["Close"].sort_index()
                p0 = float(prices.iloc[-253])
                p1 = float(prices.iloc[-1])
                if p0 > 0:
                    fwd_returns[ticker] = round(p1 / p0 - 1, 6)
                if emb_on:
                    emb_by_ticker[ticker] = point_in_time_embedding(
                        prices, end_idx=len(prices) - 1, ticker=ticker
                    )
            except Exception as e:
                logger.warning(f"Could not load price cache for {ticker}: {e}")

        if len(fwd_returns) < 3:
            return [], [], []

        median_ret = float(np.median(list(fwd_returns.values())))

        rows: list[dict]  = []
        labels: list[int] = []
        returns: list[float] = []

        for _, row in features_df.iterrows():
            ticker = str(row.get("ticker", ""))
            if ticker not in fwd_returns:
                continue
            ret = fwd_returns[ticker]

            frow: dict = {}
            for col in all_rank_cols:
                val = row.get(col)
                try:
                    frow[col] = float(val) if val is not None and np.isfinite(float(val)) else 0.5
                except (TypeError, ValueError):
                    frow[col] = 0.5

            if emb_on:
                frow.update(emb_by_ticker.get(ticker, {}))
            # _time_idx = large value so these sort after all historical windows
            frow["_time_idx"] = 999_999
            rows.append(frow)
            labels.append(1 if ret > median_ret else 0)
            returns.append(ret)

        # Standardize embedding columns cross-sectionally across the current universe.
        if emb_on and rows:
            rows = standardize_emb_columns(pd.DataFrame(rows)).to_dict("records")

        return rows, labels, returns


    # ── Last-resort fallback ───────────────────────────────────────────────

    def _fit_last_resort(self, features_df: pd.DataFrame) -> "NUMKTEnsemble":
        """
        Absolute last resort when no price history exists at all.

        Uses a SINGLE signal: 12-month price momentum rank.
        Momentum is the most robust single factor in the academic literature
        (Jegadeesh & Titman 1993). One evidence-based signal beats a cocktail
        of arbitrary hand-picked weights.

        This only triggers for brand-new deployments where no ticker has any
        cached price history. As soon as one /analyse call completes and
        prices are cached, _fit_from_price_cache() takes over next call.
        """
        logger.warning(
            "LAST-RESORT FALLBACK ACTIVE: no price history available for any ticker. "
            "Using 12m momentum rank as sole training signal. "
            "This improves automatically after the first /analyse call "
            "caches price data. Run /backtest for a fully data-driven model."
        )

        rank_cols = [c for c in features_df.columns if c.endswith("_rank")]
        if not rank_cols:
            self.is_fitted = False
            return self

        momentum_col = next(
            (c for c in ["mom_12m_rank", "mom_6m_rank", "mom_3m_rank"]
             if c in rank_cols),
            rank_cols[0]
        )

        self._feature_names = rank_cols
        X = features_df[rank_cols].fillna(0.5).values
        y = (features_df[momentum_col].fillna(0.5) >= 0.5).astype(int).values

        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            self.is_fitted = False
            return self

        # Skip CV entirely. features_df is a single cross-sectional snapshot
        # (one row per ticker, all from the same moment), not a time series.
        # TimeSeriesSplit on row-index order is meaningless here and produces a
        # fabricated accuracy number. Report 0.0 so the frontend shows "N/A"
        # rather than a misleading figure.
        self.cv_accuracy = 0.0

        self._rf_pipe.fit(X, y)
        self._gb_pipe.fit(X, y)
        self._store_feature_importances(rank_cols)

        self.is_fitted        = True
        self.n_training_rows  = len(y)
        self.training_source  = "last_resort_momentum"
        self.trained_at       = datetime.utcnow().isoformat()
        return self


    # ── Shared helpers ─────────────────────────────────────────────────────

    def _select_rank_cols(self, df: pd.DataFrame) -> list[str]:
        """Return the ranked feature columns present in df.

        Includes any encoder embedding columns (emb_*) so the head trains on
        [price ranks (+ fundamentals for snapshots) + embeddings]. When no
        encoder is loaded there are no emb_* columns and this is a no-op.
        """
        feat_cols = [c for c in FEATURE_COLS
                     if c + "_rank" in df.columns or c in df.columns]
        rank_cols = []
        for c in feat_cols:
            if c + "_rank" in df.columns:
                rank_cols.append(c + "_rank")
            elif c in df.columns:
                rank_cols.append(c)
        rank_cols.extend(sorted(
            (c for c in df.columns if c.startswith("emb_")),
            key=lambda c: int(c.split("_")[1]) if c.split("_")[1].isdigit() else 0,
        ))
        return rank_cols


    @staticmethod
    def _period_group_cv(groups: np.ndarray, n_splits: int, embargo: int = 0):
        """Time-respecting group CV: test periods are always strictly after train periods.

        Ensures all observations from the same time period are on the same
        side of every fold boundary, preventing cross-sectional leakage where
        AAPL at period 7 (train) and MSFT at period 7 (test) would share the
        same market regime.

        `embargo` (in period units) additionally drops the training periods
        immediately before each test block. When forward label windows are
        longer than the rebalance step they overlap across consecutive periods,
        so a training period within `embargo` of the test block shares part of
        its forward return with the test labels (overlapping-label leakage).
        Set embargo = forward_days // rebalance_days to make folds leak-free;
        the price-cache path uses non-overlapping windows and leaves it at 0.

        Current-snapshot rows are tagged with 999_999 and excluded from all
        folds — they're always in the final .fit() only.
        """
        # Exclude the current-snapshot sentinel (never a valid test period)
        unique_periods = sorted(p for p in np.unique(groups) if p != 999_999)
        n = len(unique_periods)
        if n < 2:
            return
        n_splits = min(n_splits, n - 1)
        # Distribute all periods after the first into n_splits test chunks.
        test_chunks = [list(c) for c in np.array_split(unique_periods[1:], n_splits) if len(c)]
        for chunk in test_chunks:
            first_test = chunk[0]
            # Embargo: training periods must end at least `embargo` periods
            # before the test block starts so their label windows don't overlap.
            train_pds  = [p for p in unique_periods if p < first_test - embargo]
            if not train_pds:
                continue
            train_idx = np.where(np.isin(groups, train_pds))[0]
            test_idx  = np.where(np.isin(groups, chunk))[0]
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


    def _run_cv_and_fit(self, X: np.ndarray, y: np.ndarray,
                        groups: np.ndarray | None = None,
                        embargo: int = 0):
        """Run CV then full fit on both pipelines.

        When period groups are supplied, uses _period_group_cv so that all
        observations from a given time period stay on the same side of every
        fold boundary (prevents cross-sectional leakage).  `embargo` (period
        units) additionally separates train/test folds when label windows
        overlap across periods.  Falls back to TimeSeriesSplit when no group
        information is available.
        """
        n_splits = min(self.cv_folds, max(2, len(y) // 4))

        if groups is not None and len(np.unique(groups)) >= 3:
            cv = list(NUMKTEnsemble._period_group_cv(groups, n_splits, embargo))
            if not cv:
                cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = TimeSeriesSplit(n_splits=n_splits)

        # Manual walk-forward OOF assembly. cross_val_predict cannot be used
        # here: walk-forward / period-group CV intentionally never tests the
        # first period or the current-snapshot sentinel rows, so the test folds
        # do NOT form a partition of all rows — cross_val_predict requires a
        # partition and raises "only works for partitions", which previously
        # failed silently and forced an in-sample fallback (fake metrics).
        # Instead we fit a clone per fold, scatter predictions into an OOF array
        # initialised to NaN, and leave uncovered rows as NaN (genuinely OOS).
        self._oof_proba: np.ndarray | None = None
        try:
            folds = cv if isinstance(cv, list) else list(cv.split(X, y))

            oof_rf = np.full(len(y), np.nan)
            oof_gb = np.full(len(y), np.nan)
            for tr_idx, te_idx in folds:
                if len(tr_idx) == 0 or len(te_idx) == 0:
                    continue
                rf = clone(self._rf_pipe).fit(X[tr_idx], y[tr_idx])
                gb = clone(self._gb_pipe).fit(X[tr_idx], y[tr_idx])
                oof_rf[te_idx] = rf.predict_proba(X[te_idx])[:, 1]
                oof_gb[te_idx] = gb.predict_proba(X[te_idx])[:, 1]

            covered = np.isfinite(oof_rf) & np.isfinite(oof_gb)
            self._oof_proba = np.full(len(y), np.nan)
            self._oof_proba[covered] = (
                self.rf_weight * oof_rf[covered] + self.gb_weight * oof_gb[covered]
            )

            if covered.sum() > 0:
                oof_class = (self._oof_proba[covered] >= 0.5).astype(int)
                self.cv_accuracy = float(np.mean(oof_class == y[covered]))
                logger.info(
                    f"OOF CV accuracy (ensemble): {self.cv_accuracy:.3f} "
                    f"on {int(covered.sum())}/{len(y)} covered rows"
                )
            else:
                self.cv_accuracy = 0.0
                logger.warning("OOF CV produced no covered rows")

            # Calibrate signal thresholds from the covered (genuinely OOS) OOF
            # predictions only. Uncovered rows are NaN and excluded by `covered`.
            covered_proba = self._oof_proba[covered]
            if len(covered_proba) >= 10:
                self._signal_thresholds = {
                    "strong_buy": float(np.percentile(covered_proba, 80)),
                    "buy":        float(np.percentile(covered_proba, 55)),
                    "hold":       float(np.percentile(covered_proba, 30)),
                }
                logger.info(
                    f"Signal thresholds calibrated from {len(covered_proba)} "
                    f"covered OOF predictions — "
                    f"STRONG BUY>{self._signal_thresholds['strong_buy']:.3f}, "
                    f"BUY>{self._signal_thresholds['buy']:.3f}, "
                    f"HOLD>{self._signal_thresholds['hold']:.3f}"
                )
        except Exception as e:
            logger.warning(f"CV failed: {e}")
            self.cv_accuracy = 0.0
            self._oof_proba  = None

        self._rf_pipe.fit(X, y)
        self._gb_pipe.fit(X, y)
        self._store_feature_importances(self._feature_names)


    def fit_pipelines(self, X: np.ndarray, y: np.ndarray,
                      feature_names: list[str] | None = None) -> "NUMKTEnsemble":
        """Fit both pipelines directly, no CV. Used for the walk-forward
        backtest where a fresh model is trained at every step purely to score
        the next out-of-sample period — CV/IC/threshold calibration would be
        wasted work there. Returns self so steps can chain fit→predict_proba."""
        if feature_names is not None:
            self._feature_names = feature_names
        self._rf_pipe.fit(X, y)
        self._gb_pipe.fit(X, y)
        self.is_fitted = True
        return self


    def _store_feature_importances(self, rank_cols: list[str]):
        rf_clf = self._rf_pipe.named_steps["clf"]
        gb_clf = self._gb_pipe.named_steps["clf"]
        self._fi_rf = dict(zip(rank_cols, rf_clf.feature_importances_))
        self._fi_gb = dict(zip(rank_cols, gb_clf.feature_importances_))


    # ── Inference ──────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            return np.full(len(X), 0.5)
        rf_p = self._rf_pipe.predict_proba(X)[:, 1]
        gb_p = self._gb_pipe.predict_proba(X)[:, 1]
        return self.rf_weight * rf_p + self.gb_weight * gb_p


    def feature_importance(self) -> list[dict]:
        if not self._fi_rf:
            return []
        blended = {
            f: round(
                self.rf_weight * self._fi_rf.get(f, 0) +
                self.gb_weight * self._fi_gb.get(f, 0),
                6
            )
            for f in set(list(self._fi_rf) + list(self._fi_gb))
        }
        return [
            {
                "feature":    k.replace("_rank", "").replace("_", " ").title(),
                "importance": v,
                "rf":         round(self._fi_rf.get(k, 0), 6),
                "gb":         round(self._fi_gb.get(k, 0), 6),
            }
            for k, v in sorted(blended.items(), key=lambda x: x[1], reverse=True)
        ][:12]


    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, name: str = "default"):
        """Persist model to disk so it survives server restarts."""
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(self, path)
        meta = {
            "name":            name,
            "trained_at":      self.trained_at,
            "cv_accuracy":     self.cv_accuracy,
            "cv_ic":           self.cv_ic,
            "n_training_rows": self.n_training_rows,
            "training_source": self.training_source,
        }
        with open(MODEL_DIR / f"{name}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Model saved to {path}")


    @classmethod
    def load(cls, name: str = "default") -> "NUMKTEnsemble | None":
        """Load a previously saved model from disk."""
        path = MODEL_DIR / f"{name}.joblib"
        if not path.exists():
            return None
        try:
            model = joblib.load(path)
            logger.info(
                f"Loaded model '{name}' — "
                f"trained {model.trained_at} on "
                f"{model.n_training_rows} rows ({model.training_source})"
            )
            return model
        except Exception as e:
            logger.error(f"Could not load model: {e}")
            return None


    @staticmethod
    def list_saved() -> list[dict]:
        """List all saved models with their metadata."""
        models = []
        for meta_file in MODEL_DIR.glob("*_meta.json"):
            try:
                with open(meta_file) as f:
                    models.append(json.load(f))
            except Exception:
                pass
        return sorted(
            models,
            key=lambda x: x.get("trained_at", ""),
            reverse=True
        )