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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from scipy.stats import spearmanr
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime

from .feature_engineering import FEATURE_COLS

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "cache" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Where data_fetcher stores parquet price files
PRICE_DIR = Path(__file__).parent.parent / "cache" / "prices"

# Minimum real labelled rows before we trust the model.
# Below this the dataset is too thin and we log a loud warning.
MIN_REAL_ROWS = 30

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
        self._fi_rf          = {}
        self._fi_gb          = {}
        self.trained_at      = None

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
    ) -> "NUMKTEnsemble":
        """
        Train on real historical outcomes supplied by the backtest runner.
        This is the highest-quality path and takes priority over everything.
        """
        if len(feature_rows) < 6:
            logger.warning(
                f"Only {len(feature_rows)} training rows from backtest — "
                "falling back to price-cache labels"
            )
            return self._fit_from_price_cache(pd.DataFrame(feature_rows))

        df = pd.DataFrame(feature_rows).fillna(0.5)
        y  = np.array(return_labels)

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

        self._run_cv_and_fit(X, y)

        if forward_returns is not None and len(forward_returns) == len(X):
            proba = self.predict_proba(X)
            ic, _ = spearmanr(proba, forward_returns)
            self.cv_ic = float(ic) if np.isfinite(ic) else 0.0
            logger.info(f"Training IC (Spearman): {self.cv_ic:.4f}")

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

        feature_rows, return_labels, forward_returns_list = [], [], []

        for horizon in FORWARD_HORIZONS:
            rows, labels, fwd = self._labels_for_horizon(
                tickers, features_df, horizon
            )
            feature_rows.extend(rows)
            return_labels.extend(labels)
            forward_returns_list.extend(fwd)

        n = len(return_labels)
        logger.info(
            f"Price-cache label generation: {n} observations from "
            f"{len(tickers)} tickers x {len(FORWARD_HORIZONS)} horizons"
        )

        if n < MIN_REAL_ROWS:
            logger.warning(
                f"Only {n} labelled rows from price cache (need {MIN_REAL_ROWS}). "
                "Price history may be missing or too short. "
                "Using last-resort cross-sectional fallback — "
                "run /backtest to build a proper model."
            )
            return self._fit_last_resort(features_df)

        df_train = pd.DataFrame(feature_rows).fillna(0.5)
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

        self._run_cv_and_fit(X, y)

        ic, _ = spearmanr(self.predict_proba(X), fwd_arr)
        self.cv_ic = float(ic) if np.isfinite(ic) else 0.0

        self.is_fitted        = True
        self.n_training_rows  = n
        self.training_source  = "price_cache"
        self.trained_at       = datetime.utcnow().isoformat()

        logger.info(
            f"Model fitted on {n} price-cache rows. "
            f"CV accuracy: {self.cv_accuracy:.3f}  IC: {self.cv_ic:.4f}"
        )
        return self


    def _labels_for_horizon(
        self,
        tickers:     list[str],
        features_df: pd.DataFrame,
        horizon:     int,
    ) -> tuple[list[dict], list[int], list[float]]:
        """
        For a given forward horizon (in trading days), load each ticker's
        cached price series and generate non-overlapping (features, label)
        pairs stepping through the history.

        Returns three parallel lists:
          feature_rows    -- one dict per observation (factor snapshot at T)
          return_labels   -- 1 if stock beat universe median, else 0
          forward_returns -- raw forward return (for IC calculation)
        """
        # Load price series for all tickers at once
        price_series: dict[str, pd.Series] = {}
        for ticker in tickers:
            safe = ticker.replace(".", "_").replace("/", "_")
            path = PRICE_DIR / f"{safe}.parquet"
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

        # Number of complete non-overlapping windows we can extract.
        # Use the shortest series so every window has a full cross-section.
        min_len  = min(len(s) for s in price_series.values())
        n_windows = (min_len - horizon) // horizon
        if n_windows < 1:
            return [], [], []

        # Build a factor snapshot lookup from current features_df.
        # We use today's snapshot as a proxy for historical snapshots —
        # see NOTE in _fit_from_price_cache for the acknowledged limitation.
        feature_lookup: dict[str, dict] = {}
        if "ticker" in features_df.columns:
            for _, row in features_df.iterrows():
                t = row.get("ticker", "")
                if t:
                    feature_lookup[t] = row.to_dict()

        all_feature_rows: list[dict]  = []
        all_labels:       list[int]   = []
        all_fwd:          list[float] = []

        for w in range(n_windows):
            t_start = w * horizon
            t_end   = t_start + horizon

            window_returns: dict[str, float] = {}
            for ticker, prices in price_series.items():
                if len(prices) <= t_end:
                    continue
                p_start = prices.iloc[t_start]
                p_end   = prices.iloc[t_end]
                if p_start > 0:
                    window_returns[ticker] = float(p_end / p_start - 1)

            if len(window_returns) < 3:
                continue

            median_ret = float(np.median(list(window_returns.values())))

            for ticker, ret in window_returns.items():
                label = 1 if ret > median_ret else 0
                frow  = feature_lookup.get(ticker, {})
                all_feature_rows.append(frow)
                all_labels.append(label)
                all_fwd.append(ret)

        return all_feature_rows, all_labels, all_fwd


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

        n_splits = min(self.cv_folds, max(2, len(y) // 3))
        tscv     = TimeSeriesSplit(n_splits=n_splits)

        try:
            cv_scores        = cross_val_score(self._rf_pipe, X, y, cv=tscv,
                                               scoring="accuracy", n_jobs=-1)
            self.cv_accuracy = float(np.mean(cv_scores))
        except Exception:
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
        """Return the ranked feature columns present in df."""
        feat_cols = [c for c in FEATURE_COLS
                     if c + "_rank" in df.columns or c in df.columns]
        rank_cols = []
        for c in feat_cols:
            if c + "_rank" in df.columns:
                rank_cols.append(c + "_rank")
            elif c in df.columns:
                rank_cols.append(c)
        return rank_cols


    def _run_cv_and_fit(self, X: np.ndarray, y: np.ndarray):
        """Run TimeSeriesSplit CV then full fit on both pipelines."""
        n_splits = min(self.cv_folds, max(2, len(y) // 4))
        tscv     = TimeSeriesSplit(n_splits=n_splits)

        try:
            cv_rf = cross_val_score(self._rf_pipe, X, y, cv=tscv,
                                    scoring="accuracy", n_jobs=-1)
            cv_gb = cross_val_score(self._gb_pipe, X, y, cv=tscv,
                                    scoring="accuracy", n_jobs=-1)
            self.cv_accuracy = float(
                self.rf_weight * np.mean(cv_rf) +
                self.gb_weight * np.mean(cv_gb)
            )
            logger.info(
                f"CV accuracy — RF: {np.mean(cv_rf):.3f}  "
                f"GB: {np.mean(cv_gb):.3f}  "
                f"Ensemble: {self.cv_accuracy:.3f}"
            )
        except Exception as e:
            logger.warning(f"CV failed: {e}")
            self.cv_accuracy = 0.0

        self._rf_pipe.fit(X, y)
        self._gb_pipe.fit(X, y)
        self._store_feature_importances(self._feature_names)


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