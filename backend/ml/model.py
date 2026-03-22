"""
model.py  (v2 — trains on real returns)
────────────────────────────────────────
NUMKTEnsemble now trains on ACTUAL historical forward returns
instead of a synthetic quality score.

How it works:
  1. For each stock, take factor snapshot at time T
  2. Measure actual price return from T to T+forward_months
  3. Label = 1 if stock beat the equal-weight universe median, 0 if not
  4. Train RF+GB to learn which factor combinations predicted outperformance
  5. Apply learned model to current snapshot to score live stocks

This is genuine machine learning on real market outcomes.
The more historical periods you feed it, the better it gets.

The model is also saved to disk so it persists between sessions.
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


class NUMKTEnsemble:
    """
    Random Forest + Gradient Boosting ensemble.
    Trained on real historical forward returns.
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

        self.is_fitted      = False
        self.cv_accuracy    = 0.0
        self.cv_ic          = 0.0     # Information Coefficient on CV data
        self.n_training_rows = 0
        self.training_source = "none" # "real_returns" or "synthetic"
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


    def fit_on_real_returns(
        self,
        feature_rows: list[dict],   # list of {feature_col: value, ...}
        return_labels: list[int],   # 1 = beat benchmark, 0 = did not
        forward_returns: list[float] | None = None,  # actual returns for IC calc
    ) -> "NUMKTEnsemble":
        """
        Train on real historical outcomes.

        feature_rows:   one dict per (stock, time_period) observation
        return_labels:  1 if stock beat equal-weight median that period
        forward_returns: actual % returns (used only for IC calculation)
        """
        if len(feature_rows) < 6:
            logger.warning(f"Only {len(feature_rows)} training rows — need at least 6")
            return self._fit_synthetic(pd.DataFrame(feature_rows))

        df = pd.DataFrame(feature_rows).fillna(0.5)
        y  = np.array(return_labels)

        # Select feature columns that exist
        feat_cols = [c for c in FEATURE_COLS
                     if c + "_rank" in df.columns or c in df.columns]
        rank_cols = []
        for c in feat_cols:
            if c + "_rank" in df.columns:
                rank_cols.append(c + "_rank")
            elif c in df.columns:
                rank_cols.append(c)

        if not rank_cols:
            logger.warning("No feature columns found")
            return self

        self._feature_names = rank_cols
        X = df[rank_cols].values

        # Balance check
        pos = y.sum()
        neg = len(y) - pos
        if pos < 2 or neg < 2:
            logger.warning(f"Poor class balance ({pos} pos, {neg} neg) — using synthetic fallback")
            return self._fit_synthetic(df)

        # TimeSeriesSplit CV — respects time ordering
        n_splits = min(self.cv_folds, max(2, len(y) // 4))
        tscv     = TimeSeriesSplit(n_splits=n_splits)

        try:
            cv_rf = cross_val_score(self._rf_pipe, X, y, cv=tscv,
                                    scoring="accuracy", n_jobs=-1)
            cv_gb = cross_val_score(self._gb_pipe, X, y, cv=tscv,
                                    scoring="accuracy", n_jobs=-1)
            self.cv_accuracy = float(
                self.rf_weight * np.mean(cv_rf) + self.gb_weight * np.mean(cv_gb)
            )
            logger.info(f"CV accuracy — RF: {np.mean(cv_rf):.3f} "
                        f"GB: {np.mean(cv_gb):.3f} "
                        f"Ensemble: {self.cv_accuracy:.3f}")
        except Exception as e:
            logger.warning(f"CV failed: {e}")
            self.cv_accuracy = 0.0

        # Full fit
        self._rf_pipe.fit(X, y)
        self._gb_pipe.fit(X, y)

        # Feature importances
        rf_clf = self._rf_pipe.named_steps["clf"]
        gb_clf = self._gb_pipe.named_steps["clf"]
        self._fi_rf = dict(zip(rank_cols, rf_clf.feature_importances_))
        self._fi_gb = dict(zip(rank_cols, gb_clf.feature_importances_))

        # IC on training data (directional check)
        if forward_returns is not None and len(forward_returns) == len(X):
            proba = self.predict_proba(X)
            ic, _ = spearmanr(proba, forward_returns)
            self.cv_ic = float(ic) if np.isfinite(ic) else 0.0
            logger.info(f"Training IC (Spearman): {self.cv_ic:.4f}")

        self.is_fitted        = True
        self.n_training_rows  = len(y)
        self.training_source  = "real_returns"
        self.trained_at       = datetime.utcnow().isoformat()

        logger.info(f"Model fitted on {len(y)} real-return rows. "
                    f"CV: {self.cv_accuracy:.3f}")
        return self


    def _fit_synthetic(self, df: pd.DataFrame) -> "NUMKTEnsemble":
        """
        Fallback: train on synthetic quality score when real returns
        are unavailable (e.g. not enough history yet).
        """
        logger.info("Using synthetic quality label as training target")

        rank_cols = [c for c in df.columns if c.endswith("_rank")]
        if not rank_cols:
            self.is_fitted = False
            return self

        self._feature_names = rank_cols

        # Build quality score from factor ranks
        quality = np.zeros(len(df))
        weights = {
            "roe_rank":            0.25,
            "net_margin_rank":     0.20,
            "mom_12m_rank":        0.20,
            "pe_ratio_rank":      -0.15,
            "debt_equity_rank":   -0.10,
            "revenue_growth_rank": 0.10,
        }
        for col, wt in weights.items():
            if col in df.columns:
                vals = df[col].fillna(0.5).values
                quality += abs(wt) * (1 - vals if wt < 0 else vals)

        y = (quality >= np.median(quality)).astype(int)

        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            self.is_fitted = False
            return self

        X = df[rank_cols].fillna(0.5).values

        n_splits = min(self.cv_folds, max(2, len(y) // 3))
        tscv     = TimeSeriesSplit(n_splits=n_splits)

        try:
            cv_scores = cross_val_score(self._rf_pipe, X, y, cv=tscv,
                                        scoring="accuracy", n_jobs=-1)
            self.cv_accuracy = float(np.mean(cv_scores))
        except Exception:
            self.cv_accuracy = 0.0

        self._rf_pipe.fit(X, y)
        self._gb_pipe.fit(X, y)

        rf_clf = self._rf_pipe.named_steps["clf"]
        gb_clf = self._gb_pipe.named_steps["clf"]
        self._fi_rf = dict(zip(rank_cols, rf_clf.feature_importances_))
        self._fi_gb = dict(zip(rank_cols, gb_clf.feature_importances_))

        self.is_fitted       = True
        self.n_training_rows = len(y)
        self.training_source = "synthetic"
        self.trained_at      = datetime.utcnow().isoformat()
        return self


    def fit(self, features_df: pd.DataFrame) -> "NUMKTEnsemble":
        """Compatibility shim — called by /analyse when no real return data exists."""
        return self._fit_synthetic(features_df)


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
            f: round(self.rf_weight * self._fi_rf.get(f, 0) +
                     self.gb_weight * self._fi_gb.get(f, 0), 6)
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


    def save(self, name: str = "default"):
        """Persist model to disk so it survives server restarts."""
        path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(self, path)
        meta = {
            "name":           name,
            "trained_at":     self.trained_at,
            "cv_accuracy":    self.cv_accuracy,
            "cv_ic":          self.cv_ic,
            "n_training_rows":self.n_training_rows,
            "training_source":self.training_source,
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
            logger.info(f"Loaded model '{name}' — "
                        f"trained {model.trained_at} on "
                        f"{model.n_training_rows} rows ({model.training_source})")
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
        return sorted(models, key=lambda x: x.get("trained_at", ""), reverse=True)