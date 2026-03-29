"""
AutoML Engine Module for AI Data Science Copilot.

Provides automated machine learning with:
- Auto problem-type detection
- Robust preprocessing pipeline
- Multi-model training & comparison
- Overfitting detection
- Bias-variance decomposition
- Model reasoning layer (why it won, when it fails, confidence score)
- Tradeoff analysis across models
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Optional XGBoost ───────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available — XGBoost models will be skipped.")

# ── Constants ──────────────────────────────────────────────────────────────────
_CLASSIFICATION_THRESHOLD = 15   # nunique ≤ this → classification candidate
_OVERFIT_GAP_WARN         = 0.10 # train-test gap above which we flag overfitting
_OVERFIT_GAP_SEVERE        = 0.20 # gap above which we flag severe overfitting
_HIGH_CARDINALITY          = 20   # nunique above which we use ordinal instead of OHE
_SKEW_THRESHOLD            = 1.0  # |skewness| above which we apply log1p to numeric
_NEAR_ZERO_VAR_THRESHOLD   = 0.01 # variance below which a column is considered constant


# ══════════════════════════════════════════════════════════════════════════════
# AutoML Engine
# ══════════════════════════════════════════════════════════════════════════════

class AutoMLEngine:
    """
    Automated Machine Learning Engine.

    Trains multiple models, selects the best performer, and provides
    a rich reasoning layer explaining *why* the winning model won,
    *when* it is likely to fail, an overfitting check, and a
    bias-variance decomposition.
    """

    # ── Base model definitions ─────────────────────────────────────────────────
    CLASSIFICATION_MODELS: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "gradient_boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    REGRESSION_MODELS: Dict[str, Any] = {
        "linear_regression": LinearRegression(),
        "ridge":             Ridge(random_state=42),
        "lasso":             Lasso(random_state=42),
        "random_forest":     RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    # ── Static model knowledge base for the reasoning layer ───────────────────
    _MODEL_PROFILES: Dict[str, Dict[str, Any]] = {
        "logistic_regression": {
            "strengths":     ["Fast to train", "Highly interpretable", "Works well when classes are linearly separable", "Low variance — rarely overfits"],
            "weaknesses":    ["Cannot capture non-linear relationships", "Sensitive to feature scale", "Struggles with high-cardinality categoricals"],
            "fails_when":    ["Features interact non-linearly", "Classes are imbalanced without reweighting", "Many irrelevant features present"],
            "bias_level":    "high",
            "variance_level": "low",
        },
        "ridge": {
            "strengths":     ["Handles multicollinearity well", "Low variance via L2 regularisation", "Fast and interpretable"],
            "weaknesses":    ["Cannot perform feature selection", "Assumes linear relationship"],
            "fails_when":    ["True relationship is non-linear", "Many irrelevant features with high coefficients"],
            "bias_level":    "medium",
            "variance_level": "low",
        },
        "lasso": {
            "strengths":     ["Built-in feature selection via L1 sparsity", "Good when few features are truly predictive"],
            "weaknesses":    ["Can under-select correlated features", "Needs careful alpha tuning"],
            "fails_when":    ["All features are informative", "Features are highly correlated"],
            "bias_level":    "medium",
            "variance_level": "low",
        },
        "linear_regression": {
            "strengths":     ["Maximally interpretable", "No hyperparameters", "Efficient on large data"],
            "weaknesses":    ["Assumes linearity", "No regularisation — can overfit with many features"],
            "fails_when":    ["True relationship is non-linear", "Outliers present in target"],
            "bias_level":    "high",
            "variance_level": "medium",
        },
        "random_forest": {
            "strengths":     ["Handles non-linearity and feature interactions", "Robust to outliers", "Built-in feature importance", "Low variance via bagging"],
            "weaknesses":    ["Slow to predict on very large datasets", "Hard to interpret individual trees", "Can memorise training data"],
            "fails_when":    ["Very high-dimensional sparse data", "Extrapolation beyond training range needed", "Tiny datasets (n < 100)"],
            "bias_level":    "low",
            "variance_level": "medium",
        },
        "gradient_boosting": {
            "strengths":     ["Often best accuracy on tabular data", "Handles missing values natively (XGBoost)", "Captures complex interactions"],
            "weaknesses":    ["Slow to train", "Many hyperparameters to tune", "Prone to overfitting on noisy data"],
            "fails_when":    ["Very small datasets", "High noise-to-signal ratio", "Real-time training required"],
            "bias_level":    "low",
            "variance_level": "high",
        },
        "xgboost": {
            "strengths":     ["State-of-the-art on most tabular benchmarks", "Handles missing values natively", "Regularised boosting reduces overfitting vs plain GBM"],
            "weaknesses":    ["Many hyperparameters", "Black-box predictions"],
            "fails_when":    ["Very small datasets", "High label noise", "Extrapolation beyond training distribution"],
            "bias_level":    "low",
            "variance_level": "medium",
        },
    }

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        cv_folds: int = 5,
    ):
        """
        Initialise the AutoML Engine.

        Args:
            random_state: Random seed for reproducibility.
            test_size:    Fraction of data held out for testing.
            cv_folds:     Number of cross-validation folds.
        """
        self.random_state = random_state
        self.test_size    = test_size
        self.cv_folds     = cv_folds

        self.problem_type:    Optional[str]        = None
        self.target_col:      Optional[str]        = None
        self.models:          Dict[str, Any]       = {}
        self.results:         Dict[str, Any]       = {}
        self.best_model:      Optional[Any]        = None
        self.best_model_name: Optional[str]        = None
        self.preprocessor:    Optional[ColumnTransformer] = None
        self.label_encoder:   Optional[LabelEncoder]      = None
        self.feature_names:   Optional[List[str]]         = None
        self.is_fitted:       bool                        = False

        # ── Add XGBoost if installed ───────────────────────────────────────────
        if XGBOOST_AVAILABLE:
            self.CLASSIFICATION_MODELS["xgboost"] = XGBClassifier(
                n_estimators=100, random_state=random_state,
                use_label_encoder=False, eval_metric="logloss",
            )
            self.REGRESSION_MODELS["xgboost"] = XGBRegressor(
                n_estimators=100, random_state=random_state,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Problem-type detection
    # ══════════════════════════════════════════════════════════════════════════

    def detect_problem_type(self, df: pd.DataFrame, target_col: str) -> str:
        """
        Infer whether the task is binary classification, multiclass classification,
        or regression from the target column's characteristics.

        Args:
            df:         Input DataFrame.
            target_col: Name of the target column.

        Returns:
            One of ``"binary_classification"``, ``"multiclass_classification"``,
            or ``"regression"``.

        Raises:
            ValueError: if *target_col* is not present in *df*.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

        series   = df[target_col].dropna()
        n_unique = series.nunique()

        if series.dtype in ("object", "category", "bool"):
            is_classification = True
        elif n_unique <= _CLASSIFICATION_THRESHOLD:
            # Integer-valued with few levels → classification
            is_classification = series.apply(lambda x: float(x).is_integer()).all()
        else:
            is_classification = False

        if is_classification:
            self.problem_type = (
                "binary_classification" if n_unique == 2 else "multiclass_classification"
            )
        else:
            self.problem_type = "regression"

        self.target_col = target_col
        logger.info("Detected problem type: %s", self.problem_type)
        return self.problem_type

    # ══════════════════════════════════════════════════════════════════════════
    # Preprocessing
    # ══════════════════════════════════════════════════════════════════════════

    def create_preprocessor(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = "median",
        categorical_strategy: str = "auto",
        scaler: str = "robust",
    ) -> ColumnTransformer:
        """
        Build a preprocessing pipeline adapted to the dataset's characteristics.

        Improvements over the original:
        - **Near-zero variance** columns are dropped before the pipeline runs.
        - **Scaler selection**: ``"robust"`` (default) is less sensitive to
          outliers than ``StandardScaler``; ``"standard"`` and ``"minmax"``
          are also accepted.
        - **High-cardinality** categoricals (nunique > ``_HIGH_CARDINALITY``)
          use ``OrdinalEncoder`` instead of one-hot encoding to avoid a
          combinatorial feature explosion.
        - **Skewed numerics** get a ``log1p`` pre-transform before scaling when
          ``|skewness| > _SKEW_THRESHOLD`` and all values are non-negative.
        - Datetime columns are auto-extracted into year/month/day/weekday
          integer features rather than being silently dropped.

        Args:
            df:                   Feature DataFrame (target column already removed).
            numeric_strategy:     Imputation strategy for numeric columns
                                  (``"median"`` | ``"mean"`` | ``"constant"``).
            categorical_strategy: ``"auto"`` (smart selection) | ``"onehot"``
                                  (always OHE) | ``"ordinal"`` (always ordinal).
            scaler:               ``"robust"`` | ``"standard"`` | ``"minmax"``.

        Returns:
            Fitted :class:`~sklearn.compose.ColumnTransformer`.
        """
        df = df.copy()

        # ── 1. Expand datetime columns ─────────────────────────────────────────
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        for col in dt_cols:
            df[f"{col}_year"]    = df[col].dt.year
            df[f"{col}_month"]   = df[col].dt.month
            df[f"{col}_day"]     = df[col].dt.day
            df[f"{col}_weekday"] = df[col].dt.weekday
            df = df.drop(columns=[col])

        # ── 2. Drop near-zero-variance columns ────────────────────────────────
        num_df   = df.select_dtypes(include=[np.number])
        low_var  = num_df.columns[num_df.var() < _NEAR_ZERO_VAR_THRESHOLD].tolist()
        if low_var:
            logger.info("Dropping near-zero-variance columns: %s", low_var)
            df = df.drop(columns=low_var)

        # ── 3. Identify column types ───────────────────────────────────────────
        numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        logger.info("Numeric features: %d  |  Categorical features: %d",
                    len(numeric_cols), len(categorical_cols))

        # ── 4. Detect skewed numerics ──────────────────────────────────────────
        skewed_cols: List[str] = []
        for col in numeric_cols:
            col_data = df[col].dropna()
            if col_data.empty:
                continue
            if abs(col_data.skew()) > _SKEW_THRESHOLD and (col_data >= 0).all():
                skewed_cols.append(col)

        non_skewed_cols = [c for c in numeric_cols if c not in skewed_cols]

        # ── 5. Scaler selection ────────────────────────────────────────────────
        scaler_map = {
            "robust":   RobustScaler(),
            "standard": StandardScaler(),
            "minmax":   MinMaxScaler(),
        }
        chosen_scaler = scaler_map.get(scaler, RobustScaler())

        # ── 6. Build numeric sub-pipelines ────────────────────────────────────
        transformers = []

        if non_skewed_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy=numeric_strategy)),
                ("scaler",  chosen_scaler.__class__()),   # fresh instance
            ])
            transformers.append(("num", num_pipe, non_skewed_cols))

        if skewed_cols:
            log_pipe = Pipeline([
                ("imputer",  SimpleImputer(strategy=numeric_strategy)),
                ("log1p",    FunctionTransformer(np.log1p, validate=False)),
                ("scaler",   chosen_scaler.__class__()),
            ])
            transformers.append(("num_skewed", log_pipe, skewed_cols))

        # ── 7. Build categorical sub-pipeline ─────────────────────────────────
        if categorical_cols:
            # Auto-select encoder based on cardinality
            if categorical_strategy == "auto":
                max_card = max(
                    (df[c].nunique() for c in categorical_cols), default=0
                )
                use_ordinal = max_card > _HIGH_CARDINALITY
            else:
                use_ordinal = categorical_strategy == "ordinal"

            encoder = (
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                if use_ordinal
                else OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            )
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("astype_str", FunctionTransformer(lambda x: x.astype(str), validate=False)), # Fix for mixed types
                ("encoder", encoder),
            ])
            transformers.append(("cat", cat_pipe, categorical_cols))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        self.preprocessor  = preprocessor
        self.feature_names = numeric_cols + categorical_cols
        self._preprocessed_df = df   # store for column access in feature importance

        return preprocessor

    # ══════════════════════════════════════════════════════════════════════════
    # Data preparation
    # ══════════════════════════════════════════════════════════════════════════

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Encode the target, build the preprocessor, and split the data.

        Args:
            df:         Full input DataFrame.
            target_col: Name of the target column.

        Returns:
            ``(X_train, X_test, y_train, y_test)`` as numpy arrays.
        """
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()

        # Target encoding
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if y.dtype == "object" or y.dtype.name == "category":
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y.astype(str))

        self.create_preprocessor(X)

        stratify = y if self.problem_type != "regression" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc  = self.preprocessor.transform(X_test)

        logger.info("Train: %s  |  Test: %s", X_train_proc.shape, X_test_proc.shape)
        return X_train_proc, X_test_proc, np.array(y_train), np.array(y_test)

    # ══════════════════════════════════════════════════════════════════════════
    # Model training
    # ══════════════════════════════════════════════════════════════════════════

    def train_models(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray,
        models_to_train: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train every candidate model and record metrics, CV scores,
        overfitting diagnostics, and bias-variance estimates.

        Args:
            X_train:         Preprocessed training features.
            X_test:          Preprocessed test features.
            y_train:         Training target.
            y_test:          Test target.
            models_to_train: Subset of model names to train; ``None`` trains all.

        Returns:
            Per-model result dicts keyed by model name.
        """
        pool = (
            self.CLASSIFICATION_MODELS
            if self.problem_type != "regression"
            else self.REGRESSION_MODELS
        )

        if models_to_train:
            pool = {k: v for k, v in pool.items() if k in models_to_train}

        results: Dict[str, Dict[str, Any]] = {}

        for name, model in pool.items():
            logger.info("Training %s …", name)
            try:
                model.fit(X_train, y_train)
                y_pred       = model.predict(X_test)
                y_pred_proba = None

                if self.problem_type != "regression" and hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)

                # Train-set predictions for overfitting check
                y_train_pred = model.predict(X_train)

                metrics_test  = self._calculate_metrics(y_test,  y_pred, y_pred_proba)
                metrics_train = self._calculate_metrics(y_train, y_train_pred)

                cv_scores = self._cross_validate(model, X_train, y_train)
                overfit   = self._check_overfitting(metrics_train, metrics_test, cv_scores)
                bv        = self._bias_variance_analysis(name, cv_scores, metrics_train, metrics_test)

                results[name] = {
                    "model":         model,
                    "metrics":       metrics_test,
                    "train_metrics": metrics_train,
                    "cv_scores":     cv_scores,
                    "cv_mean":       float(np.mean(cv_scores)),
                    "cv_std":        float(np.std(cv_scores)),
                    "predictions":   y_pred,
                    "probabilities": y_pred_proba,
                    "overfitting":   overfit,
                    "bias_variance": bv,
                }

                logger.info(
                    "%-22s  CV %.4f ± %.4f  |  overfit=%s",
                    name, np.mean(cv_scores), np.std(cv_scores), overfit["status"],
                )

            except Exception as exc:
                logger.error("Error training %s: %s", name, exc)
                results[name] = {"error": str(exc)}

        self.results      = results
        self.models       = {n: r["model"] for n, r in results.items() if "error" not in r}
        self._X_train_ref = X_train
        self._y_train_ref = y_train

        self._select_best_model()
        self.is_fitted = True
        return results

    # ══════════════════════════════════════════════════════════════════════════
    # Metrics
    # ══════════════════════════════════════════════════════════════════════════

    def _calculate_metrics(
        self,
        y_true:       np.ndarray,
        y_pred:       np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute task-appropriate evaluation metrics."""
        metrics: Dict[str, float] = {}

        if self.problem_type != "regression":
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            avg = "binary" if self.problem_type == "binary_classification" else "weighted"
            metrics["precision"] = float(precision_score(y_true, y_pred, average=avg, zero_division=0))
            metrics["recall"]    = float(recall_score(y_true,    y_pred, average=avg, zero_division=0))
            metrics["f1"]        = float(f1_score(y_true,        y_pred, average=avg, zero_division=0))

            if self.problem_type == "binary_classification" and y_pred_proba is not None:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                except Exception:
                    pass
        else:
            mse = float(mean_squared_error(y_true, y_pred))
            metrics["mse"]  = mse
            metrics["rmse"] = float(np.sqrt(mse))
            metrics["mae"]  = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"]   = float(r2_score(y_true, y_pred))
            # MAPE — guard against zero targets
            denom = np.where(np.abs(y_true) > 1e-8, np.abs(y_true), 1e-8)
            metrics["mape"] = float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

        return metrics

    # ══════════════════════════════════════════════════════════════════════════
    # Cross-validation
    # ══════════════════════════════════════════════════════════════════════════

    def _cross_validate(
        self, model, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Run stratified (classification) or plain (regression) k-fold CV."""
        if self.problem_type != "regression":
            cv      = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scoring = "accuracy"
        else:
            cv      = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scoring = "r2"

        return cross_val_score(model, X, y, cv=cv, scoring=scoring)

    # ══════════════════════════════════════════════════════════════════════════
    # Overfitting check  ← NEW
    # ══════════════════════════════════════════════════════════════════════════

    def _check_overfitting(
        self,
        train_metrics: Dict[str, float],
        test_metrics:  Dict[str, float],
        cv_scores:     np.ndarray,
    ) -> Dict[str, Any]:
        """
        Detect overfitting by comparing train vs test performance and
        measuring cross-validation variance.

        Returns a dict with:
        - ``status``: ``"none"`` | ``"mild"`` | ``"severe"``
        - ``train_score``:  primary metric on training set
        - ``test_score``:   primary metric on test set
        - ``gap``:          train_score − test_score
        - ``cv_std``:       standard deviation of CV scores
        - ``cv_stability``: ``"stable"`` | ``"unstable"``
        - ``details``:      human-readable explanation
        """
        primary = "accuracy" if self.problem_type != "regression" else "r2"
        train_s = train_metrics.get(primary, 0.0)
        test_s  = test_metrics.get(primary,  0.0)
        gap     = train_s - test_s
        cv_std  = float(np.std(cv_scores))

        if gap >= _OVERFIT_GAP_SEVERE:
            status = "severe"
        elif gap >= _OVERFIT_GAP_WARN:
            status = "mild"
        else:
            status = "none"

        cv_stability = "unstable" if cv_std > 0.05 else "stable"

        details = []
        if status == "severe":
            details.append(
                f"Train {primary} ({train_s:.3f}) is {gap:.3f} higher than test ({test_s:.3f}) — "
                "the model has memorised training data. "
                "Consider stronger regularisation, pruning depth, or more data."
            )
        elif status == "mild":
            details.append(
                f"Mild gap of {gap:.3f} between train ({train_s:.3f}) and test ({test_s:.3f}). "
                "Light regularisation tuning may help."
            )
        else:
            details.append(
                f"No overfitting detected — train/test gap is {gap:.3f}, within acceptable range."
            )

        if cv_stability == "unstable":
            details.append(
                f"CV std {cv_std:.3f} > 0.05: performance varies across folds. "
                "Consider collecting more data or simplifying the model."
            )

        return {
            "status":       status,
            "train_score":  round(train_s, 4),
            "test_score":   round(test_s,  4),
            "gap":          round(gap,     4),
            "cv_std":       round(cv_std,  4),
            "cv_stability": cv_stability,
            "details":      " ".join(details),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Bias-variance analysis  ← NEW
    # ══════════════════════════════════════════════════════════════════════════

    def _bias_variance_analysis(
        self,
        model_name:    str,
        cv_scores:     np.ndarray,
        train_metrics: Dict[str, float],
        test_metrics:  Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Reason about a model's position on the bias-variance spectrum.

        Uses the model's known profile (from ``_MODEL_PROFILES``) combined
        with observed train/test gap and CV variance to produce a
        data-driven diagnosis.

        Returns a dict with:
        - ``bias_level``:    ``"low"`` | ``"medium"`` | ``"high"``
        - ``variance_level``: ``"low"`` | ``"medium"`` | ``"high"``
        - ``dominant_error``: ``"bias"`` | ``"variance"`` | ``"balanced"``
        - ``interpretation``: plain-English paragraph
        - ``recommendation``: what to do next
        """
        profile      = self._MODEL_PROFILES.get(model_name, {})
        prior_bias   = profile.get("bias_level",     "medium")
        prior_var    = profile.get("variance_level", "medium")

        primary    = "accuracy" if self.problem_type != "regression" else "r2"
        train_s    = train_metrics.get(primary, 0.0)
        test_s     = test_metrics.get(primary,  0.0)
        gap        = train_s - test_s
        cv_std     = float(np.std(cv_scores))
        cv_mean    = float(np.mean(cv_scores))

        # ── Override priors with observed evidence ─────────────────────────────
        # High gap → high variance (model fits noise)
        if gap >= _OVERFIT_GAP_SEVERE:
            obs_variance = "high"
        elif gap >= _OVERFIT_GAP_WARN:
            obs_variance = "medium"
        else:
            obs_variance = prior_var

        # Low test score despite low gap → high bias (model underfits)
        if test_s < 0.6 and gap < _OVERFIT_GAP_WARN:
            obs_bias = "high"
        elif test_s < 0.75 and gap < _OVERFIT_GAP_WARN:
            obs_bias = "medium"
        else:
            obs_bias = prior_bias

        # Dominant error
        if obs_bias == "high" and obs_variance in ("low", "medium"):
            dominant = "bias"
        elif obs_variance == "high" and obs_bias in ("low", "medium"):
            dominant = "variance"
        else:
            dominant = "balanced"

        # Human-readable interpretation
        interp_map = {
            "bias": (
                f"{model_name} underfits the data (high bias). "
                f"CV mean {cv_mean:.3f} and test score {test_s:.3f} are both low, "
                "indicating the model's hypothesis space is too restricted to capture the true signal."
            ),
            "variance": (
                f"{model_name} overfits the data (high variance). "
                f"Training score {train_s:.3f} vs test score {test_s:.3f} (gap {gap:.3f}) shows "
                "the model captures noise rather than the underlying pattern."
            ),
            "balanced": (
                f"{model_name} sits in a balanced region. "
                f"Train {train_s:.3f} ≈ test {test_s:.3f} (gap {gap:.3f}), "
                f"CV std {cv_std:.3f}. Neither bias nor variance dominates."
            ),
        }

        rec_map = {
            "bias": "Use a more complex model, add polynomial features, or reduce regularisation.",
            "variance": "Increase regularisation, prune tree depth, add more training data, or use bagging.",
            "balanced": "This model is well-calibrated. Focus on feature engineering or hyperparameter tuning for marginal gains.",
        }

        return {
            "bias_level":    obs_bias,
            "variance_level": obs_variance,
            "dominant_error": dominant,
            "interpretation": interp_map[dominant],
            "recommendation": rec_map[dominant],
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Best-model selection
    # ══════════════════════════════════════════════════════════════════════════

    def _select_best_model(self) -> None:
        """
        Choose the best model by CV mean score, with a tiebreak that
        penalises overfitting: if two models are within 0.5% CV score,
        prefer the one with the smaller train-test gap.
        """
        valid = {k: v for k, v in self.results.items() if "error" not in v}
        if not valid:
            logger.warning("No valid models to select from.")
            return

        def _score(name: str) -> float:
            r   = valid[name]
            cv  = r["cv_mean"]
            gap = r["overfitting"]["gap"]
            # Small penalty for overfitting so tight models tie-break correctly
            return cv - 0.5 * max(0.0, gap - _OVERFIT_GAP_WARN)

        best_name = max(valid, key=_score)
        self.best_model_name = best_name
        self.best_model      = valid[best_name]["model"]

        logger.info(
            "Best model: %s  (CV %.4f, overfit=%s)",
            best_name,
            valid[best_name]["cv_mean"],
            valid[best_name]["overfitting"]["status"],
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Reasoning layer  ← NEW
    # ══════════════════════════════════════════════════════════════════════════

    def generate_model_reasoning(self) -> Dict[str, Any]:
        """
        Produce a structured reasoning report for the winning model explaining:
        - **Why it won**: data-driven evidence from metrics and CV.
        - **When it fails**: known failure modes from the model profile.
        - **Confidence score**: 0–100 score reflecting reliability of the result.
        - **Tradeoff analysis**: comparison against every other model on key axes.

        Returns:
            Dict with keys ``why_it_won``, ``when_it_fails``, ``confidence``,
            ``confidence_explanation``, ``tradeoff_analysis``.

        Raises:
            ValueError: if ``train_models`` has not been called yet.
        """
        if not self.is_fitted or self.best_model_name is None:
            raise ValueError("Call train_models() before generate_model_reasoning().")

        best_r   = self.results[self.best_model_name]
        profile  = self._MODEL_PROFILES.get(self.best_model_name, {})
        valid    = {k: v for k, v in self.results.items() if "error" not in v}
        primary  = "accuracy" if self.problem_type != "regression" else "r2"

        # ── 1. Why it won ──────────────────────────────────────────────────────
        best_cv    = best_r["cv_mean"]
        runner_ups = sorted(
            [(n, r["cv_mean"]) for n, r in valid.items() if n != self.best_model_name],
            key=lambda x: x[1], reverse=True,
        )

        why_lines = [
            f"**{self.best_model_name}** achieved the highest penalised CV score of **{best_cv:.4f}**."
        ]
        if runner_ups:
            second_name, second_cv = runner_ups[0]
            margin = best_cv - second_cv
            why_lines.append(
                f"It beat the second-best model ({second_name}, CV {second_cv:.4f}) "
                f"by **{margin:.4f}** — {'a meaningful margin' if margin > 0.02 else 'a slim margin; the difference may not be significant on a different split'}."
            )

        overfit_status = best_r["overfitting"]["status"]
        why_lines.append(
            f"Overfitting status: **{overfit_status}** "
            f"(train-test gap = {best_r['overfitting']['gap']:.4f})."
        )
        why_lines.extend(
            [f"Strength: {s}" for s in profile.get("strengths", [])[:3]]
        )

        # ── 2. When it fails ───────────────────────────────────────────────────
        failure_modes = profile.get("fails_when", [])
        bv            = best_r["bias_variance"]
        failure_lines = list(failure_modes)
        if bv["dominant_error"] == "variance":
            failure_lines.insert(0, "High variance detected — performance may degrade on out-of-distribution data.")
        if best_r["overfitting"]["cv_stability"] == "unstable":
            failure_lines.insert(0, "CV scores are unstable — results may not generalise to all data splits.")

        # ── 3. Confidence score ────────────────────────────────────────────────
        confidence, conf_notes = self._compute_confidence_score(best_r, valid, primary)

        # ── 4. Tradeoff analysis ───────────────────────────────────────────────
        tradeoff = self.tradeoff_analysis()

        return {
            "winner":               self.best_model_name,
            "why_it_won":           " ".join(why_lines),
            "when_it_fails":        failure_lines,
            "strengths":            profile.get("strengths", []),
            "weaknesses":           profile.get("weaknesses", []),
            "confidence":           confidence,
            "confidence_explanation": conf_notes,
            "bias_variance":        bv,
            "overfitting":          best_r["overfitting"],
            "tradeoff_analysis":    tradeoff,
        }

    def _compute_confidence_score(
        self,
        best_result: Dict[str, Any],
        valid_results: Dict[str, Dict[str, Any]],
        primary_metric: str,
    ) -> Tuple[int, str]:
        """
        Compute a 0–100 confidence score for the AutoML run.

        Deductions:
        - Severe overfitting:   −25
        - Mild overfitting:     −10
        - Unstable CV:          −15
        - Slim margin over second best (< 0.01): −10
        - High CV std (> 0.05): −10

        Returns:
            ``(score, explanation_string)``
        """
        score = 100
        notes: List[str] = []

        # Overfitting penalty
        of_status = best_result["overfitting"]["status"]
        if of_status == "severe":
            score -= 25
            notes.append("−25: severe overfitting detected.")
        elif of_status == "mild":
            score -= 10
            notes.append("−10: mild overfitting detected.")

        # CV stability penalty
        if best_result["overfitting"]["cv_stability"] == "unstable":
            score -= 15
            notes.append("−15: CV scores are unstable (std > 0.05).")

        # High CV std penalty
        if best_result["cv_std"] > 0.05:
            score -= 10
            notes.append(f"−10: CV std {best_result['cv_std']:.3f} indicates noisy evaluation.")

        # Slim margin penalty
        others = [r["cv_mean"] for n, r in valid_results.items() if n != self.best_model_name]
        if others and (best_result["cv_mean"] - max(others)) < 0.01:
            score -= 10
            notes.append("−10: margin over second-best model is < 0.01 — selection may be unstable.")

        # Absolute performance penalty (very low scores reduce confidence)
        cv = best_result["cv_mean"]
        if cv < 0.5:
            score -= 20
            notes.append(f"−20: absolute CV score {cv:.3f} is low — the model may be worse than a baseline.")
        elif cv < 0.65:
            score -= 10
            notes.append(f"−10: absolute CV score {cv:.3f} is below typical useful threshold.")

        score = max(0, min(100, score))
        if not notes:
            notes.append("No significant reliability concerns detected.")

        return score, "  ".join(notes)

    # ══════════════════════════════════════════════════════════════════════════
    # Tradeoff analysis  ← NEW
    # ══════════════════════════════════════════════════════════════════════════

    def tradeoff_analysis(self) -> List[Dict[str, Any]]:
        """
        Compare every trained model across four axes:
        accuracy, generalisation (CV mean), overfitting risk, and
        interpretability — and explain the tradeoff of choosing each.

        Returns:
            List of dicts, one per model, sorted by CV mean descending.
            Each dict contains ``model``, ``cv_mean``, ``test_score``,
            ``overfit_status``, ``bias_variance``, ``interpretability``,
            ``is_best``, ``tradeoff_summary``.
        """
        _interpretability_rank = {
            "logistic_regression": "high",
            "linear_regression":   "high",
            "ridge":               "high",
            "lasso":               "high",
            "random_forest":       "medium",
            "gradient_boosting":   "low",
            "xgboost":             "low",
        }

        valid  = {k: v for k, v in self.results.items() if "error" not in v}
        rows   = []
        primary = "accuracy" if self.problem_type != "regression" else "r2"

        for name, r in valid.items():
            cv       = r["cv_mean"]
            test_s   = r["metrics"].get(primary, float("nan"))
            of       = r["overfitting"]
            bv       = r["bias_variance"]
            interp   = _interpretability_rank.get(name, "medium")
            is_best  = name == self.best_model_name

            # One-sentence tradeoff summary
            parts = []
            if is_best:
                parts.append("**Best overall pick** by penalised CV score.")
            if of["status"] == "severe":
                parts.append("⚠️ Severe overfitting — needs regularisation before deployment.")
            elif of["status"] == "mild":
                parts.append("Mild overfitting — consider tuning.")
            if bv["dominant_error"] == "bias":
                parts.append("Underfitting — may benefit from a more complex model.")
            if interp == "high":
                parts.append("Highly interpretable — good for regulated domains.")
            elif interp == "low":
                parts.append("Black-box — requires SHAP/LIME for explainability.")

            rows.append({
                "model":             name,
                "cv_mean":           round(cv, 4),
                "test_score":        round(test_s, 4),
                "overfit_status":    of["status"],
                "overfit_gap":       of["gap"],
                "bias_level":        bv["bias_level"],
                "variance_level":    bv["variance_level"],
                "dominant_error":    bv["dominant_error"],
                "interpretability":  interp,
                "is_best":           is_best,
                "tradeoff_summary":  " ".join(parts) if parts else "No notable tradeoffs.",
            })

        return sorted(rows, key=lambda x: x["cv_mean"], reverse=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Learning curve  ← NEW helper for bias-variance visualisation
    # ══════════════════════════════════════════════════════════════════════════

    def compute_learning_curve(
        self, model_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute learning-curve data for a named model (defaults to best model).

        Learning curves reveal whether adding more data would reduce the
        dominant error: a high-bias model shows both curves plateauing low;
        a high-variance model shows a large gap that closes as n grows.

        Returns:
            Dict with ``train_sizes``, ``train_scores_mean``,
            ``train_scores_std``, ``test_scores_mean``, ``test_scores_std``.

        Raises:
            ValueError: if the engine has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Call train_models() first.")

        name  = model_name or self.best_model_name
        model = self.models.get(name)
        if model is None:
            raise ValueError(f"Model '{name}' not found.")

        scoring = "accuracy" if self.problem_type != "regression" else "r2"
        cv      = (
            StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            if self.problem_type != "regression"
            else KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        )

        train_sizes, train_scores, test_scores = learning_curve(
            model,
            self._X_train_ref,
            self._y_train_ref,
            cv=cv,
            scoring=scoring,
            train_sizes=np.linspace(0.1, 1.0, 8),
            n_jobs=-1,
        )

        return {
            "train_sizes":       train_sizes,
            "train_scores_mean": train_scores.mean(axis=1),
            "train_scores_std":  train_scores.std(axis=1),
            "test_scores_mean":  test_scores.mean(axis=1),
            "test_scores_std":   test_scores.std(axis=1),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # Feature importance
    # ══════════════════════════════════════════════════════════════════════════

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Extract feature importances from the best model.

        Handles tree-based ``feature_importances_`` and linear ``coef_``
        models. Returns ``None`` when the model type exposes neither.

        Returns:
            DataFrame with columns ``feature`` and ``importance``, sorted
            descending by importance, or ``None``.
        """
        if self.best_model is None:
            return None

        # Recover feature names from preprocessor
        try:
            feature_names: List[str] = []
            for _, pipe, cols in self.preprocessor.transformers_:
                if not cols:
                    continue
                last_step = pipe.steps[-1][1]
                if hasattr(last_step, "get_feature_names_out"):
                    feature_names.extend(last_step.get_feature_names_out(cols).tolist())
                else:
                    feature_names.extend(cols)
        except Exception:
            n = getattr(self.best_model, "n_features_in_", 0)
            feature_names = [f"feature_{i}" for i in range(n)]

        # Extract importances
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importances = np.abs(self.best_model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)
        else:
            logger.warning("Best model does not expose feature importances.")
            return None

        n = min(len(feature_names), len(importances))
        return (
            pd.DataFrame({"feature": feature_names[:n], "importance": importances[:n]})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Prediction
    # ══════════════════════════════════════════════════════════════════════════

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Run the full pipeline (preprocess → predict → decode) on new data.

        Args:
            X: Raw feature DataFrame (same columns as training, without target).

        Returns:
            Array of predictions; class labels are decoded for classification.

        Raises:
            ValueError: if the engine has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call auto_train() first.")

        X_proc = self.preprocessor.transform(X)
        preds  = self.best_model.predict(X_proc)

        if self.label_encoder is not None:
            preds = self.label_encoder.inverse_transform(preds)

        return preds

    # ══════════════════════════════════════════════════════════════════════════
    # Model comparison table
    # ══════════════════════════════════════════════════════════════════════════

    def get_model_comparison(self) -> pd.DataFrame:
        """
        Build a tidy DataFrame comparing all models on CV score, test metrics,
        and overfitting status.

        Returns:
            DataFrame sorted by ``cv_mean`` descending.
        """
        rows = []
        for name, r in self.results.items():
            if "error" in r:
                continue
            row = {
                "model":          name,
                "cv_mean":        r["cv_mean"],
                "cv_std":         r["cv_std"],
                "overfit_status": r["overfitting"]["status"],
                "overfit_gap":    r["overfitting"]["gap"],
                "bias":           r["bias_variance"]["bias_level"],
                "variance":       r["bias_variance"]["variance_level"],
                **r["metrics"],
            }
            rows.append(row)

        return (
            pd.DataFrame(rows)
            .sort_values("cv_mean", ascending=False)
            .reset_index(drop=True)
        )

    # ══════════════════════════════════════════════════════════════════════════
    # End-to-end entry point
    # ══════════════════════════════════════════════════════════════════════════

    def auto_train(
        self,
        df:              pd.DataFrame,
        target_col:      str,
        problem_type:    Optional[str]       = None,
        models_to_train: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete AutoML pipeline and return a rich result dict.

        Steps:
        1. Detect (or accept) problem type.
        2. Preprocess data.
        3. Train all candidate models.
        4. Generate the reasoning layer.

        Args:
            df:              Input DataFrame.
            target_col:      Name of the target column.
            problem_type:    Optional override (e.g. ``"regression"``).
            models_to_train: Optional subset of model names to run.

        Returns:
            Dict with keys:
            ``problem_type``, ``best_model``, ``model_results``,
            ``comparison``, ``feature_importance``, ``reasoning``.
        """
        logger.info("AutoML pipeline starting …")

        if problem_type:
            self.problem_type = problem_type
            self.target_col   = target_col
        else:
            self.detect_problem_type(df, target_col)

        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        results  = self.train_models(X_train, X_test, y_train, y_test, models_to_train)
        reasoning = self.generate_model_reasoning()

        logger.info("AutoML pipeline complete. Best model: %s", self.best_model_name)

        return {
            "problem_type":      self.problem_type,
            "best_model":        self.best_model_name,
            "model_results":     results,
            "comparison":        self.get_model_comparison(),
            "feature_importance": self.get_feature_importance(),
            "reasoning":         reasoning,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Module-level convenience function
# ══════════════════════════════════════════════════════════════════════════════

def quick_automl(
    df:           pd.DataFrame,
    target_col:   str,
    problem_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One-call AutoML with default settings.

    Args:
        df:           Input DataFrame.
        target_col:   Name of the target column.
        problem_type: Optional override for problem type.

    Returns:
        Full result dict from :meth:`AutoMLEngine.auto_train`.
    """
    return AutoMLEngine().auto_train(df, target_col, problem_type)