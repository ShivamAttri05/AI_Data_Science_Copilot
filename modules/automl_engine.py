"""
AutoML Engine Module for AI Data Science Copilot.

This module provides automated machine learning capabilities including
automatic problem type detection, preprocessing, model training, and comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# XGBoost (optional)
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. XGBoost models will be skipped.")


class AutoMLEngine:
    """
    Automated Machine Learning Engine.
    
    Automatically detects problem type, preprocesses data,
    trains multiple models, and selects the best performer.
    """
    
    # Define available models
    CLASSIFICATION_MODELS = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    REGRESSION_MODELS = {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    
    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        cv_folds: int = 5
    ):
        """
        Initialize the AutoML Engine.
        
        Args:
            random_state: Random seed for reproducibility
            test_size: Proportion of data for testing
            cv_folds: Number of cross-validation folds
        """
        self.random_state = random_state
        self.test_size = test_size
        self.cv_folds = cv_folds
        
        self.problem_type = None
        self.target_col = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.is_fitted = False
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.CLASSIFICATION_MODELS['xgboost'] = XGBClassifier(
                n_estimators=100, random_state=random_state, use_label_encoder=False, eval_metric='logloss'
            )
            self.REGRESSION_MODELS['xgboost'] = XGBRegressor(
                n_estimators=100, random_state=random_state
            )
    
    def detect_problem_type(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> str:
        """
        Automatically detect whether the problem is classification or regression.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Problem type string ('classification' or 'regression')
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        target_series = df[target_col]
        
        # Check if target is categorical or has few unique values
        if target_series.dtype in ['object', 'category']:
            problem_type = 'classification'
        elif target_series.nunique() < 10:
            # Check if values are integers (likely classification)
            if target_series.dropna().apply(lambda x: float(x).is_integer()).all():
                problem_type = 'classification'
            else:
                problem_type = 'regression'
        else:
            problem_type = 'regression'
        
        # Further refine classification type
        if problem_type == 'classification':
            n_classes = target_series.nunique()
            if n_classes == 2:
                self.problem_type = 'binary_classification'
            else:
                self.problem_type = 'multiclass_classification'
        else:
            self.problem_type = 'regression'
        
        self.target_col = target_col
        
        logger.info(f"Detected problem type: {self.problem_type}")
        return self.problem_type
    
    def create_preprocessor(
        self,
        df: pd.DataFrame,
        numeric_strategy: str = 'median',
        categorical_strategy: str = 'onehot'
    ) -> ColumnTransformer:
        """
        Create preprocessing pipeline for features.
        
        Args:
            df: Input DataFrame (without target)
            numeric_strategy: Strategy for numeric imputation
            categorical_strategy: Strategy for categorical encoding
            
        Returns:
            ColumnTransformer preprocessor
        """
        # Identify column types
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numeric features: {len(numeric_features)}")
        logger.info(f"Categorical features: {len(categorical_features)}")
        
        # Numeric pipeline
        numeric_transformers = [
            ('imputer', SimpleImputer(strategy=numeric_strategy)),
            ('scaler', StandardScaler())
        ]
        numeric_pipeline = Pipeline(numeric_transformers)
        
        # Categorical pipeline
        categorical_transformers = [
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ]
        
        if categorical_strategy == 'onehot':
            categorical_transformers.append(
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            )
        else:
            categorical_transformers.append(
                ('encoder', LabelEncoder())
            )
        
        categorical_pipeline = Pipeline(categorical_transformers)
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features),
                ('cat', categorical_pipeline, categorical_features)
            ],
            remainder='drop'  # Drop any other columns
        )
        
        self.preprocessor = preprocessor
        self.feature_names = numeric_features + categorical_features
        
        return preprocessor
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()
        
        # Handle target encoding for classification
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            if y.dtype == 'object':
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y.astype(str))
        
        # Create preprocessor and fit on training data
        self.create_preprocessor(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=y if self.problem_type in ['binary_classification', 'multiclass_classification'] else None
        )
        
        # Fit preprocessor and transform
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        logger.info(f"Training set: {X_train_processed.shape}, Test set: {X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def train_models(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models and evaluate performance.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            models_to_train: List of model names to train (None for all)
            
        Returns:
            Dictionary with training results for each model
        """
        # Select models based on problem type
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            available_models = self.CLASSIFICATION_MODELS
        else:
            available_models = self.REGRESSION_MODELS
        
        # Filter models if specified
        if models_to_train:
            available_models = {k: v for k, v in available_models.items() if k in models_to_train}
        
        results = {}
        
        for name, model in available_models.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                
                # Get prediction probabilities for classification
                if self.problem_type in ['binary_classification', 'multiclass_classification']:
                    if hasattr(model, 'predict_proba'):
                        y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Cross-validation
                cv_scores = self._cross_validate(model, X_train, y_train)
                
                # Store results
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'cv_scores': cv_scores,
                    'cv_mean': float(np.mean(cv_scores)),
                    'cv_std': float(np.std(cv_scores)),
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - CV Score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                results[name] = {'error': str(e)}
        
        self.results = results
        self.models = {name: res['model'] for name, res in results.items() if 'error' not in res}
        
        # Select best model
        self._select_best_model()
        
        self.is_fitted = True
        
        return results
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate appropriate metrics based on problem type.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            y_pred_proba: Predicted probabilities (for classification)
            
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            # Classification metrics
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            
            # For binary classification
            if self.problem_type == 'binary_classification':
                metrics['precision'] = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
                metrics['f1'] = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
                
                if y_pred_proba is not None:
                    try:
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                    except:
                        pass
            else:
                # Multiclass
                metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
                metrics['f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        
        else:
            # Regression metrics
            metrics['mse'] = float(mean_squared_error(y_true, y_pred))
            metrics['rmse'] = float(np.sqrt(metrics['mse']))
            metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
            metrics['r2'] = float(r2_score(y_true, y_pred))
            metrics['mape'] = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
        
        return metrics
    
    def _cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            
        Returns:
            Array of CV scores
        """
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scoring = 'r2'
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return scores
    
    def _select_best_model(self):
        """Select the best model based on CV scores."""
        if not self.results:
            logger.warning("No models trained yet")
            return
        
        # Filter out models with errors
        valid_results = {k: v for k, v in self.results.items() if 'error' not in v}
        
        if not valid_results:
            logger.warning("No valid models to select from")
            return
        
        # Select based on CV mean score
        if self.problem_type in ['binary_classification', 'multiclass_classification']:
            # Higher accuracy is better
            best_name = max(valid_results, key=lambda x: valid_results[x]['cv_mean'])
        else:
            # Higher R2 is better
            best_name = max(valid_results, key=lambda x: valid_results[x]['cv_mean'])
        
        self.best_model_name = best_name
        self.best_model = valid_results[best_name]['model']
        
        logger.info(f"Best model: {best_name} (CV Score: {valid_results[best_name]['cv_mean']:.4f})")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the best model.
        
        Returns:
            DataFrame with feature importances or None
        """
        if self.best_model is None:
            logger.warning("No best model available")
            return None
        
        # Get feature names from preprocessor
        try:
            feature_names = []
            
            # Get numeric feature names
            numeric_features = self.preprocessor.transformers_[0][2]
            feature_names.extend(numeric_features)
            
            # Get categorical feature names (from one-hot encoder)
            categorical_features = self.preprocessor.transformers_[1][2]
            cat_encoder = self.preprocessor.named_transformers_['cat'].named_steps.get('encoder')
            
            if cat_encoder and hasattr(cat_encoder, 'get_feature_names_out'):
                cat_names = cat_encoder.get_feature_names_out(categorical_features)
                feature_names.extend(cat_names)
            else:
                feature_names.extend(categorical_features)
            
        except:
            # Fallback to generic feature names
            n_features = self.best_model.n_features_in_ if hasattr(self.best_model, 'n_features_in_') else 0
            feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Get importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importances = np.abs(self.best_model.coef_)
            if importances.ndim > 1:
                importances = importances.mean(axis=0)
        else:
            logger.warning("Model does not provide feature importance")
            return None
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the best model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train_models first.")
        
        # Preprocess
        X_processed = self.preprocessor.transform(X)
        
        # Predict
        predictions = self.best_model.predict(X_processed)
        
        # Decode if classification
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get a comparison DataFrame of all trained models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for name, result in self.results.items():
            if 'error' in result:
                continue
            
            row = {
                'model': name,
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            }
            
            # Add metrics
            for metric, value in result['metrics'].items():
                row[metric] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data).sort_values('cv_mean', ascending=False)
    
    def auto_train(
        self,
        df: pd.DataFrame,
        target_col: str,
        problem_type: Optional[str] = None,
        models_to_train: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Complete AutoML pipeline: detect problem, prepare data, train models.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            problem_type: Optional problem type override
            models_to_train: Optional list of models to train
            
        Returns:
            Dictionary with complete training results
        """
        logger.info("Starting AutoML training pipeline...")
        
        # Detect or set problem type
        if problem_type:
            self.problem_type = problem_type
            self.target_col = target_col
            logger.info(f"Using specified problem type: {problem_type}")
        else:
            self.detect_problem_type(df, target_col)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df, target_col)
        
        # Train models
        results = self.train_models(X_train, X_test, y_train, y_test, models_to_train)
        
        logger.info("AutoML training complete!")
        
        return {
            'problem_type': self.problem_type,
            'best_model': self.best_model_name,
            'model_results': results,
            'comparison': self.get_model_comparison(),
            'feature_importance': self.get_feature_importance()
        }


def quick_automl(
    df: pd.DataFrame,
    target_col: str,
    problem_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick AutoML training with default settings.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        problem_type: Optional problem type override
        
    Returns:
        Dictionary with training results
    """
    engine = AutoMLEngine()
    return engine.auto_train(df, target_col, problem_type)