"""
Experiment Analysis Module for AI Data Science Copilot.

This module provides comprehensive model evaluation and visualization
including confusion matrices, ROC curves, and performance comparisons.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import visualization utilities
from utils.visualization import (
    create_confusion_matrix_plot,
    create_roc_curve_plot,
    create_prediction_vs_actual_plot,
    create_residuals_plot,
    create_model_comparison_chart,
    create_feature_importance_plot
)


class ExperimentAnalyzer:
    """
    A class to analyze and visualize model experiments.
    
    Provides comprehensive evaluation metrics and visualizations
    for both classification and regression models.
    """
    
    def __init__(
        self,
        problem_type: str,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize the Experiment Analyzer.
        
        Args:
            problem_type: Type of problem ('classification' or 'regression')
            class_names: List of class names for classification
        """
        self.problem_type = problem_type
        self.class_names = class_names
        self.results = {}
        self.is_binary = False
        
        if problem_type in ['binary_classification', 'multiclass_classification']:
            self.is_classification = True
            if problem_type == 'binary_classification':
                self.is_binary = True
        else:
            self.is_classification = False
    
    def analyze_classification_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze a classification model's performance.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary with analysis results
        """
        if not self.is_classification:
            raise ValueError("This method is for classification models only")
        
        logger.info(f"Analyzing classification model: {model_name}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        analysis = {
            'model_name': model_name,
            'confusion_matrix': cm,
            'classification_report': report,
            'accuracy': report['accuracy'],
            'macro_avg': report['macro avg'],
            'weighted_avg': report['weighted avg']
        }
        
        # ROC curve for binary classification
        if self.is_binary and y_pred_proba is not None:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            analysis['roc_curve'] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc
            }
            
            # Precision-Recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba[:, 1])
            pr_auc = auc(recall, precision)
            
            analysis['pr_curve'] = {
                'precision': precision,
                'recall': recall,
                'auc': pr_auc
            }
        
        # Per-class metrics
        if self.class_names:
            per_class = {}
            for i, class_name in enumerate(self.class_names):
                if class_name in report:
                    per_class[class_name] = report[class_name]
            analysis['per_class'] = per_class
        
        self.results[model_name] = analysis
        
        return analysis
    
    def analyze_regression_model(
        self,
        model_name: str,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze a regression model's performance.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with analysis results
        """
        if self.is_classification:
            raise ValueError("This method is for regression models only")
        
        logger.info(f"Analyzing regression model: {model_name}")
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Additional statistics
        analysis = {
            'model_name': model_name,
            'metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': np.mean(np.abs(residuals / (y_true + 1e-10))) * 100,
                'explained_variance': 1 - np.var(residuals) / np.var(y_true)
            },
            'residuals': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals)
            },
            'predictions': y_pred,
            'actuals': y_true,
            'residual_values': residuals
        }
        
        self.results[model_name] = analysis
        
        return analysis
    
    def generate_confusion_matrix_plot(
        self,
        model_name: str,
        use_plotly: bool = True
    ) -> Any:
        """
        Generate confusion matrix visualization.
        
        Args:
            model_name: Name of the model
            use_plotly: Whether to use plotly
            
        Returns:
            Visualization figure
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        cm = self.results[model_name]['confusion_matrix']
        
        return create_confusion_matrix_plot(cm, self.class_names, use_plotly)
    
    def generate_roc_curve_plot(
        self,
        model_name: str,
        use_plotly: bool = True
    ) -> Any:
        """
        Generate ROC curve visualization.
        
        Args:
            model_name: Name of the model
            use_plotly: Whether to use plotly
            
        Returns:
            Visualization figure
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        if not self.is_binary:
            logger.warning("ROC curve is only available for binary classification")
            return None
        
        roc_data = self.results[model_name].get('roc_curve')
        if not roc_data:
            logger.warning("ROC curve data not available")
            return None
        
        return create_roc_curve_plot(
            roc_data['fpr'],
            roc_data['tpr'],
            roc_data['auc'],
            use_plotly
        )
    
    def generate_prediction_plot(
        self,
        model_name: str,
        use_plotly: bool = True
    ) -> Any:
        """
        Generate prediction vs actual plot for regression.
        
        Args:
            model_name: Name of the model
            use_plotly: Whether to use plotly
            
        Returns:
            Visualization figure
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        if self.is_classification:
            logger.warning("Prediction plot is for regression models only")
            return None
        
        y_true = self.results[model_name]['actuals']
        y_pred = self.results[model_name]['predictions']
        
        return create_prediction_vs_actual_plot(y_true, y_pred, use_plotly)
    
    def generate_residuals_plot(
        self,
        model_name: str,
        use_plotly: bool = True
    ) -> Any:
        """
        Generate residuals plot for regression.
        
        Args:
            model_name: Name of the model
            use_plotly: Whether to use plotly
            
        Returns:
            Visualization figure
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        if self.is_classification:
            logger.warning("Residuals plot is for regression models only")
            return None
        
        y_true = self.results[model_name]['actuals']
        y_pred = self.results[model_name]['predictions']
        
        return create_residuals_plot(y_true, y_pred, use_plotly)
    
    def generate_model_comparison(
        self,
        metric: str = 'accuracy',
        use_plotly: bool = True
    ) -> Any:
        """
        Generate model comparison visualization.
        
        Args:
            metric: Metric to compare
            use_plotly: Whether to use plotly
            
        Returns:
            Visualization figure
        """
        if not self.results:
            logger.warning("No results available for comparison")
            return None
        
        # Extract metrics for each model
        comparison_data = {}
        
        for model_name, result in self.results.items():
            if self.is_classification:
                if metric in result:
                    comparison_data[model_name] = {metric: result[metric]}
                elif 'classification_report' in result:
                    comparison_data[model_name] = {
                        'accuracy': result['classification_report']['accuracy'],
                        'precision': result['classification_report']['weighted avg']['precision'],
                        'recall': result['classification_report']['weighted avg']['recall'],
                        'f1': result['classification_report']['weighted avg']['f1-score']
                    }
            else:
                if 'metrics' in result:
                    comparison_data[model_name] = result['metrics']
        
        return create_model_comparison_chart(comparison_data, metric, use_plotly)
    
    def generate_feature_importance_plot(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_n: int = 15,
        use_plotly: bool = True
    ) -> Any:
        """
        Generate feature importance visualization.
        
        Args:
            feature_names: List of feature names
            importance_scores: Array of importance scores
            top_n: Number of top features to show
            use_plotly: Whether to use plotly
            
        Returns:
            Visualization figure
        """
        return create_feature_importance_plot(
            feature_names,
            importance_scores,
            top_n,
            use_plotly
        )
    
    def generate_classification_report_table(
        self,
        model_name: str
    ) -> pd.DataFrame:
        """
        Generate a formatted classification report table.
        
        Args:
            model_name: Name of the model
            
        Returns:
            DataFrame with classification report
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        report = self.results[model_name]['classification_report']
        
        # Convert to DataFrame
        rows = []
        for key, values in report.items():
            if isinstance(values, dict):
                row = {'class': key}
                row.update(values)
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def generate_metrics_summary(self) -> pd.DataFrame:
        """
        Generate a summary table of all model metrics.
        
        Returns:
            DataFrame with metrics summary
        """
        if not self.results:
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, result in self.results.items():
            row = {'model': model_name}
            
            if self.is_classification:
                row['accuracy'] = result.get('accuracy', 0)
                if 'weighted_avg' in result:
                    row['precision'] = result['weighted_avg'].get('precision', 0)
                    row['recall'] = result['weighted_avg'].get('recall', 0)
                    row['f1'] = result['weighted_avg'].get('f1-score', 0)
                if 'roc_curve' in result:
                    row['auc'] = result['roc_curve'].get('auc', 0)
            else:
                if 'metrics' in result:
                    metrics = result['metrics']
                    row['rmse'] = metrics.get('rmse', 0)
                    row['mae'] = metrics.get('mae', 0)
                    row['r2'] = metrics.get('r2', 0)
                    row['mape'] = metrics.get('mape', 0)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, float]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        if not self.results:
            return None, 0
        
        best_model = None
        best_score = -np.inf
        
        for model_name, result in self.results.items():
            if self.is_classification:
                score = result.get(metric, 0)
            else:
                score = result.get('metrics', {}).get(metric, 0)
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model, best_score
    
    def export_results(self, filepath: str) -> bool:
        """
        Export analysis results to a JSON file.
        
        Args:
            filepath: Path to save file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            export_data = {}
            for model_name, result in self.results.items():
                export_data[model_name] = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        export_data[model_name][key] = value.tolist()
                    elif isinstance(value, dict):
                        export_data[model_name][key] = {
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in value.items()
                        }
                    else:
                        export_data[model_name][key] = value
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Results exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False


def quick_analyze(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    problem_type: str,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Quick analysis of model predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        problem_type: Type of problem
        y_pred_proba: Predicted probabilities (for classification)
        
    Returns:
        Dictionary with analysis results
    """
    analyzer = ExperimentAnalyzer(problem_type)
    
    if problem_type in ['binary_classification', 'multiclass_classification']:
        return analyzer.analyze_classification_model(
            'model', y_true, y_pred, y_pred_proba
        )
    else:
        return analyzer.analyze_regression_model('model', y_true, y_pred)