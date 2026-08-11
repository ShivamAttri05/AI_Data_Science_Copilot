"""
Visualization utilities for the AI Data Science Copilot.

This module provides reusable functions for creating various
visualizations used in EDA and model evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Optional, List, Dict, Tuple, Any
import streamlit as st
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set default style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def create_correlation_heatmap(
    df: pd.DataFrame,
    method: str = 'pearson',
    figsize: Tuple[int, int] = (12, 10),
    use_plotly: bool = True
) -> Any:
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df: Input DataFrame
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size for matplotlib
        use_plotly: Whether to use plotly (True) or matplotlib (False)
        
    Returns:
        Plotly figure or matplotlib figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        logger.warning("No numeric columns found for correlation heatmap")
        return None
    
    corr_matrix = numeric_df.corr(method=method)
    
    if use_plotly:
        fig = px.imshow(
            corr_matrix,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            zmin=-1,
            zmax=1,
            title=f'Correlation Heatmap ({method.capitalize()})'
        )
        fig.update_layout(height=600, width=800)
        return fig
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, ax=ax, square=True)
        ax.set_title(f'Correlation Heatmap ({method.capitalize()})')
        plt.tight_layout()
        return fig


def create_distribution_plot(
    df: pd.DataFrame,
    column: str,
    use_plotly: bool = True,
    bins: int = 30
) -> Any:
    """
    Create a distribution plot for a single column.
    
    Args:
        df: Input DataFrame
        column: Column name to plot
        use_plotly: Whether to use plotly
        bins: Number of bins for histogram
        
    Returns:
        Plotly or matplotlib figure
    """
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in DataFrame")
        return None
    
    data = df[column].dropna()
    
    if use_plotly:
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            name='Histogram',
            opacity=0.7,
            marker_color='steelblue'
        ))
        
        # KDE curve (if numeric)
        if np.issubdtype(df[column].dtype, np.number):
            from scipy import stats
            kde_x = np.linspace(data.min(), data.max(), 100)
            kde_y = stats.gaussian_kde(data)(kde_x)
            
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y * len(data) * (data.max() - data.min()) / bins,
                mode='lines',
                name='KDE',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title=f'Distribution of {column}',
            xaxis_title=column,
            yaxis_title='Count',
            bargap=0.1,
            height=500
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if np.issubdtype(df[column].dtype, np.number):
            sns.histplot(data, kde=True, bins=bins, ax=ax, color='steelblue')
        else:
            value_counts = data.value_counts().head(20)
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax)
        
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        plt.tight_layout()
        return fig


def create_missing_values_heatmap(
    df: pd.DataFrame,
    use_plotly: bool = True
) -> Any:
    """
    Create a heatmap showing missing values.
    
    Args:
        df: Input DataFrame
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly or matplotlib figure
    """
    if df.isnull().sum().sum() == 0:
        logger.info("No missing values in the dataset")
        return None
    
    # Calculate missing value statistics
    missing_stats = df.isnull().sum().sort_values(ascending=False)
    missing_stats = missing_stats[missing_stats > 0]
    
    if use_plotly:
        fig = go.Figure()
        
        # Bar chart of missing values
        fig.add_trace(go.Bar(
            x=missing_stats.index,
            y=missing_stats.values,
            marker_color='coral',
            text=missing_stats.values,
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Missing Values by Column',
            xaxis_title='Columns',
            yaxis_title='Number of Missing Values',
            xaxis_tickangle=-45,
            height=500
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        missing_stats.plot(kind='bar', ax=ax, color='coral')
        ax.set_title('Missing Values by Column')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Number of Missing Values')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig


def create_pairplot(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    hue: Optional[str] = None,
    max_cols: int = 5
) -> Any:
    """
    Create a pairplot for selected columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to include (None for auto-selection)
        hue: Column to use for color coding
        max_cols: Maximum number of columns to include
        
    Returns:
        Seaborn pairplot
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if columns is None:
        columns = numeric_df.columns[:max_cols].tolist()
    else:
        columns = [col for col in columns if col in numeric_df.columns][:max_cols]
    
    if len(columns) < 2:
        logger.warning("Need at least 2 numeric columns for pairplot")
        return None
    
    if hue and hue in df.columns:
        plot_df = df[columns + [hue]].copy()
    else:
        plot_df = df[columns].copy()
        hue = None
    
    g = sns.pairplot(plot_df, hue=hue, diag_kind='kde', corner=True)
    g.fig.suptitle('Pairplot of Numeric Features', y=1.02)
    
    return g


def create_categorical_barplot(
    df: pd.DataFrame,
    column: str,
    top_n: int = 15,
    use_plotly: bool = True
) -> Any:
    """
    Create a bar plot for categorical column value counts.
    
    Args:
        df: Input DataFrame
        column: Categorical column name
        top_n: Show top N categories
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly or matplotlib figure
    """
    if column not in df.columns:
        logger.error(f"Column '{column}' not found in DataFrame")
        return None
    
    value_counts = df[column].value_counts().head(top_n)
    
    if use_plotly:
        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index,
            orientation='h',
            title=f'Top {top_n} Values in {column}',
            labels={'x': 'Count', 'y': column},
            color=value_counts.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500, yaxis=dict(autorange="reversed"))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette='viridis')
        ax.set_title(f'Top {top_n} Values in {column}')
        ax.set_xlabel('Count')
        plt.tight_layout()
        return fig


def create_boxplot(
    df: pd.DataFrame,
    numeric_col: str,
    categorical_col: Optional[str] = None,
    use_plotly: bool = True
) -> Any:
    """
    Create a box plot for numeric column.
    
    Args:
        df: Input DataFrame
        numeric_col: Numeric column name
        categorical_col: Optional categorical column for grouping
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly or matplotlib figure
    """
    if numeric_col not in df.columns:
        logger.error(f"Column '{numeric_col}' not found in DataFrame")
        return None
    
    if use_plotly:
        if categorical_col and categorical_col in df.columns:
            fig = px.box(df, x=categorical_col, y=numeric_col, 
                        title=f'{numeric_col} by {categorical_col}')
        else:
            fig = px.box(df, y=numeric_col, title=f'Box Plot of {numeric_col}')
        fig.update_layout(height=500)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        if categorical_col and categorical_col in df.columns:
            sns.boxplot(data=df, x=categorical_col, y=numeric_col, ax=ax)
        else:
            sns.boxplot(data=df, y=numeric_col, ax=ax)
        ax.set_title(f'Box Plot of {numeric_col}')
        plt.tight_layout()
        return fig


def create_model_comparison_chart(
    results: Dict[str, Dict[str, float]],
    metric: str = 'accuracy',
    use_plotly: bool = True
) -> Any:
    """
    Create a comparison chart for model performance.
    
    Args:
        results: Dictionary with model names as keys and metrics as values
        metric: Metric to compare
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly or matplotlib figure
    """
    models = list(results.keys())
    scores = [results[model].get(metric, 0) for model in models]
    
    # Sort by score
    sorted_pairs = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
    models, scores = zip(*sorted_pairs)
    
    if use_plotly:
        fig = px.bar(
            x=list(models),
            y=list(scores),
            title=f'Model Comparison - {metric.capitalize()}',
            labels={'x': 'Model', 'y': metric.capitalize()},
            color=list(scores),
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(height=500, xaxis_tickangle=-45)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, scores, color='steelblue')
        ax.set_title(f'Model Comparison - {metric.capitalize()}')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric.capitalize())
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig


def create_feature_importance_plot(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 15,
    use_plotly: bool = True
) -> Any:
    """
    Create a feature importance plot.
    
    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        top_n: Show top N features
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly or matplotlib figure
    """
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=True).tail(top_n)
    
    if use_plotly:
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title=f'Top {top_n} Feature Importances',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=600, yaxis=dict(autorange="reversed"))
        return fig
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax, palette='viridis')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        return fig


def create_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    use_plotly: bool = True
) -> Any:
    """
    Create a confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly or matplotlib figure
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    if use_plotly:
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True
        )
        fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=500,
            width=500
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        return fig


def create_roc_curve_plot(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    use_plotly: bool = True
) -> Any:
    """
    Create an ROC curve plot.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly or matplotlib figure
    """
    if use_plotly:
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {auc_score:.3f})',
            line=dict(color='darkorange', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='navy', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1.05])
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        plt.tight_layout()
        return fig


def create_prediction_vs_actual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    use_plotly: bool = True
) -> Any:
    """
    Create a prediction vs actual scatter plot for regression.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly or matplotlib figure
    """
    if use_plotly:
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            marker=dict(color='steelblue', size=8, opacity=0.6),
            name='Predictions'
        ))
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='Predicted vs Actual Values',
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            height=500
        )
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.6, color='steelblue', edgecolors='white')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Predicted vs Actual Values')
        ax.legend()
        plt.tight_layout()
        return fig


def create_residuals_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    use_plotly: bool = True
) -> Any:
    """
    Create a residuals plot for regression evaluation.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        use_plotly: Whether to use plotly
        
    Returns:
        Plotly or matplotlib figure
    """
    residuals = y_true - y_pred
    
    if use_plotly:
        fig = make_subplots(rows=1, cols=2, 
                          subplot_titles=('Residuals vs Predicted', 'Residuals Distribution'))
        
        # Residuals vs Predicted
        fig.add_trace(go.Scatter(
            x=y_pred,
            y=residuals,
            mode='markers',
            marker=dict(color='steelblue', size=6, opacity=0.6),
            name='Residuals'
        ), row=1, col=1)
        
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residuals histogram
        fig.add_trace(go.Histogram(
            x=residuals,
            nbinsx=30,
            marker_color='steelblue',
            name='Distribution'
        ), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_xaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        fig.update_xaxes(title_text="Residuals", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        return fig
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Residuals vs Predicted
        axes[0].scatter(y_pred, residuals, alpha=0.6, color='steelblue', edgecolors='white')
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Predicted')
        
        # Residuals histogram
        axes[1].hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Residuals Distribution')
        axes[1].axvline(x=0, color='red', linestyle='--')
        
        plt.tight_layout()
        return fig