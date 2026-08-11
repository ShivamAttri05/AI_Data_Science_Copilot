"""
Exploratory Data Analysis (EDA) Engine for AI Data Science Copilot.

This module provides comprehensive EDA capabilities including
statistical summaries, visualizations, and data profiling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import visualization utilities
import sys
sys.path.append('..')
from utils.visualization import (
    create_correlation_heatmap,
    create_distribution_plot,
    create_missing_values_heatmap,
    create_pairplot,
    create_categorical_barplot,
    create_boxplot
)


class EDAEngine:
    """
    A class to perform comprehensive Exploratory Data Analysis.
    
    Provides automated statistical analysis and visualization
    generation for any dataset.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.
        
        Args:
            df: Input DataFrame for analysis
        """
        self.df = df.copy()
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
        self.analysis_results = {}
        
        logger.info(f"EDA Engine initialized with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
    
    def run_full_analysis(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete EDA analysis.
        
        Args:
            target_col: Optional target column for targeted analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        logger.info("Running full EDA analysis...")
        
        self.analysis_results = {
            'overview': self.get_overview(),
            'summary_statistics': self.get_summary_statistics(),
            'correlation_analysis': self.get_correlation_analysis(),
            'distributions': self.get_distribution_analysis(),
            'categorical_analysis': self.get_categorical_analysis(),
            'missing_analysis': self.get_missing_value_analysis(),
            'target_analysis': self.get_target_analysis(target_col) if target_col else None
        }
        
        logger.info("EDA analysis complete")
        return self.analysis_results
    
    def get_overview(self) -> Dict[str, Any]:
        """
        Get dataset overview information.
        
        Returns:
            Dictionary with overview statistics
        """
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'n_numeric': len(self.numeric_cols),
            'n_categorical': len(self.categorical_cols),
            'n_datetime': len(self.datetime_cols),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            'n_missing_total': int(self.df.isnull().sum().sum()),
            'missing_percentage': round(self.df.isnull().sum().sum() / (self.df.shape[0] * self.df.shape[1]) * 100, 2),
            'n_duplicates': int(self.df.duplicated().sum())
        }
    
    def get_summary_statistics(self) -> Dict[str, pd.DataFrame]:
        """
        Get summary statistics for all columns.
        
        Returns:
            Dictionary with statistics DataFrames
        """
        stats_dict = {}
        
        # Numeric columns
        if self.numeric_cols:
            numeric_stats = self.df[self.numeric_cols].describe()
            # Add additional statistics
            numeric_stats.loc['skewness'] = self.df[self.numeric_cols].skew()
            numeric_stats.loc['kurtosis'] = self.df[self.numeric_cols].kurtosis()
            numeric_stats.loc['missing'] = self.df[self.numeric_cols].isnull().sum()
            numeric_stats.loc['missing_pct'] = self.df[self.numeric_cols].isnull().sum() / len(self.df) * 100
            stats_dict['numeric'] = numeric_stats
        
        # Categorical columns
        if self.categorical_cols:
            cat_stats = pd.DataFrame({
                'unique_values': self.df[self.categorical_cols].nunique(),
                'most_frequent': self.df[self.categorical_cols].mode().iloc[0] if not self.df[self.categorical_cols].mode().empty else None,
                'most_frequent_count': [self.df[col].value_counts().iloc[0] if not self.df[col].value_counts().empty else 0 for col in self.categorical_cols],
                'missing': self.df[self.categorical_cols].isnull().sum(),
                'missing_pct': self.df[self.categorical_cols].isnull().sum() / len(self.df) * 100
            })
            stats_dict['categorical'] = cat_stats
        
        return stats_dict
    
    def get_correlation_analysis(self) -> Dict[str, Any]:
        """
        Get correlation analysis for numeric columns.
        
        Returns:
            Dictionary with correlation results
        """
        if len(self.numeric_cols) < 2:
            return {'error': 'Need at least 2 numeric columns for correlation analysis'}
        
        # Pearson correlation
        pearson_corr = self.df[self.numeric_cols].corr(method='pearson')
        
        # Spearman correlation
        spearman_corr = self.df[self.numeric_cols].corr(method='spearman')
        
        # Find highly correlated pairs
        high_corr_pairs = []
        threshold = 0.8
        
        for i in range(len(pearson_corr.columns)):
            for j in range(i+1, len(pearson_corr.columns)):
                corr_val = pearson_corr.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        'feature_1': pearson_corr.columns[i],
                        'feature_2': pearson_corr.columns[j],
                        'correlation': round(corr_val, 3)
                    })
        
        return {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'high_correlation_pairs': high_corr_pairs,
            'n_high_correlations': len(high_corr_pairs)
        }
    
    def get_distribution_analysis(self) -> Dict[str, Any]:
        """
        Get distribution analysis for numeric columns.
        
        Returns:
            Dictionary with distribution statistics
        """
        if not self.numeric_cols:
            return {'error': 'No numeric columns found'}
        
        distribution_stats = {}
        
        for col in self.numeric_cols:
            data = self.df[col].dropna()
            
            if len(data) == 0:
                continue
            
            # Shapiro-Wilk test for normality (sample if too large)
            from scipy import stats
            sample_data = data.sample(min(5000, len(data)), random_state=42)
            _, p_value = stats.shapiro(sample_data)
            
            distribution_stats[col] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis()),
                'min': float(data.min()),
                'max': float(data.max()),
                'range': float(data.max() - data.min()),
                'iqr': float(data.quantile(0.75) - data.quantile(0.25)),
                'normality_p_value': float(p_value),
                'is_normal': p_value > 0.05
            }
        
        return {
            'distribution_stats': distribution_stats,
            'n_normal_distributions': sum(1 for v in distribution_stats.values() if v['is_normal'])
        }
    
    def get_categorical_analysis(self) -> Dict[str, Any]:
        """
        Get analysis for categorical columns.
        
        Returns:
            Dictionary with categorical analysis results
        """
        if not self.categorical_cols:
            return {'error': 'No categorical columns found'}
        
        categorical_stats = {}
        
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts()
            
            categorical_stats[col] = {
                'n_unique': int(self.df[col].nunique()),
                'unique_ratio': round(self.df[col].nunique() / len(self.df), 4),
                'top_categories': value_counts.head(10).to_dict(),
                'entropy': self._calculate_entropy(self.df[col]),
                'is_high_cardinality': self.df[col].nunique() > 50
            }
        
        return {
            'categorical_stats': categorical_stats,
            'n_high_cardinality': sum(1 for v in categorical_stats.values() if v['is_high_cardinality'])
        }
    
    def _calculate_entropy(self, series: pd.Series) -> float:
        """
        Calculate Shannon entropy for a categorical series.
        
        Args:
            series: Input series
            
        Returns:
            Entropy value
        """
        from scipy.stats import entropy
        value_counts = series.value_counts(normalize=True)
        return float(entropy(value_counts))
    
    def get_missing_value_analysis(self) -> Dict[str, Any]:
        """
        Get detailed missing value analysis.
        
        Returns:
            Dictionary with missing value statistics
        """
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df) * 100).round(2)
        
        cols_with_missing = missing_counts[missing_counts > 0]
        
        missing_patterns = {}
        for col in cols_with_missing.index:
            missing_patterns[col] = {
                'count': int(missing_counts[col]),
                'percentage': float(missing_pct[col]),
                'pattern': 'MCAR' if missing_pct[col] < 5 else 'MAR/MNAR'  # Simplified classification
            }
        
        # Check if missing values are correlated
        missing_corr = None
        if len(cols_with_missing) > 1:
            missing_df = self.df[cols_with_missing.index].isnull().astype(int)
            missing_corr = missing_df.corr()
        
        return {
            'total_missing': int(missing_counts.sum()),
            'total_missing_percentage': round(missing_counts.sum() / (len(self.df) * len(self.df.columns)) * 100, 2),
            'columns_with_missing': len(cols_with_missing),
            'missing_patterns': missing_patterns,
            'missing_correlation': missing_corr,
            'complete_cases': int((~self.df.isnull().any(axis=1)).sum()),
            'complete_cases_percentage': round((~self.df.isnull().any(axis=1)).sum() / len(self.df) * 100, 2)
        }
    
    def get_target_analysis(self, target_col: str) -> Dict[str, Any]:
        """
        Get analysis specific to the target column.
        
        Args:
            target_col: Name of target column
            
        Returns:
            Dictionary with target analysis results
        """
        if target_col not in self.df.columns:
            return {'error': f'Target column {target_col} not found'}
        
        target_series = self.df[target_col]
        
        analysis = {
            'column': target_col,
            'dtype': str(target_series.dtype),
            'n_missing': int(target_series.isnull().sum()),
            'missing_percentage': round(target_series.isnull().sum() / len(target_series) * 100, 2)
        }
        
        # Determine if classification or regression
        if target_series.dtype in ['object', 'category'] or target_series.nunique() < 10:
            # Classification target
            analysis['problem_type'] = 'classification'
            analysis['n_classes'] = int(target_series.nunique())
            analysis['class_distribution'] = target_series.value_counts().to_dict()
            analysis['class_percentages'] = (target_series.value_counts(normalize=True) * 100).round(2).to_dict()
            
            # Check for imbalance
            class_counts = target_series.value_counts()
            imbalance_ratio = class_counts.iloc[0] / class_counts.iloc[-1] if len(class_counts) > 1 else 1
            analysis['imbalance_ratio'] = round(imbalance_ratio, 2)
            analysis['is_imbalanced'] = imbalance_ratio > 3
            
        else:
            # Regression target
            analysis['problem_type'] = 'regression'
            analysis['statistics'] = {
                'mean': float(target_series.mean()),
                'median': float(target_series.median()),
                'std': float(target_series.std()),
                'min': float(target_series.min()),
                'max': float(target_series.max()),
                'skewness': float(target_series.skew()),
                'kurtosis': float(target_series.kurtosis())
            }
            analysis['range'] = float(target_series.max() - target_series.min())
        
        return analysis
    
    def generate_correlation_heatmap(self, use_plotly: bool = True) -> Any:
        """
        Generate correlation heatmap visualization.
        
        Args:
            use_plotly: Whether to use plotly
            
        Returns:
            Visualization figure
        """
        return create_correlation_heatmap(self.df, use_plotly=use_plotly)
    
    def generate_distribution_plots(
        self,
        columns: Optional[List[str]] = None,
        max_plots: int = 6,
        use_plotly: bool = True
    ) -> List[Any]:
        """
        Generate distribution plots for numeric columns.
        
        Args:
            columns: Specific columns to plot (None for auto-selection)
            max_plots: Maximum number of plots to generate
            use_plotly: Whether to use plotly
            
        Returns:
            List of visualization figures
        """
        if columns is None:
            columns = self.numeric_cols[:max_plots]
        
        figures = []
        for col in columns:
            if col in self.df.columns:
                fig = create_distribution_plot(self.df, col, use_plotly=use_plotly)
                figures.append((col, fig))
        
        return figures
    
    def generate_categorical_plots(
        self,
        columns: Optional[List[str]] = None,
        max_plots: int = 6,
        use_plotly: bool = True
    ) -> List[Any]:
        """
        Generate bar plots for categorical columns.
        
        Args:
            columns: Specific columns to plot (None for auto-selection)
            max_plots: Maximum number of plots to generate
            use_plotly: Whether to use plotly
            
        Returns:
            List of visualization figures
        """
        if columns is None:
            columns = self.categorical_cols[:max_plots]
        
        figures = []
        for col in columns:
            if col in self.df.columns:
                fig = create_categorical_barplot(self.df, col, use_plotly=use_plotly)
                figures.append((col, fig))
        
        return figures
    
    def generate_missing_values_plot(self, use_plotly: bool = True) -> Any:
        """
        Generate missing values visualization.
        
        Args:
            use_plotly: Whether to use plotly
            
        Returns:
            Visualization figure
        """
        return create_missing_values_heatmap(self.df, use_plotly=use_plotly)
    
    def generate_pairplot(
        self,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        max_cols: int = 5
    ) -> Any:
        """
        Generate pairplot for selected columns.
        
        Args:
            columns: Columns to include
            hue: Column for color coding
            max_cols: Maximum number of columns
            
        Returns:
            Pairplot figure
        """
        return create_pairplot(self.df, columns, hue, max_cols)
    
    def generate_target_analysis_plots(
        self,
        target_col: str,
        use_plotly: bool = True
    ) -> Dict[str, Any]:
        """
        Generate plots for target column analysis.
        
        Args:
            target_col: Name of target column
            use_plotly: Whether to use plotly
            
        Returns:
            Dictionary with visualization figures
        """
        if target_col not in self.df.columns:
            return {'error': f'Target column {target_col} not found'}
        
        figures = {}
        target_series = self.df[target_col]
        
        if target_series.dtype in ['object', 'category'] or target_series.nunique() < 10:
            # Classification - bar chart
            value_counts = target_series.value_counts()
            
            if use_plotly:
                fig = px.bar(
                    x=value_counts.index.astype(str),
                    y=value_counts.values,
                    title=f'Target Distribution: {target_col}',
                    labels={'x': target_col, 'y': 'Count'},
                    color=value_counts.values,
                    color_continuous_scale='Viridis'
                )
            else:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x=value_counts.index.astype(str), y=value_counts.values, ax=ax)
                ax.set_title(f'Target Distribution: {target_col}')
                ax.set_xlabel(target_col)
                ax.set_ylabel('Count')
                plt.tight_layout()
            
            figures['target_distribution'] = fig
            
        else:
            # Regression - distribution plot
            fig = create_distribution_plot(self.df, target_col, use_plotly=use_plotly)
            figures['target_distribution'] = fig
        
        return figures
    
    def get_eda_report(self) -> str:
        """
        Generate a text report of the EDA findings.
        
        Returns:
            Formatted report string
        """
        if not self.analysis_results:
            self.run_full_analysis()
        
        overview = self.analysis_results['overview']
        
        lines = [
            "=" * 60,
            "EXPLORATORY DATA ANALYSIS REPORT",
            "=" * 60,
            "",
            f"Dataset Shape: {overview['shape'][0]:,} rows × {overview['shape'][1]} columns",
            f"Memory Usage: {overview['memory_usage_mb']} MB",
            f"Numeric Columns: {overview['n_numeric']}",
            f"Categorical Columns: {overview['n_categorical']}",
            f"Missing Values: {overview['missing_percentage']}%",
            f"Duplicate Rows: {overview['n_duplicates']:,}",
            "",
            "-" * 60,
            "CORRELATION ANALYSIS",
            "-" * 60,
        ]
        
        corr_analysis = self.analysis_results.get('correlation_analysis', {})
        if 'high_correlation_pairs' in corr_analysis:
            lines.append(f"Highly Correlated Pairs (|r| >= 0.8): {corr_analysis['n_high_correlations']}")
            for pair in corr_analysis['high_correlation_pairs'][:5]:
                lines.append(f"  - {pair['feature_1']} vs {pair['feature_2']}: r = {pair['correlation']}")
        
        lines.extend([
            "",
            "-" * 60,
            "DISTRIBUTION ANALYSIS",
            "-" * 60,
        ])
        
        dist_analysis = self.analysis_results.get('distributions', {})
        if 'distribution_stats' in dist_analysis:
            lines.append(f"Normal Distributions: {dist_analysis.get('n_normal_distributions', 0)}")
            for col, stats in list(dist_analysis['distribution_stats'].items())[:3]:
                lines.append(f"  - {col}: skew = {stats['skewness']:.2f}, "
                           f"{'normal' if stats['is_normal'] else 'non-normal'}")
        
        lines.extend([
            "",
            "-" * 60,
            "CATEGORICAL ANALYSIS",
            "-" * 60,
        ])
        
        cat_analysis = self.analysis_results.get('categorical_analysis', {})
        if 'categorical_stats' in cat_analysis:
            lines.append(f"High Cardinality Columns: {cat_analysis.get('n_high_cardinality', 0)}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


def quick_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a quick profile of the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with key profile information
    """
    engine = EDAEngine(df)
    
    return {
        'shape': df.shape,
        'column_types': {
            'numeric': len(df.select_dtypes(include=[np.number]).columns),
            'categorical': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime': len(df.select_dtypes(include=['datetime64']).columns)
        },
        'missing_pct': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'preview': df.head(5).to_dict()
    }