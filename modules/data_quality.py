"""
Data Quality Module for Explainable ML Pipeline Analyzer.

This module performs comprehensive data quality checks including
missing values, duplicates, outliers, class imbalance, small datasets,
high noise levels, and data leakage risks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """
    A class to perform comprehensive data quality checks.

    Automatically detects common data quality issues and provides
    detailed reports with recommendations.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame.

        Args:
            df: Input DataFrame to check
        """
        self.df = df.copy()
        self.report = {}
        self.issues = []
        self.warnings = []

    def check_dataset_size(self) -> Dict[str, Any]:
        """Check if dataset is too small for reliable ML modeling."""
        n_rows = len(self.df)
        n_cols = len(self.df.columns)

        result = {
            'n_rows': n_rows,
            'n_cols': n_cols,
            'sample_size_per_feature': n_rows / n_cols if n_cols > 0 else 0,
            'severity': 'none'
        }

        if n_rows < 100:
            result['severity'] = 'critical'
            self.issues.append(
                f"🚨 CRITICAL: Dataset too small ({n_rows} rows). "
                f"ML models unreliable below 100-500 rows depending on complexity."
            )
        elif n_rows < 500:
            result['severity'] = 'high'
            self.warnings.append(
                f"⚠️ HIGH RISK: Small dataset ({n_rows} rows). "
                f"Use simple models (LR, RF); XGBoost/GB may overfit."
            )
        elif n_rows < 1000:
            result['severity'] = 'medium'
            self.warnings.append(f"ℹ️ Dataset modest ({n_rows} rows). CV stability may vary.")

        return result

    def check_noise_level(self) -> Dict[str, Any]:
        """Estimate noise level via outlier prevalence and variance inconsistency."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        noise_indicators = []

        for col in numeric_cols:
            col_data = self.df[col].dropna()
            if len(col_data) < 10:
                continue

            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
            outlier_pct = outliers / len(col_data) * 100

            mean_val = col_data.mean()
            cv = col_data.std() / mean_val * 100 if mean_val != 0 else 0

            noise_indicators.append({
                'column': col,
                'outlier_pct': outlier_pct,
                'cv': cv
            })

        avg_outlier_pct = np.mean([ni['outlier_pct'] for ni in noise_indicators]) if noise_indicators else 0
        avg_cv = np.mean([ni['cv'] for ni in noise_indicators]) if noise_indicators else 0

        result = {
            'numeric_columns_analyzed': len(numeric_cols),
            'avg_outlier_pct': round(avg_outlier_pct, 2),
            'avg_cv': round(avg_cv, 2),
            'noise_indicators': noise_indicators[:10],
            'severity': 'none'
        }

        if avg_outlier_pct > 20:
            result['severity'] = 'high'
            self.warnings.append(
                f"⚠️ HIGH NOISE: {avg_outlier_pct:.1f}% outliers across numeric features. "
                f"Tree models may memorize noise; prefer regularized models."
            )
        elif avg_outlier_pct > 10:
            result['severity'] = 'medium'
            self.warnings.append(f"Moderate noise detected ({avg_outlier_pct:.1f}% outliers).")

        if avg_cv > 200:
            result['severity'] = 'high'
            self.warnings.append(
                f"⚠️ HIGH VARIABILITY: Average CV {avg_cv:.0f}%. Signals may be noisy."
            )

        return result

    def check_data_leakage_risk(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect potential data leakage risks (high target correlations, future leaks).

        Correlation with the target is only computed when the target column
        contains numeric data.  String / categorical targets are skipped so
        the method never raises ValueError trying to cast text to float.
        """
        risks = []

        if target_col and target_col in self.df.columns:
            target = self.df[target_col]

            # ── Only attempt correlation when the target is numeric ────────────
            target_is_numeric = pd.api.types.is_numeric_dtype(target)

            if target_is_numeric:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns

                for col in numeric_cols:
                    if col == target_col:
                        continue
                    try:
                        corr = self.df[col].corr(target)
                    except Exception:
                        # Guard against any residual edge-case (all-NaN column, etc.)
                        continue

                    if pd.isna(corr):
                        continue

                    if abs(corr) > 0.95:
                        risks.append({
                            'type': 'high_target_correlation',
                            'column': col,
                            'correlation': round(corr, 3),
                            'risk': 'Feature too similar to target — potential leakage or redundancy'
                        })

                # ── Temporal leakage: datetime vs numeric target ───────────────
                datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
                for date_col in datetime_cols:
                    try:
                        timestamps = pd.to_numeric(self.df[date_col], errors='coerce')
                        target_numeric = pd.to_numeric(target, errors='coerce')
                        corr = timestamps.corr(target_numeric)
                    except Exception:
                        continue

                    if pd.isna(corr):
                        continue

                    if abs(corr) > 0.8:
                        risks.append({
                            'type': 'temporal_leakage',
                            'column': date_col,
                            'correlation': round(corr, 3),
                            'risk': 'Date feature highly correlated with target — check for leakage'
                        })

            else:
                # Categorical target: note that correlation-based leakage detection
                # was skipped; flag it so the caller is aware.
                logger.info(
                    "check_data_leakage_risk: target '%s' is non-numeric (%s) — "
                    "correlation-based checks skipped.",
                    target_col, target.dtype
                )

        # ── Low samples-per-feature ratio (applies regardless of target type) ──
        if len(self.df.columns) > 0:
            info_ratio = len(self.df) / len(self.df.columns)
            if info_ratio < 5:
                risks.append({
                    'type': 'low_info_ratio',
                    'n_rows': len(self.df),
                    'n_cols': len(self.df.columns),
                    'risk': f'Only {info_ratio:.1f} samples per feature — high leakage/overfit risk'
                })

        result = {
            'risks_detected': risks,
            'n_risks': len(risks),
            'severity': (
                'none' if not risks else
                'high' if any('leak' in r['risk'].lower() for r in risks) else
                'medium'
            )
        }

        if risks:
            risk_msg = f"🚨 LEAKAGE RISKS: {len(risks)} potential issues detected"
            if result['severity'] == 'high':
                self.issues.append(risk_msg)
            else:
                self.warnings.append(risk_msg)

        return result

    def run_all_checks(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Run all data quality checks including failure detection."""
        logger.info("Running comprehensive data quality checks with failure detection...")

        self.report = {
            'overview': self._get_overview(),
            'dataset_size': self.check_dataset_size(),
            'noise_level': self.check_noise_level(),
            'data_leakage_risk': self.check_data_leakage_risk(target_col),
            'missing_values': self.check_missing_values(),
            'duplicates': self.check_duplicates(),
            'outliers': self.check_outliers(),
            'constant_columns': self.check_constant_columns(),
            'data_types': self.check_data_types(),
            'cardinality': self.check_cardinality(),
            'class_balance': self.check_class_balance(target_col) if target_col else None,
            'critical_failures': [],
            'issues': [],
            'warnings': [],
            'recommendations': []
        }

        if self.report['dataset_size']['severity'] == 'critical':
            self.report['critical_failures'].append('DATASET_TOO_SMALL')
        if self.report['noise_level']['severity'] == 'high':
            self.report['critical_failures'].append('HIGH_NOISE')
        if self.report['data_leakage_risk']['severity'] == 'high':
            self.report['critical_failures'].append('LEAKAGE_RISK')

        self._compile_issues()
        self._generate_recommendations()

        self.report['issues'] = self.issues
        self.report['warnings'] = self.warnings

        logger.info(
            "Quality check complete. Critical failures: %d",
            len(self.report['critical_failures'])
        )

        return self.report

    def _get_overview(self) -> Dict[str, Any]:
        """Get basic overview of the dataset."""
        return {
            'n_rows': len(self.df),
            'n_columns': len(self.df.columns),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024 ** 2, 2),
            'numeric_columns': len(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(self.df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': len(self.df.select_dtypes(include=['datetime64']).columns)
        }

    def check_missing_values(self) -> Dict[str, Any]:
        """Check for missing values in the dataset."""
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df) * 100).round(2)

        cols_with_missing = missing_counts[missing_counts > 0]

        total_cells = len(self.df) * len(self.df.columns)
        total_missing_pct = round(missing_counts.sum() / total_cells * 100, 2) if total_cells else 0

        result = {
            'total_missing': int(missing_counts.sum()),
            'total_missing_percentage': total_missing_pct,
            'columns_with_missing': len(cols_with_missing),
            'missing_by_column': {},
            'severity': 'none'
        }

        for col in cols_with_missing.index:
            result['missing_by_column'][col] = {
                'count': int(missing_counts[col]),
                'percentage': float(missing_pct[col])
            }

        if result['total_missing_percentage'] > 20:
            result['severity'] = 'critical'
            self.issues.append(f"Critical: {result['total_missing_percentage']}% of data is missing")
        elif result['total_missing_percentage'] > 5:
            result['severity'] = 'high'
            self.warnings.append(f"High: {result['total_missing_percentage']}% of data is missing")
        elif result['total_missing_percentage'] > 0:
            result['severity'] = 'low'

        return result

    def check_duplicates(self) -> Dict[str, Any]:
        """Check for duplicate rows in the dataset."""
        n_duplicates = int(self.df.duplicated().sum())
        duplicate_pct = round(n_duplicates / len(self.df) * 100, 2) if len(self.df) else 0

        result = {
            'n_duplicate_rows': n_duplicates,
            'percentage': float(duplicate_pct),
            'severity': 'none'
        }

        if duplicate_pct > 10:
            result['severity'] = 'critical'
            self.issues.append(f"Critical: {duplicate_pct}% duplicate rows detected")
        elif duplicate_pct > 1:
            result['severity'] = 'medium'
            self.warnings.append(f"Medium: {duplicate_pct}% duplicate rows detected")

        return result

    def check_outliers(
        self,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, Any]:
        """
        Check for outliers in numeric columns.

        Args:
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            Dictionary with outlier analysis
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns

        outliers_by_column = {}
        total_outliers = 0

        for col in numeric_cols:
            col_data = self.df[col].dropna()

            if len(col_data) < 10:
                continue

            lower_bound = upper_bound = None

            if method == 'iqr':
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(col_data))
                outlier_mask = z_scores > threshold
            else:
                continue

            n_outliers = int(outlier_mask.sum())
            outlier_pct = round(n_outliers / len(col_data) * 100, 2)

            if n_outliers > 0:
                outliers_by_column[col] = {
                    'count': n_outliers,
                    'percentage': float(outlier_pct),
                    'lower_bound': float(lower_bound) if lower_bound is not None else None,
                    'upper_bound': float(upper_bound) if upper_bound is not None else None
                }
                total_outliers += n_outliers

        avg_outlier_pct = (
            np.mean([v['percentage'] for v in outliers_by_column.values()])
            if outliers_by_column else 0
        )

        result = {
            'method': method,
            'threshold': threshold,
            'columns_analyzed': len(numeric_cols),
            'columns_with_outliers': len(outliers_by_column),
            'total_outliers': total_outliers,
            'outliers_by_column': outliers_by_column,
            'severity': 'none'
        }

        if avg_outlier_pct > 10:
            result['severity'] = 'high'
            self.warnings.append(
                f"High: Significant outliers detected in {len(outliers_by_column)} columns"
            )
        elif avg_outlier_pct > 5:
            result['severity'] = 'medium'
            self.warnings.append(
                f"Medium: Moderate outliers detected in {len(outliers_by_column)} columns"
            )
        elif outliers_by_column:
            result['severity'] = 'low'

        return result

    def check_constant_columns(self) -> Dict[str, Any]:
        """Check for constant or near-constant columns."""
        constant_cols = []
        near_constant_cols = []

        for col in self.df.columns:
            n_unique = self.df[col].nunique()
            unique_pct = n_unique / len(self.df) * 100 if len(self.df) else 0

            if n_unique <= 1:
                constant_cols.append({
                    'column': col,
                    'unique_values': int(n_unique),
                    'unique_percentage': round(unique_pct, 2)
                })
            elif unique_pct < 0.1:
                near_constant_cols.append({
                    'column': col,
                    'unique_values': int(n_unique),
                    'unique_percentage': round(unique_pct, 2)
                })

        result = {
            'constant_columns': constant_cols,
            'near_constant_columns': near_constant_cols,
            'n_constant': len(constant_cols),
            'n_near_constant': len(near_constant_cols),
            'severity': 'none'
        }

        if constant_cols:
            result['severity'] = 'high'
            self.issues.append(
                f"High: Constant columns found: {[c['column'] for c in constant_cols]}"
            )

        if near_constant_cols:
            result['severity'] = 'medium' if result['severity'] == 'none' else result['severity']
            self.warnings.append(
                f"Medium: Near-constant columns found: {[c['column'] for c in near_constant_cols]}"
            )

        return result

    def check_data_types(self) -> Dict[str, Any]:
        """Check data types and potential type mismatches."""
        type_info = {}
        potential_issues = []

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)

            if dtype == 'object':
                non_null = self.df[col].notna().sum()
                if non_null > 0:
                    numeric_attempt = pd.to_numeric(self.df[col], errors='coerce')
                    non_null_ratio = numeric_attempt.notna().sum() / non_null
                    if non_null_ratio > 0.8:
                        potential_issues.append({
                            'column': col,
                            'current_type': dtype,
                            'suggested_type': 'numeric',
                            'convertible_percentage': round(non_null_ratio * 100, 2)
                        })

            type_info[col] = {
                'current_type': dtype,
                'memory_usage': self.df[col].memory_usage(deep=True) / 1024
            }

        result = {
            'type_info': type_info,
            'potential_issues': potential_issues,
            'n_issues': len(potential_issues),
            'severity': 'low' if potential_issues else 'none'
        }

        if potential_issues:
            self.warnings.append(
                f"Low: {len(potential_issues)} columns may have incorrect data types"
            )

        return result

    def check_cardinality(self) -> Dict[str, Any]:
        """Check cardinality of categorical columns."""
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns

        cardinality_info = {}
        high_cardinality = []

        for col in categorical_cols:
            n_unique = self.df[col].nunique()
            unique_pct = round(n_unique / len(self.df) * 100, 2) if len(self.df) else 0

            cardinality_info[col] = {
                'unique_values': int(n_unique),
                'unique_percentage': float(unique_pct)
            }

            if unique_pct > 50 or n_unique > 100:
                high_cardinality.append({
                    'column': col,
                    'unique_values': int(n_unique),
                    'unique_percentage': float(unique_pct)
                })

        result = {
            'cardinality_info': cardinality_info,
            'high_cardinality_columns': high_cardinality,
            'n_high_cardinality': len(high_cardinality),
            'severity': 'low' if high_cardinality else 'none'
        }

        if high_cardinality:
            self.warnings.append(
                f"Low: {len(high_cardinality)} columns have high cardinality"
            )

        return result

    def check_class_balance(
        self,
        target_col: str,
        threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Check class balance for classification problems.

        Args:
            target_col: Name of target column
            threshold: Imbalance threshold (minority class ratio)

        Returns:
            Dictionary with class balance analysis
        """
        if target_col not in self.df.columns:
            return {'error': f'Target column {target_col} not found'}

        value_counts = self.df[target_col].value_counts()
        total = len(self.df)

        class_distribution = {
            str(cls): {
                'count': int(cnt),
                'percentage': round(cnt / total * 100, 2)
            }
            for cls, cnt in value_counts.items()
        }

        majority_count = value_counts.iloc[0]
        minority_count = value_counts.iloc[-1]
        imbalance_ratio = majority_count / minority_count if minority_count > 0 else float('inf')
        minority_pct = minority_count / total

        result = {
            'n_classes': len(value_counts),
            'class_distribution': class_distribution,
            'imbalance_ratio': round(imbalance_ratio, 2),
            'most_frequent_class': str(value_counts.index[0]),
            'least_frequent_class': str(value_counts.index[-1]),
            'severity': 'none'
        }

        if minority_pct < 0.05:
            result['severity'] = 'critical'
            self.issues.append(
                f"Critical: Severe class imbalance (minority: {minority_pct * 100:.1f}%)"
            )
        elif minority_pct < threshold:
            result['severity'] = 'high'
            self.warnings.append(
                f"High: Class imbalance detected (minority: {minority_pct * 100:.1f}%)"
            )
        elif imbalance_ratio > 3:
            result['severity'] = 'medium'
            self.warnings.append(
                f"Medium: Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1)"
            )

        return result

    def _compile_issues(self):
        """Compile all issues found during checks."""
        pass

    def _generate_recommendations(self):
        """Generate recommendations based on findings."""
        recommendations = []

        missing = self.report.get('missing_values', {})
        if missing.get('severity') in ['critical', 'high']:
            recommendations.append("Consider imputation strategies for missing values")
            recommendations.append("Columns with >50% missing values should be dropped")

        duplicates = self.report.get('duplicates', {})
        if duplicates.get('severity') in ['critical', 'medium']:
            recommendations.append("Remove duplicate rows before training")

        outliers = self.report.get('outliers', {})
        if outliers.get('severity') in ['high', 'medium']:
            recommendations.append(
                "Consider outlier treatment (clipping, transformation, or removal)"
            )

        constant = self.report.get('constant_columns', {})
        if constant.get('n_constant', 0) > 0:
            recommendations.append("Remove constant columns (no predictive power)")

        class_balance = self.report.get('class_balance', {})
        if class_balance and class_balance.get('severity') in ['critical', 'high']:
            recommendations.append(
                "Use class balancing techniques (SMOTE, class weights, or stratified sampling)"
            )

        self.report['recommendations'] = recommendations

    def get_quick_summary(self) -> str:
        """Get a quick text summary of data quality."""
        if not self.report:
            return "No quality checks performed yet."

        n_critical = sum(1 for i in self.issues if 'Critical' in i)

        lines = [
            "=" * 50,
            "DATA QUALITY SUMMARY",
            "=" * 50,
            f"Dataset Shape: {self.report['overview']['n_rows']:,} rows "
            f"× {self.report['overview']['n_columns']} columns",
            f"Memory Usage: {self.report['overview']['memory_usage_mb']} MB",
            "",
            "ISSUES FOUND:",
            f"  - Critical Issues: {n_critical}",
            f"  - Warnings: {len(self.warnings)}",
            "",
            "DETAILS:",
            f"  - Missing Values: {self.report['missing_values']['total_missing_percentage']}%",
            f"  - Duplicate Rows: {self.report['duplicates']['percentage']}%",
            f"  - Columns with Outliers: {self.report['outliers']['columns_with_outliers']}",
            f"  - Constant Columns: {self.report['constant_columns']['n_constant']}",
            "=" * 50,
        ]

        return "\n".join(lines)


def quick_quality_check(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform a quick quality check on a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with key quality metrics
    """
    total_cells = len(df) * len(df.columns)
    return {
        'missing_pct':      df.isnull().sum().sum() / total_cells * 100 if total_cells else 0,
        'duplicate_pct':    df.duplicated().sum() / len(df) * 100 if len(df) else 0,
        'n_constant_cols':  sum(1 for col in df.columns if df[col].nunique() <= 1),
        'memory_mb':        df.memory_usage(deep=True).sum() / 1024 ** 2,
    }