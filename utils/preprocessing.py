"""
Data preprocessing utilities for the AI Data Science Copilot.

This module provides reusable functions for data cleaning,
transformation, and feature engineering tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Dict, Optional, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Automatically detect numeric and categorical columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with 'numeric' and 'categorical' column lists
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Further refine: columns with few unique values might be categorical
    for col in numeric_cols[:]:
        if df[col].nunique() < 10 and df[col].nunique() / len(df) < 0.05:
            logger.info(f"Column '{col}' appears to be categorical (numeric with few unique values)")
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols
    }


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = 'median',
    categorical_strategy: str = 'most_frequent',
    drop_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        numeric_strategy: Strategy for numeric columns ('mean', 'median', 'drop')
        categorical_strategy: Strategy for categorical columns ('most_frequent', 'constant', 'drop')
        drop_threshold: Drop columns with missing ratio above this threshold
        
    Returns:
        DataFrame with handled missing values
    """
    df_clean = df.copy()
    
    # Drop columns with too many missing values
    missing_ratio = df_clean.isnull().sum() / len(df_clean)
    cols_to_drop = missing_ratio[missing_ratio > drop_threshold].index.tolist()
    
    if cols_to_drop:
        logger.info(f"Dropping columns with >{drop_threshold*100}% missing values: {cols_to_drop}")
        df_clean = df_clean.drop(columns=cols_to_drop)
    
    column_types = detect_column_types(df_clean)
    
    # Handle numeric missing values
    numeric_cols = [col for col in column_types['numeric'] if col in df_clean.columns]
    if numeric_cols and df_clean[numeric_cols].isnull().sum().sum() > 0:
        if numeric_strategy == 'drop':
            df_clean = df_clean.dropna(subset=numeric_cols)
        else:
            imputer = SimpleImputer(strategy=numeric_strategy)
            df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    
    # Handle categorical missing values
    categorical_cols = [col for col in column_types['categorical'] if col in df_clean.columns]
    if categorical_cols and df_clean[categorical_cols].isnull().sum().sum() > 0:
        if categorical_strategy == 'drop':
            df_clean = df_clean.dropna(subset=categorical_cols)
        else:
            for col in categorical_cols:
                if df_clean[col].isnull().sum() > 0:
                    if categorical_strategy == 'most_frequent':
                        fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                    else:
                        fill_value = 'Unknown'
                    df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean


def encode_categorical(
    df: pd.DataFrame,
    categorical_cols: Optional[List[str]] = None,
    method: str = 'onehot',
    drop_first: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categorical variables.
    
    Args:
        df: Input DataFrame
        categorical_cols: List of categorical columns to encode
        method: Encoding method ('onehot', 'label', 'ordinal')
        drop_first: Whether to drop first category in one-hot encoding
        
    Returns:
        Tuple of (encoded DataFrame, encoders dictionary)
    """
    df_encoded = df.copy()
    encoders = {}
    
    if categorical_cols is None:
        categorical_cols = detect_column_types(df)['categorical']
    
    for col in categorical_cols:
        if col not in df_encoded.columns:
            continue
            
        if method == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=drop_first)
            df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummies], axis=1)
            encoders[col] = {'method': 'onehot', 'categories': df[col].unique().tolist()}
            
        elif method == 'label':
            # Label encoding
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = {'method': 'label', 'encoder': le}
            
        elif method == 'ordinal':
            # Ordinal encoding (assign integer values)
            unique_values = df_encoded[col].unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            df_encoded[col] = df_encoded[col].map(mapping)
            encoders[col] = {'method': 'ordinal', 'mapping': mapping}
    
    return df_encoded, encoders


def scale_features(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    method: str = 'standard'
) -> Tuple[pd.DataFrame, Union[StandardScaler, MinMaxScaler]]:
    """
    Scale numeric features.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of numeric columns to scale
        method: Scaling method ('standard', 'minmax')
        
    Returns:
        Tuple of (scaled DataFrame, scaler object)
    """
    df_scaled = df.copy()
    
    if numeric_cols is None:
        numeric_cols = detect_column_types(df)['numeric']
    
    # Only scale columns that exist and are numeric
    numeric_cols = [col for col in numeric_cols if col in df_scaled.columns]
    
    if not numeric_cols:
        return df_scaled, None
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    
    return df_scaled, scaler


def remove_outliers(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Remove outliers from numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_cols: List of numeric columns to check
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if numeric_cols is None:
        numeric_cols = detect_column_types(df)['numeric']
    
    numeric_cols = [col for col in numeric_cols if col in df_clean.columns]
    
    if method == 'iqr':
        for col in numeric_cols:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            outliers = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            df_clean = df_clean[~outliers]
            
    elif method == 'zscore':
        from scipy import stats
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(df_clean[col].dropna()))
            df_clean = df_clean[z_scores < threshold]
    
    logger.info(f"Removed {len(df) - len(df_clean)} outlier rows")
    return df_clean


def create_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    handle_missing: bool = True,
    encode_categorical: bool = True,
    scale_numeric: bool = True,
    remove_outliers_flag: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Create a complete preprocessing pipeline.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column (if any)
        handle_missing: Whether to handle missing values
        encode_categorical: Whether to encode categorical variables
        scale_numeric: Whether to scale numeric features
        remove_outliers_flag: Whether to remove outliers
        
    Returns:
        Tuple of (preprocessed DataFrame, preprocessing info dictionary)
    """
    pipeline_info = {}
    df_processed = df.copy()
    
    # Separate target if specified
    target = None
    if target_col and target_col in df_processed.columns:
        target = df_processed[target_col].copy()
        df_processed = df_processed.drop(columns=[target_col])
        pipeline_info['target_col'] = target_col
    
    # Detect column types
    column_types = detect_column_types(df_processed)
    pipeline_info['column_types'] = column_types
    
    # Handle missing values
    if handle_missing:
        df_processed = handle_missing_values(df_processed)
        pipeline_info['missing_handled'] = True
    
    # Remove outliers
    if remove_outliers_flag:
        df_processed = remove_outliers(df_processed, column_types['numeric'])
        pipeline_info['outliers_removed'] = True
    
    # Encode categorical variables
    encoders = {}
    if encode_categorical and column_types['categorical']:
        df_processed, encoders = encode_categorical(
            df_processed, 
            column_types['categorical'],
            method='onehot'
        )
        pipeline_info['encoders'] = encoders
    
    # Scale numeric features
    scaler = None
    if scale_numeric and column_types['numeric']:
        # Get remaining numeric columns (some might be dropped during encoding)
        remaining_numeric = [col for col in column_types['numeric'] if col in df_processed.columns]
        if remaining_numeric:
            df_processed, scaler = scale_features(df_processed, remaining_numeric, method='standard')
            pipeline_info['scaler'] = scaler
    
    # Reattach target
    if target is not None:
        df_processed[target_col] = target.values[:len(df_processed)]
    
    pipeline_info['final_shape'] = df_processed.shape
    
    return df_processed, pipeline_info


def detect_target_column(
    df: pd.DataFrame,
    suggested_name: Optional[str] = None
) -> Tuple[Optional[str], str]:
    """
    Automatically detect the target column for ML tasks.
    
    Args:
        df: Input DataFrame
        suggested_name: User-suggested target column name
        
    Returns:
        Tuple of (target column name, problem type)
    """
    if suggested_name and suggested_name in df.columns:
        target_col = suggested_name
    else:
        # Heuristic: look for common target column names
        common_target_names = ['target', 'label', 'y', 'class', 'output', 'prediction', 
                              'result', 'category', 'type', 'grade', 'score', 'value']
        target_col = None
        
        for col in df.columns:
            if col.lower() in common_target_names:
                target_col = col
                break
        
        # If no common name found, use the last column as a fallback
        if target_col is None:
            target_col = df.columns[-1]
            logger.info(f"No obvious target column found. Using last column: {target_col}")
    
    # Determine problem type
    if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 10:
        if df[target_col].nunique() == 2:
            problem_type = 'binary_classification'
        else:
            problem_type = 'multiclass_classification'
    else:
        problem_type = 'regression'
    
    logger.info(f"Detected target: {target_col}, Problem type: {problem_type}")
    
    return target_col, problem_type