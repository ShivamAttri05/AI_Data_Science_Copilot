"""
Helper utilities for the AI Data Science Copilot.

This module provides general-purpose helper functions used
across the application.
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from typing import Any, Dict, Optional, List
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_dataset_id(df: pd.DataFrame) -> str:
    """
    Generate a unique identifier for a dataset based on its content.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Unique hash string
    """
    # Create a hash based on column names and shape
    content = f"{list(df.columns)}_{df.shape}_{df.head(1).values.tobytes()}"
    return hashlib.md5(content.encode()).hexdigest()[:12]


def save_json(data: Dict, filepath: str) -> bool:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        return False


def load_json(filepath: str) -> Optional[Dict]:
    """
    Load dictionary from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary or None if error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return None


def save_pickle(obj: Any, filepath: str) -> bool:
    """
    Save object to pickle file.
    
    Args:
        obj: Object to save
        filepath: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Saved pickle to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving pickle: {e}")
        return False


def load_pickle(filepath: str) -> Optional[Any]:
    """
    Load object from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Object or None if error
    """
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading pickle: {e}")
        return None


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format a number for display.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    if pd.isna(num):
        return "N/A"
    
    if abs(num) >= 1e9:
        return f"{num/1e9:.{decimals}f}B"
    elif abs(num) >= 1e6:
        return f"{num/1e6:.{decimals}f}M"
    elif abs(num) >= 1e3:
        return f"{num/1e3:.{decimals}f}K"
    else:
        return f"{num:.{decimals}f}"


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get memory usage of a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Formatted memory usage string
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    
    if memory_bytes < 1024:
        return f"{memory_bytes} B"
    elif memory_bytes < 1024**2:
        return f"{memory_bytes/1024:.2f} KB"
    elif memory_bytes < 1024**3:
        return f"{memory_bytes/1024**2:.2f} MB"
    else:
        return f"{memory_bytes/1024**3:.2f} GB"


def create_directory_structure(base_path: str) -> Dict[str, str]:
    """
    Create standard directory structure for the project.
    
    Args:
        base_path: Base directory path
        
    Returns:
        Dictionary of created directory paths
    """
    directories = {
        'models': os.path.join(base_path, 'saved_models'),
        'exports': os.path.join(base_path, 'exports'),
        'logs': os.path.join(base_path, 'logs'),
        'temp': os.path.join(base_path, 'temp'),
        'api': os.path.join(base_path, 'api')
    }
    
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return directories


def validate_email(email: str) -> bool:
    """
    Simple email validation.
    
    Args:
        email: Email string to validate
        
    Returns:
        True if valid, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def truncate_string(s: str, max_length: int = 50, suffix: str = '...') -> str:
    """
    Truncate a string to maximum length.
    
    Args:
        s: Input string
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Filename string
        
    Returns:
        File extension (lowercase)
    """
    return os.path.splitext(filename)[1].lower()


def is_valid_file_type(filename: str, allowed_extensions: List[str]) -> bool:
    """
    Check if file has valid extension.
    
    Args:
        filename: Filename to check
        allowed_extensions: List of allowed extensions
        
    Returns:
        True if valid, False otherwise
    """
    ext = get_file_extension(filename)
    return ext in allowed_extensions


def timestamp_to_datetime(timestamp: float) -> datetime:
    """
    Convert Unix timestamp to datetime.
    
    Args:
        timestamp: Unix timestamp
        
    Returns:
        Datetime object
    """
    return datetime.fromtimestamp(timestamp)


def get_current_timestamp() -> str:
    """
    Get current timestamp as formatted string.
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division by zero
        
    Returns:
        Division result or default value
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ValueError):
        return default


def convert_to_numeric(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Convert columns to numeric type where possible.
    
    Args:
        df: Input DataFrame
        columns: List of columns to convert (None for all)
        
    Returns:
        DataFrame with converted columns
    """
    df_converted = df.copy()
    
    if columns is None:
        columns = df_converted.columns
    
    for col in columns:
        if col in df_converted.columns:
            df_converted[col] = pd.to_numeric(df_converted[col], errors='ignore')
    
    return df_converted


def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with DataFrame information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': get_memory_usage(df),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'n_duplicates': df.duplicated().sum()
    }
    
    return info


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame column names by removing special characters and spaces.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned column names
    """
    import re
    
    def clean_name(name: str) -> str:
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^\w\s]', '_', str(name))
        name = re.sub(r'\s+', '_', name)
        # Remove multiple consecutive underscores
        name = re.sub(r'_+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Convert to lowercase
        return name.lower()
    
    df_clean = df.copy()
    df_clean.columns = [clean_name(col) for col in df_clean.columns]
    
    return df_clean


def sample_dataframe(df: pd.DataFrame, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Sample DataFrame if it's too large.
    
    Args:
        df: Input DataFrame
        n_samples: Maximum number of samples
        random_state: Random seed
        
    Returns:
        Sampled DataFrame
    """
    if len(df) > n_samples:
        logger.info(f"Sampling DataFrame from {len(df)} to {n_samples} rows")
        return df.sample(n=n_samples, random_state=random_state)
    return df


def estimate_reading_time(text: str, words_per_minute: int = 200) -> int:
    """
    Estimate reading time for text.
    
    Args:
        text: Text to estimate
        words_per_minute: Average reading speed
        
    Returns:
        Estimated reading time in minutes
    """
    word_count = len(text.split())
    return max(1, round(word_count / words_per_minute))


class ProgressTracker:
    """
    Simple progress tracker for long-running operations.
    """
    
    def __init__(self, total_steps: int, description: str = "Processing"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, step: int = 1):
        """Update progress by specified number of steps."""
        self.current_step += step
        progress = (self.current_step / self.total_steps) * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        logger.info(f"{self.description}: {progress:.1f}% complete ({self.current_step}/{self.total_steps}) - {elapsed:.1f}s elapsed")
    
    def get_progress(self) -> float:
        """Get current progress percentage."""
        return (self.current_step / self.total_steps) * 100
    
    def is_complete(self) -> bool:
        """Check if progress is complete."""
        return self.current_step >= self.total_steps


def make_serializable(obj: Any) -> Any:
    """
    Convert object to JSON-serializable format.
    
    Args:
        obj: Object to convert
        
    Returns:
        Serializable object
    """
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    else:
        return obj