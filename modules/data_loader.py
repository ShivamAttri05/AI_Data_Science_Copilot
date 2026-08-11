"""
Data Loading Module for AI Data Science Copilot.

This module handles loading datasets from various file formats
and provides basic dataset information.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    A class to handle data loading operations.
    
    Supports CSV and Excel file formats with automatic
    format detection and error handling.
    """
    
    def __init__(self):
        self.dataset = None
        self.file_name = None
        self.file_type = None
    
    def load_file(
        self,
        file_path: str,
        file_type: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a dataset from a file.
        
        Args:
            file_path: Path to the file
            file_type: File type ('csv', 'excel', 'xlsx', 'xls')
                      If None, will be auto-detected from extension
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        self.file_name = file_path.split('/')[-1] if '/' in file_path else file_path
        
        # Auto-detect file type if not provided
        if file_type is None:
            file_type = self._detect_file_type(self.file_name)
        
        self.file_type = file_type.lower()
        
        try:
            if self.file_type in ['csv']:
                self.dataset = self._load_csv(file_path, **kwargs)
            elif self.file_type in ['excel', 'xlsx', 'xls']:
                self.dataset = self._load_excel(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}. "
                               f"Supported types: csv, excel, xlsx, xls")
            
            logger.info(f"Successfully loaded {self.file_name}: {self.dataset.shape}")
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading file {file_name}: {e}")
            raise
    
    def load_from_upload(
        self,
        uploaded_file,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a dataset from an uploaded file object (e.g., from Streamlit).
        
        Args:
            uploaded_file: File object from upload (e.g., st.file_uploader)
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded DataFrame
        """
        self.file_name = uploaded_file.name
        file_type = self._detect_file_type(self.file_name)
        self.file_type = file_type
        
        try:
            if file_type == 'csv':
                self.dataset = pd.read_csv(uploaded_file, **kwargs)
            elif file_type in ['excel', 'xlsx', 'xls']:
                self.dataset = pd.read_excel(uploaded_file, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            logger.info(f"Successfully loaded {self.file_name}: {self.dataset.shape}")
            return self.dataset
            
        except Exception as e:
            logger.error(f"Error loading uploaded file: {e}")
            raise
    
    def _detect_file_type(self, filename: str) -> str:
        """
        Detect file type from filename extension.
        
        Args:
            filename: Name of the file
            
        Returns:
            Detected file type
        """
        extension = filename.split('.')[-1].lower()
        
        if extension == 'csv':
            return 'csv'
        elif extension in ['xlsx', 'xls', 'xlsm', 'xlsb']:
            return 'excel'
        else:
            return extension
    
    def _load_csv(
        self,
        file_path: str,
        encoding: str = 'utf-8',
        sep: str = ',',
        **kwargs
    ) -> pd.DataFrame:
        """
        Load a CSV file.
        
        Args:
            file_path: Path to CSV file
            encoding: File encoding
            sep: Column separator
            **kwargs: Additional pandas read_csv arguments
            
        Returns:
            Loaded DataFrame
        """
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=sep, **kwargs)
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning(f"UTF-8 encoding failed, trying latin-1")
            df = pd.read_csv(file_path, encoding='latin-1', sep=sep, **kwargs)
        
        return df
    
    def _load_excel(
        self,
        file_path: str,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load an Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Name or index of sheet to load
            **kwargs: Additional pandas read_excel arguments
            
        Returns:
            Loaded DataFrame
        """
        try:
            # Get sheet names if not specified
            if sheet_name is None:
                xl_file = pd.ExcelFile(file_path)
                sheet_name = xl_file.sheet_names[0]
                logger.info(f"Loading first sheet: {sheet_name}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            return df
            
        except Exception as e:
            logger.error(f"Error loading Excel file: {e}")
            raise
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get basic information about the loaded dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please load a dataset first.")
        
        info = {
            'file_name': self.file_name,
            'file_type': self.file_type,
            'n_rows': self.dataset.shape[0],
            'n_columns': self.dataset.shape[1],
            'column_names': self.dataset.columns.tolist(),
            'column_types': self.dataset.dtypes.to_dict(),
            'memory_usage': self._get_memory_usage(),
            'numeric_columns': self.dataset.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': self.dataset.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': self.dataset.select_dtypes(include=['datetime64']).columns.tolist(),
            'missing_values': self.dataset.isnull().sum().to_dict(),
            'missing_percentage': (self.dataset.isnull().sum() / len(self.dataset) * 100).round(2).to_dict(),
            'duplicate_rows': self.dataset.duplicated().sum()
        }
        
        return info
    
    def get_preview(self, n_rows: int = 5) -> pd.DataFrame:
        """
        Get a preview of the dataset.
        
        Args:
            n_rows: Number of rows to show
            
        Returns:
            DataFrame preview
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please load a dataset first.")
        
        return self.dataset.head(n_rows)
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for numeric columns.
        
        Returns:
            DataFrame with summary statistics
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please load a dataset first.")
        
        return self.dataset.describe()
    
    def _get_memory_usage(self) -> str:
        """
        Get formatted memory usage of the dataset.
        
        Returns:
            Formatted memory usage string
        """
        memory_bytes = self.dataset.memory_usage(deep=True).sum()
        
        if memory_bytes < 1024:
            return f"{memory_bytes:.2f} B"
        elif memory_bytes < 1024**2:
            return f"{memory_bytes/1024:.2f} KB"
        elif memory_bytes < 1024**3:
            return f"{memory_bytes/1024**2:.2f} MB"
        else:
            return f"{memory_bytes/1024**3:.2f} GB"
    
    def get_column_info(self, column: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific column.
        
        Args:
            column: Column name
            
        Returns:
            Dictionary with column information
        """
        if self.dataset is None:
            raise ValueError("No dataset loaded. Please load a dataset first.")
        
        if column not in self.dataset.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        col_data = self.dataset[column]
        
        info = {
            'name': column,
            'dtype': str(col_data.dtype),
            'n_unique': col_data.nunique(),
            'n_missing': col_data.isnull().sum(),
            'missing_percentage': round(col_data.isnull().sum() / len(col_data) * 100, 2)
        }
        
        if np.issubdtype(col_data.dtype, np.number):
            info.update({
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std()
            })
        else:
            info.update({
                'most_frequent': col_data.mode()[0] if not col_data.mode().empty else None,
                'top_values': col_data.value_counts().head(5).to_dict()
            })
        
        return info


def load_sample_dataset(name: str = 'iris') -> pd.DataFrame:
    """
    Load a sample dataset for testing.
    
    Args:
        name: Name of sample dataset ('iris', 'wine', 'diabetes', 'boston')
        
    Returns:
        Sample DataFrame
    """
    from sklearn.datasets import load_iris, load_wine, load_diabetes, fetch_california_housing
    
    if name == 'iris':
        data = load_iris(as_frame=True)
    elif name == 'wine':
        data = load_wine(as_frame=True)
    elif name == 'diabetes':
        data = load_diabetes(as_frame=True)
    elif name == 'california':
        data = fetch_california_housing(as_frame=True)
    else:
        raise ValueError(f"Unknown sample dataset: {name}")
    
    df = data.frame
    df['target'] = data.target
    
    logger.info(f"Loaded sample dataset '{name}': {df.shape}")
    return df


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Validate if a DataFrame is suitable for analysis.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check if empty
    if df.empty:
        issues.append("Dataset is empty")
    
    # Check if too small
    if len(df) < 10:
        issues.append(f"Dataset has only {len(df)} rows (minimum recommended: 10)")
    
    # Check for columns with all missing values
    all_missing = df.columns[df.isnull().all()].tolist()
    if all_missing:
        issues.append(f"Columns with all missing values: {all_missing}")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        issues.append(f"Constant columns: {constant_cols}")
    
    # Check for duplicate column names
    if len(df.columns) != len(set(df.columns)):
        issues.append("Dataset has duplicate column names")
    
    is_valid = len(issues) == 0
    
    return is_valid, issues