"""
AI Insights Module for AI Data Science Copilot.

This module uses Google's GenAI SDK to analyze datasets and provide
intelligent insights, feature recommendations, and ML problem identification.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the new Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-genai not installed. AI insights will be disabled. To enable, run: pip install google-genai")


class AIInsightsGenerator:
    """
    A class to generate AI-powered insights for datasets using Gemini API.
    
    Analyzes dataset metadata and provides intelligent recommendations
    for preprocessing, feature engineering, and model selection.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the AI Insights Generator.
        
        Args:
            api_key: Gemini API key (if None, will try to load from environment)
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        self.client = None
        self.model_name = 'gemini-2.5-flash'  # Upgraded to 2.5-flash as default for the new SDK
        self.is_available = False
        
        if GEMINI_AVAILABLE and self.api_key:
            try:
                # Initialize the new Client
                self.client = genai.Client(api_key=self.api_key)
                self.is_available = True
                logger.info("Gemini API configured successfully")
            except Exception as e:
                logger.error(f"Error configuring Gemini API: {e}")
        else:
            if not GEMINI_AVAILABLE:
                logger.warning("Gemini library not available")
            if not self.api_key:
                logger.warning("Gemini API key not provided")
    
    def analyze_dataset(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
        problem_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform AI-powered dataset analysis.
        
        Args:
            df: Input DataFrame
            target_col: Optional target column name
            problem_type: Optional problem type hint
            
        Returns:
            Dictionary with AI-generated insights
        """
        if not self.is_available:
            logger.warning("Gemini API not available. Returning fallback insights.")
            return self._generate_fallback_insights(df, target_col)
        
        # Prepare dataset metadata
        metadata = self._prepare_metadata(df, target_col)
        
        try:
            # Generate insights using Gemini
            insights = self._call_gemini_for_insights(metadata, target_col, problem_type)
            return insights
            
        except Exception as e:
            logger.error(f"Error generating AI insights: {e}")
            return self._generate_fallback_insights(df, target_col)
    
    def _prepare_metadata(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare dataset metadata for AI analysis.
        
        Args:
            df: Input DataFrame
            target_col: Optional target column
            
        Returns:
            Dictionary with dataset metadata
        """
        # Basic info
        metadata = {
            'shape': list(df.shape),
            'n_rows': int(df.shape[0]),
            'n_columns': int(df.shape[1]),
        }
        
        # Column information
        columns_info = []
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'n_unique': int(df[col].nunique()),
                'n_missing': int(df[col].isnull().sum()),
                'missing_pct': round(df[col].isnull().sum() / len(df) * 100, 2)
            }
            
            # Add numeric stats if applicable
            if np.issubdtype(df[col].dtype, np.number):
                col_info.update({
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else None,
                    'skewness': float(df[col].skew()) if not pd.isna(df[col].skew()) else None
                })
            else:
                # Add top values for categorical
                top_values = df[col].value_counts().head(3).to_dict()
                col_info['top_values'] = {str(k): int(v) for k, v in top_values.items()}
            
            columns_info.append(col_info)
        
        metadata['columns'] = columns_info
        
        # Target information
        if target_col and target_col in df.columns:
            target_series = df[target_col]
            metadata['target'] = {
                'name': target_col,
                'dtype': str(target_series.dtype),
                'n_unique': int(target_series.nunique()),
                'unique_ratio': round(target_series.nunique() / len(df), 4)
            }
            
            # Determine if classification or regression
            if target_series.dtype in ['object', 'category'] or target_series.nunique() < 10:
                metadata['target']['problem_type'] = 'classification'
                metadata['target']['class_distribution'] = target_series.value_counts().to_dict()
            else:
                metadata['target']['problem_type'] = 'regression'
                metadata['target']['statistics'] = {
                    'mean': float(target_series.mean()),
                    'std': float(target_series.std()),
                    'min': float(target_series.min()),
                    'max': float(target_series.max())
                }
        
        # Correlation info for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr().abs()
            # Get top correlated pairs
            high_corr = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if corr_val >= 0.7:
                        high_corr.append({
                            'feature_1': numeric_cols[i],
                            'feature_2': numeric_cols[j],
                            'correlation': round(float(corr_val), 3)
                        })
            metadata['high_correlations'] = high_corr[:5]  # Top 5
        
        return metadata
    
    def _call_gemini_for_insights(
        self,
        metadata: Dict[str, Any],
        target_col: Optional[str],
        problem_type: Optional[str]
    ) -> Dict[str, Any]:
        """
        Call Gemini API to generate insights.
        
        Args:
            metadata: Dataset metadata
            target_col: Target column name
            problem_type: Problem type hint
            
        Returns:
            Dictionary with AI-generated insights
        """
        # Construct the prompt
        prompt = self._construct_prompt(metadata, target_col, problem_type)
        
        # Call Gemini using the new SDK pattern
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            )
        )
        
        # Parse the response
        try:
            # Try to extract JSON from response
            response_text = response.text
            
            # Find JSON in the response (it might be wrapped in markdown)
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                json_str = response_text.split('```')[1].split('```')[0]
            else:
                json_str = response_text
            
            insights = json.loads(json_str.strip())
            
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Could not parse JSON from Gemini response: {e}")
            # Return the raw text as description
            insights = {
                'raw_insights': response.text,
                'parsing_error': str(e)
            }
        
        return insights
    
    def _construct_prompt(
        self,
        metadata: Dict[str, Any],
        target_col: Optional[str],
        problem_type: Optional[str]
    ) -> str:
        """
        Construct the prompt for Gemini API.
        
        Args:
            metadata: Dataset metadata
            target_col: Target column name
            problem_type: Problem type hint
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are an expert data scientist analyzing a dataset. Please analyze the following dataset metadata and provide insights in JSON format.

DATASET OVERVIEW:
- Rows: {metadata['n_rows']:,}
- Columns: {metadata['n_columns']}
"""
        
        if target_col:
            prompt += f"\nTARGET COLUMN: {target_col}\n"
            if problem_type:
                prompt += f"PROBLEM TYPE: {problem_type}\n"
        
        prompt += "\nCOLUMN DETAILS:\n"
        for col in metadata['columns'][:15]:  # Limit to 15 columns
            prompt += f"\n- {col['name']} ({col['dtype']}):\n"
            prompt += f"  - Unique values: {col['n_unique']}\n"
            prompt += f"  - Missing: {col['missing_pct']}%\n"
            
            if 'mean' in col:
                prompt += f"  - Mean: {col['mean']:.2f}, Std: {col['std']:.2f}\n"
            if 'top_values' in col:
                prompt += f"  - Top values: {list(col['top_values'].keys())[:3]}\n"
        
        if metadata.get('high_correlations'):
            prompt += "\nHIGH CORRELATIONS:\n"
            for corr in metadata['high_correlations']:
                prompt += f"- {corr['feature_1']} vs {corr['feature_2']}: {corr['correlation']}\n"
        
        prompt += """

Please provide your analysis in the following JSON format:

{
    "dataset_overview": {
        "description": "Brief description of the dataset",
        "size_assessment": "Assessment of dataset size adequacy"
    },
    "important_features": [
        {"feature": "feature_name", "importance": "high/medium/low", "reason": "why this feature is important"}
    ],
    "potential_target_columns": [
        {"column": "column_name", "suitability": "explanation of why this could be a target"}
    ],
    "data_quality_issues": [
        {"issue": "description", "severity": "high/medium/low", "recommendation": "how to fix"}
    ],
    "recommended_preprocessing": [
        {"step": "preprocessing step", "reason": "why this is needed", "priority": "high/medium/low"}
    ],
    "ml_problem_type": {
        "suggested_type": "classification/regression/clustering/etc",
        "confidence": "high/medium/low",
        "reasoning": "explanation"
    },
    "feature_engineering_suggestions": [
        {"suggestion": "feature engineering idea", "expected_impact": "high/medium/low"}
    ],
    "model_recommendations": [
        {"model": "model name", "reason": "why this model is suitable", "priority": 1}
    ],
    "key_insights": [
        "insight 1",
        "insight 2",
        "insight 3"
    ]
}

Be specific, actionable, and data-driven in your analysis. Focus on practical recommendations for a data science workflow.
"""
        
        return prompt
    
    def _generate_fallback_insights(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate fallback insights when Gemini API is not available.
        
        Args:
            df: Input DataFrame
            target_col: Optional target column
            
        Returns:
            Dictionary with rule-based insights
        """
        insights = {
            'dataset_overview': {
                'description': f'Dataset with {df.shape[0]} rows and {df.shape[1]} columns',
                'size_assessment': 'Adequate' if df.shape[0] > 1000 else 'Small but usable'
            },
            'important_features': [],
            'potential_target_columns': [],
            'data_quality_issues': [],
            'recommended_preprocessing': [],
            'ml_problem_type': {},
            'feature_engineering_suggestions': [],
            'model_recommendations': [],
            'key_insights': [],
            'note': 'AI insights unavailable. Using rule-based fallback.'
        }
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().sum() > 0].tolist()
        if missing_cols:
            insights['data_quality_issues'].append({
                'issue': f'Missing values in columns: {missing_cols}',
                'severity': 'medium',
                'recommendation': 'Consider imputation strategies'
            })
            insights['recommended_preprocessing'].append({
                'step': 'Handle missing values',
                'reason': f'{len(missing_cols)} columns have missing values',
                'priority': 'high'
            })
        
        # Check for duplicates
        if df.duplicated().sum() > 0:
            insights['data_quality_issues'].append({
                'issue': f"{df.duplicated().sum()} duplicate rows detected",
                'severity': 'low',
                'recommendation': 'Consider removing duplicates'
            })
        
        # Suggest important features (high variance numeric columns)
        if numeric_cols:
            variances = df[numeric_cols].var().sort_values(ascending=False)
            top_features = variances.head(3).index.tolist()
            for feat in top_features:
                insights['important_features'].append({
                    'feature': feat,
                    'importance': 'high',
                    'reason': 'High variance suggests informative feature'
                })
        
        # Suggest potential target columns
        if target_col:
            insights['potential_target_columns'].append({
                'column': target_col,
                'suitability': 'User-specified target column'
            })
        else:
            # Suggest last column or low-cardinality columns
            for col in df.columns:
                if df[col].nunique() < 20 and col in numeric_cols:
                    insights['potential_target_columns'].append({
                        'column': col,
                        'suitability': f'Low cardinality ({df[col].nunique()} unique values)'
                    })
        
        # Determine problem type
        if target_col and target_col in df.columns:
            target_series = df[target_col]
            if target_series.dtype in ['object', 'category'] or target_series.nunique() < 10:
                insights['ml_problem_type'] = {
                    'suggested_type': 'classification',
                    'confidence': 'high',
                    'reasoning': 'Target has discrete values'
                }
                insights['model_recommendations'] = [
                    {'model': 'Random Forest', 'reason': 'Good baseline for classification', 'priority': 1},
                    {'model': 'XGBoost', 'reason': 'High performance classifier', 'priority': 2},
                    {'model': 'Logistic Regression', 'reason': 'Simple and interpretable', 'priority': 3}
                ]
            else:
                insights['ml_problem_type'] = {
                    'suggested_type': 'regression',
                    'confidence': 'high',
                    'reasoning': 'Target has continuous values'
                }
                insights['model_recommendations'] = [
                    {'model': 'Random Forest Regressor', 'reason': 'Good baseline for regression', 'priority': 1},
                    {'model': 'XGBoost Regressor', 'reason': 'High performance regressor', 'priority': 2},
                    {'model': 'Linear Regression', 'reason': 'Simple and interpretable', 'priority': 3}
                ]
        
        # Feature engineering suggestions
        if categorical_cols:
            insights['feature_engineering_suggestions'].append({
                'suggestion': 'Encode categorical variables',
                'expected_impact': 'high'
            })
        
        if len(numeric_cols) > 1:
            insights['feature_engineering_suggestions'].append({
                'suggestion': 'Create interaction features between numeric columns',
                'expected_impact': 'medium'
            })
        
        insights['key_insights'] = [
            f'Dataset contains {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features',
            f'{len(missing_cols)} columns have missing values' if missing_cols else 'No missing values detected',
            'Consider feature scaling for numeric columns'
        ]
        
        return insights
    
    def suggest_feature_importance(
        self,
        df: pd.DataFrame,
        target_col: str
    ) -> List[Dict[str, Any]]:
        """
        Suggest potentially important features based on statistical relationships.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            List of feature importance suggestions
        """
        if target_col not in df.columns:
            return []
        
        suggestions = []
        target_series = df[target_col]
        
        # Determine if classification or regression
        is_classification = target_series.dtype in ['object', 'category'] or target_series.nunique() < 10
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != target_col]
        
        if is_classification:
            # Use ANOVA F-value for classification
            from sklearn.feature_selection import f_classif
            from sklearn.preprocessing import LabelEncoder
            
            # Prepare target
            le = LabelEncoder()
            y = le.fit_transform(target_series.astype(str))
            
            # Prepare features (handle missing values temporarily)
            X = df[numeric_cols].fillna(df[numeric_cols].median())
            
            try:
                f_values, p_values = f_classif(X, y)
                
                for i, col in enumerate(numeric_cols):
                    suggestions.append({
                        'feature': col,
                        'f_score': float(f_values[i]),
                        'p_value': float(p_values[i]),
                        'importance': 'high' if p_values[i] < 0.001 else 'medium' if p_values[i] < 0.05 else 'low'
                    })
                
                # Sort by F-score
                suggestions.sort(key=lambda x: x['f_score'], reverse=True)
                
            except Exception as e:
                logger.warning(f"Error calculating feature importance: {e}")
        
        else:
            # Use correlation for regression
            correlations = df[numeric_cols].corrwith(target_series).abs().sort_values(ascending=False)
            
            for col, corr in correlations.head(10).items():
                suggestions.append({
                    'feature': col,
                    'correlation': float(corr),
                    'importance': 'high' if corr > 0.7 else 'medium' if corr > 0.4 else 'low'
                })
        
        return suggestions
    
    def generate_preprocessing_pipeline(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a recommended preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Optional target column
            
        Returns:
            Dictionary with preprocessing recommendations
        """
        pipeline = {
            'steps': [],
            'notes': []
        }
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().sum() > 0].tolist()
        if missing_cols:
            pipeline['steps'].append({
                'name': 'Handle Missing Values',
                'description': f'Impute missing values in {len(missing_cols)} columns',
                'strategy': 'median for numeric, mode for categorical',
                'priority': 'high'
            })
        
        # Check for categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_col:
            categorical_cols = [c for c in categorical_cols if c != target_col]
        
        if categorical_cols:
            pipeline['steps'].append({
                'name': 'Encode Categorical Variables',
                'description': f'Encode {len(categorical_cols)} categorical columns',
                'strategy': 'One-hot encoding for low cardinality, target encoding for high cardinality',
                'priority': 'high'
            })
        
        # Check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col:
            numeric_cols = [c for c in numeric_cols if c != target_col]
        
        if numeric_cols:
            pipeline['steps'].append({
                'name': 'Scale Numeric Features',
                'description': f'Scale {len(numeric_cols)} numeric columns',
                'strategy': 'StandardScaler (zero mean, unit variance)',
                'priority': 'medium'
            })
        
        # Check for high cardinality
        high_cardinality = [c for c in categorical_cols if df[c].nunique() > 50]
        if high_cardinality:
            pipeline['notes'].append(f"High cardinality columns detected: {high_cardinality}. Consider target encoding or grouping rare categories.")
        
        # Check for potential outliers
        if numeric_cols:
            pipeline['steps'].append({
                'name': 'Outlier Treatment (Optional)',
                'description': 'Consider handling extreme values',
                'strategy': 'IQR method or winsorization',
                'priority': 'low'
            })
        
        return pipeline


def get_ai_insights(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    target_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to get AI insights for a dataset.
    
    Args:
        df: Input DataFrame
        api_key: Gemini API key (optional)
        target_col: Target column name (optional)
        
    Returns:
        Dictionary with AI-generated insights
    """
    generator = AIInsightsGenerator(api_key=api_key)
    return generator.analyze_dataset(df, target_col)