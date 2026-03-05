"""
AI Data Science Copilot - Modules Package

This package contains all the core modules for the AI Data Science Copilot.
"""

from .data_loader import DataLoader, load_sample_dataset, validate_dataset
from .data_quality import DataQualityChecker, quick_quality_check
from .eda_engine import EDAEngine, quick_profile
from .ai_insights import AIInsightsGenerator, get_ai_insights
from .automl_engine import AutoMLEngine, quick_automl
from .experiment_analysis import ExperimentAnalyzer, quick_analyze
from .model_deployment import ModelDeployer, deploy_model

__all__ = [
    'DataLoader',
    'load_sample_dataset',
    'validate_dataset',
    'DataQualityChecker',
    'quick_quality_check',
    'EDAEngine',
    'quick_profile',
    'AIInsightsGenerator',
    'get_ai_insights',
    'AutoMLEngine',
    'quick_automl',
    'ExperimentAnalyzer',
    'quick_analyze',
    'ModelDeployer',
    'deploy_model'
]