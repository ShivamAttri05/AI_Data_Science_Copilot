"""
AI Data Science Copilot - Utils Package

This package contains utility functions for the AI Data Science Copilot.
"""

from .preprocessing import (
    detect_column_types,
    handle_missing_values,
    encode_categorical,
    scale_features,
    remove_outliers,
    create_preprocessing_pipeline,
    detect_target_column
)

from .visualization import (
    create_correlation_heatmap,
    create_distribution_plot,
    create_missing_values_heatmap,
    create_pairplot,
    create_categorical_barplot,
    create_boxplot,
    create_model_comparison_chart,
    create_feature_importance_plot,
    create_confusion_matrix_plot,
    create_roc_curve_plot,
    create_prediction_vs_actual_plot,
    create_residuals_plot
)

from .helpers import (
    generate_dataset_id,
    save_json,
    load_json,
    save_pickle,
    load_pickle,
    format_number,
    get_memory_usage,
    create_directory_structure,
    truncate_string,
    get_file_extension,
    is_valid_file_type,
    get_current_timestamp,
    get_dataframe_info,
    clean_column_names,
    sample_dataframe,
    make_serializable,
    ProgressTracker
)

__all__ = [
    # Preprocessing
    'detect_column_types',
    'handle_missing_values',
    'encode_categorical',
    'scale_features',
    'remove_outliers',
    'create_preprocessing_pipeline',
    'detect_target_column',
    # Visualization
    'create_correlation_heatmap',
    'create_distribution_plot',
    'create_missing_values_heatmap',
    'create_pairplot',
    'create_categorical_barplot',
    'create_boxplot',
    'create_model_comparison_chart',
    'create_feature_importance_plot',
    'create_confusion_matrix_plot',
    'create_roc_curve_plot',
    'create_prediction_vs_actual_plot',
    'create_residuals_plot',
    # Helpers
    'generate_dataset_id',
    'save_json',
    'load_json',
    'save_pickle',
    'load_pickle',
    'format_number',
    'get_memory_usage',
    'create_directory_structure',
    'truncate_string',
    'get_file_extension',
    'is_valid_file_type',
    'get_current_timestamp',
    'get_dataframe_info',
    'clean_column_names',
    'sample_dataframe',
    'make_serializable',
    'ProgressTracker'
]