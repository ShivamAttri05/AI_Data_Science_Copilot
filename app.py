"""
AI Data Science Copilot - Main Streamlit Application

A comprehensive web application for automated data science workflows
including data loading, quality checks, EDA, AI insights, AutoML,
and model deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import os
import sys
from datetime import datetime

# Add modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from modules.data_loader import DataLoader, validate_dataset
from modules.data_quality import DataQualityChecker
from modules.eda_engine import EDAEngine
from modules.ai_insights import AIInsightsGenerator
from modules.automl_engine import AutoMLEngine
from modules.experiment_analysis import ExperimentAnalyzer
from modules.model_deployment import ModelDeployer

# Import utilities
from utils.helpers import get_dataframe_info, create_directory_structure

# Page configuration
st.set_page_config(
    page_title="AI Data Science Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #3498db;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #dc3545;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'quality_report' not in st.session_state:
    st.session_state.quality_report = None
if 'eda_results' not in st.session_state:
    st.session_state.eda_results = None
if 'ai_insights' not in st.session_state:
    st.session_state.ai_insights = None
if 'automl_results' not in st.session_state:
    st.session_state.automl_results = None
if 'target_col' not in st.session_state:
    st.session_state.target_col = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None


def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-header">🤖 AI Data Science Copilot</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #6c757d; font-size: 1.1rem;">
        Your intelligent assistant for end-to-end data science workflows
    </p>
    """, unsafe_allow_html=True)
    st.divider()


def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.title("📊 Navigation")
        
        page = st.radio(
            "Select a section:",
            [
                "🏠 Home",
                "📁 Dataset Upload",
                "🔍 Data Quality",
                "📈 EDA Dashboard",
                "🧠 AI Insights",
                "🤖 AutoML Training",
                "📊 Model Evaluation",
                "🚀 Deployment"
            ]
        )
        
        st.divider()
        
        # Dataset info if loaded
        if st.session_state.data_loaded and st.session_state.df is not None:
            st.subheader("📋 Dataset Info")
            st.write(f"**Rows:** {st.session_state.df.shape[0]:,}")
            st.write(f"**Columns:** {st.session_state.df.shape[1]}")
            
            if st.session_state.target_col:
                st.write(f"**Target:** {st.session_state.target_col}")
            
            if st.session_state.problem_type:
                st.write(f"**Problem:** {st.session_state.problem_type}")
        
        st.divider()
        
        # About
        st.markdown("""
        **About**
        
        AI Data Science Copilot helps you go from raw data to 
        deployed ML models with minimal coding.
        
        Built with ❤️ using Streamlit
        """)
        
        return page


def render_home():
    """Render the home page."""
    st.markdown('<h2 class="section-header">Welcome to AI Data Science Copilot</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">📁</div>
            <div class="metric-label"><strong>Upload Data</strong></div>
            <p style="font-size: 0.85rem; margin-top: 0.5rem;">
                Support for CSV and Excel files
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🤖</div>
            <div class="metric-label"><strong>AutoML</strong></div>
            <p style="font-size: 0.85rem; margin-top: 0.5rem;">
                Automated model training & selection
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🚀</div>
            <div class="metric-label"><strong>Deploy</strong></div>
            <p style="font-size: 0.85rem; margin-top: 0.5rem;">
                Export models & create APIs
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('<h3 class="section-header">Workflow Pipeline</h3>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <ol>
        <li><strong>📁 Dataset Upload</strong> - Load your CSV or Excel files</li>
        <li><strong>🔍 Data Quality Check</strong> - Automatic detection of issues</li>
        <li><strong>📈 Exploratory Data Analysis</strong> - Visualize and understand your data</li>
        <li><strong>🧠 AI Dataset Investigation</strong> - Get intelligent insights from Gemini</li>
        <li><strong>🤖 AutoML Model Training</strong> - Train multiple models automatically</li>
        <li><strong>📊 Model Evaluation</strong> - Compare and analyze model performance</li>
        <li><strong>🚀 Model Deployment</strong> - Export and deploy your best model</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start
    st.markdown('<h3 class="section-header">Quick Start</h3>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.info("👈 Start by uploading your dataset in the **Dataset Upload** section")
    else:
        st.success("✅ Dataset loaded! Continue to the next sections to analyze and model your data.")


def render_dataset_upload():
    """Render the dataset upload page."""
    st.markdown('<h2 class="section-header">📁 Dataset Upload</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Upload a CSV or Excel file to get started"
    )
    
    # Sample datasets
    st.markdown("Or try a sample dataset:")
    col1, col2, col3 = st.columns(3)
    
    sample_dataset = None
    with col1:
        if st.button("🌸 Iris Dataset"):
            sample_dataset = 'iris'
    with col2:
        if st.button("🍷 Wine Dataset"):
            sample_dataset = 'wine'
    with col3:
        if st.button("🏠 California Housing"):
            sample_dataset = 'california'
    
    # Load data
    if uploaded_file is not None or sample_dataset:
        try:
            loader = DataLoader()
            
            if uploaded_file:
                df = loader.load_from_upload(uploaded_file)
                st.success(f"✅ Successfully loaded **{uploaded_file.name}**")
            else:
                from modules.data_loader import load_sample_dataset
                df = load_sample_dataset(sample_dataset)
                st.success(f"✅ Loaded sample dataset: **{sample_dataset}**")
            
            # Validate dataset
            is_valid, issues = validate_dataset(df)
            
            if not is_valid:
                st.warning("⚠️ Dataset validation issues found:")
                for issue in issues:
                    st.write(f"- {issue}")
            
            # Store in session state
            st.session_state.df = df
            st.session_state.data_loaded = True
            
            # Display dataset info
            st.markdown('<h3 class="section-header">Dataset Overview</h3>', unsafe_allow_html=True)
            
            info = loader.get_dataset_info()
            
            # Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            with mcol1:
                st.metric("Rows", f"{info['n_rows']:,}")
            with mcol2:
                st.metric("Columns", info['n_columns'])
            with mcol3:
                st.metric("Numeric", len(info['numeric_columns']))
            with mcol4:
                st.metric("Categorical", len(info['categorical_columns']))
            
            # Preview
            st.markdown("#### Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Column information
            st.markdown("#### Column Information")
            col_info = pd.DataFrame({
                'Column': info['column_names'],
                'Type': [str(info['column_types'][col]) for col in info['column_names']],
                'Missing': [info['missing_values'][col] for col in info['column_names']],
                'Missing %': [info['missing_percentage'][col] for col in info['column_names']]
            })
            st.dataframe(col_info, use_container_width=True)
            
            # Target selection
            st.markdown("#### Target Column Selection (Optional)")
            target_col = st.selectbox(
                "Select target column (if applicable)",
                ['None'] + list(df.columns),
                help="Select the column you want to predict"
            )
            
            if target_col != 'None':
                st.session_state.target_col = target_col
                
                # Auto-detect problem type
                if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() < 10:
                    if df[target_col].nunique() == 2:
                        st.session_state.problem_type = 'binary_classification'
                    else:
                        st.session_state.problem_type = 'multiclass_classification'
                else:
                    st.session_state.problem_type = 'regression'
                
                st.info(f"Detected problem type: **{st.session_state.problem_type}**")
            
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")


def render_data_quality():
    """Render the data quality page."""
    st.markdown('<h2 class="section-header">🔍 Data Quality Report</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload a dataset first")
        return
    
    df = st.session_state.df
    
    # Run quality check
    if st.button("🔍 Run Data Quality Check", type="primary"):
        with st.spinner("Analyzing data quality..."):
            checker = DataQualityChecker(df)
            report = checker.run_all_checks(st.session_state.target_col)
            st.session_state.quality_report = report
    
    # Display report if available
    if st.session_state.quality_report:
        report = st.session_state.quality_report
        
        # Overview
        st.markdown("#### Overview")
        overview = report['overview']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", f"{overview['n_rows']:,}")
        with col2:
            st.metric("Columns", overview['n_columns'])
        with col3:
            st.metric("Memory (MB)", overview['memory_usage_mb'])
        with col4:
            st.metric("Issues", len(report['issues']))
        
        # Issues and warnings
        if report['issues']:
            st.markdown("#### ⚠️ Critical Issues")
            for issue in report['issues']:
                st.error(issue)
        
        if report['warnings']:
            st.markdown("#### ⚡ Warnings")
            for warning in report['warnings']:
                st.warning(warning)
        
        # Detailed sections
        tabs = st.tabs([
            "Missing Values",
            "Duplicates",
            "Outliers",
            "Constant Columns",
            "Recommendations"
        ])
        
        with tabs[0]:
            missing = report['missing_values']
            st.write(f"**Total Missing:** {missing['total_missing']:,} ({missing['total_missing_percentage']}%)")
            
            if missing['columns_with_missing'] > 0:
                missing_df = pd.DataFrame([
                    {'Column': col, 'Count': data['count'], 'Percentage': data['percentage']}
                    for col, data in missing['missing_by_column'].items()
                ]).sort_values('Percentage', ascending=False)
                
                st.dataframe(missing_df, use_container_width=True)
                
                # Visualization
                fig = px.bar(
                    missing_df.head(15),
                    x='Column',
                    y='Percentage',
                    title='Missing Values by Column (%)'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        with tabs[1]:
            duplicates = report['duplicates']
            st.write(f"**Duplicate Rows:** {duplicates['n_duplicate_rows']:,} ({duplicates['percentage']}%)")
            
            if duplicates['n_duplicate_rows'] > 0:
                st.warning("Consider removing duplicate rows before training")
            else:
                st.success("✅ No duplicate rows found!")
        
        with tabs[2]:
            outliers = report['outliers']
            st.write(f"**Columns with Outliers:** {outliers['columns_with_outliers']}")
            st.write(f"**Method:** {outliers['method'].upper()}")
            
            if outliers['outliers_by_column']:
                outlier_df = pd.DataFrame([
                    {'Column': col, 'Count': data['count'], 'Percentage': data['percentage']}
                    for col, data in outliers['outliers_by_column'].items()
                ]).sort_values('Percentage', ascending=False)
                
                st.dataframe(outlier_df, use_container_width=True)
            else:
                st.success("✅ No significant outliers detected!")
        
        with tabs[3]:
            constant = report['constant_columns']
            
            if constant['n_constant'] > 0:
                st.error(f"**Constant Columns:** {constant['n_constant']}")
                for col_data in constant['constant_columns']:
                    st.write(f"- {col_data['column']}")
            else:
                st.success("✅ No constant columns found!")
            
            if constant['n_near_constant'] > 0:
                st.warning(f"**Near-Constant Columns:** {constant['n_near_constant']}")
        
        with tabs[4]:
            st.markdown("#### 💡 Recommendations")
            for rec in report['recommendations']:
                st.info(rec)
    else:
        st.info("Click 'Run Data Quality Check' to analyze your dataset")


def render_eda():
    """Render the EDA dashboard page."""
    st.markdown('<h2 class="section-header">📈 EDA Dashboard</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload a dataset first")
        return
    
    df = st.session_state.df
    
    # Run EDA
    if st.button("📊 Run EDA Analysis", type="primary"):
        with st.spinner("Running exploratory data analysis..."):
            engine = EDAEngine(df)
            results = engine.run_full_analysis(st.session_state.target_col)
            st.session_state.eda_results = results
            st.session_state.eda_engine = engine
    
    # Display EDA results
    if st.session_state.eda_results:
        results = st.session_state.eda_results
        engine = st.session_state.eda_engine
        
        tabs = st.tabs([
            "Overview",
            "Statistics",
            "Correlations",
            "Distributions",
            "Categorical"
        ])
        
        with tabs[0]:
            overview = results['overview']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{overview['shape'][0]:,}")
            with col2:
                st.metric("Columns", overview['shape'][1])
            with col3:
                st.metric("Missing %", overview['missing_percentage'])
            
            st.markdown("#### Column Types")
            col_data = {
                'Type': ['Numeric', 'Categorical', 'Datetime'],
                'Count': [overview['n_numeric'], overview['n_categorical'], overview['n_datetime']]
            }
            st.bar_chart(pd.DataFrame(col_data).set_index('Type'))
        
        with tabs[1]:
            stats = results['summary_statistics']
            
            if 'numeric' in stats:
                st.markdown("#### Numeric Statistics")
                st.dataframe(stats['numeric'], use_container_width=True)
            
            if 'categorical' in stats:
                st.markdown("#### Categorical Statistics")
                st.dataframe(stats['categorical'], use_container_width=True)
        
        with tabs[2]:
            corr = results['correlation_analysis']
            
            if 'pearson_correlation' in corr:
                st.markdown("#### Correlation Heatmap")
                fig = engine.generate_correlation_heatmap(use_plotly=True)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            if 'high_correlation_pairs' in corr and corr['high_correlation_pairs']:
                st.markdown("#### Highly Correlated Pairs")
                corr_df = pd.DataFrame(corr['high_correlation_pairs'])
                st.dataframe(corr_df, use_container_width=True)
        
        with tabs[3]:
            st.markdown("#### Feature Distributions")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
            
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column to visualize", numeric_cols)
                
                fig = engine.generate_distribution_plots([selected_col], use_plotly=True)
                if fig:
                    st.plotly_chart(fig[0][1], use_container_width=True)
                
                # Distribution statistics
                dist_stats = results['distributions']['distribution_stats'].get(selected_col, {})
                if dist_stats:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean", f"{dist_stats.get('mean', 0):.2f}")
                    with col2:
                        st.metric("Std", f"{dist_stats.get('std', 0):.2f}")
                    with col3:
                        st.metric("Skewness", f"{dist_stats.get('skewness', 0):.2f}")
                    with col4:
                        st.metric("Normal?", "Yes" if dist_stats.get('is_normal') else "No")
            else:
                st.info("No numeric columns available for distribution analysis")
        
        with tabs[4]:
            st.markdown("#### Categorical Analysis")
            
            cat_cols = df.select_dtypes(include=['object', 'category']).columns[:10]
            
            if len(cat_cols) > 0:
                selected_cat = st.selectbox("Select categorical column", cat_cols)
                
                value_counts = df[selected_cat].value_counts().head(15)
                fig = px.bar(
                    x=value_counts.values,
                    y=value_counts.index,
                    orientation='h',
                    title=f'Top Categories in {selected_cat}'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No categorical columns available")
    else:
        st.info("Click 'Run EDA Analysis' to explore your dataset")


def render_ai_insights():
    """Render the AI insights page."""
    st.markdown('<h2 class="section-header">🧠 AI Dataset Investigation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload a dataset first")
        return
    
    df = st.session_state.df
    
    # API Key input
    api_key = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        help="Enter your Gemini API key for AI-powered insights. Leave blank to use environment variable."
    )
    
    # Generate insights
    if st.button("🧠 Generate AI Insights", type="primary"):
        with st.spinner("Analyzing dataset with AI..."):
            try:
                generator = AIInsightsGenerator(api_key=api_key if api_key else None)
                insights = generator.analyze_dataset(
                    df,
                    st.session_state.target_col,
                    st.session_state.problem_type
                )
                st.session_state.ai_insights = insights
            except Exception as e:
                st.error(f"Error generating insights: {e}")
    
    # Display insights
    if st.session_state.ai_insights:
        insights = st.session_state.ai_insights
        
        if 'note' in insights:
            st.info(insights['note'])
        
        # Dataset Overview
        if 'dataset_overview' in insights:
            st.markdown("#### 📋 Dataset Overview")
            overview = insights['dataset_overview']
            st.write(f"**Description:** {overview.get('description', 'N/A')}")
            st.write(f"**Size Assessment:** {overview.get('size_assessment', 'N/A')}")
        
        # Key Insights
        if 'key_insights' in insights:
            st.markdown("#### 🔑 Key Insights")
            for insight in insights['key_insights']:
                st.write(f"• {insight}")
        
        # Important Features
        if 'important_features' in insights and insights['important_features']:
            st.markdown("#### ⭐ Important Features")
            features_df = pd.DataFrame(insights['important_features'])
            st.dataframe(features_df, use_container_width=True)
        
        # ML Problem Type
        if 'ml_problem_type' in insights:
            st.markdown("#### 🎯 ML Problem Type")
            ml_type = insights['ml_problem_type']
            st.write(f"**Suggested Type:** {ml_type.get('suggested_type', 'N/A')}")
            st.write(f"**Confidence:** {ml_type.get('confidence', 'N/A')}")
            st.write(f"**Reasoning:** {ml_type.get('reasoning', 'N/A')}")
        
        # Preprocessing Recommendations
        if 'recommended_preprocessing' in insights and insights['recommended_preprocessing']:
            st.markdown("#### 🔧 Recommended Preprocessing")
            for step in insights['recommended_preprocessing']:
                with st.expander(f"{step.get('step', 'Step')} (Priority: {step.get('priority', 'N/A')})"):
                    st.write(f"**Reason:** {step.get('reason', 'N/A')}")
        
        # Model Recommendations
        if 'model_recommendations' in insights and insights['model_recommendations']:
            st.markdown("#### 🤖 Model Recommendations")
            models_df = pd.DataFrame(insights['model_recommendations'])
            st.dataframe(models_df, use_container_width=True)
        
        # Data Quality Issues
        if 'data_quality_issues' in insights and insights['data_quality_issues']:
            st.markdown("#### ⚠️ Data Quality Issues")
            for issue in insights['data_quality_issues']:
                severity = issue.get('severity', 'low')
                if severity == 'high':
                    st.error(f"**{issue.get('issue', 'Issue')}** - {issue.get('recommendation', '')}")
                elif severity == 'medium':
                    st.warning(f"**{issue.get('issue', 'Issue')}** - {issue.get('recommendation', '')}")
                else:
                    st.info(f"**{issue.get('issue', 'Issue')}** - {issue.get('recommendation', '')}")
    else:
        st.info("Click 'Generate AI Insights' to get AI-powered analysis of your dataset")


def render_automl():
    """Render the AutoML training page."""
    st.markdown('<h2 class="section-header">🤖 AutoML Training</h2>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload a dataset first")
        return
    
    if not st.session_state.target_col:
        st.warning("⚠️ Please select a target column in the Dataset Upload section")
        return
    
    df = st.session_state.df
    target_col = st.session_state.target_col
    
    # Training settings
    st.markdown("#### Training Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    with col2:
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5, 1)
    
    # Model selection
    st.markdown("#### Select Models to Train")
    
    if st.session_state.problem_type in ['binary_classification', 'multiclass_classification']:
        available_models = {
            'Logistic Regression': 'logistic_regression',
            'Random Forest': 'random_forest',
            'Gradient Boosting': 'gradient_boosting',
            'XGBoost': 'xgboost'
        }
    else:
        available_models = {
            'Linear Regression': 'linear_regression',
            'Ridge': 'ridge',
            'Random Forest': 'random_forest',
            'Gradient Boosting': 'gradient_boosting',
            'XGBoost': 'xgboost'
        }
    
    selected_models = []
    cols = st.columns(len(available_models))
    for i, (name, key) in enumerate(available_models.items()):
        with cols[i]:
            if st.checkbox(name, value=True):
                selected_models.append(key)
    
    # Train button
    if st.button("🚀 Start Training", type="primary"):
        if not selected_models:
            st.error("Please select at least one model to train")
            return
        
        with st.spinner("Training models... This may take a few minutes"):
            try:
                # Initialize AutoML engine
                engine = AutoMLEngine(
                    test_size=test_size,
                    cv_folds=cv_folds
                )
                
                # Run training
                results = engine.auto_train(
                    df,
                    target_col,
                    models_to_train=selected_models
                )
                
                st.session_state.automl_results = results
                st.session_state.best_model = engine.best_model
                st.session_state.automl_engine = engine
                
                st.success("✅ Training complete!")
                
            except Exception as e:
                st.error(f"❌ Error during training: {e}")
    
    # Display results
    if st.session_state.automl_results:
        results = st.session_state.automl_results
        engine = st.session_state.automl_engine
        
        st.markdown("#### 🏆 Training Results")
        
        # Best model
        st.success(f"**Best Model:** {results['best_model']}")
        
        # Model comparison
        comparison = results['comparison']
        st.markdown("#### Model Comparison")
        st.dataframe(comparison, use_container_width=True)
        
        # Visualization
        if st.session_state.problem_type in ['binary_classification', 'multiclass_classification']:
            metric = 'accuracy'
        else:
            metric = 'r2'
        
        fig = px.bar(
            comparison,
            x='model',
            y=metric,
            title=f'Model Comparison - {metric.upper()}'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance
        if results['feature_importance'] is not None:
            st.markdown("#### Feature Importance")
            fi_df = results['feature_importance'].head(15)
            fig = px.bar(
                fi_df,
                x='importance',
                y='feature',
                orientation='h',
                title='Top 15 Feature Importances'
            )
            st.plotly_chart(fig, use_container_width=True)


def render_model_evaluation():
    """Render the model evaluation page."""
    st.markdown('<h2 class="section-header">📊 Model Evaluation</h2>', unsafe_allow_html=True)
    
    if not st.session_state.automl_results:
        st.warning("⚠️ Please train models first in the AutoML Training section")
        return
    
    engine = st.session_state.automl_engine
    results = st.session_state.automl_results
    
    # Select model to evaluate
    model_names = list(results['model_results'].keys())
    selected_model = st.selectbox("Select model to evaluate", model_names)
    
    if selected_model:
        model_result = results['model_results'][selected_model]
        
        if 'error' in model_result:
            st.error(f"Error in model: {model_result['error']}")
            return
        
        # Initialize analyzer
        analyzer = ExperimentAnalyzer(
            problem_type=st.session_state.problem_type,
            class_names=engine.label_encoder.classes_ if engine.label_encoder else None
        )
        
        st.session_state.analyzer = analyzer
        
        # Get predictions
        y_test = model_result['predictions']  # This is actually y_pred, need to get actual y_test
        y_pred = model_result['predictions']
        y_pred_proba = model_result.get('probabilities')
        
        # For demonstration, we'll use the predictions we have
        # In a real scenario, we'd store y_test from the training process
        
        tabs = st.tabs([
            "Metrics",
            "Visualizations",
            "Detailed Report"
        ])
        
        with tabs[0]:
            st.markdown("#### Performance Metrics")
            
            metrics = model_result['metrics']
            
            if st.session_state.problem_type in ['binary_classification', 'multiclass_classification']:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                with col4:
                    st.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
                
                if 'roc_auc' in metrics:
                    st.metric("ROC AUC", f"{metrics['roc_auc']:.4f}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                with col2:
                    st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                with col3:
                    st.metric("R²", f"{metrics.get('r2', 0):.4f}")
                with col4:
                    st.metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
            
            # Cross-validation scores
            st.markdown("#### Cross-Validation Scores")
            cv_scores = model_result['cv_scores']
            st.write(f"Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
            
            fig = px.line(
                x=range(1, len(cv_scores) + 1),
                y=cv_scores,
                markers=True,
                title='Cross-Validation Scores'
            )
            fig.update_xaxes(title='Fold')
            fig.update_yaxes(title='Score')
            st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            st.markdown("#### Visualizations")
            
            if st.session_state.problem_type in ['binary_classification', 'multiclass_classification']:
                st.info("Confusion matrix and ROC curve visualizations would appear here")
                st.write("Note: Full evaluation requires storing test set labels during training")
            else:
                st.info("Prediction vs Actual and Residuals plots would appear here")
                st.write("Note: Full evaluation requires storing test set values during training")
        
        with tabs[2]:
            st.markdown("#### Detailed Report")
            st.json(model_result['metrics'])


def render_deployment():
    """Render the deployment page."""
    st.markdown('<h2 class="section-header">🚀 Model Deployment</h2>', unsafe_allow_html=True)
    
    if not st.session_state.best_model:
        st.warning("⚠️ Please train models first in the AutoML Training section")
        return
    
    engine = st.session_state.automl_engine
    
    st.markdown("#### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 💾 Save Model")
        
        model_name = st.text_input("Model name", "trained_model")
        
        if st.button("💾 Save Model", type="primary"):
            with st.spinner("Saving model..."):
                try:
                    deployer = ModelDeployer()
                    
                    # Get feature importance for metadata
                    fi = engine.get_feature_importance()
                    
                    metadata = {
                        'model_name': model_name,
                        'problem_type': st.session_state.problem_type,
                        'target_column': st.session_state.target_col,
                        'training_date': datetime.now().isoformat(),
                        'best_cv_score': float(engine.results[engine.best_model_name]['cv_mean'])
                    }
                    
                    filepath = deployer.save_model(
                        model=engine.best_model,
                        model_name=model_name,
                        preprocessor=engine.preprocessor,
                        metadata=metadata
                    )
                    
                    st.success(f"✅ Model saved to: {filepath}")
                    
                    # Provide download
                    with open(filepath, 'rb') as f:
                        st.download_button(
                            "📥 Download Model",
                            f.read(),
                            file_name=os.path.basename(filepath),
                            mime='application/octet-stream'
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error saving model: {e}")
    
    with col2:
        st.markdown("##### 🌐 Generate API")
        
        api_port = st.number_input("API Port", 8000, 9000, 8000)
        
        if st.button("🌐 Generate FastAPI Service", type="primary"):
            with st.spinner("Generating API service..."):
                try:
                    deployer = ModelDeployer()
                    
                    # First save the model
                    model_path = deployer.save_model(
                        model=engine.best_model,
                        model_name="api_model",
                        preprocessor=engine.preprocessor,
                        metadata={
                            'problem_type': st.session_state.problem_type,
                            'target_column': st.session_state.target_col
                        }
                    )
                    
                    # Get feature names
                    feature_names = engine.feature_names if engine.feature_names else []
                    
                    # Generate API
                    api_dir = deployer.generate_fastapi_service(
                        model_path=model_path,
                        feature_names=feature_names,
                        port=api_port
                    )
                    
                    st.success(f"✅ API service generated in: {api_dir}")
                    
                    # Show instructions
                    st.markdown("""
                    #### To run the API:
                    ```bash
                    cd api
                    pip install -r requirements.txt
                    uvicorn main:app --reload --port {port}
                    ```
                    
                    #### API Documentation:
                    - Swagger UI: http://localhost:{port}/docs
                    - ReDoc: http://localhost:{port}/redoc
                    """.format(port=api_port))
                    
                except Exception as e:
                    st.error(f"❌ Error generating API: {e}")


def main():
    """Main application function."""
    render_header()
    page = render_sidebar()
    
    # Route to appropriate page
    if page == "🏠 Home":
        render_home()
    elif page == "📁 Dataset Upload":
        render_dataset_upload()
    elif page == "🔍 Data Quality":
        render_data_quality()
    elif page == "📈 EDA Dashboard":
        render_eda()
    elif page == "🧠 AI Insights":
        render_ai_insights()
    elif page == "🤖 AutoML Training":
        render_automl()
    elif page == "📊 Model Evaluation":
        render_model_evaluation()
    elif page == "🚀 Deployment":
        render_deployment()


if __name__ == "__main__":
    main()