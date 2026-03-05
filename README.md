# 🤖 AI Data Science Copilot

A comprehensive, modular Streamlit-based web application that automates the typical data science workflow from raw data to deployed machine learning models.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Features

- **📁 Dataset Upload** - Support for CSV and Excel files with automatic format detection
- **🔍 Data Quality Check** - Automatic detection of missing values, duplicates, outliers, and class imbalance
- **📈 Exploratory Data Analysis** - Comprehensive statistical analysis and interactive visualizations
- **🧠 AI Dataset Investigation** - Gemini-powered intelligent insights and recommendations
- **🤖 AutoML Model Training** - Automatic problem type detection and model comparison
- **📊 Model Evaluation** - Detailed performance metrics and visualizations
- **🚀 Model Deployment** - Export models and generate FastAPI prediction services

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/ai-data-science-copilot.git
cd ai-data-science-copilot
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Gemini API key (optional)
# Get your API key from: https://makersuite.google.com/app/apikey
```

## 🚀 Quick Start

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

### Sample Workflow

1. **Upload Data** - Drag and drop your CSV or Excel file
2. **Check Quality** - Run automated data quality checks
3. **Explore** - Generate EDA visualizations and statistics
4. **Get Insights** - Use AI to analyze your dataset
5. **Train Models** - Run AutoML to train and compare models
6. **Evaluate** - Review model performance metrics
7. **Deploy** - Export your best model or generate an API

## 📁 Project Structure

```
ai_data_science_copilot/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── README.md                       # Project documentation
│
├── modules/                        # Core functionality modules
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loading functionality
│   ├── data_quality.py            # Data quality checks
│   ├── eda_engine.py              # Exploratory data analysis
│   ├── ai_insights.py             # Gemini AI integration
│   ├── automl_engine.py           # Automated ML training
│   ├── experiment_analysis.py     # Model evaluation
│   └── model_deployment.py        # Model export and deployment
│
└── utils/                          # Utility functions
    ├── __init__.py
    ├── preprocessing.py           # Data preprocessing utilities
    ├── visualization.py           # Visualization helpers
    └── helpers.py                 # General helper functions
```

## 📖 Usage Guide

### 1. Dataset Upload

Upload your dataset in CSV or Excel format. The application will automatically:
- Detect file format
- Parse column types
- Show data preview
- Display basic statistics

```python
# Programmatic usage
from modules.data_loader import DataLoader

loader = DataLoader()
df = loader.load_file('data.csv')
info = loader.get_dataset_info()
```

### 2. Data Quality Check

Automatically detect common data quality issues:
- Missing values (with percentage)
- Duplicate rows
- Outliers (using IQR method)
- Constant columns
- Class imbalance

```python
from modules.data_quality import DataQualityChecker

checker = DataQualityChecker(df)
report = checker.run_all_checks(target_col='target')
print(checker.get_quick_summary())
```

### 3. Exploratory Data Analysis

Generate comprehensive EDA including:
- Summary statistics
- Correlation heatmaps
- Distribution plots
- Categorical value counts
- Missing value visualizations

```python
from modules.eda_engine import EDAEngine

engine = EDAEngine(df)
results = engine.run_full_analysis(target_col='target')
print(engine.get_eda_report())
```

### 4. AI Dataset Investigation

Get intelligent insights powered by Google's Gemini API:
- Important feature identification
- Target column suggestions
- Data quality recommendations
- Preprocessing steps
- Model recommendations

```python
from modules.ai_insights import AIInsightsGenerator

generator = AIInsightsGenerator(api_key='your_api_key')
insights = generator.analyze_dataset(df, target_col='target')
```

### 5. AutoML Model Training

Automatically train and compare multiple models:

**Classification Models:**
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

**Regression Models:**
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

```python
from modules.automl_engine import AutoMLEngine

engine = AutoMLEngine()
results = engine.auto_train(df, target_col='target')
print(f"Best model: {results['best_model']}")
```

### 6. Model Evaluation

Comprehensive evaluation with:
- Confusion matrices
- ROC curves (classification)
- Prediction vs actual plots (regression)
- Feature importance
- Cross-validation scores

```python
from modules.experiment_analysis import ExperimentAnalyzer

analyzer = ExperimentAnalyzer(problem_type='classification')
analysis = analyzer.analyze_classification_model(
    'model_name', y_true, y_pred, y_pred_proba
)
```

### 7. Model Deployment

Export and deploy your models:

```python
from modules.model_deployment import ModelDeployer

deployer = ModelDeployer()

# Save model
model_path = deployer.save_model(
    model=engine.best_model,
    model_name='my_model',
    preprocessor=engine.preprocessor,
    metadata={'problem_type': 'classification'}
)

# Generate FastAPI service
api_dir = deployer.generate_fastapi_service(
    model_path=model_path,
    feature_names=['feature1', 'feature2'],
    port=8000
)
```

## 🔌 API Reference

### Generated FastAPI Endpoints

When you generate a FastAPI service, the following endpoints are available:

#### Health Check
```
GET /health
```

#### Model Information
```
GET /model_info
```

#### Single Prediction
```
POST /predict
Content-Type: application/json

{
    "feature1": 1.0,
    "feature2": 2.0,
    ...
}
```

#### Batch Prediction
```
POST /predict_batch
Content-Type: application/json

{
    "inputs": [
        {"feature1": 1.0, "feature2": 2.0},
        {"feature1": 3.0, "feature2": 4.0}
    ]
}
```

#### Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Google Gemini API Key (optional)
GEMINI_API_KEY=your_api_key_here

# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
RANDOM_STATE=42
TEST_SIZE=0.2
CV_FOLDS=5
```

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and add it to your `.env` file

## 🐛 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Make sure you're in the project directory
cd ai_data_science_copilot

# Reinstall dependencies
pip install -r requirements.txt
```

#### Gemini API Not Working
- Check that your API key is correct
- Verify the `.env` file is in the project root
- Ensure you have internet connectivity

#### Memory Issues with Large Datasets
- Reduce dataset size before uploading
- Increase available memory
- Use sampling for initial exploration

#### Model Training Fails
- Check for missing values in target column
- Verify data types are correct
- Ensure sufficient data (minimum 10 rows recommended)

### Getting Help

If you encounter issues:
1. Check the application logs for error messages
2. Verify your environment setup
3. Ensure all dependencies are installed correctly
4. Try with a sample dataset first

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black modules/ utils/ app.py

# Lint code
flake8 modules/ utils/ app.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - For the amazing web app framework
- [scikit-learn](https://scikit-learn.org/) - For machine learning algorithms
- [Google Gemini](https://deepmind.google/technologies/gemini/) - For AI-powered insights
- [Plotly](https://plotly.com/) - For interactive visualizations

## 📧 Contact

For questions or suggestions, please open an issue on GitHub or contact the maintainers.

---

**Happy Data Science! 🚀**