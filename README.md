# 🔬 Explainable ML Pipeline Analyzer

[![Streamlit](https://img.shields.io/badge/Streamlit-FF6B35?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plot.ly)

A **production-ready Streamlit application** for transparent end-to-end ML workflows. Automatically detects issues, provides reasoned model recommendations with failure analysis, and deploys production APIs.

## ✨ Features

| Stage | Capabilities |
|-------|--------------|
| **📁 Data Upload** | CSV/Excel + sample datasets (Iris, Wine, Housing) |
| **🧹 Data Cleaning** | **NEW** Feature importance guidance, smart imputation, outlier Winsorizing (1.5×IQR) |
| **🔍 Data Quality** | Missing values, duplicates, outliers, leakage risks, noise detection |
| **📈 EDA Dashboard** | Distributions, correlations, normality tests, pairplots |
| **🧠 AI Insights** | Gemini-powered dataset investigation & recommendations |
| **🤖 AutoML** | Multi-model CV w/ **reasoning layer** (confidence scores, bias-variance tradeoff analysis) |
| **📊 Evaluation** | Confusion matrices, ROC, residuals, learning curves, **bias-variance diagnostics** |
| **🔮 Predictions** | **ENHANCED** Single/batch w/ confidence, probabilities, **feature contributions** |
| **🚀 Deployment** | **ADVANCED** Model export + auto-generated **FastAPI** (`/predict`, `/batch`, `/explain`, `/drift`) |

### 🧠 The Reasoning Layer (Unique)

Unlike basic AutoML tools, this provides **explainable model selection** with **0-100 confidence scores**:

```
🏆 Best: XGBoost (CV 0.87 ± 0.03) 🎯 Confidence: 92/100
   ✅ Why it won: Beat RF by 0.03; mild overfit (gap=0.08); optimal bias-variance balance
   ⚠️ Fails when: High noise/small datasets (<1K rows)
   📊 Bias: low | Variance: medium | Dominant error: balanced
   🔄 Tradeoff: Stable CV (σ=0.03), interpretable coefficients
```

## 📁 Project Structure

```
AI_Data_Science_Copilot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── modules/              # Core ML engines
│   ├── data_loader.py    # Dataset loading & validation
│   ├── data_quality.py   # Quality checks w/ failure detection
│   ├── eda_engine.py     # Comprehensive EDA
│   ├── ai_insights.py    # Gemini-powered analysis
│   ├── automl_engine.py  # AutoML w/ reasoning layer ⭐
│   ├── experiment_analysis.py # Model diagnostics
│   └── model_deployment.py    # FastAPI service generation
├── utils/                # Utility functions
│   ├── helpers.py        # General utilities
│   ├── preprocessing.py  # Data cleaning pipeline
│   └── visualization.py  # EDA/model viz
└── saved_models/         # Exported models & APIs
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/ShivamAttri05/AI_Data_Science_Copilot.git
cd AI_Data_Science_Copilot
pip install -r requirements.txt
```

**Optional**: Gemini API key for AI insights
```bash
echo GEMINI_API_KEY=your_key_here > .env
```

### 2. Launch

```bash
streamlit run app.py
```

Open **http://localhost:8501** 🎉

### 3. Workflow

1. **📁 Upload** → CSV/Excel or sample datasets
2. **🧹 Clean** → **NEW** Guided cleaning w/ feature importance
3. **🔍 Quality** → Auto-issue detection & fixes
4. **📈 EDA** → Interactive distributions + correlations
5. **🧠 AI Insights** → Gemini-powered recommendations
6. **🤖 AutoML** → Multi-model CV + reasoning/confidence
7. **📊 Evaluate** → Bias-variance + overfitting diagnostics
8. **🔮 Predict** → Confidence + feature contributions
9. **🚀 Deploy** → **Production FastAPI** (`/predict` + `/drift` + `/explain`)

## 🛠️ Advanced Usage

### 🆕 Model Deployment (Production-Ready)

Auto-generates complete **FastAPI service** with input validation, confidence scores, and monitoring:

```
api/                    # Auto-created production API
├── main.py            # FastAPI app w/ full OpenAPI docs
├── schemas.py         # **Per-feature** Pydantic models
├── predictor.py       # Validation + confidence + explanations
├── requirements.txt   # FastAPI/Uvicorn/Pydantic
└── Dockerfile         # Container-ready
```

**Run locally**:
```bash
cd api && uvicorn main:app --port 8000 --reload
```
Swagger: `http://localhost:8000/docs`

#### **Endpoints**:
| Method | Path | Features |
|--------|------|----------|
| `POST` | `/predict` | Single pred + **confidence** + **feature contributions** |
| `POST` | `/predict_batch` | **Bulk** predictions w/ per-row probabilities |
| `POST` | `/explain` | **SHAP-style** feature attribution (coef/importance-based) |
| `POST` | `/drift` | **Drift detection** (z-score vs training distribution) |
| `GET`  | `/model_info` | Feature list + model metadata |

**Example** `/predict`:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 35, "income": 65000, "score": 0.85}'
```
```json
{
  "prediction": 1,
  "confidence": 0.92,
  "feature_contributions": {"age": 0.42, "income": -0.18},
  "top_positive": ["age"],
  "interpretation": "Age was the strongest positive driver"
}
```

**Drift Alert**:
```bash
curl -X POST "http://localhost:8000/drift" \
  -H "Content-Type: application/json" \
  -d '{"age": 85, "income": 2e6}'
```
```json
{
  "drifted_features": ["income"],
  "drift_scores": {"income": 4.2},
  "level": "severe",
  "recommendation": "Retrain recommended"
}
```

### Sample Output

```json
{{
  "prediction": 1,
  "confidence": 0.92,
  "interpretation": "High confidence prediction: '1' (92%).",
  "probabilities": [{{"label": "1", "probability": 0.92}}]
}}
```

## 🔧 Technical Highlights

### 🆕 AutoML Reasoning Layer **(v2 improvements)**
```
🏆 Penalized CV: CV score - overfit_penalty(0.5×gap)
🔍 Bias-variance decomposition w/ recommendations
⚠️ Overfitting: none/mild/severe (train-test gap + CV σ)
🎯 Confidence: 0-100 score (stability + margin + absolute perf)
📊 Tradeoffs: 8-axis comparison table across all models
```

### 🆕 Production Deployment Features
```
✅ Input validation (per-feature type/range/nulls)
✅ Confidence/probabilities per prediction
✅ Feature attribution (linear coef/tree importance)
✅ Statistical drift detection (z-score vs reference stats)
✅ Batch endpoint (N×parallel predictions)
✅ Auto-generated Pydantic schemas + predictor.py
✅ Docker-ready w/ health checks
```

### 🧹 Data Cleaning Pipeline
```
✅ Smart imputation: median(skewed)/mean(normal), mode(cat)
✅ Winsorizing: 1.5×IQR outlier capping (preserves rows)
✅ RF baseline importance for feature selection
✅ Zero-variance dropping + datetime extraction
✅ Skew handling: log1p transform (non-negative numerics)
```

### Data Quality Checks
```
✅ Dataset size adequacy + noise estimation
✅ Leakage risk + class imbalance warnings
✅ High-cardinality categorical detection
✅ Near-constant column warnings
```

## 📊 Supported Algorithms **(14 total)**

| Classification | Regression |
|----------------|------------|
| Logistic Regression | Linear Regression |
| **Naive Bayes** | **Ridge** |
| **KNN** | **Lasso** |
| **SVM** | **ElasticNet** |
| Random Forest | **KNN** |
| Gradient Boosting | **SVM** |
| **XGBoost** ⭐ | Random Forest |
| **LightGBM** ⭐ | Gradient Boosting |
| | **XGBoost** ⭐ |
| | **LightGBM** ⭐ |

**⭐ State-of-the-art boosters** (xgboost/lightgbm optional — install from requirements.txt)

## 🤝 Contributing

1. Fork & clone
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
4. Add features/PR

**New modules**: Place in `modules/` following existing patterns.

## 🙏 Acknowledgments

Built with ❤️ using:
- [Streamlit](https://streamlit.io)
- [scikit-learn](https://scikit-learn.org)
- [Plotly](https://plot.ly)
- [Google Gemini](https://ai.google.dev)

---

⭐ **Star this repo if it helps your ML workflow!** ⭐
