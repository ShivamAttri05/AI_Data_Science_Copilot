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
| **🔍 Data Quality** | Missing values, duplicates, outliers, leakage risks, noise detection |
| **📈 EDA Dashboard** | Distributions, correlations, normality tests, pairplots |
| **🧠 AI Insights** | Gemini-powered dataset investigation & recommendations |
| **🤖 AutoML** | Multi-model CV w/ **reasoning layer** (overfit/bias-variance) |
| **📊 Evaluation** | Confusion matrices, ROC, residuals, learning curves |
| **🔮 Predictions** | Single/batch predictions w/ confidence scores |
| **🚀 Deployment** | Model export + FastAPI service generation w/ drift detection |

### 🧠 The Reasoning Layer (Unique)

Unlike basic AutoML tools, this provides **explainable model selection**:

```
🏆 Best: XGBoost (CV 0.87 ± 0.03)
   ✅ Why it won: Beat RF by 0.03; mild overfit; low bias/high variance balance
   ⚠️ Fails when: High noise/small datasets
   🎯 Confidence: 92/100 (stable CV, clear margin over alternatives)
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

1. **Upload dataset** (CSV/Excel) or use samples
2. **Data Quality** → Fix issues automatically flagged
3. **EDA** → Explore distributions/correlations
4. **AI Insights** → Get intelligent recommendations
5. **AutoML** → Train models w/ reasoning layer
6. **Evaluate** → Diagnostics (bias-variance, overfitting)
7. **Predict** → Single/batch w/ confidence scores
8. **Deploy** → Export model + auto-generate FastAPI service

## 🛠️ Advanced Usage

### Model Deployment

```python
# Generated API structure (auto-created)
api/
├── main.py          # FastAPI app w/ /predict, /explain, /drift
├── schemas.py       # Pydantic input/output models
├── predictor.py     # Prediction logic w/ confidence
├── requirements.txt
└── Dockerfile

# Run API
cd api && uvicorn main:app --port 8000
```

**Endpoints**:
- `POST /predict` → Single prediction + confidence + explanation
- `POST /predict_batch` → Bulk predictions
- `POST /drift` → Detects statistical shift vs training data
- `POST /explain` → Feature contributions

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

### AutoML Reasoning Layer
```
- Penalized CV selection (CV score - overfit penalty)
- Bias-variance decomposition
- Overfitting diagnostics (train-test gap + CV std)
- Model failure modes & confidence score (0-100)
- Tradeoff analysis across all models
```

### Data Quality Checks
```
✅ Dataset size adequacy
✅ Noise estimation (outlier % + CV)
✅ Leakage risk detection
✅ Class imbalance
✅ High-cardinality warnings
```

## 📊 Supported Algorithms

| Classification | Regression |
|----------------|------------|
| Logistic Regression | Linear Regression |
| Random Forest | Ridge/Lasso |
| Gradient Boosting | Random Forest |
| **XGBoost** (if installed) | **XGBoost** |

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
