# Explainable ML Pipeline Analyzer 🔬

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-brightgreen)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**Explainable ML Pipeline Analyzer** accelerates your data science workflow with **transparent decision-making**, **critical failure detection**, and **reasoned model recommendations**. 

Built with original logic for dataset diagnostics, bias-variance analysis, tradeoff reasoning, and production deployment – not just automation, but intelligent analysis that explains *why* models win/lose and *when* to avoid them (e.g. \"XGBoost fails on small datasets <500 rows\").

## 🚀 **Key Features**

| Stage | Feature | Description |
|-------|---------|-------------|
| **🚨 Failure Detection** | Critical Risks | **NEW** Detects small datasets (<100 rows), high noise, data leakage risks – blocks bad modeling |
| **📁 Data Upload** | CSV/Excel + Validation | Drag-and-drop with instant quality warnings |
| **🔍 Data Quality** | 10+ Automated Checks | Missing, duplicates, outliers + **small data/noise/leakage** detection |
| **📈 EDA** | Interactive Dashboards | Correlations, distributions, drill-down analysis |
| **🧠 Reasoning Engine** | Model Tradeoffs | \"XGBoost wins but fails on small data\" + confidence scores |
| **🤖 Transparent AutoML** | Multi-Model w/ Explainability | Why each model wins/loses + bias-variance decomposition |
| **📊 Diagnostics** | Overfitting + CV Analysis | Production-grade model health checks |
| **🚀 Production Deploy** | FastAPI + Docker Ready | One-click API generation w/ Swagger docs |

## 🎯 **End-to-End Workflow**

```
1. Upload Dataset → 2. Quality Check → 3. EDA → 4. AI Insights
    ↓
5. AutoML Training → 6. Model Selection → 7. Evaluation → 8. Deploy
```

**Pipeline Confidence Score**: Proprietary algorithm rates model reliability (0-100) considering overfitting, CV stability, and absolute performance.

## 📱 **Live Demo**

```
streamlit run app.py
```

**Navigation Sidebar** → **9 Automated Sections** → **Production-Ready Results**

### **Sample Output**
```
✅ Best Model: XGBoost (CV 0.947 ± 0.012)
🏆 Pipeline Confidence: 92/100
💡 Why it won: Highest penalized CV + low overfitting
⚠️ Fails when: High noise, extrapolation needed
```

## 🏗️ **Project Structure**

```
AI_Data_Science_Copilot/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Dependencies
├── README.md              # This file
├── .gitignore
├── modules/               # Core business logic
│   ├── data_loader.py     # CSV/Excel loading + validation
│   ├── data_quality.py    # Missing values, outliers, constants
│   ├── eda_engine.py      # Comprehensive EDA with Plotly
│   ├── ai_insights.py     # Gemini-powered dataset analysis
│   ├── automl_engine.py   # Multi-model AutoML + reasoning layer
│   ├── experiment_analysis.py # Model diagnostics + bias-variance
│   └── model_deployment.py # FastAPI generation + serialization
└── utils/                 # Reusable utilities
    ├── helpers.py         # File I/O, progress tracking, validators
    ├── preprocessing.py   # Imputation, encoding, scaling
    └── visualization.py   # Plotly + Matplotlib charts
```

## 🛠️ **Tech Stack**

| Component | Technology |
|-----------|------------|
| **Web Framework** | Streamlit (interactive UI) |
| **AutoML** | scikit-learn, XGBoost |
| **AI Insights** | Google Gemini API |
| **Visualizations** | Plotly, Seaborn, Matplotlib |
| **Deployment** | FastAPI + Uvicorn + Docker |
| **Data Processing** | pandas, numpy |
| **Preprocessing** | scikit-learn Pipeline |

```txt
streamlit>=1.28.0
pandas>=2.0.0      numpy>=1.24.0
scikit-learn>=1.3.0 xgboost>=2.0.0
plotly>=5.15.0      fastapi>=0.104.0
google-generativeai>=0.3.0
```

## ⚙️ **Quick Start**

### **1. Clone & Install**
```bash
git clone <repo>
cd AI_Data_Science_Copilot
pip install -r requirements.txt
```

### **2. Run Application**
```bash
streamlit run app.py
```

### **3. (Optional) AI Insights**
```bash
# Set Gemini API key
export GEMINI_API_KEY="your-api-key-here"
```

### **4. Upload Dataset & Go!**
- **CSV/Excel** → **Auto Quality Check** → **AI Insights** → **AutoML** → **Deploy API**

## 🎨 **Advanced Features**

### **1. AutoML Reasoning Layer** ⭐
```
Why XGBoost won:
• Highest penalized CV: 0.947 ± 0.012
• Beat RF by 0.023 margin
• Mild overfitting (gap: 0.08)
• Low variance, medium bias

Tradeoffs vs alternatives:
• RF: Similar accuracy, higher interpretability
• Logistic: Faster but linear bias
```

### **2. Bias-Variance Decomposition**
```
Model Health:
├── Bias: Medium    → Add polynomial features
├── Variance: Low   → Robust to noise
└ Dominant Error: Balanced
```

### **3. Production Deployment**
```
✅ Model saved: trained_model_20241201.pkl
✅ API generated: ./api/
✅ Docker ready: docker build -t ml-api .
```

**API Endpoints**:
```
POST /predict     → Single prediction
POST /predict_batch → Batch prediction
GET  /docs        → Swagger UI
```

## 🔬 **Technical Highlights**

1. **Smart Preprocessing**: Auto-detects skewness, cardinality, datetime → log1p, ordinal encoding, feature extraction
2. **Overfitting Detection**: Train-test gap + CV stability analysis
3. **Pipeline Confidence**: 0-100 score with explainability
4. **Model Registry**: Timestamped saves + metadata tracking
5. **Interactive EDA**: Clickable Plotly charts + drill-downs

## 📈 **Supported Algorithms**

| Classification | Regression |
|----------------|------------|
| Logistic Regression | Linear Regression |
| Random Forest | Ridge |
| Gradient Boosting | Random Forest |
| **XGBoost** | **Gradient Boosting** |
| | **XGBoost** |

## 🤝 **Contributing**

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push & PR!

## 🙏 **Acknowledgments**

Built with ❤️ using:
- [Streamlit](https://streamlit.io)
- [scikit-learn](https://scikit-learn.org)
- [Google Gemini API](https://ai.google.dev)
- [Plotly](https://plotly.com)
- [FastAPI](https://fastapi.tiangolo.com)

---