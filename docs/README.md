# 🚀 TMIV v3.0 ULTRA PRO

**The Most Intelligent Visualizer** - Production-Grade AutoML Platform

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-3.0.0-orange.svg)](CHANGELOG.md)

## 🎯 What's New in v3.0

### 🧠 Neural Networks
- **PyTorch & TensorFlow** integration
- **AutoML Neural** - automatic architecture search
- Early stopping, learning rate scheduling
- TensorBoard logging

### ⚡ Hyperparameter Optimization
- **Optuna** (TPE + Pruning)
- **Genetic Algorithm**
- **Bayesian Optimization**
- Multi-objective optimization

### 📈 Time Series Forecasting
- **Prophet** (Facebook)
- **ARIMA/SARIMA** with auto selection
- **LSTM** deep learning
- Confidence intervals & diagnostics

### 🔄 MLOps
- **MLflow** experiment tracking
- **Drift Detection** (KS test, PSI, concept drift)
- **Model Registry** with versioning
- **Auto Retrainer** with triggers

### 🎨 Feature Engineering
- **Auto Feature Generator** (polynomial, interactions, datetime)
- **Feature Selector** (variance, correlation, MI, RFE)
- **Feature Store** with versioning

### 🔍 Explainability
- **SHAP** values & visualizations
- **LIME** local explanations
- **What-If Analysis** (counterfactuals)
- **Fairness Checker** (bias detection)

---

## 📦 Installation

### Prerequisites
```bash
Python 3.9+
Node.js 16+
```

### Quick Install
```bash
# Clone repository
git clone https://github.com/your-org/tmiv.git
cd tmiv

# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### Full Install (with all features)
```bash
# Backend with all ML libraries
pip install -r requirements-full.txt

# Includes:
# - PyTorch (CUDA 11.8)
# - TensorFlow
# - Prophet
# - SHAP
# - Optuna
# - MLflow
```

---

## 🚀 Quick Start

### 1. Start Backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 2. Start Frontend
```bash
cd frontend
npm start
```

### 3. Open Browser
```
http://localhost:3000
```

---

## 💡 Usage Examples

### Neural Network Training
```python
from backend.neural_nets.pytorch_trainer import PyTorchTrainer

trainer = PyTorchTrainer(
    problem_type='binary_classification',
    hidden_sizes=[128, 64, 32],
    max_epochs=100
)

trainer.fit(X_train, y_train)
predictions = trainer.predict(X_test)
```

### Hyperparameter Optimization
```python
from backend.optimization.optuna_tuner import OptunaTuner
from sklearn.ensemble import RandomForestClassifier

tuner = OptunaTuner(
    model_class=RandomForestClassifier,
    n_trials=50
)

results = tuner.optimize(X, y)
best_model = results['best_model']
```

### Time Series Forecasting
```python
from backend.timeseries.prophet_forecaster import ProphetForecaster

forecaster = ProphetForecaster()
forecaster.fit(df, date_col='ds', target_col='y')
forecast = forecaster.predict(periods=30)
```

### Drift Detection
```python
from backend.mlops.drift_detector import DriftDetector

detector = DriftDetector(reference_data=X_train)
report = detector.full_drift_report(model, X_new, y_new)

if report['overall_drift_detected']:
    print("⚠️ DRIFT DETECTED - Retrain recommended!")
```

### Model Explainability
```python
from backend.explainability.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(model, X_train.sample(100))
shap_values = explainer.explain(X_test)

# Feature importance
importance = explainer.get_feature_importance(X_test)

# Visualizations
explainer.plot_summary(X_test)
explainer.plot_waterfall(X_test, sample_idx=0)
```

---

## 📖 Documentation

- **[Complete Features Guide](docs/V3_FEATURES.md)** - All v3.0 features
- **[API Reference](docs/API.md)** - REST API documentation
- **[Tutorials](docs/tutorials/)** - Step-by-step guides
- **[Architecture](docs/ARCHITECTURE.md)** - System design

---

## 🧪 Testing
```bash
# Run all tests
pytest

# Run specific module
pytest tests/test_neural_nets.py

# With coverage
pytest --cov=backend tests/
```

---

## 🎯 Key Features

### ✅ Data Processing
- CSV, Excel, JSON support
- Automatic type detection
- Missing value handling
- Outlier detection

### ✅ Visualization
- 15+ chart types
- Interactive plots (Plotly)
- Correlation heatmaps
- Distribution analysis

### ✅ Machine Learning
- 20+ algorithms
- AutoML pipeline
- Ensemble methods
- Deep learning (PyTorch, TensorFlow)

### ✅ Model Evaluation
- Comprehensive metrics
- Cross-validation
- ROC curves, confusion matrix
- Learning curves

### ✅ Production Ready
- MLflow experiment tracking
- Model registry & versioning
- Drift detection
- Auto retraining

---

## 🏗️ Architecture
```
TMIV v3.0
├── backend/
│   ├── neural_nets/          # PyTorch, TensorFlow, AutoML
│   ├── optimization/          # Optuna, Genetic, Bayesian
│   ├── timeseries/            # Prophet, ARIMA, LSTM
│   ├── mlops/                 # MLflow, Drift, Registry
│   ├── feature_engineering/   # Auto Features, Selector, Store
│   ├── explainability/        # SHAP, LIME, What-If, Fairness
│   ├── api/                   # FastAPI endpoints
│   └── models/                # ML model wrappers
└── frontend/
    ├── components/            # React components
    ├── pages/                 # Application pages
    └── utils/                 # Utilities
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=your-org/tmiv&type=Date)](https://star-history.com/#your-org/tmiv&Date)

---

## 📧 Contact

- **Email**: support@tmiv.ai
- **Discord**: [Join our community](https://discord.gg/tmiv)
- **Twitter**: [@TMIV_AI](https://twitter.com/TMIV_AI)

---

## 🙏 Acknowledgments

Built with:
- PyTorch, TensorFlow
- SHAP, LIME
- Prophet, Optuna
- MLflow
- FastAPI, React

---

**Made with ❤️ by the TMIV Team**