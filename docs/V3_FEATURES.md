# TMIV v3.0 ULTRA PRO - Complete Feature Documentation

## 🚀 Overview

TMIV v3.0 introduces **cutting-edge ML capabilities** transforming it into a production-grade AutoML platform with enterprise features.

---

## 📦 New Modules

### 1. Neural Networks (`backend/neural_nets/`)

#### PyTorch Integration
- **Automatic architecture design** based on dataset characteristics
- **Early stopping** and learning rate scheduling
- **Gradient clipping** for stable training
- **TensorBoard logging** for experiment tracking
- Support for classification and regression

**Example Usage:**
```python
from backend.neural_nets.pytorch_trainer import PyTorchTrainer

trainer = PyTorchTrainer(
    problem_type='binary_classification',
    hidden_sizes=[128, 64, 32],
    learning_rate=0.001,
    max_epochs=100
)

trainer.fit(X_train, y_train, verbose=True)
predictions = trainer.predict(X_test)
```

#### TensorFlow/Keras Integration
- **Keras Sequential API** for rapid prototyping
- **Callbacks**: EarlyStopping, ReduceLROnPlateau
- **Multi-output support**

#### AutoML Neural Networks
- **Architecture search** with cross-validation
- **Hyperparameter tuning** for neural networks
- **Automatic framework selection**

**Example:**
```python
from backend.neural_nets.automl_neural import AutoMLNeural

automl = AutoMLNeural(
    problem_type='classification',
    framework='pytorch',
    max_trials=10
)

results = automl.fit(X, y)
best_model = results['best_model']
```

---

### 2. Hyperparameter Optimization (`backend/optimization/`)

#### Optuna Integration
- **Tree-structured Parzen Estimator (TPE)** sampling
- **Median pruning** for early stopping bad trials
- **Parallel optimization** support
- **Visualization**: optimization history, parameter importance

**Supported Models:**
- Random Forest, Gradient Boosting
- XGBoost, LightGBM, CatBoost
- SVM, KNN, Neural Networks

**Example:**
```python
from backend.optimization.optuna_tuner import OptunaTuner
from sklearn.ensemble import RandomForestClassifier

tuner = OptunaTuner(
    model_class=RandomForestClassifier,
    problem_type='classification',
    n_trials=50
)

results = tuner.optimize(X, y)
best_model = results['best_model']
```

#### Genetic Algorithm Optimizer
- **Evolution-based** parameter search
- **Tournament selection**
- **Crossover and mutation** operators
- **Elite preservation**

**Example:**
```python
from backend.optimization.genetic_optimizer import GeneticOptimizer

param_space = {
    'n_estimators': {'range': (50, 500), 'type': 'int'},
    'max_depth': {'range': (3, 30), 'type': 'int'}
}

optimizer = GeneticOptimizer(
    model_class=RandomForestClassifier,
    param_space=param_space,
    population_size=20,
    n_generations=10
)

results = optimizer.optimize(X, y)
```

#### Bayesian Optimization
- **Gaussian Process** surrogate model
- **Acquisition functions**: EI, UCB, PI
- **Sequential model-based optimization**

---

### 3. Time Series Forecasting (`backend/timeseries/`)

#### Prophet (Facebook)
- **Automatic seasonality detection** (daily, weekly, yearly)
- **Holiday effects**
- **Trend changepoints**
- **Uncertainty intervals**

**Example:**
```python
from backend.timeseries.prophet_forecaster import ProphetForecaster

forecaster = ProphetForecaster()
forecaster.fit(df, date_col='ds', target_col='y')

forecast = forecaster.predict(periods=30)
```

#### ARIMA/SARIMA
- **Auto ARIMA** with automatic (p,d,q) selection
- **Seasonal ARIMA** support
- **Stationarity testing**
- **Diagnostics plots**

**Example:**
```python
from backend.timeseries.arima_forecaster import ARIMAForecaster

forecaster = ARIMAForecaster(auto=True, seasonal=True)
forecaster.fit(series)

forecast, conf_int = forecaster.predict(steps=30, return_conf_int=True)
```

#### LSTM Deep Learning
- **Sequence-to-sequence** forecasting
- **Multivariate support**
- **Attention mechanism** (optional)

**Example:**
```python
from backend.timeseries.lstm_forecaster import LSTMForecaster

forecaster = LSTMForecaster(
    lookback=30,
    forecast_horizon=7,
    lstm_units=[50, 50]
)

forecaster.fit(df_multivariate)
forecast = forecaster.predict(last_sequence)
```

---

### 4. MLOps (`backend/mlops/`)

#### MLflow Integration
- **Experiment tracking** with automatic logging
- **Model registry** with versioning
- **Artifact storage** (models, plots, datasets)
- **Run comparison**

**Example:**
```python
from backend.mlops.mlflow_integration import MLflowTracker

tracker = MLflowTracker(experiment_name='my_experiment')

tracker.start_run(run_name='random_forest_v1')
tracker.log_params({'n_estimators': 100, 'max_depth': 10})
tracker.log_metrics({'accuracy': 0.95, 'f1': 0.93})
tracker.log_model(model, registered_model_name='my_model')
tracker.end_run()
```

#### Drift Detection
- **Data drift**: Kolmogorov-Smirnov test, PSI
- **Concept drift**: Performance degradation monitoring
- **Feature-wise drift analysis**
- **Automated alerts**

**Example:**
```python
from backend.mlops.drift_detector import DriftDetector

detector = DriftDetector(
    reference_data=X_train,
    reference_target=y_train
)

# Full drift report
report = detector.full_drift_report(
    model=model,
    current_data=X_new,
    current_target=y_new
)

if report['overall_drift_detected']:
    print("⚠️ DRIFT DETECTED!")
    print(f"Alerts: {report['alerts']}")
```

#### Model Registry
- **Local model versioning**
- **Stage management** (staging, production, archived)
- **Metadata tracking**
- **Model lineage**

**Example:**
```python
from backend.mlops.model_registry import ModelRegistry

registry = ModelRegistry()

# Register model
version = registry.register_model(
    model=my_model,
    model_name='credit_risk_model',
    stage='staging',
    metadata={'accuracy': 0.95}
)

# Promote to production
registry.update_stage('credit_risk_model', version, 'production')

# Load production model
prod_model = registry.load_model('credit_risk_model', stage='production')
```

#### Auto Retrainer
- **Scheduled retraining**
- **Drift-triggered retraining**
- **Performance-triggered retraining**
- **Automatic deployment**
- **Rollback mechanism**

**Example:**
```python
from backend.mlops.auto_retrainer import AutoRetrainer

def train_fn(data):
    model = RandomForestClassifier()
    model.fit(data[features], data[target])
    return model

retrainer = AutoRetrainer(
    model_name='my_model',
    training_function=train_fn,
    registry=registry,
    retraining_frequency_days=7,
    auto_deploy=True
)

# Set baseline for drift detection
retrainer.set_baseline(X_train, y_train)

# Check if retraining needed
needed, reasons = retrainer.check_retraining_needed(X_new, y_new, model)

if needed:
    new_model = retrainer.retrain_model(training_data, reason=reasons[0])
```

---

### 5. Feature Engineering (`backend/feature_engineering/`)

#### Auto Feature Generator
- **Polynomial features** (degree 2, 3)
- **Interaction features** (pairwise multiplication/division)
- **Datetime features** (year, month, day, day of week, etc.)
- **Aggregation features** (row-wise statistics)

**Example:**
```python
from backend.feature_engineering.auto_features import AutoFeatureGenerator

generator = AutoFeatureGenerator(
    max_polynomial_degree=2,
    max_interactions=100,
    include_datetime_features=True
)

df_enhanced = generator.fit_transform(df)
print(f"Generated {len(generator.get_generated_features())} new features")
```

#### Feature Selector
- **Variance threshold** filtering
- **Correlation filtering** (remove highly correlated)
- **Mutual information** ranking
- **Recursive Feature Elimination (RFE)**
- **L1-based selection** (Lasso)

**Example:**
```python
from backend.feature_engineering.feature_selector import FeatureSelector

selector = FeatureSelector(
    problem_type='classification',
    correlation_threshold=0.95
)

X_selected, selected_features = selector.select_features(
    X, y,
    methods=['variance', 'correlation', 'mutual_info']
)

# Get importance ranking
ranking = selector.get_feature_importance_ranking()
```

#### Feature Store
- **Local feature storage** with Parquet
- **Versioning** of feature sets
- **Metadata tracking**
- **Feature lineage**
- **Merge multiple feature sets**

**Example:**
```python
from backend.feature_engineering.feature_store import FeatureStore

store = FeatureStore()

# Save feature set
version = store.save_feature_set(
    df=features_df,
    feature_set_name='customer_features',
    metadata={'source': 'transaction_data'}
)

# Load feature set
features = store.load_feature_set('customer_features', version=version)

# List all feature sets
all_sets = store.list_feature_sets()
```

---

### 6. Explainability (`backend/explainability/`)

#### SHAP Explainer
- **SHAP values** for individual predictions
- **Global feature importance**
- **Summary plots**, dependence plots
- **Force plots**, waterfall plots
- Support for tree, linear, and black-box models

**Example:**
```python
from backend.explainability.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(
    model=model,
    X_background=X_train.sample(100)
)

# Explain predictions
shap_values = explainer.explain(X_test)

# Feature importance
importance = explainer.get_feature_importance(X_test)

# Explain single prediction
explanation = explainer.explain_prediction(X_test, sample_idx=0, top_k=5)

# Plots
explainer.plot_summary(X_test)
explainer.plot_waterfall(X_test, sample_idx=0)
```

#### LIME Explainer
- **Local explanations** using linear approximations
- **Tabular, text, image** support
- **Feature contribution** visualization

**Example:**
```python
from backend.explainability.lime_explainer import LIMEExplainer

explainer = LIMEExplainer(
    model=model,
    X_train=X_train,
    mode='classification'
)

# Explain instance
instance = X_test.iloc[0].values
explanation = explainer.explain_instance(instance, num_features=10)

# Get feature importance
importance = explainer.get_feature_importance(explanation)
```

#### What-If Analyzer
- **Counterfactual explanations**: "What changes give different outcome?"
- **Feature perturbation analysis**
- **Minimal change recommendations**
- **Actionable insights**

**Example:**
```python
from backend.explainability.whatif_analyzer import WhatIfAnalyzer

analyzer = WhatIfAnalyzer(
    model=model,
    X_train=X_train
)

# Find counterfactual
instance = X_test.iloc[0]
counterfactual = analyzer.find_counterfactual(
    instance=instance,
    desired_class=1,
    max_changes=3
)

print(f"Changes needed: {counterfactual['changes']}")

# Analyze feature impact
impact = analyzer.analyze_feature_impact(instance, feature='age', n_steps=20)

# Minimal change recommendations
recommendations = analyzer.recommend_minimal_changes(instance, desired_class=1)
```

#### Fairness Checker
- **Demographic parity**
- **Equal opportunity**
- **Disparate impact**
- **Fairness metrics** per protected attribute

**Example:**
```python
from backend.explainability.fairness_checker import FairnessChecker

checker = FairnessChecker(
    sensitive_features=['gender', 'race']
)

# Check fairness
fairness_metrics = checker.check_fairness(
    y_true=y_test,
    y_pred=predictions,
    X=X_test,
    protected_attribute='gender'
)

# Full report
report = checker.generate_fairness_report(y_test, predictions, X_test)
```

---

## 🎯 API Endpoints (v3)

### Neural Networks
- `POST /api/v3/neural-network/train` - Train PyTorch/TensorFlow model

### Optimization
- `POST /api/v3/optimization/tune` - Hyperparameter tuning (Optuna/Genetic/Bayesian)

### Time Series
- `POST /api/v3/timeseries/forecast` - Forecasting (Prophet/ARIMA/LSTM)

### MLOps
- `POST /api/v3/mlops/detect-drift` - Drift detection
- `POST /api/v3/mlops/register-model` - Model registry
- `GET /api/v3/mlops/list-models` - List registered models

### Feature Engineering
- `POST /api/v3/features/auto-generate` - Auto feature generation
- `POST /api/v3/features/select` - Feature selection

### Explainability
- `POST /api/v3/explainability/explain` - Model explanations (SHAP/LIME/What-If)

---

## 🔧 Configuration

### Environment Variables
```bash
# MLflow
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_ARTIFACT_ROOT=./mlartifacts

# Neural Networks
PYTORCH_DEVICE=cuda  # or cpu
TENSORFLOW_GPU_MEMORY_GROWTH=True

# Optimization
OPTUNA_N_JOBS=1
GENETIC_POPULATION_SIZE=20

# Time Series
PROPHET_WEEKLY_SEASONALITY=auto
ARIMA_AUTO_SELECT=True

# Feature Store
FEATURE_STORE_PATH=./artifacts/feature_store

# Model Registry
MODEL_REGISTRY_PATH=./artifacts/registry
```

---

## 📊 Performance Benchmarks

### Neural Networks
- **PyTorch**: ~2-5x faster than sklearn on large datasets (>100K samples)
- **AutoML Neural**: Finds optimal architecture in 10-20 trials

### Hyperparameter Optimization
- **Optuna**: 30-50% better than GridSearch in same time
- **Genetic**: Good for discrete/categorical parameters
- **Bayesian**: Best for expensive model training

### Time Series
- **Prophet**: Best for business data with seasonality
- **ARIMA**: Good for stationary series
- **LSTM**: Best for complex multivariate patterns

---

## 🚨 Best Practices

### Neural Networks
1. **Start with AutoML** to find good architecture
2. **Use early stopping** to prevent overfitting
3. **Monitor loss curves** in TensorBoard
4. **Scale features** before training

### Optimization
1. **Use Optuna** for general hyperparameter tuning
2. **Use Genetic** for discrete/combinatorial search spaces
3. **Use Bayesian** when function evaluations are expensive
4. **Set reasonable n_trials** (50-100 for Optuna)

### Time Series
1. **Check stationarity** before ARIMA
2. **Prophet works best** with at least 1 year of data
3. **LSTM needs lots of data** (>1000 samples)
4. **Cross-validate** time series properly (time-based splits)

### MLOps
1. **Always track experiments** with MLflow
2. **Set up drift detection** for production models
3. **Use model registry** for version control
4. **Automate retraining** with AutoRetrainer

### Feature Engineering
1. **Generate features first**, then select
2. **Remove highly correlated** features
3. **Use domain knowledge** for feature creation
4. **Version feature sets** in Feature Store

### Explainability
1. **Use SHAP** for global and local explanations
2. **Use LIME** when SHAP is too slow
3. **Use What-If** for actionable insights
4. **Check fairness** for sensitive applications

---

## 🐛 Troubleshooting

### Common Issues

**PyTorch CUDA not available:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
# Install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Prophet installation fails:**
```bash
# Install dependencies first
pip install pystan
pip install prophet
```

**SHAP slow on large datasets:**
```python
# Use TreeExplainer for tree models (fast)
# Use sample for KernelExplainer
explainer = SHAPExplainer(model, X_background=X_train.sample(100))
```

**Optuna parallel warnings:**
```python
# Set n_jobs=1 for stability
tuner = OptunaTuner(..., n_jobs=1)
```

---

## 📚 References

### Papers
- SHAP: "A Unified Approach to Interpreting Model Predictions" (Lundberg & Lee, 2017)
- Prophet: "Forecasting at Scale" (Taylor & Letham, 2018)
- Optuna: "Optuna: A Next-generation Hyperparameter Optimization Framework" (Akiba et al., 2019)

### Documentation
- PyTorch: https://pytorch.org/docs/
- SHAP: https://shap.readthedocs.io/
- Prophet: https://facebook.github.io/prophet/
- MLflow: https://mlflow.org/docs/
- Optuna: https://optuna.readthedocs.io/

---

## 🎓 Tutorials

See `docs/tutorials/` for detailed tutorials:
- `neural_networks_tutorial.md`
- `optimization_tutorial.md`
- `timeseries_tutorial.md`
- `mlops_tutorial.md`
- `explainability_tutorial.md`

---

## 🔮 Roadmap v3.1

- [ ] Distributed training (PyTorch DDP)
- [ ] Model serving with FastAPI
- [ ] Real-time drift monitoring dashboard
- [ ] AutoML for time series
- [ ] Transformer models integration
- [ ] Federated learning support
- [ ] A/B testing framework
- [ ] Cost-sensitive learning

---

**Version**: 3.0.0  
**Last Updated**: 2024-01-20  
**Maintainer**: TMIV Team