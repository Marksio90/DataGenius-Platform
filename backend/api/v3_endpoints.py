"""
TMIV v3.0 API Endpoints - nowe zaawansowane funkcjonalności.

Endpointy:
- Neural Networks (PyTorch, TensorFlow, AutoML)
- Hyperparameter Optimization (Optuna, Genetic, Bayesian)
- Time Series (Prophet, ARIMA, LSTM)
- MLOps (MLflow, Drift Detection, Auto Retraining)
- Feature Engineering
- Explainability (SHAP, LIME, What-If)
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from backend.neural_nets.pytorch_trainer import PyTorchTrainer
from backend.neural_nets.automl_neural import AutoMLNeural
from backend.optimization.optuna_tuner import OptunaTuner
from backend.optimization.genetic_optimizer import GeneticOptimizer
from backend.optimization.bayesian_opt import BayesianOptimizer
from backend.timeseries.prophet_forecaster import ProphetForecaster
from backend.timeseries.arima_forecaster import ARIMAForecaster
from backend.mlops.drift_detector import DriftDetector
from backend.mlops.model_registry import ModelRegistry
from backend.feature_engineering.auto_features import AutoFeatureGenerator
from backend.feature_engineering.feature_selector import FeatureSelector
from backend.explainability.shap_explainer import SHAPExplainer
from backend.explainability.whatif_analyzer import WhatIfAnalyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v3", tags=["v3"])


# ==================== REQUEST/RESPONSE MODELS ====================

class NeuralNetworkRequest(BaseModel):
    """Request dla neural network training."""
    problem_type: str
    framework: str = 'pytorch'  # 'pytorch' lub 'tensorflow'
    hidden_sizes: Optional[List[int]] = None
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100
    auto_tune: bool = False


class OptimizationRequest(BaseModel):
    """Request dla hyperparameter optimization."""
    model_type: str  # 'random_forest', 'xgboost', etc.
    optimizer: str = 'optuna'  # 'optuna', 'genetic', 'bayesian'
    n_trials: int = 50
    cv_folds: int = 5
    problem_type: str


class TimeSeriesRequest(BaseModel):
    """Request dla time series forecasting."""
    method: str  # 'prophet', 'arima', 'lstm'
    forecast_periods: int = 30
    date_column: str = 'ds'
    target_column: str = 'y'


class DriftDetectionRequest(BaseModel):
    """Request dla drift detection."""
    method: str = 'all'  # 'ks', 'psi', 'concept', 'all'
    threshold_ks: float = 0.05
    threshold_psi: float = 0.1


class ExplainabilityRequest(BaseModel):
    """Request dla model explainability."""
    method: str  # 'shap', 'lime', 'whatif'
    sample_index: Optional[int] = 0
    top_k: int = 10


# ==================== NEURAL NETWORKS ====================

@router.post("/neural-network/train")
async def train_neural_network(
    request: NeuralNetworkRequest,
    session_id: str
):
    """
    Trenuje neural network (PyTorch lub TensorFlow).
    
    Args:
        request: Training request
        session_id: Session ID
        
    Returns:
        Dict: Training results
    """
    try:
        logger.info(f"Neural network training request: {request.framework}")
        
        # Load data from session
        from backend.data_handler import DataHandler
        dh = DataHandler()
        data = dh.get_session_data(session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Session data not found")
        
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # AutoML Neural?
        if request.auto_tune:
            logger.info("Using AutoML Neural for architecture search")
            
            automl = AutoMLNeural(
                problem_type=request.problem_type,
                framework=request.framework,
                max_trials=10,
                cv_folds=3
            )
            
            results = automl.fit(X_train.values, y_train.values, verbose=True)
            
            # Evaluate
            from sklearn.metrics import accuracy_score, r2_score
            
            y_pred = automl.predict(X_test.values)
            
            if 'classification' in request.problem_type:
                score = accuracy_score(y_test, y_pred)
                metric_name = 'accuracy'
            else:
                score = r2_score(y_test, y_pred)
                metric_name = 'r2'
            
            return {
                'status': 'success',
                'framework': request.framework,
                'auto_tune': True,
                'best_params': results['best_params'],
                'best_cv_score': results['best_score'],
                f'test_{metric_name}': score,
                'n_trials': len(results['trial_history'])
            }
        
        else:
            # Manual training
            if request.framework == 'pytorch':
                trainer = PyTorchTrainer(
                    problem_type=request.problem_type,
                    hidden_sizes=request.hidden_sizes,
                    learning_rate=request.learning_rate,
                    batch_size=request.batch_size,
                    max_epochs=request.max_epochs
                )
            elif request.framework == 'tensorflow':
                from backend.neural_nets.tensorflow_trainer import TensorFlowTrainer
                
                trainer = TensorFlowTrainer(
                    problem_type=request.problem_type,
                    hidden_sizes=request.hidden_sizes,
                    learning_rate=request.learning_rate,
                    batch_size=request.batch_size,
                    max_epochs=request.max_epochs
                )
            else:
                raise HTTPException(status_code=400, detail="Unknown framework")
            
            # Train
            history = trainer.fit(X_train.values, y_train.values, verbose=True)
            
            # Evaluate
            from sklearn.metrics import accuracy_score, r2_score
            
            y_pred = trainer.predict(X_test.values)
            
            if 'classification' in request.problem_type:
                score = accuracy_score(y_test, y_pred)
                metric_name = 'accuracy'
            else:
                score = r2_score(y_test, y_pred)
                metric_name = 'r2'
            
            return {
                'status': 'success',
                'framework': request.framework,
                'architecture': request.hidden_sizes,
                f'test_{metric_name}': score,
                'training_history': {
                    'train_loss': history['train_loss'][-10:],  # Last 10
                    'val_loss': history['val_loss'][-10:]
                }
            }
    
    except Exception as e:
        logger.error(f"Error in neural network training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== HYPERPARAMETER OPTIMIZATION ====================

@router.post("/optimization/tune")
async def hyperparameter_tuning(
    request: OptimizationRequest,
    session_id: str
):
    """
    Hyperparameter tuning używając Optuna/Genetic/Bayesian.
    
    Args:
        request: Optimization request
        session_id: Session ID
        
    Returns:
        Dict: Optimization results
    """
    try:
        logger.info(f"Hyperparameter tuning: {request.optimizer}")
        
        # Load data
        from backend.data_handler import DataHandler
        dh = DataHandler()
        data = dh.get_session_data(session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Session data not found")
        
        X_train = data['X_train']
        y_train = data['y_train']
        
        # Get model class
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        import xgboost as xgb
        
        if request.model_type == 'random_forest':
            if 'classification' in request.problem_type:
                model_class = RandomForestClassifier
            else:
                model_class = RandomForestRegressor
        elif request.model_type == 'xgboost':
            if 'classification' in request.problem_type:
                model_class = xgb.XGBClassifier
            else:
                model_class = xgb.XGBRegressor
        else:
            raise HTTPException(status_code=400, detail="Unknown model type")
        
        # Optimize
        if request.optimizer == 'optuna':
            tuner = OptunaTuner(
                model_class=model_class,
                problem_type=request.problem_type,
                n_trials=request.n_trials,
                cv_folds=request.cv_folds
            )
            
            results = tuner.optimize(X_train.values, y_train.values, verbose=True)
            
            return {
                'status': 'success',
                'optimizer': 'optuna',
                'best_params': results['best_params'],
                'best_score': results['best_score'],
                'n_trials': results['n_trials']
            }
        
        elif request.optimizer == 'genetic':
            # Define param space for genetic
            param_space = {
                'n_estimators': {'range': (50, 500), 'type': 'int'},
                'max_depth': {'range': (3, 30), 'type': 'int'},
                'min_samples_split': {'range': (2, 20), 'type': 'int'}
            }
            
            optimizer = GeneticOptimizer(
                model_class=model_class,
                param_space=param_space,
                problem_type=request.problem_type,
                population_size=20,
                n_generations=request.n_trials // 20
            )
            
            results = optimizer.optimize(X_train.values, y_train.values, verbose=True)
            
            return {
                'status': 'success',
                'optimizer': 'genetic',
                'best_params': results['best_params'],
                'best_score': results['best_score'],
                'n_generations': len(results['population_history'])
            }
        
        elif request.optimizer == 'bayesian':
            # Define param space
            param_space = {
                'n_estimators': {'range': (50, 500), 'type': 'int'},
                'max_depth': {'range': (3, 30), 'type': 'int'}
            }
            
            optimizer = BayesianOptimizer(
                model_class=model_class,
                param_space=param_space,
                problem_type=request.problem_type,
                n_iterations=request.n_trials
            )
            
            results = optimizer.optimize(X_train.values, y_train.values, verbose=True)
            
            return {
                'status': 'success',
                'optimizer': 'bayesian',
                'best_params': results['best_params'],
                'best_score': results['best_score'],
                'n_evaluations': results['n_evaluations']
            }
        
        else:
            raise HTTPException(status_code=400, detail="Unknown optimizer")
    
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== TIME SERIES ====================

@router.post("/timeseries/forecast")
async def time_series_forecast(
    request: TimeSeriesRequest,
    session_id: str
):
    """
    Time series forecasting.
    
    Args:
        request: Forecast request
        session_id: Session ID
        
    Returns:
        Dict: Forecast results
    """
    try:
        logger.info(f"Time series forecasting: {request.method}")
        
        # Load data
        from backend.data_handler import DataHandler
        dh = DataHandler()
        data = dh.get_session_data(session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Session data not found")
        
        df = data['df']
        
        # Forecast
        if request.method == 'prophet':
            forecaster = ProphetForecaster()
            
            forecaster.fit(
                df,
                date_col=request.date_column,
                target_col=request.target_column
            )
            
            forecast = forecaster.predict(periods=request.forecast_periods)
            
            return {
                'status': 'success',
                'method': 'prophet',
                'forecast': forecast.to_dict('records')
            }
        
        elif request.method == 'arima':
            forecaster = ARIMAForecaster(auto=True)
            
            series = df.set_index(request.date_column)[request.target_column]
            
            forecaster.fit(series)
            
            forecast, conf_int = forecaster.predict(
                steps=request.forecast_periods,
                return_conf_int=True
            )
            
            return {
                'status': 'success',
                'method': 'arima',
                'order': forecaster.get_params()['order'],
                'forecast': forecast.to_dict(),
                'confidence_intervals': conf_int.to_dict() if conf_int is not None else None
            }
        
        elif request.method == 'lstm':
            from backend.timeseries.lstm_forecaster import LSTMForecaster
            
            # Prepare multivariate data
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df_numeric = df[numeric_cols]
            
            forecaster = LSTMForecaster(
                lookback=30,
                forecast_horizon=request.forecast_periods,
                epochs=50
            )
            
            forecaster.fit(df_numeric, verbose=0)
            
            # Forecast
            last_sequence = df_numeric.tail(30).values
            forecast = forecaster.predict(last_sequence)
            
            return {
                'status': 'success',
                'method': 'lstm',
                'forecast': forecast.tolist()
            }
        
        else:
            raise HTTPException(status_code=400, detail="Unknown forecasting method")
    
    except Exception as e:
        logger.error(f"Error in time series forecasting: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== DRIFT DETECTION ====================

@router.post("/mlops/detect-drift")
async def detect_drift(
    request: DriftDetectionRequest,
    session_id: str
):
    """
    Wykrywa data/concept drift.
    
    Args:
        request: Drift detection request
        session_id: Session ID
        
    Returns:
        Dict: Drift detection results
    """
    try:
        logger.info(f"Drift detection: {request.method}")
        
        # Load data
        from backend.data_handler import DataHandler
        dh = DataHandler()
        data = dh.get_session_data(session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Session data not found")
        
        X_train = data['X_train']
        X_test = data['X_test']
        y_test = data.get('y_test')
        
        # Initialize detector
        detector = DriftDetector(
            reference_data=X_train,
            threshold_ks=request.threshold_ks,
            threshold_psi=request.threshold_psi
        )
        
        # Detect
        if request.method == 'all':
            # Full report (requires model)
            model = data.get('best_model')
            
            if model is None:
                raise HTTPException(status_code=400, detail="No trained model found")
            
            report = detector.full_drift_report(
                model=model,
                current_data=X_test,
                current_target=y_test
            )
            
            return {
                'status': 'success',
                'report': report
            }
        
        elif request.method == 'ks':
            result = detector.detect_data_drift_ks(X_test)
            
            return {
                'status': 'success',
                'method': 'kolmogorov_smirnov',
                'result': result
            }
        
        elif request.method == 'psi':
            result = detector.detect_data_drift_psi(X_test)
            
            return {
                'status': 'success',
                'method': 'population_stability_index',
                'result': result
            }
        
        else:
            raise HTTPException(status_code=400, detail="Unknown drift detection method")
    
    except Exception as e:
        logger.error(f"Error in drift detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== EXPLAINABILITY ====================

@router.post("/explainability/explain")
async def explain_model(
    request: ExplainabilityRequest,
    session_id: str
):
    """
    Wyjaśnia predykcje modelu.
    
    Args:
        request: Explainability request
        session_id: Session ID
        
    Returns:
        Dict: Explanation results
    """
    try:
        logger.info(f"Model explanation: {request.method}")
        
        # Load data
        from backend.data_handler import DataHandler
        dh = DataHandler()
        data = dh.get_session_data(session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Session data not found")
        
        model = data.get('best_model')
        X_test = data['X_test']
        
        if model is None:
            raise HTTPException(status_code=400, detail="No trained model found")
        
        # Explain
        if request.method == 'shap':
            explainer = SHAPExplainer(
                model=model,
                X_background=data['X_train'].sample(min(100, len(data['X_train'])))
            )
            
            # Explain single instance
            shap_values = explainer.explain(X_test.iloc[[request.sample_index]])
            
            explanation = explainer.explain_prediction(
                X_test,
                sample_idx=request.sample_index,
                top_k=request.top_k
            )
            
            # Feature importance
            importance = explainer.get_feature_importance(X_test)
            
            return {
                'status': 'success',
                'method': 'shap',
                'explanation': explanation,
                'global_importance': importance.head(request.top_k).to_dict('records')
            }
        
        elif request.method == 'lime':
            from backend.explainability.lime_explainer import LIMEExplainer
            
            explainer = LIMEExplainer(
                model=model,
                X_train=data['X_train'],
                mode='classification' if 'classification' in data.get('problem_type', '') else 'regression'
            )
            
            # Explain instance
            instance = X_test.iloc[request.sample_index].values
            explanation = explainer.explain_instance(instance, num_features=request.top_k)
            
            # Extract
            importance = explainer.get_feature_importance(explanation)
            
            return {
                'status': 'success',
                'method': 'lime',
                'feature_importance': importance.to_dict('records')
            }
        
        elif request.method == 'whatif':
            analyzer = WhatIfAnalyzer(
                model=model,
                X_train=data['X_train']
            )
            
            # Counterfactual for instance
            instance = X_test.iloc[request.sample_index]
            
            # Assume binary classification, find counterfactual for opposite class
            current_pred = model.predict(instance.values.reshape(1, -1))[0]
            desired_class = 1 - current_pred  # Flip
            
            counterfactual = analyzer.find_counterfactual(
                instance=instance,
                desired_class=desired_class,
                max_changes=3
            )
            
            return {
                'status': 'success',
                'method': 'what_if',
                'counterfactual': counterfactual
            }
        
        else:
            raise HTTPException(status_code=400, detail="Unknown explainability method")
    
    except Exception as e:
        logger.error(f"Error in model explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== FEATURE ENGINEERING ====================

@router.post("/features/auto-generate")
async def auto_generate_features(session_id: str):
    """
    Automatycznie generuje nowe features.
    
    Args:
        session_id: Session ID
        
    Returns:
        Dict: Generated features info
    """
    try:
        logger.info("Auto feature generation")
        
        # Load data
        from backend.data_handler import DataHandler
        dh = DataHandler()
        data = dh.get_session_data(session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Session data not found")
        
        df = data['df']
        
        # Generate features
        generator = AutoFeatureGenerator(
            max_polynomial_degree=2,
            max_interactions=50,
            include_datetime_features=True,
            include_aggregations=True
        )
        
        df_enhanced = generator.fit_transform(df)
        
        # Update session data
        data['df'] = df_enhanced
        dh.save_session_data(session_id, data)
        
        return {
            'status': 'success',
            'original_features': len(df.columns),
            'new_features': len(df_enhanced.columns) - len(df.columns),
            'total_features': len(df_enhanced.columns),
            'generated_features': generator.get_generated_features()[:20]  # Top 20
        }
    
    except Exception as e:
        logger.error(f"Error in auto feature generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/select")
async def select_features(
    session_id: str,
    n_features: int = 20
):
    """
    Selekcja najlepszych features.
    
    Args:
        session_id: Session ID
        n_features: Liczba features do wyboru
        
    Returns:
        Dict: Selected features info
    """
    try:
        logger.info(f"Feature selection: top {n_features}")
        
        # Load data
        from backend.data_handler import DataHandler
        dh = DataHandler()
        data = dh.get_session_data(session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Session data not found")
        
        X_train = data['X_train']
        y_train = data['y_train']
        
        # Select
        selector = FeatureSelector(
            problem_type=data.get('problem_type', 'classification'),
            n_features_to_select=n_features
        )
        
        X_selected, selected_features = selector.select_features(
            X_train,
            y_train.values,
            methods=['variance', 'correlation', 'mutual_info']
        )
        
        # Ranking
        ranking = selector.get_feature_importance_ranking()
        
        return {
            'status': 'success',
            'original_features': len(X_train.columns),
            'selected_features': len(selected_features),
            'features': selected_features,
            'importance_ranking': ranking.head(20).to_dict('records')
        }
    
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MODEL REGISTRY ====================

@router.post("/mlops/register-model")
async def register_model(
    session_id: str,
    model_name: str,
    stage: str = 'staging'
):
    """
    Rejestruje model w registry.
    
    Args:
        session_id: Session ID
        model_name: Nazwa modelu
        stage: Stage ('staging', 'production')
        
    Returns:
        Dict: Registration info
    """
    try:
        logger.info(f"Registering model: {model_name}")
        
        # Load data
        from backend.data_handler import DataHandler
        dh = DataHandler()
        data = dh.get_session_data(session_id)
        
        if data is None:
            raise HTTPException(status_code=404, detail="Session data not found")
        
        model = data.get('best_model')
        
        if model is None:
            raise HTTPException(status_code=400, detail="No trained model found")
        
        # Register
        registry = ModelRegistry()
        
        version = registry.register_model(
            model=model,
            model_name=model_name,
            stage=stage,
            metadata={
                'session_id': session_id,
                'problem_type': data.get('problem_type'),
                'metrics': data.get('metrics', {})
            }
        )
        
        return {
            'status': 'success',
            'model_name': model_name,
            'version': version,
            'stage': stage
        }
    
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mlops/list-models")
async def list_models(stage: Optional[str] = None):
    """
    Listuje zarejestrowane modele.
    
    Args:
        stage: Filtr stage (None = wszystkie)
        
    Returns:
        Dict: Lista modeli
    """
    try:
        registry = ModelRegistry()
        
        models = registry.list_models(stage=stage)
        
        return {
            'status': 'success',
            'n_models': len(models),
            'models': models
        }
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))