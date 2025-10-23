"""
Asynchroniczne trenowanie modeli ML.

Funkcjonalności:
- Trening wielu modeli równolegle
- Progress tracking
- Timeout handling
- Cross-validation
- Model evaluation
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from backend.error_handler import MLTrainingException, handle_errors
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Import gradient boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost nie jest dostępny")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM nie jest dostępny")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost nie jest dostępny")


class AsyncMLTrainer:
    """
    Trener modeli ML z asynchronicznym wykonywaniem.
    
    Trenuje wiele modeli równolegle i zbiera wyniki.
    """

    def __init__(self, problem_type: str, training_plan: Dict):
        """
        Inicjalizacja trainera.

        Args:
            problem_type: Typ problemu ML
            training_plan: Plan trenowania z training_plan.py
        """
        self.problem_type = problem_type
        self.training_plan = training_plan
        self.results: List[Dict] = []
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None

    def get_model_class(self, model_name: str) -> Optional[Any]:
        """
        Zwraca klasę modelu na podstawie nazwy.

        Args:
            model_name: Nazwa modelu

        Returns:
            Optional[Any]: Klasa modelu lub None

        Example:
            >>> trainer = AsyncMLTrainer('binary_classification', {})
            >>> model_cls = trainer.get_model_class('LogisticRegression')
            >>> model_cls is not None
            True
        """
        is_classification = 'classification' in self.problem_type

        model_map = {
            # Classification
            'LogisticRegression': LogisticRegression if is_classification else None,
            'RandomForest': RandomForestClassifier if is_classification else RandomForestRegressor,
            'GradientBoosting': GradientBoostingClassifier if is_classification else GradientBoostingRegressor,
            'DecisionTree': DecisionTreeClassifier if is_classification else DecisionTreeRegressor,
            'ExtraTrees': ExtraTreesClassifier if is_classification else ExtraTreesRegressor,
            'KNN': KNeighborsClassifier if is_classification else KNeighborsRegressor,
            'SVC': SVC if is_classification else None,
            'SVR': SVR if not is_classification else None,
            
            # Regression
            'LinearRegression': LinearRegression if not is_classification else None,
            'Ridge': Ridge if not is_classification else None,
            'Lasso': Lasso if not is_classification else None,
            'ElasticNet': ElasticNet if not is_classification else None,
            
            # Gradient Boosting
            'XGBoost': self._get_xgboost_class(is_classification) if XGBOOST_AVAILABLE else None,
            'LightGBM': self._get_lightgbm_class(is_classification) if LIGHTGBM_AVAILABLE else None,
            'CatBoost': self._get_catboost_class(is_classification) if CATBOOST_AVAILABLE else None,
        }

        return model_map.get(model_name)

    def _get_xgboost_class(self, is_classification: bool):
        """Zwraca odpowiednią klasę XGBoost."""
        return xgb.XGBClassifier if is_classification else xgb.XGBRegressor

    def _get_lightgbm_class(self, is_classification: bool):
        """Zwraca odpowiednią klasę LightGBM."""
        return lgb.LGBMClassifier if is_classification else lgb.LGBMRegressor

    def _get_catboost_class(self, is_classification: bool):
        """Zwraca odpowiednią klasę CatBoost."""
        return cb.CatBoostClassifier if is_classification else cb.CatBoostRegressor

    def get_default_params(self, model_name: str) -> Dict:
        """
        Zwraca domyślne parametry dla modelu.

        Args:
            model_name: Nazwa modelu

        Returns:
            Dict: Słownik parametrów

        Example:
            >>> trainer = AsyncMLTrainer('binary_classification', {})
            >>> params = trainer.get_default_params('RandomForest')
            >>> 'random_state' in params
            True
        """
        random_state = self.training_plan.get('random_state', 42)
        n_jobs = self.training_plan.get('n_jobs', -1)

        base_params = {
            'random_state': random_state,
        }

        # Parametry specyficzne dla modeli
        model_params = {
            'LogisticRegression': {
                **base_params,
                'max_iter': 1000,
                'n_jobs': n_jobs,
            },
            'RandomForest': {
                **base_params,
                'n_estimators': 100,
                'n_jobs': n_jobs,
            },
            'GradientBoosting': {
                **base_params,
                'n_estimators': 100,
            },
            'DecisionTree': {
                **base_params,
            },
            'ExtraTrees': {
                **base_params,
                'n_estimators': 100,
                'n_jobs': n_jobs,
            },
            'KNN': {
                'n_neighbors': 5,
                'n_jobs': n_jobs,
            },
            'SVC': {
                **base_params,
                'probability': True,  # Ważne dla ROC curves
            },
            'SVR': {
                **base_params,
            },
            'LinearRegression': {
                'n_jobs': n_jobs,
            },
            'Ridge': {
                **base_params,
            },
            'Lasso': {
                **base_params,
            },
            'ElasticNet': {
                **base_params,
            },
            'XGBoost': {
                **base_params,
                'n_estimators': 100,
                'n_jobs': n_jobs,
                'verbosity': 0,
            },
            'LightGBM': {
                **base_params,
                'n_estimators': 100,
                'n_jobs': n_jobs,
                'verbose': -1,
            },
            'CatBoost': {
                **base_params,
                'iterations': 100,
                'verbose': False,
            },
        }

        return model_params.get(model_name, base_params)

    @handle_errors(show_in_ui=False)
    def train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cv_folds: int = 5
    ) -> Dict:
        """
        Trenuje pojedynczy model z cross-validation.

        Args:
            model_name: Nazwa modelu
            X_train: Features treningowe
            y_train: Target treningowy
            X_test: Features testowe
            y_test: Target testowy
            cv_folds: Liczba foldów CV

        Returns:
            Dict: Wyniki treningu

        Example:
            >>> from sklearn.datasets import make_classification
            >>> X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            >>> X_train, X_test = X[:80], X[80:]
            >>> y_train, y_test = y[:80], y[80:]
            >>> trainer = AsyncMLTrainer('binary_classification', {'random_state': 42})
            >>> result = trainer.train_single_model('LogisticRegression', X_train, y_train, X_test, y_test)
            >>> 'model' in result
            True
        """
        start_time = time.time()
        logger.info(f"Trening modelu: {model_name}")

        # Pobierz klasę modelu
        model_class = self.get_model_class(model_name)
        if model_class is None:
            raise MLTrainingException(f"Model {model_name} nie jest dostępny")

        # Pobierz parametry
        params = self.get_default_params(model_name)

        # Utwórz model
        try:
            model = model_class(**params)
        except Exception as e:
            raise MLTrainingException(f"Błąd tworzenia modelu {model_name}: {e}")

        # Cross-validation
        try:
            cv_scores = self._perform_cross_validation(
                model, X_train, y_train, cv_folds
            )
        except Exception as e:
            logger.warning(f"Cross-validation failed for {model_name}: {e}")
            cv_scores = {}

        # Trening na pełnym zbiorze treningowym
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            raise MLTrainingException(f"Błąd treningu modelu {model_name}: {e}")

        # Ewaluacja na zbiorze testowym
        try:
            test_metrics = self._evaluate_model(model, X_test, y_test)
        except Exception as e:
            logger.warning(f"Evaluation failed for {model_name}: {e}")
            test_metrics = {}

        training_time = time.time() - start_time

        result = {
            'model_name': model_name,
            'model': model,
            'cv_scores': cv_scores,
            'test_metrics': test_metrics,
            'training_time': training_time,
            'params': params,
        }

        logger.info(f"Model {model_name} wytrenowany w {training_time:.2f}s")
        return result

    def _perform_cross_validation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int
    ) -> Dict:
        """
        Wykonuje cross-validation.

        Args:
            model: Model do walidacji
            X: Features
            y: Target
            cv_folds: Liczba foldów

        Returns:
            Dict: Wyniki CV
        """
        metrics = self.training_plan.get('metrics', {})
        primary_metric = metrics.get('primary', 'accuracy')
        secondary_metrics = metrics.get('secondary', [])

        # Przygotuj scoring
        scoring = [primary_metric] + secondary_metrics

        try:
            cv_results = cross_validate(
                model, X, y,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=1,  # Już korzystamy z ThreadPoolExecutor
                return_train_score=False
            )

            # Agreguj wyniki
            cv_scores = {}
            for metric in scoring:
                test_key = f'test_{metric}'
                if test_key in cv_results:
                    scores = cv_results[test_key]
                    cv_scores[metric] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'scores': scores.tolist(),
                    }

            return cv_scores

        except Exception as e:
            logger.warning(f"Cross-validation error: {e}")
            return {}

    def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Ewaluuje model na zbiorze testowym.

        Args:
            model: Wytrenowany model
            X: Features
            y: Target

        Returns:
            Dict: Metryki testowe
        """
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            mean_absolute_error,
            mean_squared_error,
            precision_score,
            r2_score,
            recall_score,
            roc_auc_score,
        )

        metrics = {}

        try:
            # Predykcje
            y_pred = model.predict(X)

            if 'classification' in self.problem_type:
                # Metryki klasyfikacji
                metrics['accuracy'] = float(accuracy_score(y, y_pred))

                # Dla binarnej: więcej metryk
                if self.problem_type == 'binary_classification':
                    metrics['precision'] = float(precision_score(y, y_pred, zero_division=0))
                    metrics['recall'] = float(recall_score(y, y_pred, zero_division=0))
                    metrics['f1'] = float(f1_score(y, y_pred, zero_division=0))

                    # ROC AUC jeśli dostępne predict_proba
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X)[:, 1]
                        metrics['roc_auc'] = float(roc_auc_score(y, y_proba))

                # Dla wieloklasowej
                elif 'multiclass' in self.problem_type:
                    metrics['f1_weighted'] = float(f1_score(y, y_pred, average='weighted', zero_division=0))
                    metrics['f1_macro'] = float(f1_score(y, y_pred, average='macro', zero_division=0))

            else:
                # Metryki regresji
                metrics['r2'] = float(r2_score(y, y_pred))
                metrics['mae'] = float(mean_absolute_error(y, y_pred))
                metrics['rmse'] = float(np.sqrt(mean_squared_error(y, y_pred)))
                metrics['mse'] = float(mean_squared_error(y, y_pred))

        except Exception as e:
            logger.warning(f"Metrics calculation error: {e}")

        return metrics

    @handle_errors(show_in_ui=False)
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        max_workers: int = 4,
        timeout_per_model: int = 300
    ) -> List[Dict]:
        """
        Trenuje wszystkie modele asynchronicznie.

        Args:
            X_train: Features treningowe
            y_train: Target treningowy
            X_test: Features testowe
            y_test: Target testowy
            max_workers: Maksymalna liczba workerów
            timeout_per_model: Timeout na model (sekundy)

        Returns:
            List[Dict]: Lista wyników wszystkich modeli

        Example:
            >>> from sklearn.datasets import make_classification
            >>> X, y = make_classification(n_samples=100, n_features=10, random_state=42)
            >>> X_train, X_test = X[:80], X[80:]
            >>> y_train, y_test = y[:80], y[80:]
            >>> plan = {'models': ['LogisticRegression'], 'random_state': 42, 'metrics': {'primary': 'accuracy', 'secondary': []}}
            >>> trainer = AsyncMLTrainer('binary_classification', plan)
            >>> results = trainer.train_all_models(X_train, y_train, X_test, y_test)
            >>> len(results) > 0
            True
        """
        models_to_train = self.training_plan.get('models', [])
        cv_folds = self.training_plan.get('cv_strategy', {}).get('n_splits', 5)

        logger.info(f"Rozpoczęcie treningu {len(models_to_train)} modeli z {max_workers} workerami")

        results = []
        failed_models = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit wszystkie taski
            future_to_model = {}
            for model_name in models_to_train:
                future = executor.submit(
                    self.train_single_model,
                    model_name,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    cv_folds
                )
                future_to_model[future] = model_name

            # Zbierz wyniki
            for future in as_completed(future_to_model, timeout=timeout_per_model * len(models_to_train)):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=timeout_per_model)
                    results.append(result)
                    logger.info(f"✓ Model {model_name} ukończony")
                except TimeoutError:
                    logger.error(f"✗ Model {model_name} przekroczył timeout ({timeout_per_model}s)")
                    failed_models.append(model_name)
                except Exception as e:
                    logger.error(f"✗ Model {model_name} nie powiódł się: {e}")
                    failed_models.append(model_name)

        self.results = results

        if failed_models:
            logger.warning(f"Modele, które nie powiodły się: {failed_models}")

        # Znajdź najlepszy model
        self._find_best_model()

        logger.info(f"Trening zakończony: {len(results)} modeli sukces, {len(failed_models)} niepowodzeń")

        return results

    def _find_best_model(self) -> None:
        """Znajduje najlepszy model na podstawie metryki głównej."""
        if not self.results:
            return

        primary_metric = self.training_plan.get('metrics', {}).get('primary', 'accuracy')

        # Pobierz wartości metryki
        model_scores = []
        for result in self.results:
            # Najpierw sprawdź test_metrics
            score = result.get('test_metrics', {}).get(primary_metric)
            # Jeśli nie ma, sprawdź cv_scores
            if score is None:
                score = result.get('cv_scores', {}).get(primary_metric, {}).get('mean')

            if score is not None:
                # Obsługa metryk "negatywnych" (RMSE, MAE)
                if primary_metric.startswith('neg_'):
                    score = -score

                model_scores.append((result['model_name'], score, result))

        if model_scores:
            # Znajdź maksimum (dla większości metryk wyższe = lepsze)
            # Dla RMSE/MAE już odwróciliśmy znak
            best_name, best_score, best_result = max(model_scores, key=lambda x: x[1])

            self.best_model_name = best_name
            self.best_model = best_result['model']

            logger.info(f"Najlepszy model: {best_name} ({primary_metric}={best_score:.4f})")

    def get_results(self) -> List[Dict]:
        """
        Zwraca wyniki wszystkich modeli.

        Returns:
            List[Dict]: Lista wyników
        """
        return self.results

    def get_best_model(self) -> Tuple[Optional[str], Optional[Any]]:
        """
        Zwraca najlepszy model.

        Returns:
            Tuple: (nazwa_modelu, model)
        """
        return self.best_model_name, self.best_model