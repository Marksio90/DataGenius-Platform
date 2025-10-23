"""
Optuna Hyperparameter Tuner.

Funkcjonalności:
- Automatyczny tuning dla wszystkich modeli
- Multi-objective optimization
- Pruning nieefektywnych trials
- Wizualizacje (importance, history)
- Parallel optimization
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import cross_val_score

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)

# Suppress optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaTuner:
    """
    Optuna-based hyperparameter tuner.
    
    Wspiera:
    - Wszystkie modele sklearn/XGBoost/LightGBM/CatBoost
    - Single-objective i multi-objective
    - Pruning
    - Distributed optimization
    """

    def __init__(
        self,
        model_class: Any,
        problem_type: str,
        n_trials: int = 50,
        cv_folds: int = 5,
        scoring: str = 'accuracy',
        direction: str = 'maximize',
        n_jobs: int = -1,
        random_state: int = 42,
        enable_pruning: bool = True
    ):
        """
        Inicjalizacja tunera.

        Args:
            model_class: Klasa modelu (np. RandomForestClassifier)
            problem_type: Typ problemu
            n_trials: Liczba trials
            cv_folds: Liczba foldów CV
            scoring: Metryka do optymalizacji
            direction: 'maximize' lub 'minimize'
            n_jobs: Liczba procesów (-1 = all)
            random_state: Random state
            enable_pruning: Czy włączyć pruning
        """
        self.model_class = model_class
        self.problem_type = problem_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.direction = direction
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.enable_pruning = enable_pruning

        # Best results
        self.best_params = None
        self.best_score = None
        self.best_model = None
        self.study = None

        logger.info(f"Optuna Tuner zainicjalizowany dla {model_class.__name__}")

    def _get_param_space(self, trial: optuna.Trial) -> Dict:
        """
        Definiuje przestrzeń parametrów dla danego modelu.

        Args:
            trial: Optuna trial

        Returns:
            Dict: Parametry do przetestowania
        """
        model_name = self.model_class.__name__

        params = {}

        # Random Forest
        if 'RandomForest' in model_name:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 30)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
            params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

        # Gradient Boosting
        elif 'GradientBoosting' in model_name:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)

        # XGBoost
        elif 'XGB' in model_name:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 10)
            params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            params['gamma'] = trial.suggest_float('gamma', 0, 5)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0, 10)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0, 10)

        # LightGBM
        elif 'LGBM' in model_name or 'LightGBM' in model_name:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 500)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            params['num_leaves'] = trial.suggest_int('num_leaves', 20, 150)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['min_child_samples'] = trial.suggest_int('min_child_samples', 5, 100)
            params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 0, 10)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 0, 10)

        # CatBoost
        elif 'CatBoost' in model_name:
            params['iterations'] = trial.suggest_int('iterations', 50, 500)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            params['depth'] = trial.suggest_int('depth', 4, 10)
            params['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 1, 10)
            params['border_count'] = trial.suggest_int('border_count', 32, 255)
            params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0, 1)

        # Logistic Regression
        elif 'LogisticRegression' in model_name:
            params['C'] = trial.suggest_float('C', 0.001, 100, log=True)
            params['penalty'] = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet', None])
            if params['penalty'] == 'elasticnet':
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
                params['solver'] = 'saga'
            elif params['penalty'] == 'l1':
                params['solver'] = 'saga'
            elif params['penalty'] is None:
                params['solver'] = 'saga'

        # Ridge/Lasso
        elif 'Ridge' in model_name or 'Lasso' in model_name:
            params['alpha'] = trial.suggest_float('alpha', 0.001, 100, log=True)

        # SVM
        elif 'SVC' in model_name or 'SVR' in model_name:
            params['C'] = trial.suggest_float('C', 0.001, 100, log=True)
            params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
            if params['kernel'] == 'rbf':
                params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])
            elif params['kernel'] == 'poly':
                params['degree'] = trial.suggest_int('degree', 2, 5)

        # KNN
        elif 'KNeighbors' in model_name:
            params['n_neighbors'] = trial.suggest_int('n_neighbors', 3, 30)
            params['weights'] = trial.suggest_categorical('weights', ['uniform', 'distance'])
            params['metric'] = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])

        # Decision Tree
        elif 'DecisionTree' in model_name:
            params['max_depth'] = trial.suggest_int('max_depth', 3, 30)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)

        # Neural Networks (sklearn MLPClassifier/MLPRegressor)
        elif 'MLP' in model_name:
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_layer_sizes = []
            for i in range(n_layers):
                size = trial.suggest_int(f'n_units_l{i}', 32, 512)
                hidden_layer_sizes.append(size)
            params['hidden_layer_sizes'] = tuple(hidden_layer_sizes)
            params['activation'] = trial.suggest_categorical('activation', ['relu', 'tanh'])
            params['alpha'] = trial.suggest_float('alpha', 0.0001, 0.1, log=True)
            params['learning_rate'] = trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])

        return params

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """
        Funkcja celu dla Optuna.

        Args:
            trial: Optuna trial
            X: Features
            y: Target

        Returns:
            float: Score
        """
        # Get params
        params = self._get_param_space(trial)

        # Add fixed params
        params['random_state'] = self.random_state

        # Special handling dla niektórych modeli
        model_name = self.model_class.__name__

        if 'CatBoost' in model_name:
            params['verbose'] = False
            params['allow_writing_files'] = False

        if 'LGBM' in model_name or 'LightGBM' in model_name:
            params['verbose'] = -1

        if 'XGB' in model_name:
            params['verbosity'] = 0

        # Create model
        try:
            model = self.model_class(**params)
        except Exception as e:
            logger.warning(f"Błąd tworzenia modelu z params {params}: {e}")
            raise optuna.exceptions.TrialPruned()

        # Cross-validation with pruning
        try:
            scores = []
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)

                # Score
                if 'classification' in self.problem_type:
                    score = model.score(X_val, y_val)
                else:
                    from sklearn.metrics import r2_score
                    y_pred = model.predict(X_val)
                    score = r2_score(y_val, y_pred)

                scores.append(score)

                # Pruning
                if self.enable_pruning:
                    trial.report(np.mean(scores), fold)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            return np.mean(scores)

        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Błąd w trial: {e}")
            raise optuna.exceptions.TrialPruned()

    @handle_errors(show_in_ui=False)
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Uruchamia optymalizację.

        Args:
            X: Features
            y: Target
            verbose: Czy wyświetlać progress

        Returns:
            Dict: Wyniki optymalizacji
        """
        logger.info(f"Rozpoczęcie optymalizacji Optuna (n_trials={self.n_trials})")

        # Sampler
        sampler = TPESampler(seed=self.random_state)

        # Pruner
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3) if self.enable_pruning else None

        # Study
        self.study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
            pruner=pruner
        )

        # Optimize
        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            n_jobs=1,  # Musi być 1 dla prawidłowego CV
            show_progress_bar=verbose
        )

        # Best params
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        if verbose:
            logger.info(f"Najlepsze parametry: {self.best_params}")
            logger.info(f"Najlepszy score: {self.best_score:.4f}")

        # Train final model
        final_params = self.best_params.copy()
        final_params['random_state'] = self.random_state

        # Special handling
        model_name = self.model_class.__name__

        if 'CatBoost' in model_name:
            final_params['verbose'] = False
            final_params['allow_writing_files'] = False

        if 'LGBM' in model_name or 'LightGBM' in model_name:
            final_params['verbose'] = -1

        if 'XGB' in model_name:
            final_params['verbosity'] = 0

        self.best_model = self.model_class(**final_params)
        self.best_model.fit(X, y)

        logger.info("Optymalizacja zakończona")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': self.best_model,
            'study': self.study,
            'n_trials': len(self.study.trials)
        }

    def get_best_model(self) -> Any:
        """Zwraca najlepszy model."""
        if self.best_model is None:
            raise ValueError("Optymalizacja nie została uruchomiona - użyj optimize() najpierw")

        return self.best_model

    def get_importance(self) -> Dict[str, float]:
        """
        Zwraca importance parametrów.

        Returns:
            Dict: Importance poszczególnych parametrów
        """
        if self.study is None:
            raise ValueError("Optymalizacja nie została uruchomiona")

        try:
            importance = optuna.importance.get_param_importances(self.study)
            return importance
        except Exception as e:
            logger.warning(f"Nie udało się obliczyć importance: {e}")
            return {}

    def plot_optimization_history(self):
        """Generuje wykres historii optymalizacji."""
        if self.study is None:
            raise ValueError("Optymalizacja nie została uruchomiona")

        try:
            import plotly

            fig = optuna.visualization.plot_optimization_history(self.study)
            return fig
        except ImportError:
            logger.warning("Plotly nie jest zainstalowane - brak wizualizacji")
            return None

    def plot_param_importances(self):
        """Generuje wykres importance parametrów."""
        if self.study is None:
            raise ValueError("Optymalizacja nie została uruchomiona")

        try:
            import plotly

            fig = optuna.visualization.plot_param_importances(self.study)
            return fig
        except ImportError:
            logger.warning("Plotly nie jest zainstalowane - brak wizualizacji")
            return None

    def plot_parallel_coordinate(self):
        """Generuje wykres parallel coordinate."""
        if self.study is None:
            raise ValueError("Optymalizacja nie została uruchomiona")

        try:
            import plotly

            fig = optuna.visualization.plot_parallel_coordinate(self.study)
            return fig
        except ImportError:
            logger.warning("Plotly nie jest zainstalowane - brak wizualizacji")
            return None


class MultiObjectiveOptunaTuner:
    """
    Multi-objective optimization z Optuna.
    
    Pozwala optymalizować kilka metryk jednocześnie (np. accuracy + speed).
    """

    def __init__(
        self,
        model_class: Any,
        problem_type: str,
        objectives: List[str],
        n_trials: int = 50,
        cv_folds: int = 5,
        directions: List[str] = None,
        random_state: int = 42
    ):
        """
        Inicjalizacja multi-objective tunera.

        Args:
            model_class: Klasa modelu
            problem_type: Typ problemu
            objectives: Lista celów do optymalizacji ['accuracy', 'training_time']
            n_trials: Liczba trials
            cv_folds: Liczba foldów CV
            directions: Lista kierunków optymalizacji ['maximize', 'minimize']
            random_state: Random state
        """
        self.model_class = model_class
        self.problem_type = problem_type
        self.objectives = objectives
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.directions = directions or ['maximize'] * len(objectives)
        self.random_state = random_state

        self.study = None
        self.pareto_front = None

        logger.info(f"Multi-objective Optuna Tuner zainicjalizowany")

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> Tuple[float, ...]:
        """
        Multi-objective funkcja celu.

        Returns:
            Tuple: Wartości dla każdego celu
        """
        # Użyj tej samej logiki co OptunaTuner
        tuner = OptunaTuner(
            model_class=self.model_class,
            problem_type=self.problem_type,
            n_trials=1,
            cv_folds=self.cv_folds,
            random_state=self.random_state
        )

        params = tuner._get_param_space(trial)
        params['random_state'] = self.random_state

        model = self.model_class(**params)

        # Measure objectives
        results = []

        for objective in self.objectives:
            if objective == 'accuracy' or objective == 'score':
                # CV score
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X, y, cv=self.cv_folds, n_jobs=1)
                results.append(np.mean(scores))

            elif objective == 'training_time':
                # Measure training time
                import time
                start = time.time()
                model.fit(X, y)
                elapsed = time.time() - start
                results.append(elapsed)

            elif objective == 'model_size':
                # Estimate model size (number of parameters/trees)
                model.fit(X, y)
                if hasattr(model, 'n_estimators'):
                    size = model.n_estimators
                elif hasattr(model, 'coef_'):
                    size = model.coef_.size
                else:
                    size = 1000  # Default
                results.append(size)

        return tuple(results)

    @handle_errors(show_in_ui=False)
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Uruchamia multi-objective optimization.

        Returns:
            Dict: Wyniki z Pareto front
        """
        logger.info("Rozpoczęcie multi-objective optimization")

        # Study
        self.study = optuna.create_study(
            directions=self.directions,
            sampler=TPESampler(seed=self.random_state)
        )

        # Optimize
        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            show_progress_bar=verbose
        )

        # Get Pareto front
        self.pareto_front = self.study.best_trials

        logger.info(f"Znaleziono {len(self.pareto_front)} rozwiązań na Pareto front")

        return {
            'pareto_front': self.pareto_front,
            'n_solutions': len(self.pareto_front),
            'study': self.study
        }

    def get_pareto_front_models(self, X: np.ndarray, y: np.ndarray) -> List[Any]:
        """
        Trenuje modele dla wszystkich rozwiązań z Pareto front.

        Returns:
            List[Any]: Lista modeli
        """
        if self.pareto_front is None:
            raise ValueError("Optymalizacja nie została uruchomiona")

        models = []

        for trial in self.pareto_front:
            params = trial.params.copy()
            params['random_state'] = self.random_state

            model = self.model_class(**params)
            model.fit(X, y)
            models.append(model)

        return models