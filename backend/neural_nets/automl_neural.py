"""
AutoML Neural Networks - automatyczny dobór architektury i hyperparametrów.

Funkcjonalności:
- Grid search po architekturach
- Automatyczny tuning
- Wybór najlepszego modelu
- Ensemble neural networks
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import KFold

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class AutoMLNeural:
    """
    AutoML dla Neural Networks.
    
    Automatycznie:
    - Dobiera architekturę
    - Tunuje hyperparametry
    - Trenuje i wybiera najlepszy model
    """

    def __init__(
        self,
        problem_type: str,
        framework: str = 'pytorch',
        max_trials: int = 10,
        cv_folds: int = 3,
        max_epochs_per_trial: int = 50,
        random_state: int = 42
    ):
        """
        Inicjalizacja AutoML Neural.

        Args:
            problem_type: Typ problemu
            framework: Framework ('pytorch' lub 'tensorflow')
            max_trials: Maksymalna liczba prób
            cv_folds: Liczba foldów CV
            max_epochs_per_trial: Maksymalna liczba epok per trial
            random_state: Random state
        """
        self.problem_type = problem_type
        self.framework = framework
        self.max_trials = max_trials
        self.cv_folds = cv_folds
        self.max_epochs_per_trial = max_epochs_per_trial
        self.random_state = random_state

        # Best model
        self.best_model = None
        self.best_params = None
        self.best_score = None

        # Trial history
        self.trial_history = []

        logger.info(f"AutoML Neural zainicjalizowany (framework={framework})")

    def _get_search_space(self) -> List[Dict]:
        """
        Definiuje przestrzeń przeszukiwania.

        Returns:
            List[Dict]: Lista konfiguracji do przetestowania
        """
        search_space = []

        # Architektura
        architectures = [
            [128, 64],
            [256, 128, 64],
            [512, 256, 128],
            [128, 64, 32],
            [256, 128, 64, 32],
        ]

        # Learning rates
        learning_rates = [0.001, 0.0001]

        # Dropout rates
        dropout_rates = [0.2, 0.3, 0.5]

        # Batch sizes
        batch_sizes = [16, 32, 64]

        # Kombinacje (wybieramy max_trials najlepszych)
        for arch in architectures:
            for lr in learning_rates:
                for dropout in dropout_rates:
                    for batch_size in batch_sizes:
                        search_space.append({
                            'hidden_sizes': arch,
                            'learning_rate': lr,
                            'dropout_rate': dropout,
                            'batch_size': batch_size,
                        })

        # Losowo wybierz max_trials konfiguracji
        np.random.seed(self.random_state)
        indices = np.random.choice(
            len(search_space),
            size=min(self.max_trials, len(search_space)),
            replace=False
        )

        return [search_space[i] for i in indices]

    @handle_errors(show_in_ui=False)
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Trenuje modele i wybiera najlepszy.

        Args:
            X: Features
            y: Target
            verbose: Czy wyświetlać progress

        Returns:
            Dict: Wyniki AutoML
        """
        logger.info(f"Rozpoczęcie AutoML Neural (max_trials={self.max_trials})")

        search_space = self._get_search_space()

        if verbose:
            logger.info(f"Przestrzeń przeszukiwania: {len(search_space)} konfiguracji")

        # Cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        best_score = -np.inf
        best_params = None

        for trial_idx, params in enumerate(search_space):
            if verbose:
                logger.info(f"Trial {trial_idx+1}/{len(search_space)}: {params}")

            # CV scores
            cv_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Train model
                if self.framework == 'pytorch':
                    from backend.neural_nets.pytorch_trainer import PyTorchTrainer

                    trainer = PyTorchTrainer(
                        problem_type=self.problem_type,
                        hidden_sizes=params['hidden_sizes'],
                        learning_rate=params['learning_rate'],
                        dropout_rate=params['dropout_rate'],
                        batch_size=params['batch_size'],
                        max_epochs=self.max_epochs_per_trial,
                        early_stopping_patience=5
                    )

                    trainer.fit(
                        X_train_fold,
                        y_train_fold,
                        X_val_fold,
                        y_val_fold,
                        verbose=False
                    )

                    # Evaluate
                    if 'classification' in self.problem_type:
                        y_pred = trainer.predict(X_val_fold)
                        score = (y_pred == y_val_fold).mean()
                    else:
                        from sklearn.metrics import r2_score
                        y_pred = trainer.predict(X_val_fold)
                        score = r2_score(y_val_fold, y_pred)

                elif self.framework == 'tensorflow':
                    from backend.neural_nets.tensorflow_trainer import TensorFlowTrainer

                    trainer = TensorFlowTrainer(
                        problem_type=self.problem_type,
                        hidden_sizes=params['hidden_sizes'],
                        learning_rate=params['learning_rate'],
                        dropout_rate=params['dropout_rate'],
                        batch_size=params['batch_size'],
                        max_epochs=self.max_epochs_per_trial,
                        early_stopping_patience=5
                    )

                    trainer.fit(
                        X_train_fold,
                        y_train_fold,
                        X_val_fold,
                        y_val_fold,
                        verbose=0
                    )

                    # Evaluate
                    if 'classification' in self.problem_type:
                        y_pred = trainer.predict(X_val_fold)
                        score = (y_pred == y_val_fold).mean()
                    else:
                        from sklearn.metrics import r2_score
                        y_pred = trainer.predict(X_val_fold)
                        score = r2_score(y_val_fold, y_pred)

                cv_scores.append(score)

            # Mean CV score
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)

            if verbose:
                logger.info(f"  CV Score: {mean_score:.4f} (±{std_score:.4f})")

            # Track trial
            self.trial_history.append({
                'trial': trial_idx + 1,
                'params': params,
                'cv_scores': cv_scores,
                'mean_score': mean_score,
                'std_score': std_score
            })

            # Update best
            if mean_score > best_score:
                best_score = mean_score
                best_params = params

                if verbose:
                    logger.info(f"  ✨ New best score: {best_score:.4f}")

        # Train final model on full data with best params
        logger.info(f"Trening finalnego modelu z najlepszymi parametrami: {best_params}")

        if self.framework == 'pytorch':
            from backend.neural_nets.pytorch_trainer import PyTorchTrainer

            self.best_model = PyTorchTrainer(
                problem_type=self.problem_type,
                hidden_sizes=best_params['hidden_sizes'],
                learning_rate=best_params['learning_rate'],
                dropout_rate=best_params['dropout_rate'],
                batch_size=best_params['batch_size'],
                max_epochs=self.max_epochs_per_trial * 2,  # More epochs for final
                early_stopping_patience=10
            )

        elif self.framework == 'tensorflow':
            from backend.neural_nets.tensorflow_trainer import TensorFlowTrainer

            self.best_model = TensorFlowTrainer(
                problem_type=self.problem_type,
                hidden_sizes=best_params['hidden_sizes'],
                learning_rate=best_params['learning_rate'],
                dropout_rate=best_params['dropout_rate'],
                batch_size=best_params['batch_size'],
                max_epochs=self.max_epochs_per_trial * 2,
                early_stopping_patience=10
            )

        self.best_model.fit(X, y, verbose=verbose)

        self.best_params = best_params
        self.best_score = best_score

        logger.info("AutoML Neural zakończony")

        return {
            'best_params': best_params,
            'best_score': best_score,
            'trial_history': self.trial_history,
            'best_model': self.best_model
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predykcja najlepszym modelem."""
        if self.best_model is None:
            raise ValueError("Model nie został wytrenowany - uruchom fit() najpierw")

        return self.best_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predykcja prawdopodobieństw."""
        if self.best_model is None:
            raise ValueError("Model nie został wytrenowany")

        return self.best_model.predict_proba(X)

    def get_best_architecture(self) -> Dict:
        """Zwraca najlepszą architekturę."""
        return {
            'params': self.best_params,
            'score': self.best_score,
            'framework': self.framework
        }


# Sklearn-compatible wrapper
class AutoMLNeuralNetwork:
    """Sklearn-compatible wrapper dla AutoML Neural."""

    def __init__(
        self,
        problem_type: str,
        framework: str = 'pytorch',
        max_trials: int = 10,
        cv_folds: int = 3,
        random_state: int = 42
    ):
        self.problem_type = problem_type
        self.framework = framework
        self.max_trials = max_trials
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.automl = None

    def fit(self, X, y):
        """Trenuje AutoML."""
        self.automl = AutoMLNeural(
            problem_type=self.problem_type,
            framework=self.framework,
            max_trials=self.max_trials,
            cv_folds=self.cv_folds,
            random_state=self.random_state
        )

        self.automl.fit(X, y, verbose=False)

        return self

    def predict(self, X):
        """Predykcja."""
        if self.automl is None:
            raise ValueError("Model nie został wytrenowany")

        return self.automl.predict(X)

    def predict_proba(self, X):
        """Predykcja prawdopodobieństw."""
        if self.automl is None:
            raise ValueError("Model nie został wytrenowany")

        return self.automl.predict_proba(X)