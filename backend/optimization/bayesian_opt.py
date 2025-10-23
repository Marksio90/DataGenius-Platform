"""
Bayesian Optimization - gradient-free optimization.

Funkcjonalności:
- Gaussian Process-based optimization
- Acquisition functions (EI, UCB, PI)
- Hyperparameter tuning
- Sequential model-based optimization
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import cross_val_score
from scipy.stats import norm

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class BayesianOptimizer:
    """
    Bayesian Optimization dla hyperparameter tuning.
    
    Algorytm:
    1. Inicjalizacja losowych punktów
    2. Fit Gaussian Process na zebranych punktach
    3. Użyj acquisition function do wyboru następnego punktu
    4. Ewaluuj punkt
    5. Powtarzaj kroki 2-4
    """

    def __init__(
        self,
        model_class: Any,
        param_space: Dict,
        problem_type: str,
        n_initial_points: int = 5,
        n_iterations: int = 20,
        acquisition_function: str = 'ei',
        xi: float = 0.01,
        cv_folds: int = 5,
        scoring: str = 'accuracy',
        random_state: int = 42
    ):
        """
        Inicjalizacja Bayesian Optimizer.

        Args:
            model_class: Klasa modelu
            param_space: Przestrzeń parametrów
            problem_type: Typ problemu
            n_initial_points: Liczba początkowych losowych punktów
            n_iterations: Liczba iteracji BO
            acquisition_function: Funkcja akwizycji ('ei', 'ucb', 'pi')
            xi: Exploration-exploitation trade-off (dla EI/PI)
            cv_folds: Liczba foldów CV
            scoring: Metryka
            random_state: Random state
        """
        self.model_class = model_class
        self.param_space = param_space
        self.problem_type = problem_type
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.acquisition_function = acquisition_function
        self.xi = xi
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state

        # Gaussian Process
        kernel = Matern(nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state
        )

        # History
        self.X_observed = []  # Parametry (znormalizowane)
        self.y_observed = []  # Scores
        self.params_observed = []  # Parametry (oryginalne)

        # Best
        self.best_params = None
        self.best_score = None

        # Bounds (dla normalizacji)
        self.bounds = []
        self.param_names = []

        self._setup_bounds()

        np.random.seed(random_state)

        logger.info(f"Bayesian Optimizer zainicjalizowany dla {model_class.__name__}")

    def _setup_bounds(self):
        """Konfiguruje bounds dla parametrów."""
        for param_name, param_def in self.param_space.items():
            self.param_names.append(param_name)

            if isinstance(param_def, dict):
                if 'range' in param_def:
                    low, high = param_def['range']
                    if param_def.get('log', False):
                        # Log scale - użyj log bounds
                        self.bounds.append((np.log(low), np.log(high)))
                    else:
                        self.bounds.append((low, high))
                elif 'choices' in param_def:
                    # Categorical - użyj index bounds
                    self.bounds.append((0, len(param_def['choices']) - 1))
            elif isinstance(param_def, (list, tuple)):
                # Simple choices
                self.bounds.append((0, len(param_def) - 1))

        self.bounds = np.array(self.bounds)

    def _normalize_params(self, params: Dict) -> np.ndarray:
        """
        Normalizuje parametry do [0, 1].

        Args:
            params: Parametry oryginalne

        Returns:
            np.ndarray: Parametry znormalizowane
        """
        x = np.zeros(len(self.param_names))

        for i, param_name in enumerate(self.param_names):
            value = params[param_name]
            param_def = self.param_space[param_name]

            low, high = self.bounds[i]

            if isinstance(param_def, dict):
                if 'range' in param_def:
                    if param_def.get('log', False):
                        value = np.log(value)
                    x[i] = (value - low) / (high - low)
                elif 'choices' in param_def:
                    # Index
                    idx = param_def['choices'].index(value)
                    x[i] = idx / (high - low)
            elif isinstance(param_def, (list, tuple)):
                idx = param_def.index(value)
                x[i] = idx / (high - low)

        return x

    def _denormalize_params(self, x: np.ndarray) -> Dict:
        """
        Denormalizuje parametry z [0, 1] do oryginalnych wartości.

        Args:
            x: Parametry znormalizowane

        Returns:
            Dict: Parametry oryginalne
        """
        params = {}

        for i, param_name in enumerate(self.param_names):
            param_def = self.param_space[param_name]

            low, high = self.bounds[i]
            normalized_value = x[i]

            if isinstance(param_def, dict):
                if 'range' in param_def:
                    value = normalized_value * (high - low) + low

                    if param_def.get('log', False):
                        value = np.exp(value)

                    if param_def.get('type') == 'int':
                        value = int(np.round(value))

                    params[param_name] = value

                elif 'choices' in param_def:
                    idx = int(np.round(normalized_value * (high - low)))
                    idx = np.clip(idx, 0, len(param_def['choices']) - 1)
                    params[param_name] = param_def['choices'][idx]

            elif isinstance(param_def, (list, tuple)):
                idx = int(np.round(normalized_value * (high - low)))
                idx = np.clip(idx, 0, len(param_def) - 1)
                params[param_name] = param_def[idx]

        return params

    def _random_params(self) -> Dict:
        """Generuje losowe parametry."""
        x = np.random.uniform(0, 1, len(self.param_names))
        return self._denormalize_params(x)

    def _evaluate_params(self, params: Dict, X: np.ndarray, y: np.ndarray) -> float:
        """Ewaluuje parametry (CV score)."""
        model_params = params.copy()
        model_params['random_state'] = self.random_state

        try:
            model = self.model_class(**model_params)

            scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds,
                scoring=self.scoring,
                n_jobs=1
            )

            score = np.mean(scores)

        except Exception as e:
            logger.warning(f"Błąd ewaluacji: {e}")
            score = -np.inf

        return score

    def _acquisition_ei(self, X: np.ndarray) -> np.ndarray:
        """
        Expected Improvement acquisition function.

        Args:
            X: Candidate points (normalized)

        Returns:
            np.ndarray: EI values
        """
        mu, sigma = self.gp.predict(X, return_std=True)

        # Current best
        y_best = np.max(self.y_observed)

        # EI calculation
        with np.errstate(divide='warn'):
            improvement = mu - y_best - self.xi
            Z = improvement / sigma
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def _acquisition_ucb(self, X: np.ndarray, kappa: float = 2.576) -> np.ndarray:
        """
        Upper Confidence Bound acquisition function.

        Args:
            X: Candidate points
            kappa: Exploration parameter

        Returns:
            np.ndarray: UCB values
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu + kappa * sigma

    def _acquisition_pi(self, X: np.ndarray) -> np.ndarray:
        """
        Probability of Improvement acquisition function.

        Args:
            X: Candidate points

        Returns:
            np.ndarray: PI values
        """
        mu, sigma = self.gp.predict(X, return_std=True)

        y_best = np.max(self.y_observed)

        with np.errstate(divide='warn'):
            Z = (mu - y_best - self.xi) / sigma
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0

        return pi

    def _propose_location(self) -> np.ndarray:
        """
        Proponuje następną lokalizację do ewaluacji.

        Returns:
            np.ndarray: Normalized parameters
        """
        # Random search over normalized space
        n_samples = 10000
        X_random = np.random.uniform(0, 1, size=(n_samples, len(self.param_names)))

        # Compute acquisition function
        if self.acquisition_function == 'ei':
            acq_values = self._acquisition_ei(X_random)
        elif self.acquisition_function == 'ucb':
            acq_values = self._acquisition_ucb(X_random)
        elif self.acquisition_function == 'pi':
            acq_values = self._acquisition_pi(X_random)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")

        # Select best
        best_idx = np.argmax(acq_values)

        return X_random[best_idx]

    @handle_errors(show_in_ui=False)
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Uruchamia Bayesian Optimization.

        Args:
            X: Features
            y: Target
            verbose: Czy wyświetlać progress

        Returns:
            Dict: Wyniki optymalizacji
        """
        logger.info(f"Rozpoczęcie Bayesian Optimization (n_iter={self.n_iterations})")

        # Initial random points
        for i in range(self.n_initial_points):
            params = self._random_params()
            score = self._evaluate_params(params, X, y)

            x = self._normalize_params(params)

            self.X_observed.append(x)
            self.y_observed.append(score)
            self.params_observed.append(params)

            if verbose:
                logger.info(f"Initial point {i+1}/{self.n_initial_points}: score={score:.4f}")

        # Bayesian Optimization loop
        for iteration in range(self.n_iterations):
            # Fit GP
            X_observed_array = np.array(self.X_observed)
            y_observed_array = np.array(self.y_observed)

            self.gp.fit(X_observed_array, y_observed_array)

            # Propose next point
            x_next = self._propose_location()
            params_next = self._denormalize_params(x_next)

            # Evaluate
            score = self._evaluate_params(params_next, X, y)

            # Update
            self.X_observed.append(x_next)
            self.y_observed.append(score)
            self.params_observed.append(params_next)

            # Track best
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.best_params = params_next.copy()

            if verbose:
                logger.info(
                    f"Iteration {iteration+1}/{self.n_iterations}: "
                    f"score={score:.4f}, best={self.best_score:.4f}"
                )

        # Train final model
        final_params = self.best_params.copy()
        final_params['random_state'] = self.random_state

        best_model = self.model_class(**final_params)
        best_model.fit(X, y)

        logger.info(f"Bayesian Optimization zakończona - Best score: {self.best_score:.4f}")

        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_model': best_model,
            'n_evaluations': len(self.y_observed),
            'observed_params': self.params_observed,
            'observed_scores': self.y_observed
        }

    def get_best_model(self) -> Any:
        """Zwraca najlepszy model."""
        if self.best_params is None:
            raise ValueError("Optymalizacja nie została uruchomiona")

        params = self.best_params.copy()
        params['random_state'] = self.random_state

        model = self.model_class(**params)

        return model

    def plot_convergence(self):
        """Generuje wykres zbieżności."""
        if not self.y_observed:
            raise ValueError("Optymalizacja nie została uruchomiona")

        import matplotlib.pyplot as plt

        iterations = list(range(1, len(self.y_observed) + 1))
        observed_scores = self.y_observed

        # Cumulative best
        cumulative_best = []
        current_best = -np.inf

        for score in observed_scores:
            current_best = max(current_best, score)
            cumulative_best.append(current_best)

        plt.figure(figsize=(10, 6))
        plt.plot(iterations, observed_scores, 'bo-', alpha=0.5, label='Observed')
        plt.plot(iterations, cumulative_best, 'r-', linewidth=2, label='Best So Far')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Bayesian Optimization Convergence')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        return plt.gcf()