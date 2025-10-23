"""
SHAP Explainer - SHapley Additive exPlanations.

Funkcjonalności:
- SHAP values dla individual predictions
- Global feature importance
- Summary plots
- Dependence plots
- Force plots
- Waterfall plots
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SHAP nie jest zainstalowane: pip install shap")

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    SHAP-based model explainer.
    
    SHAP wartości pokazują wkład każdej feature w predykcję.
    Obsługuje:
    - Tree-based models (TreeExplainer)
    - Linear models (LinearExplainer)
    - Black-box models (KernelExplainer)
    """

    def __init__(
        self,
        model: Any,
        X_background: Optional[pd.DataFrame] = None,
        model_type: str = 'auto'
    ):
        """
        Inicjalizacja SHAP explainer.

        Args:
            model: Model do wyjaśnienia
            X_background: Background data (sample dla KernelExplainer)
            model_type: Typ explainera ('auto', 'tree', 'linear', 'kernel')
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP nie jest zainstalowane")

        self.model = model
        self.X_background = X_background
        self.model_type = model_type

        # Initialize explainer
        self.explainer = self._create_explainer()

        # Cache
        self.shap_values = None
        self.expected_value = None

        logger.info(f"SHAP Explainer zainicjalizowany (type={self.explainer.__class__.__name__})")

    def _create_explainer(self) -> Any:
        """Tworzy odpowiedni SHAP explainer."""
        model_class_name = self.model.__class__.__name__

        # Auto-detect explainer type
        if self.model_type == 'auto':
            # Tree-based models
            if any(x in model_class_name for x in ['XGB', 'LGBM', 'LightGBM', 'CatBoost', 'RandomForest', 'GradientBoosting']):
                self.model_type = 'tree'
            # Linear models
            elif any(x in model_class_name for x in ['Linear', 'Logistic', 'Ridge', 'Lasso']):
                self.model_type = 'linear'
            # Default to kernel
            else:
                self.model_type = 'kernel'

        # Create explainer
        if self.model_type == 'tree':
            explainer = shap.TreeExplainer(self.model)

        elif self.model_type == 'linear':
            explainer = shap.LinearExplainer(self.model, self.X_background)

        elif self.model_type == 'kernel':
            if self.X_background is None:
                raise ValueError("X_background is required for KernelExplainer")

            # Use sample for efficiency
            if len(self.X_background) > 100:
                background_sample = shap.sample(self.X_background, 100)
            else:
                background_sample = self.X_background

            explainer = shap.KernelExplainer(self.model.predict, background_sample)

        else:
            raise ValueError(f"Unknown explainer type: {self.model_type}")

        return explainer

    @handle_errors(show_in_ui=False)
    def explain(
        self,
        X: pd.DataFrame,
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Oblicza SHAP values.

        Args:
            X: Data do wyjaśnienia
            check_additivity: Czy sprawdzić addytywność (wolniejsze)

        Returns:
            np.ndarray: SHAP values
        """
        logger.info(f"Obliczanie SHAP values dla {len(X)} samples...")

        self.shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)

        # Handle classification (multi-class)
        if isinstance(self.shap_values, list):
            # Multi-class: return SHAP values for class 1 (positive class)
            self.shap_values = self.shap_values[1]

        # Expected value
        if hasattr(self.explainer, 'expected_value'):
            self.expected_value = self.explainer.expected_value
            if isinstance(self.expected_value, (list, np.ndarray)):
                self.expected_value = self.expected_value[1]  # Positive class

        logger.info("SHAP values obliczone")

        return self.shap_values

    def get_feature_importance(
        self,
        X: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Zwraca global feature importance.

        Args:
            X: Data (jeśli SHAP values nie były obliczone)

        Returns:
            pd.DataFrame: Feature importance ranking
        """
        if self.shap_values is None:
            if X is None:
                raise ValueError("SHAP values nie zostały obliczone - podaj X")
            self.explain(X)

        # Mean absolute SHAP values
        importance = np.abs(self.shap_values).mean(axis=0)

        # Create DataFrame
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(self.shap_values.shape[1])]

        df_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })

        df_importance = df_importance.sort_values('importance', ascending=False)

        return df_importance

    def plot_summary(
        self,
        X: pd.DataFrame,
        plot_type: str = 'dot',
        max_display: int = 20
    ):
        """
        Generuje summary plot.

        Args:
            X: Data
            plot_type: Typ plotu ('dot', 'bar', 'violin')
            max_display: Maksymalna liczba features do pokazania

        Returns:
            Figure: Matplotlib figure
        """
        if self.shap_values is None:
            self.explain(X)

        import matplotlib.pyplot as plt

        shap.summary_plot(
            self.shap_values,
            X,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )

        return plt.gcf()

    def plot_dependence(
        self,
        feature: str,
        X: pd.DataFrame,
        interaction_feature: Optional[str] = 'auto'
    ):
        """
        Generuje dependence plot.

        Args:
            feature: Feature do analizy
            X: Data
            interaction_feature: Feature do interaction ('auto' lub nazwa)

        Returns:
            Figure: Matplotlib figure
        """
        if self.shap_values is None:
            self.explain(X)

        import matplotlib.pyplot as plt

        shap.dependence_plot(
            feature,
            self.shap_values,
            X,
            interaction_index=interaction_feature,
            show=False
        )

        return plt.gcf()

    def plot_force(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0
    ):
        """
        Generuje force plot dla pojedynczej predykcji.

        Args:
            X: Data
            sample_idx: Index sample do wyjaśnienia

        Returns:
            SHAP visualization object
        """
        if self.shap_values is None:
            self.explain(X)

        # Force plot
        force_plot = shap.force_plot(
            self.expected_value,
            self.shap_values[sample_idx],
            X.iloc[sample_idx],
            matplotlib=False
        )

        return force_plot

    def plot_waterfall(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        max_display: int = 10
    ):
        """
        Generuje waterfall plot.

        Args:
            X: Data
            sample_idx: Index sample
            max_display: Maksymalna liczba features

        Returns:
            Figure: Matplotlib figure
        """
        if self.shap_values is None:
            self.explain(X)

        import matplotlib.pyplot as plt

        # Create Explanation object
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.expected_value,
            data=X.iloc[sample_idx].values,
            feature_names=X.columns.tolist()
        )

        shap.plots.waterfall(explanation, max_display=max_display, show=False)

        return plt.gcf()

    def explain_prediction(
        self,
        X: pd.DataFrame,
        sample_idx: int = 0,
        top_k: int = 5
    ) -> Dict:
        """
        Wyjaśnia pojedynczą predykcję.

        Args:
            X: Data
            sample_idx: Index sample
            top_k: Top K najważniejszych features

        Returns:
            Dict: Wyjaśnienie
        """
        if self.shap_values is None:
            self.explain(X)

        # Get SHAP values for sample
        sample_shap = self.shap_values[sample_idx]

        # Top K features
        top_indices = np.argsort(np.abs(sample_shap))[-top_k:][::-1]

        explanation = {
            'sample_index': sample_idx,
            'expected_value': self.expected_value,
            'prediction': self.expected_value + sample_shap.sum(),
            'top_features': []
        }

        for idx in top_indices:
            feature_name = X.columns[idx]
            feature_value = X.iloc[sample_idx, idx]
            shap_value = sample_shap[idx]

            explanation['top_features'].append({
                'feature': feature_name,
                'value': feature_value,
                'shap_value': shap_value,
                'contribution': 'positive' if shap_value > 0 else 'negative'
            })

        return explanation