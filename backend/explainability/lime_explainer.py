"""
LIME Explainer - Local Interpretable Model-agnostic Explanations.

Funkcjonalności:
- Local explanations dla predictions
- Feature importance per prediction
- Tabular, text, image support
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LIME nie jest zainstalowane: pip install lime")

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    LIME-based local explainer.
    
    LIME wyjaśnia predykcje przez aproksymację modelu
    lokalnym interpretowalnym modelem (linear).
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        mode: str = 'classification',
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[int]] = None
    ):
        """
        Inicjalizacja LIME explainer.

        Args:
            model: Model do wyjaśnienia
            X_train: Training data (do określenia baseline)
            mode: 'classification' lub 'regression'
            feature_names: Nazwy features
            categorical_features: Indeksy categorical features
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME nie jest zainstalowane")

        self.model = model
        self.X_train = X_train
        self.mode = mode

        # Feature names
        if feature_names is None:
            if isinstance(X_train, pd.DataFrame):
                feature_names = X_train.columns.tolist()
            else:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        self.feature_names = feature_names

        # Initialize explainer
        self.explainer = LimeTabularExplainer(
            training_data=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
            feature_names=feature_names,
            mode=mode,
            categorical_features=categorical_features,
            random_state=42
        )

        logger.info(f"LIME Explainer zainicjalizowany (mode={mode})")

    @handle_errors(show_in_ui=False)
    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> Any:
        """
        Wyjaśnia pojedynczą instancję.

        Args:
            instance: Sample do wyjaśnienia (1D array)
            num_features: Liczba features w wyjaśnieniu
            num_samples: Liczba samples dla aproksymacji

        Returns:
            LIME Explanation object
        """
        # Prediction function
        if self.mode == 'classification':
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict

        # Explain
        explanation = self.explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        return explanation

    def explain_batch(
        self,
        X: pd.DataFrame,
        num_features: int = 10,
        num_samples: int = 5000
    ) -> List[Dict]:
        """
        Wyjaśnia batch samples.

        Args:
            X: Data
            num_features: Liczba features
            num_samples: Liczba samples

        Returns:
            List[Dict]: Lista wyjaśnień
        """
        logger.info(f"Wyjaśnianie {len(X)} samples...")

        explanations = []

        for idx in range(len(X)):
            instance = X.iloc[idx].values if isinstance(X, pd.DataFrame) else X[idx]

            try:
                exp = self.explain_instance(instance, num_features, num_samples)

                # Extract explanation
                explanation_dict = {
                    'sample_index': idx,
                    'features': []
                }

                for feature, weight in exp.as_list():
                    explanation_dict['features'].append({
                        'feature': feature,
                        'weight': weight
                    })

                explanations.append(explanation_dict)

            except Exception as e:
                logger.warning(f"Błąd wyjaśniania sample {idx}: {e}")

        logger.info("Wyjaśnianie zakończone")

        return explanations

    def plot_explanation(self, explanation: Any):
        """
        Generuje wykres wyjaśnienia.

        Args:
            explanation: LIME Explanation object

        Returns:
            Figure: Matplotlib figure
        """
        fig = explanation.as_pyplot_figure()
        return fig

    def get_feature_importance(
        self,
        explanation: Any
    ) -> pd.DataFrame:
        """
        Zwraca feature importance z wyjaśnienia.

        Args:
            explanation: LIME Explanation object

        Returns:
            pd.DataFrame: Feature importance
        """
        features = []
        weights = []

        for feature, weight in explanation.as_list():
            features.append(feature)
            weights.append(weight)

        df = pd.DataFrame({
            'feature': features,
            'weight': weights
        })

        df['abs_weight'] = df['weight'].abs()
        df = df.sort_values('abs_weight', ascending=False)

        return df