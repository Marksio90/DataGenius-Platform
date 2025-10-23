"""
What-If Analyzer - analiza what-if scenarios.

Funkcjonalności:
- Counterfactual explanations
- Feature perturbation analysis
- Minimal change recommendations
- Actionable insights
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class WhatIfAnalyzer:
    """
    What-If scenario analyzer.
    
    Odpowiada na pytania typu:
    - Co muszę zmienić żeby dostać kredyt?
    - Jakie minimalne zmiany dadzą inną predykcję?
    - Jak feature X wpływa na outcome?
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        feature_ranges: Optional[Dict] = None
    ):
        """
        Inicjalizacja What-If analyzer.

        Args:
            model: Model
            X_train: Training data (do określenia ranges)
            feature_ranges: Zakresy features (Dict: feature -> (min, max))
        """
        self.model = model
        self.X_train = X_train

        # Determine feature ranges
        if feature_ranges is None:
            self.feature_ranges = {}

            for col in X_train.columns:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    self.feature_ranges[col] = (
                        X_train[col].min(),
                        X_train[col].max()
                    )
        else:
            self.feature_ranges = feature_ranges

        logger.info("What-If Analyzer zainicjalizowany")

    @handle_errors(show_in_ui=False)
    def find_counterfactual(
        self,
        instance: pd.Series,
        desired_class: int,
        max_changes: int = 3,
        features_to_change: Optional[List[str]] = None,
        n_iterations: int = 100
    ) -> Dict:
        """
        Znajduje counterfactual explanation.

        Args:
            instance: Sample do zmiany
            desired_class: Pożądana klasa
            max_changes: Maksymalna liczba zmian
            features_to_change: Features dozwolone do zmiany (None = wszystkie)
            n_iterations: Liczba iteracji

        Returns:
            Dict: Counterfactual explanation
        """
        logger.info("Szukanie counterfactual explanation...")

        if features_to_change is None:
            features_to_change = list(self.feature_ranges.keys())

        # Current prediction
        current_pred = self.model.predict(instance.values.reshape(1, -1))[0]

        if current_pred == desired_class:
            return {
                'success': True,
                'message': 'Instance już ma pożądaną klasę',
                'changes': []
            }

        best_counterfactual = None
        best_distance = np.inf

        # Random search
        for _ in range(n_iterations):
            # Create modified instance
            modified = instance.copy()

            # Randomly select features to change
            n_changes = np.random.randint(1, max_changes + 1)
            features_selected = np.random.choice(
                features_to_change,
                size=min(n_changes, len(features_to_change)),
                replace=False
            )

            # Apply changes
            for feature in features_selected:
                if feature in self.feature_ranges:
                    min_val, max_val = self.feature_ranges[feature]
                    new_value = np.random.uniform(min_val, max_val)
                    modified[feature] = new_value

            # Check prediction
            pred = self.model.predict(modified.values.reshape(1, -1))[0]

            if pred == desired_class:
                # Calculate distance
                distance = euclidean_distances(
                    instance.values.reshape(1, -1),
                    modified.values.reshape(1, -1)
                )[0, 0]

                if distance < best_distance:
                    best_distance = distance
                    best_counterfactual = modified

        if best_counterfactual is None:
            return {
                'success': False,
                'message': f'Nie znaleziono counterfactual po {n_iterations} iteracjach'
            }

        # Extract changes
        changes = []

        for feature in instance.index:
            if instance[feature] != best_counterfactual[feature]:
                changes.append({
                    'feature': feature,
                    'original_value': instance[feature],
                    'new_value': best_counterfactual[feature],
                    'change': best_counterfactual[feature] - instance[feature]
                })

        return {
            'success': True,
            'original_prediction': current_pred,
            'new_prediction': desired_class,
            'distance': best_distance,
            'changes': changes,
            'counterfactual': best_counterfactual
        }

    def analyze_feature_impact(
        self,
        instance: pd.Series,
        feature: str,
        n_steps: int = 20
    ) -> pd.DataFrame:
        """
        Analizuje wpływ zmiany feature na predykcję.

        Args:
            instance: Sample
            feature: Feature do analizy
            n_steps: Liczba kroków

        Returns:
            pd.DataFrame: Impact analysis
        """
        if feature not in self.feature_ranges:
            raise ValueError(f"Feature {feature} nie ma określonego range")

        min_val, max_val = self.feature_ranges[feature]

        # Generate values
        values = np.linspace(min_val, max_val, n_steps)

        # Predictions
        predictions = []

        for val in values:
            modified = instance.copy()
            modified[feature] = val

            pred = self.model.predict(modified.values.reshape(1, -1))[0]

            # If classification, get probability
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(modified.values.reshape(1, -1))[0]
                predictions.append({
                    'feature_value': val,
                    'prediction': pred,
                    'probability': proba.max(),
                    'probabilities': proba
                })
            else:
                predictions.append({
                    'feature_value': val,
                    'prediction': pred
                })

        df_impact = pd.DataFrame(predictions)

        return df_impact

    def recommend_minimal_changes(
        self,
        instance: pd.Series,
        desired_class: int,
        features_to_change: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Rekomenduje minimalne zmiany dla osiągnięcia desired_class.

        Args:
            instance: Sample
            desired_class: Pożądana klasa
            features_to_change: Features dozwolone do zmiany

        Returns:
            List[Dict]: Lista rekomendacji
        """
        logger.info("Generowanie rekomendacji minimalnych zmian...")

        if features_to_change is None:
            features_to_change = list(self.feature_ranges.keys())

        recommendations = []

        # Try changing each feature individually
        for feature in features_to_change:
            if feature not in self.feature_ranges:
                continue

            min_val, max_val = self.feature_ranges[feature]

            # Try different values
            for val in np.linspace(min_val, max_val, 10):
                modified = instance.copy()
                modified[feature] = val

                pred = self.model.predict(modified.values.reshape(1, -1))[0]

                if pred == desired_class:
                    distance = abs(val - instance[feature])

                    recommendations.append({
                        'feature': feature,
                        'original_value': instance[feature],
                        'recommended_value': val,
                        'change': val - instance[feature],
                        'distance': distance
                    })

        # Sort by distance
        recommendations.sort(key=lambda x: x['distance'])

        return recommendations

    def plot_feature_impact(
        self,
        instance: pd.Series,
        feature: str,
        n_steps: int = 20
    ):
        """
        Generuje wykres wpływu feature.

        Args:
            instance: Sample
            feature: Feature
            n_steps: Liczba kroków

        Returns:
            Figure: Matplotlib figure
        """
        import matplotlib.pyplot as plt

        df_impact = self.analyze_feature_impact(instance, feature, n_steps)

        fig, ax = plt.subplots(figsize=(10, 6))

        if 'probability' in df_impact.columns:
            ax.plot(df_impact['feature_value'], df_impact['probability'], 'b-', linewidth=2)
            ax.set_ylabel('Prediction Probability')
        else:
            ax.plot(df_impact['feature_value'], df_impact['prediction'], 'b-', linewidth=2)
            ax.set_ylabel('Prediction')

        # Mark current value
        ax.axvline(instance[feature], color='r', linestyle='--', label='Current Value')

        ax.set_xlabel(f'{feature} Value')
        ax.set_title(f'Impact of {feature} on Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        return fig