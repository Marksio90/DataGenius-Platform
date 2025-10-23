"""
Drift Detector - wykrywanie data drift i concept drift.

Funkcjonalności:
- Data drift (Kolmogorov-Smirnov test)
- Concept drift (model performance degradation)
- Feature drift detection
- Population Stability Index (PSI)
- Alerts i monitoring
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, mean_squared_error

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detector dla data drift i concept drift.
    
    Metody:
    - Kolmogorov-Smirnov test (data drift)
    - Population Stability Index (PSI)
    - Performance monitoring (concept drift)
    - Feature-wise drift
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        reference_target: Optional[np.ndarray] = None,
        reference_predictions: Optional[np.ndarray] = None,
        threshold_ks: float = 0.05,
        threshold_psi: float = 0.1,
        threshold_performance: float = 0.05
    ):
        """
        Inicjalizacja drift detector.

        Args:
            reference_data: Dane referencyjne (train/baseline)
            reference_target: Target referencyjny (opcjonalny)
            reference_predictions: Predykcje referencyjne (opcjonalny)
            threshold_ks: Threshold dla KS test (p-value)
            threshold_psi: Threshold dla PSI
            threshold_performance: Threshold dla degradacji performance
        """
        self.reference_data = reference_data
        self.reference_target = reference_target
        self.reference_predictions = reference_predictions
        self.threshold_ks = threshold_ks
        self.threshold_psi = threshold_psi
        self.threshold_performance = threshold_performance

        # Compute reference statistics
        self.reference_stats = self._compute_stats(reference_data)

        logger.info("Drift Detector zainicjalizowany")

    def _compute_stats(self, data: pd.DataFrame) -> Dict:
        """
        Oblicza statystyki dla danych.

        Args:
            data: DataFrame

        Returns:
            Dict: Statystyki dla każdej kolumny
        """
        stats_dict = {}

        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats_dict[col] = {
                    'type': 'numeric',
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'quantiles': data[col].quantile([0.25, 0.5, 0.75]).to_dict()
                }
            else:
                # Categorical
                value_counts = data[col].value_counts(normalize=True)
                stats_dict[col] = {
                    'type': 'categorical',
                    'categories': value_counts.index.tolist(),
                    'proportions': value_counts.values.tolist()
                }

        return stats_dict

    @handle_errors(show_in_ui=False)
    def detect_data_drift_ks(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None
    ) -> Dict:
        """
        Wykrywa data drift używając Kolmogorov-Smirnov test.

        Args:
            current_data: Aktualne dane
            features: Lista features do sprawdzenia (None = wszystkie numeryczne)

        Returns:
            Dict: Wyniki testu dla każdej feature
        """
        if features is None:
            features = [
                col for col in self.reference_data.columns
                if pd.api.types.is_numeric_dtype(self.reference_data[col])
            ]

        results = {}

        for feature in features:
            if feature not in current_data.columns:
                logger.warning(f"Feature {feature} nie istnieje w current_data")
                continue

            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()

            # KS test
            ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)

            drift_detected = p_value < self.threshold_ks

            results[feature] = {
                'ks_statistic': ks_statistic,
                'p_value': p_value,
                'drift_detected': drift_detected,
                'drift_severity': 'high' if p_value < 0.01 else 'medium' if p_value < 0.05 else 'low'
            }

            if drift_detected:
                logger.warning(
                    f"Data drift wykryty dla {feature}: "
                    f"KS={ks_statistic:.4f}, p-value={p_value:.4f}"
                )

        # Summary
        n_drifted = sum(1 for r in results.values() if r['drift_detected'])
        drift_ratio = n_drifted / len(results) if results else 0

        summary = {
            'n_features_tested': len(results),
            'n_features_drifted': n_drifted,
            'drift_ratio': drift_ratio,
            'overall_drift': drift_ratio > 0.2,  # >20% features drifted
            'feature_results': results
        }

        return summary

    @handle_errors(show_in_ui=False)
    def detect_data_drift_psi(
        self,
        current_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        n_bins: int = 10
    ) -> Dict:
        """
        Wykrywa data drift używając Population Stability Index (PSI).

        Args:
            current_data: Aktualne dane
            features: Lista features do sprawdzenia
            n_bins: Liczba binów

        Returns:
            Dict: PSI dla każdej feature
        """
        if features is None:
            features = [
                col for col in self.reference_data.columns
                if pd.api.types.is_numeric_dtype(self.reference_data[col])
            ]

        results = {}

        for feature in features:
            if feature not in current_data.columns:
                logger.warning(f"Feature {feature} nie istnieje w current_data")
                continue

            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()

            # Create bins based on reference data
            bins = np.linspace(ref_values.min(), ref_values.max(), n_bins + 1)
            bins[0] = -np.inf
            bins[-1] = np.inf

            # Compute distributions
            ref_dist, _ = np.histogram(ref_values, bins=bins)
            curr_dist, _ = np.histogram(curr_values, bins=bins)

            # Normalize
            ref_dist = ref_dist / ref_dist.sum()
            curr_dist = curr_dist / curr_dist.sum()

            # Avoid division by zero
            ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
            curr_dist = np.where(curr_dist == 0, 0.0001, curr_dist)

            # Calculate PSI
            psi = np.sum((curr_dist - ref_dist) * np.log(curr_dist / ref_dist))

            drift_detected = psi > self.threshold_psi

            results[feature] = {
                'psi': psi,
                'drift_detected': drift_detected,
                'drift_severity': 'high' if psi > 0.2 else 'medium' if psi > 0.1 else 'low'
            }

            if drift_detected:
                logger.warning(f"Data drift wykryty dla {feature}: PSI={psi:.4f}")

        # Summary
        n_drifted = sum(1 for r in results.values() if r['drift_detected'])
        drift_ratio = n_drifted / len(results) if results else 0

        summary = {
            'n_features_tested': len(results),
            'n_features_drifted': n_drifted,
            'drift_ratio': drift_ratio,
            'overall_drift': drift_ratio > 0.2,
            'feature_results': results
        }

        return summary

    @handle_errors(show_in_ui=False)
    def detect_concept_drift(
        self,
        model: Any,
        current_data: pd.DataFrame,
        current_target: np.ndarray,
        problem_type: str = 'classification'
    ) -> Dict:
        """
        Wykrywa concept drift (degradacja performance).

        Args:
            model: Model do ewaluacji
            current_data: Aktualne features
            current_target: Aktualny target
            problem_type: Typ problemu

        Returns:
            Dict: Wyniki concept drift
        """
        if self.reference_target is None or self.reference_predictions is None:
            logger.warning("Brak danych referencyjnych - concept drift niemożliwy")
            return {'concept_drift_detected': False, 'reason': 'no_reference_data'}

        # Current predictions
        current_predictions = model.predict(current_data)

        # Compute metrics
        if 'classification' in problem_type:
            ref_metric = accuracy_score(self.reference_target, self.reference_predictions)
            curr_metric = accuracy_score(current_target, current_predictions)
            metric_name = 'accuracy'
        else:
            ref_metric = -mean_squared_error(self.reference_target, self.reference_predictions)
            curr_metric = -mean_squared_error(current_target, current_predictions)
            metric_name = 'neg_mse'

        # Performance degradation
        degradation = ref_metric - curr_metric
        degradation_pct = (degradation / abs(ref_metric)) if ref_metric != 0 else 0

        drift_detected = degradation_pct > self.threshold_performance

        result = {
            'concept_drift_detected': drift_detected,
            'reference_metric': ref_metric,
            'current_metric': curr_metric,
            'degradation': degradation,
            'degradation_pct': degradation_pct,
            'metric_name': metric_name,
            'drift_severity': 'high' if degradation_pct > 0.1 else 'medium' if degradation_pct > 0.05 else 'low'
        }

        if drift_detected:
            logger.warning(
                f"Concept drift wykryty: "
                f"{metric_name} spadło z {ref_metric:.4f} do {curr_metric:.4f} "
                f"({degradation_pct*100:.1f}%)"
            )

        return result

    def full_drift_report(
        self,
        model: Any,
        current_data: pd.DataFrame,
        current_target: Optional[np.ndarray] = None,
        problem_type: str = 'classification'
    ) -> Dict:
        """
        Generuje pełny raport drift.

        Args:
            model: Model
            current_data: Aktualne dane
            current_target: Aktualny target (opcjonalny)
            problem_type: Typ problemu

        Returns:
            Dict: Kompletny raport drift
        """
        logger.info("Generowanie pełnego raportu drift...")

        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_drift_ks': None,
            'data_drift_psi': None,
            'concept_drift': None,
            'overall_drift_detected': False,
            'alerts': []
        }

        # Data drift - KS test
        try:
            ks_result = self.detect_data_drift_ks(current_data)
            report['data_drift_ks'] = ks_result

            if ks_result['overall_drift']:
                report['alerts'].append({
                    'type': 'data_drift_ks',
                    'severity': 'high',
                    'message': f"Data drift wykryty (KS test): {ks_result['n_features_drifted']}/{ks_result['n_features_tested']} features"
                })

        except Exception as e:
            logger.warning(f"Błąd KS test: {e}")

        # Data drift - PSI
        try:
            psi_result = self.detect_data_drift_psi(current_data)
            report['data_drift_psi'] = psi_result

            if psi_result['overall_drift']:
                report['alerts'].append({
                    'type': 'data_drift_psi',
                    'severity': 'high',
                    'message': f"Data drift wykryty (PSI): {psi_result['n_features_drifted']}/{psi_result['n_features_tested']} features"
                })

        except Exception as e:
            logger.warning(f"Błąd PSI test: {e}")

        # Concept drift
        if current_target is not None and model is not None:
            try:
                concept_result = self.detect_concept_drift(
                    model,
                    current_data,
                    current_target,
                    problem_type
                )
                report['concept_drift'] = concept_result

                if concept_result['concept_drift_detected']:
                    report['alerts'].append({
                        'type': 'concept_drift',
                        'severity': concept_result['drift_severity'],
                        'message': f"Concept drift wykryty: performance spadł o {concept_result['degradation_pct']*100:.1f}%"
                    })

            except Exception as e:
                logger.warning(f"Błąd concept drift: {e}")

        # Overall
        report['overall_drift_detected'] = len(report['alerts']) > 0

        logger.info(f"Raport drift wygenerowany: {len(report['alerts'])} alertów")

        return report