"""
Fairness Checker - analiza fairness i bias w modelach.

Funkcjonalności:
- Demographic parity
- Equal opportunity
- Disparate impact
- Fairness metrics
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class FairnessChecker:
    """
    Checker dla fairness i bias w modelach ML.
    
    Metryki fairness:
    - Demographic Parity
    - Equal Opportunity
    - Equalized Odds
    - Disparate Impact
    """

    def __init__(
        self,
        sensitive_features: List[str]
    ):
        """
        Inicjalizacja fairness checker.

        Args:
            sensitive_features: Lista sensitive features (np. 'gender', 'race')
        """
        self.sensitive_features = sensitive_features

        logger.info(f"Fairness Checker zainicjalizowany (sensitive={sensitive_features})")

    @handle_errors(show_in_ui=False)
    def check_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: pd.DataFrame,
        protected_attribute: str
    ) -> Dict:
        """
        Sprawdza fairness względem protected attribute.

        Args:
            y_true: True labels
            y_pred: Predictions
            X: Features (zawiera protected_attribute)
            protected_attribute: Nazwa protected attribute

        Returns:
            Dict: Fairness metrics
        """
        logger.info(f"Sprawdzanie fairness względem {protected_attribute}...")

        if protected_attribute not in X.columns:
            raise ValueError(f"Protected attribute {protected_attribute} nie istnieje w X")

        # Get unique groups
        groups = X[protected_attribute].unique()

        metrics = {
            'protected_attribute': protected_attribute,
            'groups': {},
            'demographic_parity': None,
            'equal_opportunity': None,
            'disparate_impact': None
        }

        # Compute metrics per group
        for group in groups:
            mask = X[protected_attribute] == group

            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            # Positive rate
            positive_rate = (y_pred_group == 1).mean() if len(y_pred_group) > 0 else 0

            # True positive rate (if binary classification)
            if len(np.unique(y_true)) == 2:
                tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            else:
                tpr = None
                fpr = None

            metrics['groups'][str(group)] = {
                'n_samples': int(mask.sum()),
                'positive_rate': float(positive_rate),
                'tpr': float(tpr) if tpr is not None else None,
                'fpr': float(fpr) if fpr is not None else None
            }

        # Demographic Parity (difference in positive rates)
        positive_rates = [m['positive_rate'] for m in metrics['groups'].values()]
        metrics['demographic_parity'] = float(max(positive_rates) - min(positive_rates))

        # Equal Opportunity (difference in TPR)
        if all(m['tpr'] is not None for m in metrics['groups'].values()):
            tprs = [m['tpr'] for m in metrics['groups'].values()]
            metrics['equal_opportunity'] = float(max(tprs) - min(tprs))

        # Disparate Impact (ratio of positive rates)
        if min(positive_rates) > 0:
            metrics['disparate_impact'] = float(min(positive_rates) / max(positive_rates))

        logger.info("Fairness check zakończony")

        return metrics

    def generate_fairness_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: pd.DataFrame
    ) -> Dict:
        """
        Generuje pełny raport fairness dla wszystkich sensitive features.

        Args:
            y_true: True labels
            y_pred: Predictions
            X: Features

        Returns:
            Dict: Kompletny raport
        """
        logger.info("Generowanie raportu fairness...")

        report = {
            'overall_accuracy': float((y_true == y_pred).mean()),
            'fairness_by_attribute': {}
        }

        for attr in self.sensitive_features:
            if attr in X.columns:
                metrics = self.check_fairness(y_true, y_pred, X, attr)
                report['fairness_by_attribute'][attr] = metrics

        # Overall fairness score
        dp_violations = []
        eo_violations = []

        for metrics in report['fairness_by_attribute'].values():
            if metrics['demographic_parity'] is not None:
                dp_violations.append(metrics['demographic_parity'] > 0.1)
            if metrics['equal_opportunity'] is not None:
                eo_violations.append(metrics['equal_opportunity'] > 0.1)

        report['fairness_violations'] = {
            'demographic_parity': int(sum(dp_violations)),
            'equal_opportunity': int(sum(eo_violations))
        }

        logger.info("Raport fairness wygenerowany")

        return report