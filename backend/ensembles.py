"""
Moduł budowy modeli ensemble.

Funkcjonalności:
- Voting Classifier/Regressor
- Stacking
- Wybór najlepszych modeli do ensemble
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier, VotingRegressor

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """
    Builder dla modeli ensemble.
    
    Tworzy ensemble z najlepszych modeli.
    """

    def __init__(self, problem_type: str):
        """
        Inicjalizacja buildera.

        Args:
            problem_type: Typ problemu ML
        """
        self.problem_type = problem_type
        self.is_classification = 'classification' in problem_type

    @handle_errors(show_in_ui=False)
    def build_voting_ensemble(
        self,
        models: List[Tuple[str, Any]],
        voting: str = 'soft'
    ) -> Optional[Any]:
        """
        Buduje Voting Ensemble.

        Args:
            models: Lista krotek (nazwa, model)
            voting: Typ votingu ('soft' lub 'hard' dla klasyfikacji)

        Returns:
            Optional[Any]: Model ensemble lub None

        Example:
            >>> from sklearn.linear_model import LogisticRegression
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> models = [('lr', LogisticRegression()), ('dt', DecisionTreeClassifier())]
            >>> builder = EnsembleBuilder('binary_classification')
            >>> ensemble = builder.build_voting_ensemble(models)
            >>> ensemble is not None
            True
        """
        if len(models) < 2:
            logger.warning("Za mało modeli do utworzenia ensemble (minimum 2)")
            return None

        try:
            if self.is_classification:
                # Sprawdź czy wszystkie modele mają predict_proba dla soft voting
                if voting == 'soft':
                    for name, model in models:
                        if not hasattr(model, 'predict_proba'):
                            logger.warning(f"Model {name} nie ma predict_proba - zmiana na hard voting")
                            voting = 'hard'
                            break

                ensemble = VotingClassifier(
                    estimators=models,
                    voting=voting,
                    n_jobs=-1
                )
            else:
                ensemble = VotingRegressor(
                    estimators=models,
                    n_jobs=-1
                )

            logger.info(f"Voting ensemble utworzony: {len(models)} modeli, voting={voting}")
            return ensemble

        except Exception as e:
            logger.error(f"Błąd tworzenia voting ensemble: {e}")
            return None

    @handle_errors(show_in_ui=False)
    def build_stacking_ensemble(
        self,
        models: List[Tuple[str, Any]],
        final_estimator: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Buduje Stacking Ensemble.

        Args:
            models: Lista krotek (nazwa, model) - base estimators
            final_estimator: Meta-model (None = domyślny)

        Returns:
            Optional[Any]: Model ensemble lub None

        Example:
            >>> from sklearn.linear_model import LogisticRegression
            >>> from sklearn.tree import DecisionTreeClassifier
            >>> models = [('lr', LogisticRegression()), ('dt', DecisionTreeClassifier())]
            >>> builder = EnsembleBuilder('binary_classification')
            >>> ensemble = builder.build_stacking_ensemble(models)
            >>> ensemble is not None
            True
        """
        if len(models) < 2:
            logger.warning("Za mało modeli do utworzenia stacking ensemble (minimum 2)")
            return None

        try:
            if self.is_classification:
                ensemble = StackingClassifier(
                    estimators=models,
                    final_estimator=final_estimator,
                    cv=3,
                    n_jobs=-1
                )
            else:
                ensemble = StackingRegressor(
                    estimators=models,
                    final_estimator=final_estimator,
                    cv=3,
                    n_jobs=-1
                )

            logger.info(f"Stacking ensemble utworzony: {len(models)} base models")
            return ensemble

        except Exception as e:
            logger.error(f"Błąd tworzenia stacking ensemble: {e}")
            return None

    def select_top_models(
        self,
        results: List[dict],
        top_n: int = 3,
        metric: str = 'accuracy'
    ) -> List[Tuple[str, Any]]:
        """
        Wybiera top N modeli do ensemble.

        Args:
            results: Lista wyników modeli
            top_n: Liczba modeli do wybrania
            metric: Metryka do rankingu

        Returns:
            List[Tuple]: Lista (nazwa, model)

        Example:
            >>> results = [
            ...     {'model_name': 'M1', 'model': 'model1', 'test_metrics': {'accuracy': 0.9}},
            ...     {'model_name': 'M2', 'model': 'model2', 'test_metrics': {'accuracy': 0.8}}
            ... ]
            >>> builder = EnsembleBuilder('binary_classification')
            >>> top = builder.select_top_models(results, top_n=2)
            >>> len(top) <= 2
            True
        """
        # Sortuj według metryki
        scored_models = []
        for result in results:
            score = result.get('test_metrics', {}).get(metric)
            if score is None:
                score = result.get('cv_scores', {}).get(metric, {}).get('mean')

            if score is not None:
                # Obsługa metryk negatywnych
                if metric.startswith('neg_'):
                    score = -score
                scored_models.append((result['model_name'], result['model'], score))

        # Sortuj od najlepszego
        scored_models.sort(key=lambda x: x[2], reverse=True)

        # Wybierz top N
        top_models = [(name, model) for name, model, score in scored_models[:top_n]]

        logger.info(f"Wybrano {len(top_models)} najlepszych modeli dla ensemble")
        return top_models