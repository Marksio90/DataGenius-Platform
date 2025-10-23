"""
Feature Selector - selekcja najlepszych features.

Funkcjonalności:
- Variance threshold
- Correlation filtering
- Mutual information
- Recursive feature elimination (RFE)
- Feature importance-based selection
- L1-based selection (Lasso)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LogisticRegression

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Selektor features.
    
    Metody selekcji:
    - Variance threshold
    - Correlation
    - Mutual information
    - RFE
    - Model-based (feature importance, L1)
    """

    def __init__(
        self,
        problem_type: str = 'classification',
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        n_features_to_select: Optional[int] = None
    ):
        """
        Inicjalizacja feature selector.

        Args:
            problem_type: Typ problemu
            variance_threshold: Threshold dla variance filtering
            correlation_threshold: Threshold dla correlation filtering
            n_features_to_select: Docelowa liczba features (None = auto)
        """
        self.problem_type = problem_type
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.n_features_to_select = n_features_to_select

        self.selected_features = []
        self.feature_scores = {}

        logger.info("Feature Selector zainicjalizowany")

    @handle_errors(show_in_ui=False)
    def select_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        methods: List[str] = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Przeprowadza selekcję features.

        Args:
            X: Features DataFrame
            y: Target
            methods: Lista metod do użycia (None = wszystkie)

        Returns:
            Tuple[pd.DataFrame, List[str]]: (X_selected, selected_feature_names)
        """
        if methods is None:
            methods = ['variance', 'correlation', 'mutual_info']

        logger.info(f"Selekcja features: {len(X.columns)} początkowych features")

        selected_mask = np.ones(len(X.columns), dtype=bool)

        # 1. Variance threshold
        if 'variance' in methods:
            variance_mask = self._select_by_variance(X)
            selected_mask &= variance_mask
            logger.info(f"Po variance: {selected_mask.sum()} features")

        # 2. Correlation filtering
        if 'correlation' in methods:
            correlation_mask = self._select_by_correlation(X.loc[:, selected_mask])
            # Update mask
            temp_mask = selected_mask.copy()
            temp_mask[selected_mask] = correlation_mask
            selected_mask = temp_mask
            logger.info(f"Po correlation: {selected_mask.sum()} features")

        # 3. Mutual information
        if 'mutual_info' in methods:
            mi_scores = self._select_by_mutual_info(
                X.loc[:, selected_mask],
                y
            )
            self.feature_scores['mutual_info'] = mi_scores

        # 4. RFE (if requested and n_features_to_select specified)
        if 'rfe' in methods and self.n_features_to_select:
            rfe_mask = self._select_by_rfe(
                X.loc[:, selected_mask],
                y
            )
            temp_mask = selected_mask.copy()
            temp_mask[selected_mask] = rfe_mask
            selected_mask = temp_mask
            logger.info(f"Po RFE: {selected_mask.sum()} features")

        # Final selection
        self.selected_features = X.columns[selected_mask].tolist()

        X_selected = X.loc[:, selected_mask]

        logger.info(f"Selekcja zakończona: {len(self.selected_features)} features wybranych")

        return X_selected, self.selected_features

    def _select_by_variance(self, X: pd.DataFrame) -> np.ndarray:
        """Selekcja przez variance threshold."""
        selector = VarianceThreshold(threshold=self.variance_threshold)

        try:
            selector.fit(X)
            mask = selector.get_support()
        except Exception as e:
            logger.warning(f"Błąd variance selection: {e}")
            mask = np.ones(len(X.columns), dtype=bool)

        return mask

    def _select_by_correlation(self, X: pd.DataFrame) -> np.ndarray:
        """Selekcja przez korelację (usuwa wysoko skorelowane)."""
        # Compute correlation matrix
        corr_matrix = X.corr().abs()

        # Upper triangle
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation > threshold
        to_drop = [
            column for column in upper.columns
            if any(upper[column] > self.correlation_threshold)
        ]

        mask = ~X.columns.isin(to_drop)

        return mask.values

    def _select_by_mutual_info(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Selekcja przez mutual information."""
        if 'classification' in self.problem_type:
            mi_scores = mutual_info_classif(X, y, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y, random_state=42)

        # Create dict
        mi_dict = dict(zip(X.columns, mi_scores))

        return mi_dict

    def _select_by_rfe(
        self,
        X: pd.DataFrame,
        y: np.ndarray
    ) -> np.ndarray:
        """Selekcja przez RFE."""
        # Choose estimator
        if 'classification' in self.problem_type:
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)

        # RFE
        selector = RFE(
            estimator,
            n_features_to_select=self.n_features_to_select,
            step=1
        )

        try:
            selector.fit(X, y)
            mask = selector.get_support()
        except Exception as e:
            logger.warning(f"Błąd RFE: {e}")
            mask = np.ones(len(X.columns), dtype=bool)

        return mask

    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """
        Zwraca ranking features według mutual information.

        Returns:
            pd.DataFrame: Ranking
        """
        if 'mutual_info' not in self.feature_scores:
            return pd.DataFrame()

        df_ranking = pd.DataFrame(
            list(self.feature_scores['mutual_info'].items()),
            columns=['feature', 'score']
        )

        df_ranking = df_ranking.sort_values('score', ascending=False)

        return df_ranking

    def select_top_k_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        k: int = 20
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Wybiera top K features według mutual information.

        Args:
            X: Features
            y: Target
            k: Liczba features

        Returns:
            Tuple[pd.DataFrame, List[str]]: (X_selected, selected_features)
        """
        if 'classification' in self.problem_type:
            selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(mutual_info_regression, k=min(k, X.shape[1]))

        X_selected = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()].tolist()

        logger.info(f"Wybrano top {len(selected_features)} features")

        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features