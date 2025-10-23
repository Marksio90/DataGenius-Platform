"""
Auto Feature Generator - automatyczne generowanie features.

Funkcjonalności:
- Polynomial features
- Interaction features
- Aggregation features
- Date/time features
- Text features (TF-IDF)
- Target encoding
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


class AutoFeatureGenerator:
    """
    Automatyczny generator features.
    
    Tworzy nowe features na podstawie istniejących:
    - Polynomial transformations
    - Feature interactions
    - Aggregations
    - Date/time extractions
    """

    def __init__(
        self,
        max_polynomial_degree: int = 2,
        max_interactions: int = 100,
        include_datetime_features: bool = True,
        include_aggregations: bool = True
    ):
        """
        Inicjalizacja auto feature generator.

        Args:
            max_polynomial_degree: Maksymalny stopień polynomial
            max_interactions: Maksymalna liczba interaction features
            include_datetime_features: Czy generować datetime features
            include_aggregations: Czy generować aggregation features
        """
        self.max_polynomial_degree = max_polynomial_degree
        self.max_interactions = max_interactions
        self.include_datetime_features = include_datetime_features
        self.include_aggregations = include_aggregations

        self.generated_features = []
        self.polynomial_transformer = None

        logger.info("Auto Feature Generator zainicjalizowany")

    @handle_errors(show_in_ui=False)
    def fit_transform(
        self,
        df: pd.DataFrame,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        datetime_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generuje nowe features.

        Args:
            df: DataFrame wejściowy
            numeric_cols: Lista kolumn numerycznych (None = auto-detect)
            categorical_cols: Lista kolumn kategorycznych (None = auto-detect)
            datetime_cols: Lista kolumn datetime (None = auto-detect)

        Returns:
            pd.DataFrame: DataFrame z nowymi features
        """
        logger.info("Rozpoczęcie generowania features...")

        df_output = df.copy()

        # Auto-detect column types
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if datetime_cols is None:
            datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        # 1. Polynomial features
        if self.max_polynomial_degree > 1 and len(numeric_cols) > 0:
            df_poly = self._generate_polynomial_features(
                df[numeric_cols],
                numeric_cols
            )
            df_output = pd.concat([df_output, df_poly], axis=1)

        # 2. Interaction features
        if self.max_interactions > 0 and len(numeric_cols) > 1:
            df_interact = self._generate_interaction_features(
                df[numeric_cols],
                numeric_cols
            )
            df_output = pd.concat([df_output, df_interact], axis=1)

        # 3. Datetime features
        if self.include_datetime_features and len(datetime_cols) > 0:
            df_datetime = self._generate_datetime_features(
                df[datetime_cols],
                datetime_cols
            )
            df_output = pd.concat([df_output, df_datetime], axis=1)

        # 4. Aggregation features
        if self.include_aggregations and len(numeric_cols) > 0:
            df_agg = self._generate_aggregation_features(
                df[numeric_cols],
                numeric_cols
            )
            df_output = pd.concat([df_output, df_agg], axis=1)

        # 5. Categorical encoding features (target encoding handled separately)
        # Skipped here - use preprocessing pipeline

        logger.info(f"Wygenerowano {len(df_output.columns) - len(df.columns)} nowych features")

        return df_output

    def _generate_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Generuje polynomial features."""
        logger.info(f"Generowanie polynomial features (degree={self.max_polynomial_degree})...")

        # Limit columns if too many
        if len(columns) > 10:
            columns = columns[:10]
            logger.warning("Ograniczono do 10 kolumn dla polynomial features")

        self.polynomial_transformer = PolynomialFeatures(
            degree=self.max_polynomial_degree,
            include_bias=False,
            interaction_only=False
        )

        # Transform
        poly_array = self.polynomial_transformer.fit_transform(df[columns])

        # Get feature names
        feature_names = self.polynomial_transformer.get_feature_names_out(columns)

        # Remove original features (already in df)
        new_features_mask = [name not in columns for name in feature_names]
        poly_array = poly_array[:, new_features_mask]
        feature_names = feature_names[new_features_mask]

        df_poly = pd.DataFrame(
            poly_array,
            columns=feature_names,
            index=df.index
        )

        self.generated_features.extend(feature_names.tolist())

        logger.info(f"Wygenerowano {len(feature_names)} polynomial features")

        return df_poly

    def _generate_interaction_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Generuje interaction features (pairwise)."""
        logger.info("Generowanie interaction features...")

        new_features = {}

        # Generate pairwise interactions
        count = 0

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if count >= self.max_interactions:
                    break

                col1, col2 = columns[i], columns[j]

                # Multiplication
                feature_name = f"{col1}_x_{col2}"
                new_features[feature_name] = df[col1] * df[col2]

                # Division (with safety)
                feature_name = f"{col1}_div_{col2}"
                new_features[feature_name] = df[col1] / (df[col2] + 1e-8)

                count += 2

            if count >= self.max_interactions:
                break

        df_interact = pd.DataFrame(new_features, index=df.index)

        self.generated_features.extend(df_interact.columns.tolist())

        logger.info(f"Wygenerowano {len(df_interact.columns)} interaction features")

        return df_interact

    def _generate_datetime_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Generuje datetime features."""
        logger.info("Generowanie datetime features...")

        new_features = {}

        for col in columns:
            # Ensure datetime
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')

            # Extract components
            new_features[f"{col}_year"] = df[col].dt.year
            new_features[f"{col}_month"] = df[col].dt.month
            new_features[f"{col}_day"] = df[col].dt.day
            new_features[f"{col}_dayofweek"] = df[col].dt.dayofweek
            new_features[f"{col}_quarter"] = df[col].dt.quarter
            new_features[f"{col}_is_weekend"] = (df[col].dt.dayofweek >= 5).astype(int)

            # Time features
            if df[col].dt.hour.notna().any():
                new_features[f"{col}_hour"] = df[col].dt.hour
                new_features[f"{col}_minute"] = df[col].dt.minute

        df_datetime = pd.DataFrame(new_features, index=df.index)

        self.generated_features.extend(df_datetime.columns.tolist())

        logger.info(f"Wygenerowano {len(df_datetime.columns)} datetime features")

        return df_datetime

    def _generate_aggregation_features(
        self,
        df: pd.DataFrame,
        columns: List[str]
    ) -> pd.DataFrame:
        """Generuje aggregation features (row-wise statistics)."""
        logger.info("Generowanie aggregation features...")

        new_features = {}

        # Row-wise statistics
        new_features['row_mean'] = df[columns].mean(axis=1)
        new_features['row_std'] = df[columns].std(axis=1)
        new_features['row_min'] = df[columns].min(axis=1)
        new_features['row_max'] = df[columns].max(axis=1)
        new_features['row_median'] = df[columns].median(axis=1)
        new_features['row_sum'] = df[columns].sum(axis=1)
        new_features['row_range'] = new_features['row_max'] - new_features['row_min']

        df_agg = pd.DataFrame(new_features, index=df.index)

        self.generated_features.extend(df_agg.columns.tolist())

        logger.info(f"Wygenerowano {len(df_agg.columns)} aggregation features")

        return df_agg

    def get_generated_features(self) -> List[str]:
        """Zwraca listę wygenerowanych features."""
        return self.generated_features

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transformuje nowy dataset używając wcześniej wygenerowanych features.

        Args:
            df: DataFrame do transformacji

        Returns:
            pd.DataFrame: Transformowany DataFrame
        """
        # TODO: Implement transform logic
        # Wymaga zapisania state z fit_transform
        raise NotImplementedError("Transform not yet implemented")