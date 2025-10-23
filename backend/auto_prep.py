"""
Automatyczny preprocessing - wrapper łączący wszystkie kroki.

Funkcjonalności:
- Orchestracja preprocessingu
- Przygotowanie danych do treningu
- Cache preprocessingu
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from backend.error_handler import handle_errors
from backend.runtime_preprocessor import RuntimePreprocessor
from backend.utils_target import detect_problem_type

logger = logging.getLogger(__name__)


@handle_errors(show_in_ui=False)
def auto_preprocess(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RuntimePreprocessor, Dict]:
    """
    Automatyczny preprocessing danych.

    Args:
        df: DataFrame
        target_col: Nazwa kolumny target
        test_size: Rozmiar zbioru testowego

    Returns:
        Tuple: (X_train, X_test, y_train, y_test, preprocessor, metadata)

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10], 'target': [0, 1, 0, 1, 0]})
        >>> X_tr, X_te, y_tr, y_te, prep, meta = auto_preprocess(df, 'target')
        >>> len(X_tr) > 0
        True
    """
    logger.info("Rozpoczęcie automatycznego preprocessingu")

    # Wykryj typ problemu
    problem_type, problem_meta = detect_problem_type(df, target_col)
    logger.info(f"Typ problemu: {problem_type}")

    # Utwórz preprocessor
    preprocessor = RuntimePreprocessor(problem_type)

    # Split train/test
    X_train_df, X_test_df, y_train, y_test = preprocessor.split_data(
        df, target_col, test_size=test_size
    )

    # Buduj i fituj preprocessor features
    preprocessor.build_preprocessor(X_train_df)
    X_train, X_test, feature_names = preprocessor.fit_transform(X_train_df, X_test_df)

    # Enkoduj target
    y_train_enc, y_test_enc = preprocessor.encode_target(y_train, y_test)

    metadata = {
        "problem_type": problem_type,
        "problem_metadata": problem_meta,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "feature_names": feature_names,
        "class_names": preprocessor.get_class_names(),
    }

    logger.info(f"Preprocessing zakończony: {metadata['n_features']} features, "
                f"{metadata['n_train']} train, {metadata['n_test']} test")

    return X_train, X_test, y_train_enc, y_test_enc, preprocessor, metadata