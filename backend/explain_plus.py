"""
Moduł zaawansowanej interpretowalności modeli.

Funkcjonalności:
- Feature importance (różne metody)
- Permutation importance
- Agregacja FI z wielu modeli
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


@handle_errors(show_in_ui=False)
def get_feature_importance_from_model(
    model: Any,
    feature_names: List[str]
) -> Optional[pd.DataFrame]:
    """
    Pobiera feature importance z modelu (jeśli dostępne).

    Args:
        model: Wytrenowany model
        feature_names: Lista nazw features

    Returns:
        Optional[pd.DataFrame]: DataFrame z importances lub None

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        >>> model = RandomForestClassifier(random_state=42)
        >>> model.fit(X, y)
        >>> fi = get_feature_importance_from_model(model, [f'f{i}' for i in range(5)])
        >>> fi is not None
        True
    """
    try:
        # Próbuj różne atrybuty
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Dla modeli liniowych - wartość bezwzględna współczynników
            coef = model.coef_
            if len(coef.shape) > 1:
                # Multi-class: średnia po klasach
                importances = np.abs(coef).mean(axis=0)
            else:
                importances = np.abs(coef)
        else:
            logger.debug(f"Model {type(model).__name__} nie ma feature_importances ani coef")
            return None

        # Utwórz DataFrame
        df_fi = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })

        df_fi = df_fi.sort_values('importance', ascending=False)

        return df_fi

    except Exception as e:
        logger.warning(f"Błąd pobierania feature importance: {e}")
        return None


@handle_errors(show_in_ui=False)
def compute_permutation_importance(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42
) -> Optional[pd.DataFrame]:
    """
    Oblicza permutation importance.

    Args:
        model: Wytrenowany model
        X: Features
        y: Target
        feature_names: Lista nazw features
        n_repeats: Liczba powtórzeń permutacji
        random_state: Random state

    Returns:
        Optional[pd.DataFrame]: DataFrame z importances lub None

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        >>> model = RandomForestClassifier(random_state=42)
        >>> model.fit(X, y)
        >>> pi = compute_permutation_importance(model, X, y, [f'f{i}' for i in range(5)])
        >>> pi is not None
        True
    """
    try:
        logger.info("Obliczanie permutation importance...")

        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        df_pi = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std,
        })

        df_pi = df_pi.sort_values('importance_mean', ascending=False)

        logger.info("Permutation importance obliczone")
        return df_pi

    except Exception as e:
        logger.warning(f"Błąd obliczania permutation importance: {e}")
        return None


@handle_errors(show_in_ui=False)
def aggregate_feature_importance(
    results: List[Dict],
    feature_names: List[str],
    method: str = 'mean'
) -> pd.DataFrame:
    """
    Agreguje feature importance z wielu modeli.

    Args:
        results: Lista wyników modeli
        feature_names: Lista nazw features
        method: Metoda agregacji ('mean', 'median', 'max')

    Returns:
        pd.DataFrame: Zagregowane feature importance

    Example:
        >>> results = []  # Przykładowe wyniki
        >>> features = ['f1', 'f2', 'f3']
        >>> # agg_fi = aggregate_feature_importance(results, features)
    """
    # Zbierz wszystkie feature importances
    all_importances = []

    for result in results:
        model = result.get('model')
        if model is None:
            continue

        fi = get_feature_importance_from_model(model, feature_names)
        if fi is not None:
            all_importances.append(fi.set_index('feature')['importance'])

    if not all_importances:
        logger.warning("Brak feature importances do agregacji")
        # Zwróć puste
        return pd.DataFrame({'feature': feature_names, 'importance': 0.0})

    # Utwórz DataFrame z wszystkimi importances
    df_all = pd.DataFrame(all_importances).T

    # Agreguj
    if method == 'mean':
        aggregated = df_all.mean(axis=1)
    elif method == 'median':
        aggregated = df_all.median(axis=1)
    elif method == 'max':
        aggregated = df_all.max(axis=1)
    else:
        aggregated = df_all.mean(axis=1)

    # Utwórz wynikowy DataFrame
    df_result = pd.DataFrame({
        'feature': aggregated.index,
        'importance': aggregated.values
    })

    df_result = df_result.sort_values('importance', ascending=False)

    logger.info(f"Feature importance zagregowane ({method}) z {len(all_importances)} modeli")

    return df_result


def get_top_features(
    df_fi: pd.DataFrame,
    top_n: int = 10
) -> List[str]:
    """
    Zwraca top N najważniejszych features.

    Args:
        df_fi: DataFrame z feature importance
        top_n: Liczba features

    Returns:
        List[str]: Lista nazw features

    Example:
        >>> df = pd.DataFrame({'feature': ['a', 'b', 'c'], 'importance': [0.5, 0.3, 0.2]})
        >>> top = get_top_features(df, top_n=2)
        >>> len(top)
        2
    """
    return df_fi.head(top_n)['feature'].tolist()