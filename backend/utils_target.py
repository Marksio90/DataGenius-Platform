"""
Detekcja kolumny target i typu problemu ML.

Funkcjonalności:
- Automatyczna detekcja kolumny target
- Określenie typu problemu (klasyfikacja/regresja)
- Wykrywanie sygnałów time series
- Walidacja target
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.error_handler import handle_errors

logger = logging.getLogger(__name__)


@handle_errors(show_in_ui=False)
def detect_target_column(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None
) -> Optional[str]:
    """
    Automatycznie wykrywa kolumnę target.

    Heurystyki:
    - Nazwy: target, label, class, y, outcome, prediction, result
    - Kolumny binarne/kategoryczne z małą liczbą unikalnych wartości
    - Ostatnia kolumna (jeśli numeryczna lub kategoryczna)

    Args:
        df: DataFrame
        exclude_cols: Kolumny do wykluczenia

    Returns:
        Optional[str]: Nazwa kolumny target lub None

    Example:
        >>> df = pd.DataFrame({'feature': [1, 2, 3], 'target': [0, 1, 0]})
        >>> target = detect_target_column(df)
        >>> target
        'target'
    """
    exclude_cols = exclude_cols or []

    # Heurystyki nazw
    target_names = [
        'target', 'label', 'class', 'y', 'outcome',
        'prediction', 'result', 'output', 'target_variable'
    ]

    for col in df.columns:
        if col in exclude_cols:
            continue

        col_lower = col.lower()
        for target_name in target_names:
            if target_name in col_lower:
                logger.info(f"Target wykryty na podstawie nazwy: {col}")
                return col

    # Sprawdź kolumny binarne/kategoryczne
    for col in df.columns:
        if col in exclude_cols:
            continue

        n_unique = df[col].nunique()

        # Binarna kolumna
        if n_unique == 2:
            logger.info(f"Target wykryty jako kolumna binarna: {col}")
            return col

        # Kategoryczna z małą liczbą klas (2-20)
        if 2 < n_unique <= 20 and not pd.api.types.is_numeric_dtype(df[col]):
            logger.info(f"Target wykryty jako kolumna kategoryczna: {col}")
            return col

    # Ostatnia kolumna (jeśli sensowna)
    last_col = df.columns[-1]
    if last_col not in exclude_cols:
        n_unique = df[last_col].nunique()
        if n_unique > 1:  # Nie constant
            logger.info(f"Target wykryty jako ostatnia kolumna: {last_col}")
            return last_col

    logger.warning("Nie udało się automatycznie wykryć kolumny target")
    return None


@handle_errors(show_in_ui=False)
def detect_problem_type(df: pd.DataFrame, target_col: str) -> Tuple[str, Dict]:
    """
    Określa typ problemu ML na podstawie kolumny target.

    Typy:
    - 'binary_classification': 2 klasy
    - 'multiclass_classification': >2 klasy
    - 'regression': wartości ciągłe

    Args:
        df: DataFrame
        target_col: Nazwa kolumny target

    Returns:
        Tuple[str, Dict]: (typ_problemu, metadane)

    Example:
        >>> df = pd.DataFrame({'target': [0, 1, 0, 1]})
        >>> problem_type, meta = detect_problem_type(df, 'target')
        >>> problem_type
        'binary_classification'
    """
    if target_col not in df.columns:
        raise ValueError(f"Kolumna {target_col} nie istnieje w DataFrame")

    target_series = df[target_col].dropna()
    n_unique = target_series.nunique()

    metadata = {
        "n_unique": n_unique,
        "n_samples": len(target_series),
        "n_missing": df[target_col].isna().sum(),
    }

    # Klasyfikacja binarna
    if n_unique == 2:
        metadata["classes"] = sorted(target_series.unique().tolist())
        metadata["class_distribution"] = target_series.value_counts().to_dict()
        logger.info(f"Wykryto klasyfikację binarną: {metadata['classes']}")
        return "binary_classification", metadata

    # Klasyfikacja wieloklasowa
    if 2 < n_unique <= 50 and not pd.api.types.is_numeric_dtype(target_series):
        metadata["classes"] = sorted(target_series.unique().tolist())
        metadata["class_distribution"] = target_series.value_counts().to_dict()
        logger.info(f"Wykryto klasyfikację wieloklasową: {n_unique} klas")
        return "multiclass_classification", metadata

    # Regresja (wartości numeryczne lub dużo unikalnych wartości)
    if pd.api.types.is_numeric_dtype(target_series) or n_unique > 50:
        metadata["min"] = float(target_series.min())
        metadata["max"] = float(target_series.max())
        metadata["mean"] = float(target_series.mean())
        metadata["std"] = float(target_series.std())
        logger.info(f"Wykryto regresję: zakres [{metadata['min']:.2f}, {metadata['max']:.2f}]")
        return "regression", metadata

    # Domyślnie: klasyfikacja wieloklasowa
    metadata["classes"] = sorted(target_series.unique().tolist())
    metadata["class_distribution"] = target_series.value_counts().to_dict()
    logger.info(f"Domyślnie: klasyfikacja wieloklasowa ({n_unique} klas)")
    return "multiclass_classification", metadata


@handle_errors(show_in_ui=False)
def detect_timeseries_signals(df: pd.DataFrame) -> Dict:
    """
    Wykrywa sygnały time series w danych.

    Sprawdza:
    - Kolumny dat
    - Monotoniczność indexu
    - Regularność próbkowania

    Args:
        df: DataFrame

    Returns:
        Dict: Informacje o time series

    Example:
        >>> df = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=10)})
        >>> ts_info = detect_timeseries_signals(df)
        >>> ts_info['has_datetime_column']
        True
    """
    ts_info = {
        "has_datetime_column": False,
        "datetime_columns": [],
        "index_is_datetime": False,
        "index_is_monotonic": False,
        "is_likely_timeseries": False,
        "warnings": [],
    }

    # Sprawdź kolumny dat
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if datetime_cols:
        ts_info["has_datetime_column"] = True
        ts_info["datetime_columns"] = datetime_cols

    # Sprawdź index
    if pd.api.types.is_datetime64_any_dtype(df.index):
        ts_info["index_is_datetime"] = True

    if df.index.is_monotonic_increasing or df.index.is_monotonic_decreasing:
        ts_info["index_is_monotonic"] = True

    # Określ czy to likely time series
    if ts_info["has_datetime_column"] or ts_info["index_is_datetime"]:
        ts_info["is_likely_timeseries"] = True

        if not ts_info["index_is_monotonic"]:
            ts_info["warnings"].append(
                "Index nie jest monotoniczna - rozważ sortowanie dla time series"
            )

    if ts_info["is_likely_timeseries"]:
        logger.info("Wykryto sygnały time series w danych")
        ts_info["warnings"].append(
            "Wykryto dane czasowe - rozważ użycie TimeSeriesSplit w CV"
        )

    return ts_info


@handle_errors(show_in_ui=False)
def validate_target_column(
    df: pd.DataFrame,
    target_col: str,
    min_samples_per_class: int = 5
) -> Tuple[bool, List[str]]:
    """
    Waliduje kolumnę target pod kątem przydatności do ML.

    Args:
        df: DataFrame
        target_col: Nazwa kolumny target
        min_samples_per_class: Minimalna liczba próbek na klasę

    Returns:
        Tuple[bool, List[str]]: (czy_poprawny, lista_ostrzeżeń)

    Example:
        >>> df = pd.DataFrame({'target': [0, 1, 0, 1, 0, 1]})
        >>> is_valid, warnings = validate_target_column(df, 'target')
        >>> is_valid
        True
    """
    warnings = []

    if target_col not in df.columns:
        return False, [f"Kolumna {target_col} nie istnieje"]

    target_series = df[target_col]

    # Sprawdź braki
    n_missing = target_series.isna().sum()
    if n_missing > 0:
        warnings.append(
            f"Kolumna target zawiera {n_missing} brakujących wartości "
            f"({n_missing / len(df) * 100:.1f}%)"
        )

    # Sprawdź constant
    if target_series.nunique() == 1:
        return False, ["Kolumna target ma tylko jedną unikalną wartość - nie można trenować modelu"]

    # Dla klasyfikacji: sprawdź rozkład klas
    problem_type, _ = detect_problem_type(df, target_col)

    if "classification" in problem_type:
        value_counts = target_series.value_counts()

        # Sprawdź minimalną liczbę próbek na klasę
        min_class_count = value_counts.min()
        if min_class_count < min_samples_per_class:
            warnings.append(
                f"Najmniej liczna klasa ma tylko {min_class_count} próbek "
                f"(minimum: {min_samples_per_class})"
            )

        # Sprawdź balance klas
        max_ratio = value_counts.max() / value_counts.min()
        if max_ratio > 10:
            warnings.append(
                f"Silnie niezbalansowane klasy (ratio: {max_ratio:.1f}:1) - "
                "rozważ użycie technik balansowania"
            )

    is_valid = len([w for w in warnings if "nie można trenować" in w]) == 0

    return is_valid, warnings