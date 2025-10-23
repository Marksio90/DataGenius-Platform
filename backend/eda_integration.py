"""
Moduł analizy eksploracyjnej danych (EDA).

Funkcjonalności:
- Statystyki opisowe
- Analiza rozkładów
- Macierz korelacji
- Wykrywanie outlierów
- Analiza brakujących wartości
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from backend.error_handler import EDAException, handle_errors

logger = logging.getLogger(__name__)


@handle_errors(show_in_ui=False)
def compute_basic_statistics(df: pd.DataFrame) -> Dict:
    """
    Oblicza podstawowe statystyki dla DataFrame.

    Args:
        df: DataFrame do analizy

    Returns:
        Dict: Słownik ze statystykami

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        >>> stats = compute_basic_statistics(df)
        >>> stats['n_rows']
        3
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    stats_dict = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "n_datetime": len(datetime_cols),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
        "memory_usage_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
        "duplicated_rows": int(df.duplicated().sum()),
    }

    # Analiza brakujących wartości
    missing_info = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            missing_info.append({
                "column": col,
                "n_missing": int(n_missing),
                "pct_missing": float(n_missing / len(df) * 100),
            })

    stats_dict["missing_values"] = missing_info
    stats_dict["total_missing_cells"] = int(df.isna().sum().sum())

    return stats_dict


@handle_errors(show_in_ui=False)
def compute_numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Oblicza szczegółowe statystyki dla kolumn numerycznych.

    Args:
        df: DataFrame do analizy

    Returns:
        pd.DataFrame: DataFrame ze statystykami

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        >>> stats_df = compute_numeric_statistics(df)
        >>> 'mean' in stats_df.columns
        True
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return pd.DataFrame()

    stats_list = []

    for col in numeric_df.columns:
        col_data = numeric_df[col].dropna()

        if len(col_data) == 0:
            continue

        col_stats = {
            "column": col,
            "count": len(col_data),
            "mean": float(col_data.mean()),
            "std": float(col_data.std()),
            "min": float(col_data.min()),
            "25%": float(col_data.quantile(0.25)),
            "50%": float(col_data.median()),
            "75%": float(col_data.quantile(0.75)),
            "max": float(col_data.max()),
            "skewness": float(col_data.skew()),
            "kurtosis": float(col_data.kurtosis()),
            "n_unique": int(col_data.nunique()),
        }

        # Outliers (IQR method)
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

        col_stats["n_outliers"] = int(n_outliers)
        col_stats["pct_outliers"] = float(n_outliers / len(col_data) * 100)

        stats_list.append(col_stats)

    return pd.DataFrame(stats_list)


@handle_errors(show_in_ui=False)
def compute_categorical_statistics(df: pd.DataFrame, max_categories: int = 50) -> List[Dict]:
    """
    Oblicza statystyki dla kolumn kategorycznych.

    Args:
        df: DataFrame do analizy
        max_categories: Maksymalna liczba kategorii do szczegółowej analizy

    Returns:
        List[Dict]: Lista słowników ze statystykami kategorii

    Example:
        >>> df = pd.DataFrame({'cat': ['a', 'b', 'a', 'c']})
        >>> stats = compute_categorical_statistics(df)
        >>> len(stats) > 0
        True
    """
    categorical_df = df.select_dtypes(include=['object'])

    if categorical_df.empty:
        return []

    stats_list = []

    for col in categorical_df.columns:
        col_data = categorical_df[col].dropna()

        if len(col_data) == 0:
            continue

        value_counts = col_data.value_counts()
        n_unique = len(value_counts)

        col_stats = {
            "column": col,
            "n_unique": n_unique,
            "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
            "most_common_freq": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
            "most_common_pct": float(value_counts.iloc[0] / len(col_data) * 100) if len(value_counts) > 0 else 0,
        }

        # Top kategorie (jeśli nie za dużo)
        if n_unique <= max_categories:
            top_categories = []
            for val, count in value_counts.head(10).items():
                top_categories.append({
                    "value": str(val),
                    "count": int(count),
                    "pct": float(count / len(col_data) * 100),
                })
            col_stats["top_categories"] = top_categories
        else:
            col_stats["top_categories"] = None
            col_stats["warning"] = f"Zbyt wiele kategorii ({n_unique}) do wyświetlenia"

        stats_list.append(col_stats)

    return stats_list


@handle_errors(show_in_ui=False)
def compute_correlation_matrix(
    df: pd.DataFrame,
    method: str = "pearson",
    min_correlation: float = 0.0
) -> Optional[pd.DataFrame]:
    """
    Oblicza macierz korelacji dla kolumn numerycznych.

    Args:
        df: DataFrame do analizy
        method: Metoda korelacji ('pearson', 'spearman', 'kendall')
        min_correlation: Minimalna wartość korelacji do uwzględnienia

    Returns:
        Optional[pd.DataFrame]: Macierz korelacji lub None

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 4, 6]})
        >>> corr = compute_correlation_matrix(df)
        >>> corr is not None
        True
    """
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] < 2:
        logger.warning("Zbyt mało kolumn numerycznych do obliczenia korelacji")
        return None

    try:
        corr_matrix = numeric_df.corr(method=method)

        # Filtruj niskie korelacje
        if min_correlation > 0:
            mask = np.abs(corr_matrix) >= min_correlation
            corr_matrix = corr_matrix.where(mask)

        return corr_matrix

    except Exception as e:
        logger.error(f"Błąd obliczania korelacji: {e}")
        return None


@handle_errors(show_in_ui=False)
def find_high_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7
) -> List[Tuple[str, str, float]]:
    """
    Znajduje pary kolumn z wysoką korelacją.

    Args:
        corr_matrix: Macierz korelacji
        threshold: Próg korelacji (wartość bezwzględna)

    Returns:
        List[Tuple]: Lista krotek (col1, col2, correlation)

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [2, 4, 6], 'c': [1, 1, 1]})
        >>> corr = df.corr()
        >>> high_corr = find_high_correlations(corr, threshold=0.9)
        >>> len(high_corr) > 0
        True
    """
    high_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]

            if pd.notna(corr_value) and abs(corr_value) >= threshold:
                high_corr.append((col1, col2, float(corr_value)))

    # Sortuj po wartości bezwzględnej korelacji
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)

    return high_corr


@handle_errors(show_in_ui=False)
def detect_distribution_type(series: pd.Series) -> str:
    """
    Wykrywa typ rozkładu danych numerycznych.

    Args:
        series: Seria danych do analizy

    Returns:
        str: Typ rozkładu ('normal', 'skewed', 'bimodal', 'uniform', 'unknown')

    Example:
        >>> s = pd.Series(np.random.normal(0, 1, 1000))
        >>> dist_type = detect_distribution_type(s)
        >>> dist_type in ['normal', 'skewed', 'bimodal', 'uniform', 'unknown']
        True
    """
    clean_data = series.dropna()

    if len(clean_data) < 20:
        return "unknown"

    try:
        # Test normalności (Shapiro-Wilk dla małych próbek, Anderson dla większych)
        if len(clean_data) < 5000:
            _, p_value = stats.shapiro(clean_data)
            if p_value > 0.05:
                return "normal"

        # Skośność
        skewness = clean_data.skew()
        if abs(skewness) > 1:
            return "skewed"

        # Kurtoza (bimodal często ma ujemną kurtozę)
        kurtosis = clean_data.kurtosis()
        if kurtosis < -1:
            return "bimodal"

        # Uniformity test (prosta heurystyka)
        hist, _ = np.histogram(clean_data, bins=10)
        cv = np.std(hist) / np.mean(hist) if np.mean(hist) > 0 else float('inf')
        if cv < 0.3:
            return "uniform"

        return "unknown"

    except Exception as e:
        logger.debug(f"Błąd wykrywania rozkładu: {e}")
        return "unknown"


def perform_eda(df: pd.DataFrame, include_profiling: bool = False) -> Dict:
    """
    Główna funkcja wykonująca pełną analizę EDA.

    Args:
        df: DataFrame do analizy
        include_profiling: Czy dołączyć profiling (ydata-profiling) - może być czasochłonne

    Returns:
        Dict: Kompletny raport EDA

    Raises:
        EDAException: Gdy analiza nie powiedzie się

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        >>> eda_report = perform_eda(df)
        >>> 'basic_stats' in eda_report
        True
    """
    if df.empty:
        raise EDAException("DataFrame jest pusty - nie można wykonać EDA")

    logger.info(f"Rozpoczęcie analizy EDA dla DataFrame: {df.shape}")

    eda_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "basic_stats": compute_basic_statistics(df),
        "numeric_stats": compute_numeric_statistics(df).to_dict('records'),
        "categorical_stats": compute_categorical_statistics(df),
    }

    # Korelacje
    corr_matrix = compute_correlation_matrix(df)
    if corr_matrix is not None:
        eda_report["correlation_matrix"] = corr_matrix
        eda_report["high_correlations"] = find_high_correlations(corr_matrix)
    else:
        eda_report["correlation_matrix"] = None
        eda_report["high_correlations"] = []

    # Analiza rozkładów dla kolumn numerycznych
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    distribution_types = {}
    for col in numeric_cols:
        distribution_types[col] = detect_distribution_type(df[col])

    eda_report["distribution_types"] = distribution_types

    # Profiling (opcjonalnie)
    if include_profiling:
        try:
            from ydata_profiling import ProfileReport
            logger.info("Generowanie profilu ydata-profiling (może zająć chwilę)...")
            profile = ProfileReport(df, minimal=True, explorative=False)
            eda_report["profile_html"] = profile.to_html()
            logger.info("Profil wygenerowany pomyślnie")
        except ImportError:
            logger.warning("ydata-profiling nie jest dostępny")
            eda_report["profile_html"] = None
        except Exception as e:
            logger.warning(f"Błąd generowania profilu: {e}")
            eda_report["profile_html"] = None
    else:
        eda_report["profile_html"] = None

    logger.info("Analiza EDA zakończona pomyślnie")

    return eda_report