"""
Sanityzacja i normalizacja typów danych w DataFrame.

Funkcjonalności:
- Automatyczna detekcja i konwersja typów
- Unikalizacja nazw kolumn
- Konwersja kategorii na stringi
- Obsługa dat
- Sanityzacja nazw kolumn
"""

import logging
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from backend.error_handler import DataValidationException, handle_errors

logger = logging.getLogger(__name__)


@handle_errors(show_in_ui=False)
def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanityzuje i unikalizuje nazwy kolumn.

    Operacje:
    - Usuwa białe znaki wiodące/końcowe
    - Zamienia wielokrotne spacje na pojedyncze
    - Unikalizuje duplikaty (dodaje sufiks _1, _2, etc.)

    Args:
        df: DataFrame do przetworzenia

    Returns:
        pd.DataFrame: DataFrame z sanityzowanymi nazwami

    Example:
        >>> df = pd.DataFrame({' col ': [1], 'col': [2]})
        >>> df_clean = sanitize_column_names(df)
        >>> df_clean.columns.tolist()
        ['col', 'col_1']
    """
    # Usuń białe znaki i wielokrotne spacje
    clean_names = [re.sub(r'\s+', ' ', str(col).strip()) for col in df.columns]
    
    # Unikalizuj nazwy
    seen = {}
    unique_names = []
    
    for name in clean_names:
        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
    
    df.columns = unique_names
    logger.info(f"Sanityzacja kolumn: {len(set(clean_names))} unikalnych nazw")
    
    return df


@handle_errors(show_in_ui=False)
def detect_and_convert_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Wykrywa i konwertuje typy danych kolumn.

    Konwersje:
    - Stringi numeryczne → numeric
    - Daty → datetime
    - Kategorie → object (string)
    - Boolean → boolean

    Args:
        df: DataFrame do przetworzenia

    Returns:
        Tuple[pd.DataFrame, Dict]: (DataFrame, raport konwersji)

    Example:
        >>> df = pd.DataFrame({'a': ['1', '2', '3'], 'b': ['2021-01-01', '2021-01-02', '2021-01-03']})
        >>> df_converted, report = detect_and_convert_types(df)
        >>> df_converted['a'].dtype
        dtype('int64')
    """
    conversion_report = {
        "converted": [],
        "failed": [],
        "unchanged": [],
    }
    
    df_result = df.copy()
    
    for col in df.columns:
        original_dtype = str(df[col].dtype)
        
        try:
            # Kategorie → string
            if pd.api.types.is_categorical_dtype(df[col]):
                df_result[col] = df[col].astype(str)
                conversion_report["converted"].append({
                    "column": col,
                    "from": original_dtype,
                    "to": "object",
                    "reason": "category_to_string"
                })
                continue
            
            # Próba konwersji na numeric
            if df[col].dtype == 'object':
                # Sprawdź czy to liczby
                numeric_converted = pd.to_numeric(df[col], errors='coerce')
                non_null_ratio = numeric_converted.notna().sum() / len(df)
                
                if non_null_ratio > 0.8:  # Więcej niż 80% to liczby
                    df_result[col] = numeric_converted
                    conversion_report["converted"].append({
                        "column": col,
                        "from": original_dtype,
                        "to": str(numeric_converted.dtype),
                        "reason": "string_to_numeric"
                    })
                    continue
                
                # Sprawdź czy to daty
                try:
                    date_converted = pd.to_datetime(df[col], errors='coerce')
                    date_ratio = date_converted.notna().sum() / len(df)
                    
                    if date_ratio > 0.8:
                        df_result[col] = date_converted
                        conversion_report["converted"].append({
                            "column": col,
                            "from": original_dtype,
                            "to": "datetime64[ns]",
                            "reason": "string_to_datetime"
                        })
                        continue
                except:
                    pass
            
            # Bez zmian
            conversion_report["unchanged"].append({
                "column": col,
                "dtype": original_dtype
            })
        
        except Exception as e:
            logger.warning(f"Nie udało się skonwertować kolumny {col}: {e}")
            conversion_report["failed"].append({
                "column": col,
                "dtype": original_dtype,
                "error": str(e)
            })
    
    logger.info(f"Konwersja typów: {len(conversion_report['converted'])} skonwertowanych, "
                f"{len(conversion_report['failed'])} błędów")
    
    return df_result, conversion_report


@handle_errors(show_in_ui=False)
def ensure_categories_as_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konwertuje wszystkie kolumny kategoryczne na stringi.

    Args:
        df: DataFrame do przetworzenia

    Returns:
        pd.DataFrame: DataFrame ze stringami zamiast kategorii

    Example:
        >>> df = pd.DataFrame({'cat': pd.Categorical(['a', 'b', 'a'])})
        >>> df_str = ensure_categories_as_strings(df)
        >>> df_str['cat'].dtype
        dtype('O')
    """
    df_result = df.copy()
    
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            df_result[col] = df[col].astype(str)
            logger.debug(f"Konwersja kategorii na string: {col}")
    
    return df_result


def sanitize_dataframe(df: pd.DataFrame, aggressive: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Główna funkcja sanityzacji DataFrame.

    Wykonuje:
    1. Sanityzację nazw kolumn
    2. Detekcję i konwersję typów
    3. Konwersję kategorii na stringi
    4. Opcjonalnie: usunięcie kolumn stałych

    Args:
        df: DataFrame do sanityzacji
        aggressive: Czy wykonać agresywne czyszczenie (usuń constant cols, etc.)

    Returns:
        Tuple[pd.DataFrame, Dict]: (DataFrame, raport sanityzacji)

    Raises:
        DataValidationException: Gdy DataFrame jest pusty po sanityzacji

    Example:
        >>> df = pd.DataFrame({' A ': [1, 2], 'A': [3, 4], 'cat': pd.Categorical(['x', 'y'])})
        >>> df_clean, report = sanitize_dataframe(df)
        >>> len(df_clean.columns)
        3
    """
    if df.empty:
        raise DataValidationException("DataFrame jest pusty - nie można sanityzować")
    
    logger.info(f"Rozpoczęcie sanityzacji: {df.shape}")
    
    report = {
        "original_shape": df.shape,
        "operations": [],
    }
    
    # 1. Sanityzacja nazw kolumn
    df_clean = sanitize_column_names(df)
    if not df.columns.equals(df_clean.columns):
        report["operations"].append("column_names_sanitized")
    
    # 2. Detekcja i konwersja typów
    df_clean, conversion_report = detect_and_convert_types(df_clean)
    if conversion_report["converted"]:
        report["operations"].append("types_converted")
        report["conversions"] = conversion_report
    
    # 3. Kategorie → stringi
    df_clean = ensure_categories_as_strings(df_clean)
    
    # 4. Agresywne czyszczenie (opcjonalnie)
    if aggressive:
        # Usuń kolumny stałe
        constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() == 1]
        if constant_cols:
            df_clean = df_clean.drop(columns=constant_cols)
            report["operations"].append(f"removed_{len(constant_cols)}_constant_columns")
            report["removed_constant_columns"] = constant_cols
        
        # Usuń kolumny z >95% braków
        high_missing_cols = [
            col for col in df_clean.columns 
            if df_clean[col].isna().sum() / len(df_clean) > 0.95
        ]
        if high_missing_cols:
            df_clean = df_clean.drop(columns=high_missing_cols)
            report["operations"].append(f"removed_{len(high_missing_cols)}_high_missing_columns")
            report["removed_high_missing_columns"] = high_missing_cols
    
    report["final_shape"] = df_clean.shape
    
    if df_clean.empty:
        raise DataValidationException(
            "DataFrame jest pusty po sanityzacji - sprawdź dane wejściowe"
        )
    
    logger.info(f"Sanityzacja zakończona: {df_clean.shape}")
    
    return df_clean, report


def get_dtype_summary(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Zwraca podsumowanie typów danych w DataFrame.

    Args:
        df: DataFrame do analizy

    Returns:
        Dict: Słownik z kolumnami pogrupowanymi po typach

    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y'], 'c': [1.5, 2.5]})
        >>> summary = get_dtype_summary(df)
        >>> 'int64' in summary
        True
    """
    dtype_groups = {}
    
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        
        if dtype_str not in dtype_groups:
            dtype_groups[dtype_str] = []
        
        dtype_groups[dtype_str].append(col)
    
    return dtype_groups