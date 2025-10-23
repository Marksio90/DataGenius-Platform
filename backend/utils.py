"""
Podstawowe funkcje pomocnicze używane w całej aplikacji.

Funkcjonalności:
- Hashowanie dataframe dla cache
- Formatowanie liczb
- Generowanie timestampów
- Sanityzacja nazw plików
"""

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def hash_dataframe(df: pd.DataFrame) -> str:
    """
    Generuje hash dataframe dla celów cache.

    Args:
        df: DataFrame do zahashowania

    Returns:
        str: Hash SHA256 dataframe

    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> hash_df = hash_dataframe(df)
        >>> len(hash_df)
        64
    """
    try:
        # Konwersja do bytes poprzez wartości i kolumny
        df_bytes = pd.util.hash_pandas_object(df, index=True).values
        combined = f"{df.shape}_{df.columns.tolist()}_{df_bytes.sum()}".encode()
        return hashlib.sha256(combined).hexdigest()
    except Exception as e:
        # Fallback na prostszy hash
        simple_hash = f"{df.shape}_{df.columns.tolist()}".encode()
        return hashlib.sha256(simple_hash).hexdigest()


def format_number(value: Union[int, float], decimals: int = 2) -> str:
    """
    Formatuje liczbę do czytelnej postaci.

    Args:
        value: Wartość do sformatowania
        decimals: Liczba miejsc po przecinku

    Returns:
        str: Sformatowana wartość

    Example:
        >>> format_number(1234.5678, 2)
        '1,234.57'
    """
    if pd.isna(value):
        return "N/A"

    if isinstance(value, (int, np.integer)):
        return f"{value:,}"

    return f"{value:,.{decimals}f}"


def sanitize_filename(filename: str) -> str:
    """
    Sanityzuje nazwę pliku usuwając niebezpieczne znaki.

    Args:
        filename: Nazwa pliku do sanityzacji

    Returns:
        str: Bezpieczna nazwa pliku

    Example:
        >>> sanitize_filename("my file@2024.csv")
        'my_file_2024.csv'
    """
    # Usuń niebezpieczne znaki
    safe_name = re.sub(r'[^\w\s\-\.]', '_', filename)
    # Usuń wielokrotne spacje/underscores
    safe_name = re.sub(r'[\s_]+', '_', safe_name)
    # Ograniczenie długości
    name_parts = safe_name.rsplit('.', 1)
    if len(name_parts) == 2:
        name, ext = name_parts
        name = name[:200]  # Max 200 znaków dla nazwy
        safe_name = f"{name}.{ext}"
    else:
        safe_name = safe_name[:250]

    return safe_name


def get_timestamp(format_str: str = "%Y%m%d_%H%M%S") -> str:
    """
    Generuje timestamp w podanym formacie.

    Args:
        format_str: Format daty (domyślnie: YYYYMMDD_HHMMSS)

    Returns:
        str: Timestamp

    Example:
        >>> ts = get_timestamp()
        >>> len(ts)
        15
    """
    return datetime.now().strftime(format_str)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Bezpieczne dzielenie z obsługą dzielenia przez zero.

    Args:
        numerator: Licznik
        denominator: Mianownik
        default: Wartość domyślna przy dzieleniu przez 0

    Returns:
        float: Wynik dzielenia lub wartość domyślna

    Example:
        >>> safe_divide(10, 2)
        5.0
        >>> safe_divide(10, 0)
        0.0
    """
    try:
        if denominator == 0 or pd.isna(denominator):
            return default
        result = numerator / denominator
        return default if pd.isna(result) else result
    except (ZeroDivisionError, TypeError):
        return default


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Zapewnia istnienie katalogu, tworząc go jeśli nie istnieje.

    Args:
        path: Ścieżka do katalogu

    Returns:
        Path: Ścieżka Path do katalogu

    Example:
        >>> p = ensure_dir("outputs/test")
        >>> p.exists()
        True
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def get_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """
    Zwraca informacje o zużyciu pamięci przez DataFrame.

    Args:
        df: DataFrame do analizy

    Returns:
        Dict: Słownik z informacjami o pamięci

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> info = get_memory_usage(df)
        >>> 'total_mb' in info
        True
    """
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 ** 2)

    return {
        "total_bytes": f"{memory_bytes:,}",
        "total_mb": f"{memory_mb:.2f}",
        "per_column": df.memory_usage(deep=True).to_dict(),
    }


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Obcina string do maksymalnej długości.

    Args:
        text: Tekst do obcięcia
        max_length: Maksymalna długość
        suffix: Sufiks dodawany do obciętego tekstu

    Returns:
        str: Obcięty tekst

    Example:
        >>> truncate_string("Very long text here", 10)
        'Very lo...'
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def get_column_stats_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Zwraca podstawowe statystyki DataFrame.

    Args:
        df: DataFrame do analizy

    Returns:
        Dict: Słownik ze statystykami

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        >>> stats = get_column_stats_summary(df)
        >>> stats['n_rows']
        3
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_numeric": len(numeric_cols),
        "n_categorical": len(categorical_cols),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_percentage": float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
    }