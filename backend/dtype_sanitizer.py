"""
dtype_sanitizer.py
Docstring (PL): Normalizuje typy danych w DataFrame: kategorie→string, próba konwersji dat,
wyraźne raportowanie zmian typów.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

def sanitize_dtypes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Docstring (PL): Zwraca (df_sanitized, report) z listą zmian typów.
    """
    df = df.copy()
    report = {"changes": {}}

    # Kategorie → string
    for col in df.select_dtypes(include=["category"]).columns:
        report["changes"][col] = {"from": "category", "to": "string"}
        df[col] = df[col].astype("string")

    # Próbna konwersja kolumn (object) do daty lub liczby tam gdzie ma sens
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            # Czy wygląda na datę?
            sample = df[col].dropna().astype(str).head(50)
            looks_like_date = sample.str.contains(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[./]\d{1,2}[./]\d{2,4}", regex=True).mean() > 0.6
            if looks_like_date:
                try:
                    df[col] = pd.to_datetime(df[col], errors="raise", utc=False, infer_datetime_format=True)
                    report["changes"][col] = {"from": "object/string", "to": "datetime64[ns]"}
                    continue
                except Exception:
                    pass
            # Próba konwersji do liczby
            try:
                coerced = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="raise")
                # Akceptuj tylko, jeśli rzeczywiście dużo udało się przekonwertować
                if coerced.notna().mean() > 0.9:
                    df[col] = coerced
                    report["changes"][col] = {"from": "object/string", "to": "float64"}
            except Exception:
                # zostaje string
                df[col] = df[col].astype("string")

    # Upewnij się, że wszystkie pozostałe obiekty są stringami (nie Python object)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("string")
        report["changes"].setdefault(col, {"from": "object", "to": "string"})

    return df, report