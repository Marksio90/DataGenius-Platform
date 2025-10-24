"""
file_upload.py
Docstring (PL): Bezpieczne wczytywanie plików danych (CSV/XLSX/Parquet/JSON) z limitami rozmiaru
i czytelnymi kodami błędów TMIV-IO-xxx. Nie loguje zawartości danych. Zwraca (df, report).
"""

from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import io
import os
import json
import pandas as pd

# Stałe kodów błędów
TMIV_OK = "OK"
ERR_TOO_LARGE = "TMIV-IO-001"
ERR_UNSUPPORTED = "TMIV-IO-002"
ERR_PARSE = "TMIV-IO-003"
ERR_MISSING_ENGINE = "TMIV-IO-004"

SUPPORTED_EXT = {".csv", ".json", ".parquet", ".pq", ".xlsx"}

def _get_ext(name: str) -> str:
    name = name.lower()
    for ext in SUPPORTED_EXT:
        if name.endswith(ext):
            return ext if ext != ".pq" else ".parquet"
    return os.path.splitext(name)[1].lower()

def load_from_path(path: str, max_mb: int = 200) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Docstring (PL): Wczytuje dane z podanej ścieżki. Limit rozmiaru domyślnie 200 MB.
    Obsługiwane: CSV, JSON, Parquet, XLSX.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb > max_mb:
        return pd.DataFrame(), {"code": ERR_TOO_LARGE, "message": f"Plik przekracza limit {max_mb} MB", "size_mb": round(size_mb,2)}
    ext = _get_ext(path)
    if ext not in SUPPORTED_EXT:
        return pd.DataFrame(), {"code": ERR_UNSUPPORTED, "message": f"Nieobsługiwane rozszerzenie: {ext}"}
    try:
        if ext == ".csv":
            df = pd.read_csv(path)
        elif ext == ".json":
            df = pd.read_json(path, lines=False)
        elif ext == ".parquet":
            try:
                df = pd.read_parquet(path)  # wymaga pyarrow lub fastparquet
            except Exception as e:
                return pd.DataFrame(), {"code": ERR_MISSING_ENGINE, "message": "Brak silnika Parquet (pyarrow/fastparquet).", "details": str(e)}
        elif ext == ".xlsx":
            try:
                df = pd.read_excel(path, engine=None)  # openpyxl
            except Exception as e:
                return pd.DataFrame(), {"code": ERR_MISSING_ENGINE, "message": "Brak silnika Excel (openpyxl).", "details": str(e)}
        else:
            return pd.DataFrame(), {"code": ERR_UNSUPPORTED, "message": f"Nieobsługiwane rozszerzenie: {ext}"}
    except Exception as e:
        return pd.DataFrame(), {"code": ERR_PARSE, "message": "Błąd parsowania pliku.", "details": str(e)}
    return df, {"code": TMIV_OK, "message": "Wczytano pomyślnie", "rows": len(df), "cols": len(df.columns)}

def load_from_bytes(name: str, data: bytes, max_mb: int = 200) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Docstring (PL): Wczytuje dane z wejścia bajtowego (np. upload w Streamlit).
    """
    size_mb = len(data) / (1024 * 1024)
    if size_mb > max_mb:
        return pd.DataFrame(), {"code": ERR_TOO_LARGE, "message": f"Plik przekracza limit {max_mb} MB", "size_mb": round(size_mb,2)}
    ext = _get_ext(name)
    if ext not in SUPPORTED_EXT:
        return pd.DataFrame(), {"code": ERR_UNSUPPORTED, "message": f"Nieobsługiwane rozszerzenie: {ext}"}
    bio = io.BytesIO(data)
    try:
        if ext == ".csv":
            df = pd.read_csv(bio)
        elif ext == ".json":
            df = pd.read_json(bio, lines=False)
        elif ext == ".parquet":
            try:
                df = pd.read_parquet(bio)
            except Exception as e:
                return pd.DataFrame(), {"code": ERR_MISSING_ENGINE, "message": "Brak silnika Parquet (pyarrow/fastparquet).", "details": str(e)}
        elif ext == ".xlsx":
            try:
                df = pd.read_excel(bio, engine=None)
            except Exception as e:
                return pd.DataFrame(), {"code": ERR_MISSING_ENGINE, "message": "Brak silnika Excel (openpyxl).", "details": str(e)}
        else:
            return pd.DataFrame(), {"code": ERR_UNSUPPORTED, "message": f"Nieobsługiwane rozszerzenie: {ext}"}
    except Exception as e:
        return pd.DataFrame(), {"code": ERR_PARSE, "message": "Błąd parsowania pliku.", "details": str(e)}
    return df, {"code": TMIV_OK, "message": "Wczytano pomyślnie", "rows": len(df), "cols": len(df.columns)}