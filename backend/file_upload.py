# backend/file_upload.py
"""
Robust file loader for TMIV – Advanced ML Platform.

Obsługiwane formaty wejścia:
- CSV/TSV/PSV (auto-separatory, wykrywanie kodowania, BOM)
- Excel (XLS/XLSX; wybór pierwszego niepustego arkusza)
- JSON (records/lines/array → normalizacja do tabeli)
- Parquet (pyarrow/fastparquet – jeśli dostępne)

Public API
----------
load_any_file(uploaded) -> tuple[pd.DataFrame, str]
    Przyjmuje:
      - Streamlit UploadedFile,
      - ścieżkę (str/Path),
      - plikopodobny (binary).
    Zwraca: (DataFrame, nazwa_pliku)

load_example_dataset() -> pd.DataFrame
normalize_columns(df) -> pd.DataFrame
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path
from typing import Any, BinaryIO, Tuple

import pandas as pd

# --------------------------
# Nazwa/rozszerzenie
# --------------------------

_TEXT_EXTS = {".csv", ".tsv", ".txt", ".psv", ".json"}
_BINARY_EXTS = {".xlsx", ".xls", ".parquet"}


def _get_name(obj: Any) -> str:
    # Streamlit UploadedFile: has .name
    n = getattr(obj, "name", None)
    if isinstance(n, str) and n.strip():
        return Path(n).name
    # Path-like
    if isinstance(obj, (str, os.PathLike)):
        return Path(obj).name
    return "uploaded_data"


def _get_bytes(obj: Any) -> bytes:
    """
    Extract raw bytes from UploadedFile / path / file-like / bytes.
    """
    if isinstance(obj, bytes):
        return obj
    if isinstance(obj, (str, os.PathLike)):
        return Path(obj).read_bytes()
    # Streamlit UploadedFile or BinaryIO-like
    if hasattr(obj, "read"):
        pos = None
        try:
            pos = obj.tell()
        except Exception:
            pos = None
        data = obj.read()
        try:
            # try reset for potential re-use
            if pos is not None:
                obj.seek(pos)
        except Exception:
            pass
        return data
    raise TypeError("Unsupported input type; pass a path, bytes, or file-like object")


def _ext_from_name(name: str) -> str:
    e = Path(name).suffix.lower()
    # heurystyka: csv-like
    if e in {".txt"} and re.search(r"\.(csv|tsv|psv)\.txt$", name.lower()):
        e = "." + name.lower().split(".")[-2]
    return e


# --------------------------
# Kodowanie i tekst
# --------------------------

def _detect_encoding(data: bytes) -> str:
    # Prefer 'utf-8-sig' to strip BOM
    try:
        data.decode("utf-8-sig")
        return "utf-8-sig"
    except Exception:
        pass
    # Try plain utf-8
    try:
        data.decode("utf-8")
        return "utf-8"
    except Exception:
        pass
    # Try chardet if available
    try:
        import chardet  # type: ignore

        guess = chardet.detect(data or b"")
        enc = (guess.get("encoding") or "latin1").strip()
        return enc
    except Exception:
        return "latin1"


def _decode_text(data: bytes, preferred: str | None = None) -> str:
    encs = [preferred] if preferred else []
    encs.extend(["utf-8-sig", "utf-8", "latin1"])
    seen = set()
    for enc in encs:
        if not enc or enc in seen:
            continue
        seen.add(enc)
        try:
            return data.decode(enc)
        except Exception:
            continue
    # Last resort: replace errors
    return data.decode("utf-8", errors="replace")


# --------------------------
# Normalizacja kolumn
# --------------------------

def _to_snake(name: str) -> str:
    s = str(name).strip()
    s = s.replace("/", "_").replace("-", "_")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_{2,}", "_", s)
    s = s.strip("_").lower()
    return s or "col"


def _ensure_unique(cols: list[str]) -> list[str]:
    out = []
    used: set[str] = set()
    for c in cols:
        base = c
        k = 1
        while c in used:
            c = f"{base}_{k}"
            k += 1
        out.append(c)
        used.add(c)
    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = [_to_snake(c) for c in df.columns]
    new_cols = _ensure_unique(new_cols)
    df.columns = new_cols
    return df


# --------------------------
# Parsowanie formatów
# --------------------------

def _read_csv_like(text: str, *, name: str) -> pd.DataFrame:
    # Try automatic separator inference with Python engine (sep=None)
    try:
        return pd.read_csv(io.StringIO(text), sep=None, engine="python")
    except Exception:
        pass

    # Fallback attempts: common delimiters
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            # sanity: at least 1 col (non-empty)
            if df.shape[1] >= 1:
                return df
        except Exception:
            continue
    raise ValueError(f"Nie udało się wczytać pliku jako CSV/TSV (plik: {name}).")


def _read_json_like(text: str, *, name: str) -> pd.DataFrame:
    # Try pandas fast paths
    try:
        # JSON Lines (one object per line)
        return pd.read_json(io.StringIO(text), lines=True)
    except Exception:
        pass
    try:
        # Regular records array
        return pd.read_json(io.StringIO(text))
    except Exception:
        pass

    # Manual loads → normalize
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return pd.json_normalize(obj)
        if isinstance(obj, dict):
            # if dict of lists / dict: make one-row frame
            return pd.json_normalize(obj)
    except Exception:
        pass
    raise ValueError(f"Nie udało się wczytać pliku JSON (plik: {name}).")


def _read_excel_bytes(data: bytes, *, name: str) -> pd.DataFrame:
    # Requires openpyxl/xlrd depending on file type
    try:
        # All sheets -> choose largest non-empty
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
        if not sheets:
            raise ValueError("Brak arkuszy w pliku Excel.")
        # pick the first non-empty (or largest by rows*cols)
        non_empty = [(k, v) for k, v in sheets.items() if isinstance(v, pd.DataFrame) and not v.empty]
        if non_empty:
            # score by size
            k, df = max(non_empty, key=lambda kv: kv[1].shape[0] * max(1, kv[1].shape[1]))
            return df
        # else return first sheet even if empty
        return next(iter(sheets.values()))
    except ImportError as e:
        raise ImportError(
            "Do odczytu XLSX/XLS wymagane są pakiety 'openpyxl' (xlsx) i/lub 'xlrd' (xls). "
            "Zainstaluj je i spróbuj ponownie."
        ) from e
    except Exception as e:
        raise ValueError(f"Nie udało się wczytać pliku Excel ({name}): {e}") from e


def _read_parquet_bytes(data: bytes, *, name: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(io.BytesIO(data))
    except ImportError as e:
        raise ImportError(
            "Do odczytu Parquet wymagany jest 'pyarrow' lub 'fastparquet'. "
            "Zainstaluj jeden z nich i spróbuj ponownie."
        ) from e
    except Exception as e:
        raise ValueError(f"Nie udało się wczytać pliku Parquet ({name}): {e}") from e


# --------------------------
# Główne API
# --------------------------

def load_any_file(uploaded: Any) -> Tuple[pd.DataFrame, str]:
    """
    Wczytaj plik dowolnego wspieranego typu i zwróć (df, nazwa_pliku).
    Parametr `uploaded` może być: UploadedFile, ścieżka, bytes lub file-like.
    """
    name = _get_name(uploaded)
    raw = _get_bytes(uploaded)
    ext = _ext_from_name(name)

    # Tekstowe: wykryj kodowanie -> dekoduj do str
    if ext in _TEXT_EXTS or not ext:
        enc = _detect_encoding(raw)
        text = _decode_text(raw, preferred=enc)
        if ext in {".json"} or name.lower().endswith(".json"):
            df = _read_json_like(text, name=name)
        else:
            df = _read_csv_like(text, name=name)
        if df.empty:
            raise ValueError(f"Wczytano pusty plik: {name}")
        return df, name

    # Binaria
    if ext in {".xlsx", ".xls"}:
        df = _read_excel_bytes(raw, name=name)
        if df.empty and df.shape[1] == 0:
            raise ValueError(f"Plik Excel ({name}) nie zawiera danych.")
        return df, name

    if ext == ".parquet":
        df = _read_parquet_bytes(raw, name=name)
        if df.empty and df.shape[1] == 0:
            raise ValueError(f"Plik Parquet ({name}) nie zawiera danych.")
        return df, name

    # Nieznane – spróbuj najpierw CSV, potem JSON
    try:
        enc = _detect_encoding(raw)
        text = _decode_text(raw, preferred=enc)
        return _read_csv_like(text, name=name), name
    except Exception:
        try:
            enc = _detect_encoding(raw)
            text = _decode_text(raw, preferred=enc)
            return _read_json_like(text, name=name), name
        except Exception:
            raise ValueError(f"Nieobsługiwany format pliku lub uszkodzony plik: {name}")


def load_example_dataset() -> pd.DataFrame:
    """
    Załaduj przykładowy dataset `data/avocado.csv` (repozytorium).
    """
    # Szukaj względem root projektu lub bieżącej ścieżki
    candidates = [
        Path("data/avocado.csv"),
        Path(__file__).resolve().parent.parent / "data" / "avocado.csv",
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            return df
    # Awaryjnie – mały dataframe inline (nie powinno się zdarzyć w repo)
    return pd.DataFrame(
        {
            "Date": ["2015-12-27", "2015-12-20", "2015-12-13"],
            "AveragePrice": [1.33, 1.35, 0.93],
            "Total Volume": [64236.62, 54876.98, 118220.22],
            "type": ["conventional", "conventional", "conventional"],
            "year": [2015, 2015, 2015],
            "region": ["Albany", "Albany", "Albany"],
        }
    )


__all__ = [
    "load_any_file",
    "load_example_dataset",
    "normalize_columns",
]
