"""
runtime_preprocessor.py
Docstring (PL): Zapewnia unikalne, przycięte nazwy kolumn oraz tworzy fingerprint ramki danych
(kształt + typy + próbka). Rozwiązuje kolizje nazw przez sufiksy _1, _2, ...
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import hashlib
import pandas as pd

def _unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        base = c.strip()
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out

def dataframe_fingerprint(df: pd.DataFrame, sample_n: int = 20) -> str:
    """
    Docstring (PL): Tworzy stabilny odcisk (fingerprint) ramki na podstawie kształtu, typów i próbki.
    """
    shape = (len(df), len(df.columns))
    dtypes = tuple((c, str(t)) for c, t in df.dtypes.items())
    sample = df.head(sample_n).to_csv(index=False)
    h = hashlib.sha256()
    h.update(str(shape).encode())
    h.update(str(dtypes).encode())
    h.update(sample.encode())
    return h.hexdigest()

def preprocess_runtime(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Docstring (PL): Zwraca (df2, meta) po przycięciu nazw i rozwiązaniu kolizji.
    """
    df2 = df.copy()
    old_cols = list(df2.columns)
    new_cols = _unique_columns([str(c).strip() for c in old_cols])
    df2.columns = new_cols
    meta = {
        "renamed": {o: n for o, n in zip(old_cols, new_cols) if o != n},
        "fingerprint": dataframe_fingerprint(df2),
        "rows": len(df2),
        "cols": len(new_cols),
    }
    return df2, meta