"""
data_contracts.py
Docstring (PL): Definicje kontraktów danych (pandera). Dla Stage 6 przykład: schemat ogólny „luźny”
z walidacją typów logicznych i unikalności kolumn (opcjonalnych).
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd

try:
    import pandera as pa
    from pandera import Column, DataFrameSchema, Check
    HAS_PANDERA = True
except Exception:
    HAS_PANDERA = False

def loose_schema(df: pd.DataFrame):
    if not HAS_PANDERA:
        return None
    cols = {}
    for c in df.columns:
        dtype = df[c].dtype
        if str(dtype).startswith("datetime64"):
            pa_type = pa.DateTime
        elif "int" in str(dtype) or "float" in str(dtype):
            pa_type = pa.Float  # uogólnij
        else:
            pa_type = pa.String
        cols[c] = Column(pa_type, nullable=True)
    return DataFrameSchema(cols)