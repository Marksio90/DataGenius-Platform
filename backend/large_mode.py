
from __future__ import annotations
import pandas as pd

def estimate_mem_mb(df: pd.DataFrame) -> float:
    try:
        return float(df.memory_usage(deep=True).sum() / (1024*1024))
    except Exception:
        return float(df.memory_usage().sum() / (1024*1024))

def detect_large_mode(df: pd.DataFrame) -> bool:
    rows = len(df)
    mem = estimate_mem_mb(df)
    return bool(rows > 500_000 or mem > 512.0)
