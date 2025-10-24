
from __future__ import annotations
import pandas as pd
from typing import Dict, Any

from backend.large_mode import detect_large_mode, estimate_mem_mb

def simple_profile(df: pd.DataFrame, max_cols: int = 50) -> Dict[str, Any]:
    """Lekki profil EDA (czasowo bez ciężkich bibliotek)."""
    large = detect_large_mode(df)
    df_view = df.sample(min(len(df), 100_000), random_state=42) if large else df
    summary = {
        "shape": list(df.shape),
        "columns": df_view.columns.tolist()[:max_cols],
        "dtypes": {c: str(t) for c,t in df_view.dtypes.items()},
        "missing_pct": {c: float(df_view[c].isna().mean()) for c in df_view.columns[:max_cols]},
        "memory_mb": float(estimate_mem_mb(df)),
        "large_mode": large,
    }
    return summary
