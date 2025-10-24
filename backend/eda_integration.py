"""
eda_integration.py (Stage 6: Polars LTS gated)
"""
from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
import numpy as np

def quick_eda(df: pd.DataFrame, topk: int = 10, bins: int = 20, use_polars: bool = False) -> Dict[str, Any]:
    """
    Docstring (PL): Zwraca słownik z podstawowymi statystykami EDA. Może użyć Polars (jeśli zainstalowany) dla przyspieszenia.
    """
    if use_polars:
        try:
            import polars as pl
            pdf = df
            pldf = pl.from_pandas(pdf, include_index=False)
            # describe
            desc = pldf.describe().to_pandas().set_index("statistic").to_dict()
            # korelacje — tylko dla numerycznych
            num_cols = [c for c in pdf.columns if np.issubdtype(pdf[c].dtype, np.number)]
            corr = {}
            if num_cols:
                plnum = pl.from_pandas(pdf[num_cols])
                corr_mat = plnum.to_pandas().corr(numeric_only=True).fillna(0.0).to_dict()
                corr = corr_mat
            # kategorie/top
            categorical_cols = [c for c in pdf.columns if c not in num_cols and not pd.api.types.is_datetime64_any_dtype(pdf[c])]
            cats = {}
            for c in categorical_cols:
                vc = pdf[c].astype("string").value_counts(dropna=False).head(topk)
                cats[c] = vc.to_dict()
            # hists
            hists = {}
            for c in num_cols:
                counts, edges = np.histogram(pdf[c].dropna().values, bins=bins)
                hists[c] = {"counts": counts.tolist(), "edges": edges.tolist()}
            datetime_cols = list(pdf.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)
            return {
                "rows": len(pdf),
                "cols": len(pdf.columns),
                "numeric_cols": num_cols,
                "categorical_cols": categorical_cols,
                "datetime_cols": datetime_cols,
                "describe": desc,
                "correlation": corr,
                "categoricals_top": cats,
                "histograms": hists,
                "note": "Profil via Polars (gated).",
            }
        except Exception:
            # fallback do wersji pandas
            pass

    # Fallback do poprzedniej implementacji (pandas)
    import pandas as _pd, numpy as _np
    df = df.copy()
    numeric_cols = list(df.select_dtypes(include=[_np.number]).columns)
    categorical_cols = [c for c in df.columns if c not in numeric_cols and not _pd.api.types.is_datetime64_any_dtype(df[c])]
    datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns)
    desc = df[numeric_cols].describe().to_dict() if numeric_cols else {}
    corr = df[numeric_cols].corr(numeric_only=True).fillna(0.0).to_dict() if numeric_cols else {}
    cats = {}
    for c in categorical_cols:
        vc = df[c].astype("string").value_counts(dropna=False).head(topk)
        cats[c] = vc.to_dict()
    hists = {}
    for c in numeric_cols:
        counts, edges = _np.histogram(df[c].dropna().values, bins=bins)
        hists[c] = {"counts": counts.tolist(), "edges": edges.tolist()}
    return {
        "rows": len(df),
        "cols": len(df.columns),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "describe": desc,
        "correlation": corr,
        "categoricals_top": cats,
        "histograms": hists,
        "note": "Profil via pandas.",
    }