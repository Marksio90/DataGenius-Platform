# core/services/eda_service.py
"""
EDAService – eksploracyjna analiza danych (szybkie statystyki, korelacje, rozkłady)
dla TMIV – Advanced ML Platform.

Implementuje minimalny kontrakt `IEDAService`:
- overview(df)       -> podstawowe statystyki i skrócone podsumowania
- correlations(df)   -> macierz korelacji numerycznych (Pearson)
- distributions(df)  -> dane do histogramów (num) i wykresów słupkowych (cat)
- profile_html(df)   -> delegacja do backend.profiling_eda (opcjonalna)

Uwaga:
- Brak twardych zależności UI/Streamlit.
- Zwracane struktury są JSON-friendly (dict/list/float/int/str).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Profiling (opcjonalny)
try:  # pragma: no cover
    from backend.profiling_eda import build_profile_html as _build_profile_html  # type: ignore
except Exception:  # pragma: no cover
    _build_profile_html = None  # type: ignore


class EDAService:
    # =========================
    # Public API
    # =========================

    def overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Zwraca lekki, JSON-owalny przegląd danych:
        {
          "shape": {"rows": int, "cols": int},
          "dtypes": {col: "dtype"},
          "missing": {col: {"count": int, "pct": float}},
          "numeric_summary": {col: {"mean":..,"std":..,"min":..,"q25":..,"median":..,"q75":..,"max":..}},
          "categorical_summary": {col: {"unique": int, "top": [{"value": str, "count": int}], "top_k": int}}
        }
        """
        if df is None or df.empty:
            return {
                "shape": {"rows": 0, "cols": 0},
                "dtypes": {},
                "missing": {},
                "numeric_summary": {},
                "categorical_summary": {},
            }

        rows, cols = int(df.shape[0]), int(df.shape[1])

        dtypes = {c: str(t) for c, t in df.dtypes.items()}

        # Missing
        miss_cnt = df.isna().sum()
        missing: Dict[str, Any] = {}
        for c, cnt in miss_cnt.items():
            missing[str(c)] = {"count": int(cnt), "pct": (float(cnt) / rows * 100.0) if rows else 0.0}

        # Numeric summary
        num_cols = list(df.select_dtypes(include=["number"]).columns)
        numeric_summary: Dict[str, Any] = {}
        if num_cols:
            desc = df[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).transpose()
            # Mapujemy spójne nazwy
            for c, row in desc.iterrows():
                numeric_summary[str(c)] = {
                    "mean": _f(row.get("mean")),
                    "std": _f(row.get("std")),
                    "min": _f(row.get("min")),
                    "q25": _f(row.get("25%")),
                    "median": _f(row.get("50%")),
                    "q75": _f(row.get("75%")),
                    "max": _f(row.get("max")),
                }

        # Categorical summary
        cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
        # „niskokardynalne inty” traktuj jako kategorie, jeśli <=20 unikalnych i <20% card/rows
        for c in df.columns.difference(num_cols + cat_cols):
            s = df[c]
            if pd.api.types.is_integer_dtype(s):
                nunique = int(s.nunique(dropna=True))
                if rows > 0 and nunique <= 20 and (nunique / rows) < 0.2:
                    cat_cols.append(c)

        categorical_summary: Dict[str, Any] = {}
        for c in cat_cols:
            s = df[c]
            vc = s.astype("string").value_counts(dropna=False).head(20)
            top = [{"value": ("" if pd.isna(v) else str(v)), "count": int(cnt)} for v, cnt in vc.items()]
            categorical_summary[str(c)] = {"unique": int(s.nunique(dropna=True)), "top": top, "top_k": 20}

        return {
            "shape": {"rows": rows, "cols": cols},
            "dtypes": dtypes,
            "missing": missing,
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
        }

    def correlations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Macierz korelacji Pearsona dla kolumn numerycznych.
        Zwraca pusty DataFrame, jeśli brak kolumn numerycznych.
        """
        if df is None or df.empty:
            return pd.DataFrame()
        num = df.select_dtypes(include=["number"])
        if num.shape[1] == 0:
            return pd.DataFrame()
        # bezpiecznie: ignoruj kolumny o zerowym odchyleniu
        safe = num.loc[:, num.std(numeric_only=True) > 0]
        if safe.shape[1] == 0:
            return pd.DataFrame(index=num.columns, columns=num.columns, dtype=float)
        return safe.corr(method="pearson")

    def distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Przygotuj dane do wizualizacji rozkładów:
        {
          "numeric": {col: {"bins": [...], "counts": [...]}},
          "categorical": {col: {"values": [...], "counts": [...]}},
        }
        """
        if df is None or df.empty:
            return {"numeric": {}, "categorical": {}}

        out_num: Dict[str, Any] = {}
        out_cat: Dict[str, Any] = {}

        # Numeric histograms (Freedman–Diaconis fallback to Sturges)
        for c in df.select_dtypes(include=["number"]).columns:
            s = pd.to_numeric(df[c], errors="coerce")
            arr = s.to_numpy()
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            bins = _fd_bins(arr)
            counts, edges = np.histogram(arr, bins=bins)
            out_num[str(c)] = {"bins": edges.astype(float).tolist(), "counts": counts.astype(int).tolist()}

        # Categorical bars (top 50)
        cat_cols = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
        for c in cat_cols:
            s = df[c].astype("string")
            vc = s.value_counts(dropna=False).head(50)
            values = [("" if pd.isna(v) else str(v)) for v in vc.index.tolist()]
            out_cat[str(c)] = {"values": values, "counts": [int(x) for x in vc.tolist()]}

        return {"numeric": out_num, "categorical": out_cat}

    def profile_html(self, df: pd.DataFrame) -> Optional[str]:
        """
        Delegacja do backend.profiling_eda (o ile zainstalowane). Zwraca absolutną ścieżkę HTML lub None.
        """
        if _build_profile_html is None or df is None or df.empty:
            return None
        try:
            return _build_profile_html(df)
        except Exception:
            return None


# =========================
# Helpers
# =========================

def _f(x: Any) -> float | None:
    """Bezpieczna konwersja do float (None dla NaN)."""
    try:
        if x is None:
            return None
        xv = float(x)
        if np.isnan(xv):
            return None
        return xv
    except Exception:
        return None


def _fd_bins(arr: np.ndarray) -> int:
    """
    Liczba binów wg reguły Freedmana–Diaconisa (z fallbackiem do Sturges).
    """
    arr = np.asarray(arr, dtype=float)
    n = arr.size
    if n < 2:
        return 1
    iqr = np.subtract(*np.percentile(arr, [75, 25]))
    if iqr <= 0 or not np.isfinite(iqr):
        # Sturges
        return max(1, int(np.ceil(np.log2(n) + 1)))
    h = 2 * iqr * (n ** (-1 / 3))
    if h <= 0 or not np.isfinite(h):
        return max(1, int(np.ceil(np.log2(n) + 1)))
    bins = int(np.ceil((arr.max() - arr.min()) / h))
    return max(1, min(200, bins))  # limit dla stabilności


__all__ = ["EDAService"]
