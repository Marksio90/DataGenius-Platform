# backend/profiling_eda.py
"""
Profiling EDA (HTML) – TMIV Advanced ML Platform.

Lekki wrapper budujący interaktywny raport HTML (ydata-profiling lub pandas-profiling – jeśli dostępne).
Zapewnia:
- bezpieczne próbkowanie dużych zbiorów (w wierszach i kolumnach),
- cache’owanie wyniku po odcisku danych,
- zapis do stałej lokalizacji (cache/artifacts/profiles),
- brak twardej zależności – gdy biblioteka nie jest zainstalowana, zwracane jest None.

Public API
----------
build_profile_html(
    df: pd.DataFrame,
    *,
    title: str = "TMIV – EDA Profile",
    minimal: bool = True,
    sample_rows: int = 100_000,
    max_cols: int = 200,
    correlations: tuple[str, ...] = ("pearson", "spearman"),
) -> str | None
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .cache_manager import cache_result, cached_path, df_fingerprint

# Spróbuj „ydata-profiling”, a jeśli brak – legacy alias „pandas_profiling”
_ProfileReport = None
try:  # pragma: no cover
    from ydata_profiling import ProfileReport as _YDataProfileReport  # type: ignore

    _ProfileReport = _YDataProfileReport
except Exception:  # pragma: no cover
    try:
        from pandas_profiling import ProfileReport as _PandasProfileReport  # type: ignore

        _ProfileReport = _PandasProfileReport
    except Exception:
        _ProfileReport = None  # brak profilera – funkcja zwróci None


def _sample_dataframe(df: pd.DataFrame, *, sample_rows: int, max_cols: int, rs: int = 42) -> pd.DataFrame:
    """Bezpieczne próbkowanie wierszy i kolumn."""
    out = df
    # ogranicz kolumny: preferuj numeryczne + kategoryczne z największą wariancją / unikalnością
    if out.shape[1] > max_cols:
        num = out.select_dtypes(include=["number"]).columns.tolist()
        cat = out.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # numeryczne – wariancja malejąco
        num_scores = (
            out[num].var(numeric_only=True).sort_values(ascending=False).head(max_cols // 2).index.tolist() if num else []
        )
        # kategoryczne – liczba unikalnych malejąco
        cat_scores = (
            out[cat].nunique(dropna=True).sort_values(ascending=False).head(max_cols - len(num_scores)).index.tolist()
            if cat
            else []
        )
        keep = num_scores + cat_scores
        if not keep:  # awaryjnie: pierwsze N kolumn
            keep = out.columns[:max_cols].tolist()
        out = out.loc[:, keep]

    # ogranicz wiersze
    if len(out) > sample_rows:
        out = out.sample(sample_rows, random_state=rs)

    return out


@cache_result(namespace="eda_profile_html", ttl=3600)
def build_profile_html(
    df: pd.DataFrame,
    *,
    title: str = "TMIV – EDA Profile",
    minimal: bool = True,
    sample_rows: int = 100_000,
    max_cols: int = 200,
    correlations: Sequence[str] = ("pearson", "spearman"),
) -> str | None:
    """
    Zbuduj raport HTML z ydata/pandas-profiling i zwróć ABSOLUTNĄ ścieżkę do pliku.
    Jeśli brak wymaganych pakietów – zwróci None.
    """
    if _ProfileReport is None:
        return None
    if df is None or df.empty:
        return None

    # Próbkowanie defensywne
    df_small = _sample_dataframe(df, sample_rows=int(sample_rows), max_cols=int(max_cols))

    # Ustawienia korelacji
    corr_cfg = {m: {"calculate": True} for m in correlations}

    profile = _ProfileReport(
        df_small,
        title=title,
        explorative=not minimal,
        minimal=minimal,
        correlations=corr_cfg,
        progress_bar=False,
        lazy=False,
    )

    out_name = f"profile_{df_fingerprint(df_small)}.html"
    out_path = cached_path("profiles", out_name)
    profile.to_file(str(out_path))
    return str(out_path)


__all__ = ["build_profile_html"]
