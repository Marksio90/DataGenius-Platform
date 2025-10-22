# backend/eda_integration.py
"""
EDA integration layer for TMIV – Advanced ML Platform.

This module provides **fast, cacheable** exploratory data analysis utilities that other
layers (UI/services) can call. It is dependency-light and gracefully degrades when
optional packages (e.g., ydata-profiling) are unavailable.

Key features
------------
- Shape, dtypes, memory usage, missing values, nunique per column
- Numeric summary (describe) and outlier heuristics (IQR-based)
- Top-K category frequencies
- Correlation matrix (pearson/spearman/kendall) and top correlated pairs
- Optional HTML profiling report via ydata-profiling (if installed)
- Results are memoized with robust cache keys (incl. DataFrame fingerprints)

Public API
----------
- eda_overview(df, *, top_k_cat=10, max_cols=200) -> dict
- numeric_describe(df, *, percentiles=(.05,.25,.5,.75,.95)) -> pd.DataFrame
- category_topk(df, *, top_k=10, max_cols=50) -> dict[str, list[tuple[str, int]]]
- corr_matrix(df, *, method="pearson") -> pd.DataFrame
- corr_top_pairs(df, *, method="pearson", min_abs=0.2, max_pairs=50) -> list[dict]
- build_profile_report(df, *, title="TMIV EDA Report", minimal=True) -> str | None

Notes
-----
These functions do not render UI; they return data structures for the frontend layer
(see frontend/ui_panels.py) or file paths for downloadable artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .cache_manager import cache_result, df_fingerprint, cached_path


# =========================
# Core helpers
# =========================


def _safe_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric-only frame with finite values (drop columns that become empty)."""
    num = df.select_dtypes(include=["number"]).copy()
    if num.empty:
        return num
    # Replace inf with nan, then drop cols with all-nan
    num.replace([np.inf, -np.inf], np.nan, inplace=True)
    num = num.dropna(axis=1, how="all")
    return num


def _memory_usage_cols(df: pd.DataFrame) -> dict[str, int]:
    try:
        mu = df.memory_usage(deep=True).to_dict()
        # Drop the "Index" key if present
        mu.pop("Index", None)
        return {str(k): int(v) for k, v in mu.items()}
    except Exception:
        return {str(c): -1 for c in df.columns}


def _outlier_iqr_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return per-numeric-column outlier ratio using the 1.5*IQR rule."""
    out: dict[str, dict[str, float]] = {}
    num = _safe_numeric_df(df)
    if num.empty:
        return out
    for c in num.columns:
        s = num[c].dropna()
        if s.empty:
            out[c] = {"ratio": 0.0, "q1": np.nan, "q3": np.nan}
            continue
        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        if iqr <= 0:
            out[c] = {"ratio": 0.0, "q1": float(q1), "q3": float(q3)}
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        ratio = float(((s < lo) | (s > hi)).mean())
        out[c] = {"ratio": ratio, "q1": float(q1), "q3": float(q3)}
    return out


# =========================
# EDA overview
# =========================


@cache_result(namespace="eda", ttl=600)
def eda_overview(
    df: pd.DataFrame,
    *,
    top_k_cat: int = 10,
    max_cols: int = 200,
) -> dict:
    """
    Return a compact EDA overview for the first `max_cols` columns.

    Output dict keys:
      - shape: (rows, cols)
      - dtypes: {col: dtype}
      - memory_bytes_by_col: {col: int}
      - missing_by_col: {col: int}
      - missing_pct_by_col: {col: float}
      - nunique_by_col: {col: int}
      - category_topk: {col: [(value, count), ...]} for object/category
      - outliers_iqr: {col: {ratio, q1, q3}} for numeric
    """
    if df is None or df.empty:
        return {
            "shape": (0, 0),
            "dtypes": {},
            "memory_bytes_by_col": {},
            "missing_by_col": {},
            "missing_pct_by_col": {},
            "nunique_by_col": {},
            "category_topk": {},
            "outliers_iqr": {},
        }

    # Limit columns for heavy ops
    cols = df.columns[:max_cols]
    sub = df.loc[:, cols].copy()

    dtypes = {str(c): str(sub[c].dtype) for c in sub.columns}
    missing = {str(c): int(sub[c].isna().sum()) for c in sub.columns}
    miss_pct = {str(c): float(sub[c].isna().mean() * 100.0) for c in sub.columns}
    nunique = {str(c): int(sub[c].nunique(dropna=True)) for c in sub.columns}
    mem = _memory_usage_cols(sub)

    # Category top-k
    cat_cols = sub.select_dtypes(include=["object", "category"]).columns.tolist()
    topk: dict[str, list[tuple[str, int]]] = {}
    for c in cat_cols:
        vc = sub[c].astype("string").fillna("<NA>").value_counts(dropna=False).head(top_k_cat)
        topk[str(c)] = [(str(idx), int(cnt)) for idx, cnt in vc.items()]

    # Outliers (numeric)
    out_iqr = _outlier_iqr_stats(sub)

    return {
        "shape": (int(len(sub)), int(sub.shape[1])),
        "dtypes": dtypes,
        "memory_bytes_by_col": mem,
        "missing_by_col": missing,
        "missing_pct_by_col": miss_pct,
        "nunique_by_col": nunique,
        "category_topk": topk,
        "outliers_iqr": out_iqr,
    }


# =========================
# Numeric describe
# =========================


@cache_result(namespace="eda", ttl=600)
def numeric_describe(
    df: pd.DataFrame,
    *,
    percentiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> pd.DataFrame:
    """
    Return pandas .describe() for numeric columns with custom percentiles.
    """
    num = _safe_numeric_df(df)
    if num.empty:
        return pd.DataFrame()
    try:
        pct = sorted(set([p for p in percentiles if 0 < p < 1]))
        desc = num.describe(percentiles=pct).T
        return desc
    except Exception:
        return num.describe().T


# =========================
# Category top-k
# =========================


@cache_result(namespace="eda", ttl=600)
def category_topk(
    df: pd.DataFrame,
    *,
    top_k: int = 10,
    max_cols: int = 50,
) -> dict[str, list[tuple[str, int]]]:
    """
    Compute top-k frequency tables for categorical columns.
    """
    cols = df.columns[:max_cols]
    sub = df.loc[:, cols]
    cats = sub.select_dtypes(include=["object", "category"]).columns.tolist()
    out: dict[str, list[tuple[str, int]]] = {}
    for c in cats:
        vc = sub[c].astype("string").fillna("<NA>").value_counts(dropna=False).head(top_k)
        out[str(c)] = [(str(k), int(v)) for k, v in vc.items()]
    return out


# =========================
# Correlations
# =========================


@cache_result(namespace="eda", ttl=600)
def corr_matrix(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
) -> pd.DataFrame:
    """
    Compute correlation matrix for numeric columns.
    method: "pearson" | "spearman" | "kendall"
    """
    num = _safe_numeric_df(df)
    if num.empty or num.shape[1] < 2:
        return pd.DataFrame()
    method = method.lower()
    if method not in {"pearson", "spearman", "kendall"}:
        method = "pearson"
    try:
        # Use pairwise complete obs
        return num.corr(method=method)
    except Exception:
        return pd.DataFrame()


@cache_result(namespace="eda", ttl=600)
def corr_top_pairs(
    df: pd.DataFrame,
    *,
    method: str = "pearson",
    min_abs: float = 0.2,
    max_pairs: int = 50,
) -> list[dict]:
    """
    Return top correlated pairs (i<j), sorted by |corr| desc, filtered by min_abs.
    Each item: {"feature_1": str, "feature_2": str, "corr": float}
    """
    cm = corr_matrix(df, method=method)
    if cm.empty:
        return []
    feats = cm.columns.tolist()
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            v = float(cm.iat[i, j])
            if np.isnan(v) or abs(v) < float(min_abs):
                continue
            pairs.append((feats[i], feats[j], v))
    pairs.sort(key=lambda t: abs(t[2]), reverse=True)
    out = [{"feature_1": a, "feature_2": b, "corr": float(c)} for a, b, c in pairs[:max_pairs]]
    return out


# =========================
# ydata-profiling (optional)
# =========================


@cache_result(namespace="eda_profile", ttl=3600)
def build_profile_report(
    df: pd.DataFrame,
    *,
    title: str = "TMIV EDA Report",
    minimal: bool = True,
) -> str | None:
    """
    Generate an HTML profiling report using ydata-profiling, if available.

    Returns
    -------
    str | None : absolute path to the generated HTML file, or None when unavailable.
    """
    try:
        from ydata_profiling import ProfileReport  # type: ignore
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Keep it safe for big frames: sample columns/rows lightly
    df_small = df.copy()
    if df_small.shape[0] > 100_000:
        df_small = df_small.sample(100_000, random_state=42)
    if df_small.shape[1] > 200:
        df_small = df_small.iloc[:, :200]

    profile = ProfileReport(
        df_small,
        title=title,
        explorative=not minimal,
        minimal=minimal,
        correlations={"pearson": {"calculate": True}, "spearman": {"calculate": True}},
        lazy=False,
    )

    out_path = cached_path("profiles", f"profile_{df_fingerprint(df_small)}.html")
    profile.to_file(str(out_path))
    return str(out_path)
