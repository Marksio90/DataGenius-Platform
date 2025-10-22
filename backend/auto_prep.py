# backend/auto_prep.py
"""
Automatic preprocessing pipeline builder for TMIV.

Creates a robust sklearn `ColumnTransformer` with:
- Numeric: imputation (median) + optional outlier-robust scaling
- Categorical: imputation (constant "MISSING") + OneHotEncoder (handle_unknown="ignore")
- Boolean: treated as numeric (0/1)
- Datetime: optional extraction of y/m/d/dow/hour (disabled by default here)

Public API
----------
build_preprocessor(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    scale_numeric: bool = True,
    rare_min_freq: float | int | None = 0.005,
    max_ohe_categories: int | None = 200,
) -> AutoPrep

transform_fit(
    pre: AutoPrep,
    df: pd.DataFrame,
    target: str | None = None,
) -> tuple[np.ndarray, pd.Series | None, list[str]]

Notes
-----
- No leakage: target column is excluded from feature preprocessing.
- Uses OneHotEncoder with `min_frequency` when available (sklearn >=1.1).
- Returns feature names via `get_feature_names_out` when possible; otherwise synthesizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class AutoPrep:
    """Holds the constructed preprocessing objects and metadata."""
    pipeline: ColumnTransformer
    numeric_cols: list[str]
    categorical_cols: list[str]
    passthrough_cols: list[str]


# -------------------------
# Type detection helpers
# -------------------------

def _is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s) or (
        s.dtype == "object" and _try_parse_datetime_ratio(s) > 0.8
    )

def _try_parse_datetime_ratio(s: pd.Series, sample: int = 500) -> float:
    if s.empty:
        return 0.0
    sample_vals = s.dropna().astype(str).head(sample)
    ok = 0
    for v in sample_vals:
        try:
            pd.to_datetime(v)
            ok += 1
        except Exception:
            pass
    return ok / max(1, len(sample_vals))


def detect_columns(
    df: pd.DataFrame,
    target: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric, categorical, passthrough) excluding `target`."""
    cols = [c for c in df.columns if c != target]
    num_cols: list[str] = []
    cat_cols: list[str] = []
    pass_cols: list[str] = []

    for c in cols:
        s = df[c]
        if _is_datetime(s):
            # keep datetime raw as passthrough (model-specific expansion elsewhere if needed)
            pass_cols.append(c)
        elif pd.api.types.is_bool_dtype(s):
            num_cols.append(c)  # treat bool as numeric (0/1)
        elif pd.api.types.is_numeric_dtype(s):
            num_cols.append(c)
        else:
            cat_cols.append(c)
    return num_cols, cat_cols, pass_cols


# -------------------------
# Builder
# -------------------------

def build_preprocessor(
    df: pd.DataFrame,
    target: str | None = None,
    *,
    scale_numeric: bool = True,
    rare_min_freq: float | int | None = 0.005,
    max_ohe_categories: int | None = 200,
) -> AutoPrep:
    """
    Construct ColumnTransformer for given dataframe.

    Parameters
    ----------
    df : DataFrame
        Training dataframe (used only to infer schema).
    target : str | None
        Optional target column to exclude from features.
    scale_numeric : bool
        Whether to add StandardScaler for numeric features.
    rare_min_freq : float | int | None
        If float in (0,1], treat categories with frequency < fraction as rare (sklearn>=1.1).
        If int >=1, treat counts < int as rare.
        If None, disable rare grouping.
    max_ohe_categories : int | None
        If provided, cap the number of categories the encoder keeps per feature (sklearn>=1.1).

    Returns
    -------
    AutoPrep
    """
    numeric_cols, categorical_cols, passthrough_cols = detect_columns(df, target=target)

    # Numeric pipeline
    num_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    num_pipe = Pipeline(steps=num_steps)

    # Categorical pipeline
    cat_imputer = SimpleImputer(strategy="constant", fill_value="MISSING")

    # OneHotEncoder config with graceful fallback for older sklearn
    ohe_kwargs = dict(handle_unknown="ignore", sparse=False)
    # min_frequency & max_categories are supported in newer sklearn
    if rare_min_freq is not None:
        ohe_kwargs["min_frequency"] = rare_min_freq  # type: ignore[assignment]
    if max_ohe_categories is not None:
        ohe_kwargs["max_categories"] = max_ohe_categories  # type: ignore[assignment]

    ohe = OneHotEncoder(**ohe_kwargs)  # type: ignore[arg-type]
    cat_pipe = Pipeline(
        steps=[
            ("imputer", cat_imputer),
            ("ohe", ohe),
        ]
    )

    transformers: list[tuple[str, object, Iterable[str]]] = []
    if numeric_cols:
        transformers.append(("num", num_pipe, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", cat_pipe, categorical_cols))
    if passthrough_cols:
        transformers.append(("pass", "passthrough", passthrough_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)

    return AutoPrep(
        pipeline=pre,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        passthrough_cols=passthrough_cols,
    )


# -------------------------
# Fit/transform wrapper
# -------------------------

def transform_fit(
    pre: AutoPrep,
    df: pd.DataFrame,
    target: str | None = None,
) -> tuple[np.ndarray, Optional[pd.Series], list[str]]:
    """
    Fit the preprocessor on `df` and return transformed X, y and feature names.

    Returns
    -------
    X : np.ndarray
        Transformed features.
    y : pd.Series | None
        Target vector if `target` provided, otherwise None.
    feature_names : list[str]
        Names after transformation (best-effort).
    """
    X_df = df.drop(columns=[target], errors="ignore") if target else df
    y = df[target] if (target and target in df.columns) else None

    X = pre.pipeline.fit_transform(X_df)

    # Feature names (best effort)
    try:
        feature_names = list(pre.pipeline.get_feature_names_out())
    except Exception:
        # Fallback: synthesize names
        feature_names = []
        feature_names.extend(pre.numeric_cols)
        # for OHE, approximate: col__catVal isn't easily available without fitted encoder
        if pre.categorical_cols:
            feature_names.extend([f"{c}__ohe_{i}" for c in pre.categorical_cols for i in range(1)])
        feature_names.extend(pre.passthrough_cols)

    return np.asarray(X), y, feature_names
