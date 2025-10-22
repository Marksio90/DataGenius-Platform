from __future__ import annotations

from backend.safe_utils import truthy_df_safe

# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Common names seen for targets; extended a bit but keeps originals first
TARGET_HINTS = [
    "target", "label", "y", "class",
    "price", "amount", "sales", "churn", "clicked", "default",
    "response", "outcome", "target_var", "targetvariable"
]

_ID_LIKE = {"id", "uuid", "guid", "index", "rowid", "recordid"}


def _looks_like_id(colname: str, s: pd.Series, n: int) -> bool:
    name = str(colname).strip().lower()
    if any(k == name or name.endswith(k) or name.startswith(k) for k in _ID_LIKE):
        return True
    try:
        nun = int(s.nunique(dropna=True))
        # nearly-unique columns are typically identifiers
        if n > 0 and (nun >= 0.98 * n):
            return True
    except Exception:
        pass
    return False


def _is_datetime_like(s: pd.Series) -> bool:
    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            return True
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            # try cheap sample conversion
            sample = s.dropna().astype(str).head(50)
            if sample.empty:
                return False
            conv = pd.to_datetime(sample, errors="coerce", utc=True)
            ok_ratio = conv.notna().mean()
            return ok_ratio >= 0.8
    except Exception:
        return False
    return False


def choose_target(df: pd.DataFrame, provided: Optional[str] = None) -> Optional[str]:
    """
    Heuristics to pick a target column:
      1) If `provided` is a valid column → use it.
      2) Exact name match against TARGET_HINTS (case-insensitive).
      3) Prefer the last column if it's not ID/datetime-like and has reasonable cardinality.
      4) Otherwise, pick a column with moderate cardinality (2..min(100, 0.9*n)),
         skipping clear ID-like or datetime-like columns.
    Returns column name or None.
    """
    if df is None or df.shape[1] == 0:
        return None

    cols = list(df.columns)
    n = len(df)

    if truthy_df_safe(provided) and provided in cols:
        return provided

    # 2) name hints (exact match, case-insensitive)
    lower_map = {str(c).lower(): c for c in cols}
    for h in TARGET_HINTS:
        if h in lower_map:
            cand = lower_map[h]
            if not _looks_like_id(cand, df[cand], n) and not _is_datetime_like(df[cand]):
                return cand

    # 3) last column heuristic
    last = cols[-1]
    try:
        nunique = df[last].nunique(dropna=True)
    except Exception:
        nunique = 0
    if (
        last in df.columns
        and 1 < nunique < n
        and not _looks_like_id(last, df[last], n)
        and not _is_datetime_like(df[last])
    ):
        return last

    # 4) moderate cardinality candidates
    try:
        cards = {c: df[c].nunique(dropna=True) for c in cols}
    except Exception:
        cards = {c: 0 for c in cols}
    upper = int(min(100, 0.9 * n)) if n > 0 else 100
    candidates = [
        c for c, k in cards.items()
        if 2 <= int(k) <= upper
        and not _looks_like_id(c, df[c], n)
        and not _is_datetime_like(df[c])
    ]
    if truthy_df_safe(candidates):
        # prefer non-float strings/booleans first (often target labels)
        def _score(col: str) -> Tuple[int, int]:
            s = df[col]
            # lower score is better
            is_cat = int(pd.api.types.is_bool_dtype(s) or pd.api.types.is_string_dtype(s) or s.dtype == "object")
            return (1 - is_cat, int(cards.get(col, 0)))  # favor smaller cardinality too

        candidates.sort(key=_score)
        return candidates[0]

    return None


def infer_problem_type(y: pd.Series) -> str:
    """
    Decide classification vs regression.
    - bool dtype -> classification
    - <=2 unique -> classification
    - <=20 unique and ratio < 0.05 of n -> classification
    - otherwise regression
    """
    if y is None or len(y) == 0:
        return "regression"
    try:
        if pd.api.types.is_bool_dtype(y):
            return "classification"
    except Exception:
        pass

    try:
        u = int(y.nunique(dropna=True))
    except Exception:
        # if we cannot compute, assume regression
        return "regression"

    if u <= 2:
        return "classification"
    if u <= 20 and (u / max(len(y), 1)) < 0.05:
        return "classification"
    return "regression"


def is_timeseries(df: pd.DataFrame) -> bool:
    """
    True if index is DatetimeIndex or any column is datetime-like
    (dtype datetime or converts successfully for most values),
    or column name hints 'date'/'time'/'timestamp' and series is convertible.
    """
    if df is None or df.shape[1] == 0:
        return False
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            return True
    except Exception:
        pass
    for c in df.columns:
        lc = str(c).lower()
        if _is_datetime_like(df[c]):
            return True
        if any(k in lc for k in ("date", "time", "timestamp")) and _is_datetime_like(df[c]):
            return True
    return False


def _safe_n_splits_for_stratified(y: pd.Series, desired: int) -> Tuple[int, bool]:
    """
    Returns (n_splits, ok_stratified). If any class has < n_splits samples,
    reduce n_splits or signal to fallback to KFold.
    """
    try:
        vc = y.value_counts(dropna=False)
        min_class = int(vc.min()) if not vc.empty else 0
        n_splits = max(2, min(desired, min_class)) if min_class > 0 else 2
        ok = min_class >= n_splits and n_splits >= 2
        return (n_splits, ok)
    except Exception:
        return (max(2, min(desired, 5)), False)


def make_training_plan(X: pd.DataFrame, y: pd.Series, quick: bool = False) -> Dict[str, Any]:
    """
    Builds a lightweight training plan dict:
      - problem_type
      - cv strategy (time series / stratified / kfold) with safe n_splits
      - primary & extra metrics (binary vs multiclass aware)
      - algo order (with speed-aware ordering for big datasets)
      - budget & sampling
      - explainability toggles
    """
    n = int(len(X) if X is not None else 0)
    p = int(X.shape[1] if X is not None else 0)
    plan: Dict[str, Any] = {
        "n": n,
        "p": p,
        "quick": bool(quick),
    }

    ptype = infer_problem_type(y if y is not None else pd.Series(dtype="float"))
    plan["problem_type"] = ptype

    # ── CV strategy ───────────────────────────────────────────────────────────
    desired_splits = 5
    if n < 3:
        desired_splits = 2
    if is_timeseries(X if X is not None else pd.DataFrame()):
        ns = max(2, min(desired_splits, max(2, n - 1)))  # TimeSeriesSplit requires n_splits < n_samples
        plan["cv"] = {"strategy": "TimeSeriesSplit", "n_splits": ns}
    else:
        if ptype == "classification":
            ns, ok = _safe_n_splits_for_stratified(y, desired_splits)
            if ok:
                plan["cv"] = {"strategy": "StratifiedKFold", "n_splits": ns, "shuffle": True, "random_state": 42}
            else:
                ns2 = max(2, min(desired_splits, n))
                plan["cv"] = {"strategy": "KFold", "n_splits": ns2, "shuffle": True, "random_state": 42}
        else:
            ns = max(2, min(desired_splits, n))
            plan["cv"] = {"strategy": "KFold", "n_splits": ns, "shuffle": True, "random_state": 42}

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Try to detect binary vs multiclass for nicer defaults
    try:
        u = int(y.nunique(dropna=True))
    except Exception:
        u = 0
    is_binary = (ptype == "classification") and (u <= 2 and u >= 2)

    if ptype == "classification":
        plan["metrics_primary"] = "f1"
        if is_binary:
            plan["metrics_extra"] = [
                "accuracy", "balanced_accuracy", "roc_auc", "average_precision",
                "mcc", "cohen_kappa", "f1_weighted"
            ]
        else:
            plan["metrics_extra"] = [
                "accuracy", "balanced_accuracy", "roc_auc_ovr",
                "average_precision", "mcc", "cohen_kappa", "f1_weighted"
            ]
        algos = [
            "LightGBM", "XGBoost", "CatBoost",
            "RandomForest", "LogisticRegression",
            "LinearSVC", "KNN"
        ]
    else:
        plan["metrics_primary"] = "rmse"
        plan["metrics_extra"] = ["mae", "r2", "mape", "medae", "rmsle", "explained_variance"]
        algos = [
            "LightGBM", "XGBoost", "CatBoost",
            "RandomForest", "LinearRegression", "ElasticNet", "KNN"
        ]

    # ── Budget & speed ────────────────────────────────────────────────────────
    if quick or n > 300_000:
        plan["budget"] = {"search": "random", "n_iter": 20, "n_jobs": -1}
        plan["sampling"] = {"mode": "subsample", "max_rows": 120_000}
        # prefer fast algos first; CatBoost often slower -> move to back
        plan["algos_order"] = [a for a in algos if a != "CatBoost"] + ["CatBoost"]
    elif n > 80_000:
        plan["budget"] = {"search": "random", "n_iter": 40, "n_jobs": -1}
        plan["sampling"] = {"mode": "subsample", "max_rows": 200_000}
        plan["algos_order"] = algos
    else:
        plan["budget"] = {"search": "grid_light", "n_iter": 0, "n_jobs": -1}
        plan["sampling"] = {"mode": "full"}
        plan["algos_order"] = algos

    # ── Explainability ────────────────────────────────────────────────────────
    plan["explainability"] = {"shap": True, "permutation_importance": True, "pdp_ice": True}
    return plan
