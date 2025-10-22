# backend/utils.py
"""
General utilities for TMIV – Advanced ML Platform.

What’s inside (zero heavy deps; all optional imports are guarded)
----------------------------------------------------------------
- Global seeding for reproducibility (`set_global_seed`)
- Run IDs & timestamps (`make_run_id`, `utc_now_iso`)
- DataFrame helpers:
    * `is_categorical_series`, `numeric_columns`, `categorical_columns`
    * `dataframe_signature` (tiny summary), `fingerprint_df` (stable hash; alias)
- Splits:
    * `stratified_train_valid_split_indices`
    * `temporal_train_valid_split_indices`
    * `tscv_indices` (generator over TimeSeriesSplit)
- Array guards & conversions: `to_numpy_2d`
- Small utils: `safe_jsonable`, `Timer` context manager

Notes
-----
- This module deliberately avoids any Streamlit/UI imports.
- The canonical DataFrame fingerprint lives in `backend.cache_manager`. We re-export a fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, train_test_split

# Try to import the canonical fingerprint from cache_manager; provide a fallback otherwise
try:  # pragma: no cover
    from .cache_manager import df_fingerprint as _df_fingerprint  # type: ignore
except Exception:  # pragma: no cover
    _df_fingerprint = None  # type: ignore


# =========================
# Time / IDs
# =========================

def utc_now_iso() -> str:
    """UTC timestamp in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_run_id(prefix: str = "run") -> str:
    """Generate a compact run id like 'run_2025-10-22_13-45-07'."""
    return f"{prefix}_{time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime())}"


# =========================
# Reproducibility
# =========================

def set_global_seed(seed: int = 42) -> Dict[str, bool]:
    """
    Seed common libraries (random, numpy, torch*, tensorflow*).
    Returns a dict with which libs were actually seeded.
    """
    seeded = {"python": True, "numpy": True, "torch": False, "tensorflow": False, "jax": False}
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # torch (optional)
    try:  # pragma: no cover
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        try:
            torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
        except Exception:
            pass
        seeded["torch"] = True
    except Exception:
        pass

    # tensorflow (optional)
    try:  # pragma: no cover
        import tensorflow as tf  # type: ignore

        try:
            tf.random.set_seed(seed)  # TF2
        except Exception:
            pass
        seeded["tensorflow"] = True
    except Exception:
        pass

    # jax (optional)
    try:  # pragma: no cover
        import jax  # type: ignore

        key = jax.random.PRNGKey(seed)  # noqa: F841
        seeded["jax"] = True
    except Exception:
        pass

    return seeded


# =========================
# DataFrame helpers
# =========================

def is_categorical_series(s: pd.Series) -> bool:
    """Return True if series behaves like categorical (object/category/bool or low cardinality ints)."""
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
        return True
    if pd.api.types.is_integer_dtype(s):
        nunique = int(s.nunique(dropna=True))
        total = int(s.notna().sum())
        if total > 0 and nunique <= 20 and nunique / total < 0.2:
            return True
    return False


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include=["number"]).columns)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    cols_obj = list(df.select_dtypes(include=["object", "category", "bool"]).columns)
    # low-cardinality ints also count as cats
    low_ints = [c for c in df.columns if c not in cols_obj and is_categorical_series(df[c])]
    return list(dict.fromkeys(cols_obj + low_ints))


def dataframe_signature(df: pd.DataFrame) -> dict:
    """Tiny signature: shapes & dtypes info for quick display/logging."""
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    return {"rows": int(df.shape[0]), "cols": int(df.shape[1]), "dtypes": dtypes}


def fingerprint_df(df: pd.DataFrame) -> str:
    """
    Stable, short fingerprint of a DataFrame (structure + small content sample).
    Uses cache_manager.df_fingerprint if available; otherwise a lightweight hash.
    """
    if _df_fingerprint is not None:
        try:
            return _df_fingerprint(df)  # type: ignore[misc]
        except Exception:
            pass

    # Fallback: hash of columns + dtypes + a small random sample (first 1k rows)
    sample = df.head(1000) if len(df) > 1000 else df
    payload = {
        "cols": list(map(str, sample.columns)),
        "dtypes": [str(t) for t in sample.dtypes],
        "shape": list(df.shape),
        "preview": sample.astype(str).to_dict(orient="list"),
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return h[:12].upper()


# =========================
# Splits
# =========================

def stratified_train_valid_split_indices(
    y: Sequence[Any],
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return train/valid indices using sklearn train_test_split with stratify.
    If y is not suitable for stratification, falls back to plain split.
    """
    y_arr = np.asarray(y)
    idx = np.arange(len(y_arr))
    stratify = None
    try:
        # valid stratify needs at least 2 classes, each with >= 2 samples after split
        if pd.Series(y_arr).nunique(dropna=True) >= 2:
            stratify = y_arr
    except Exception:
        stratify = None
    train_idx, valid_idx = train_test_split(
        idx, test_size=float(test_size), random_state=random_state, stratify=stratify
    )
    return np.asarray(train_idx), np.asarray(valid_idx)


def temporal_train_valid_split_indices(
    df: pd.DataFrame,
    time_col: str,
    test_size: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sort by `time_col` and split by the tail proportion (no leakage).
    """
    t = pd.to_datetime(df[time_col], errors="coerce")
    order = np.argsort(t.values.astype("datetime64[ns]"))
    n = len(order)
    cut = max(1, int(round(n * (1.0 - float(test_size)))))
    train_idx = order[:cut]
    valid_idx = order[cut:]
    return np.asarray(train_idx), np.asarray(valid_idx)


def tscv_indices(
    n_samples: int,
    *,
    n_splits: int = 5,
    max_train_size: int | None = None,
    test_size: int | None = None,
    gap: int = 0,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yield (train_idx, valid_idx) from sklearn's TimeSeriesSplit.
    """
    tscv = TimeSeriesSplit(
        n_splits=int(n_splits),
        max_train_size=max_train_size,
        test_size=test_size,
        gap=gap,
    )
    X_dummy = np.empty((n_samples, 1))
    for train_idx, valid_idx in tscv.split(X_dummy):
        yield np.asarray(train_idx), np.asarray(valid_idx)


# =========================
# Arrays & misc
# =========================

def to_numpy_2d(X: Any) -> np.ndarray:
    """Convert pandas/array-like to a 2D numpy array."""
    if hasattr(X, "values") and hasattr(X, "shape"):
        arr = np.asarray(X.values)
    else:
        arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def safe_jsonable(obj: Any, *, max_list: int = 2000) -> Any:
    """
    Convert objects to JSON-safe structures (trim very long lists).
    Intended for logs/telemetry payloads.
    """
    try:
        json.dumps(obj)
        return obj  # already jsonable
    except Exception:
        pass

    if isinstance(obj, dict):
        return {str(k): safe_jsonable(v, max_list=max_list) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        if len(seq) > max_list:
            seq = seq[:max_list] + ["<...>"]
        return [safe_jsonable(v, max_list=max_list) for v in seq]
    if isinstance(obj, (np.ndarray,)):
        arr = obj
        if arr.ndim == 0:
            return obj.item()
        if arr.size > max_list:
            return [safe_jsonable(x, max_list=max_list) for x in arr.ravel()[:max_list]] + ["<...>"]
        return [safe_jsonable(x, max_list=max_list) for x in arr.ravel().tolist()]
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        try:
            return json.loads(obj.to_json(orient="split"))
        except Exception:
            return {"type": str(type(obj)), "shape": list(obj.shape) if hasattr(obj, "shape") else None}
    return str(obj)


# Simple timing context manager
@dataclass
class Timer:
    name: str = "timer"
    start: float = 0.0
    elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start


__all__ = [
    "utc_now_iso",
    "make_run_id",
    "set_global_seed",
    "is_categorical_series",
    "numeric_columns",
    "categorical_columns",
    "dataframe_signature",
    "fingerprint_df",
    "stratified_train_valid_split_indices",
    "temporal_train_valid_split_indices",
    "tscv_indices",
    "to_numpy_2d",
    "safe_jsonable",
    "Timer",
]
