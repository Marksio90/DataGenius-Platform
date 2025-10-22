# backend/runtime_preprocessor.py
"""
Runtime preprocessor helpers for TMIV – Advanced ML Platform.

Focus
-----
Utilities that act **at runtime** (on the current session / dataset), complementing
the static feature pipeline from `backend.auto_prep`. They do not render UI.

What you get
------------
- Target handling for regression/classification:
  * `TargetTransformer` with auto strategy (none / log1p / yeo-johnson* / standardize)
  * Fast skewness-based heuristics to pick a transform for regression targets
  * Safe inverse transform
- Class/sample weighting:
  * `compute_class_weights` (dict) and `build_sample_weight` (vector) for clf
- Numeric scaling patch:
  * `patch_numeric_scaler(pre, kind)` to swap StandardScaler↔RobustScaler inside ColumnTransformer
- Array/data guards:
  * `replace_infinite_with_nan(X)` and `ensure_2d(X)`

(*) yeo-johnson uses scikit-learn's PowerTransformer when available; falls back to 'none'.

Public API
----------
- suggest_target_transform(y, problem_type) -> dict
- TargetTransformer(strategy="auto").fit(y).transform(y).inverse_transform(y_t)
- compute_class_weights(y, method="balanced") -> dict[label, weight]
- build_sample_weight(y, method="balanced", max_weight=25.0) -> np.ndarray | None
- patch_numeric_scaler(pre: AutoPrep, kind="standard"|"robust"|"none") -> AutoPrep
- replace_infinite_with_nan(X) -> np.ndarray
- ensure_2d(X) -> np.ndarray
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, RobustScaler

try:
    from sklearn.preprocessing import PowerTransformer  # yeo-johnson
except Exception:  # pragma: no cover
    PowerTransformer = None  # type: ignore

try:
    from sklearn.utils.class_weight import compute_class_weight
except Exception:  # pragma: no cover
    compute_class_weight = None  # type: ignore

from .auto_prep import AutoPrep


# =========================
# Array / value guards
# =========================

def replace_infinite_with_nan(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    arr[np.isinf(arr)] = np.nan
    return arr


def ensure_2d(X: np.ndarray | Sequence[float]) -> np.ndarray:
    arr = np.asarray(X)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


# =========================
# Target transform (regression)
# =========================

def _safe_skew(y: np.ndarray) -> float:
    try:
        import scipy.stats as st  # type: ignore

        return float(st.skew(y[~np.isnan(y)]))
    except Exception:
        # pandas fallback
        try:
            return float(pd.Series(y).skew(skipna=True))
        except Exception:
            return 0.0


def suggest_target_transform(
    y: Sequence[float] | pd.Series,
    problem_type: str | None = None,
) -> Dict[str, str | float]:
    """
    Heuristic suggestion for regression target transformation.
    Returns: {"strategy": str, "reason": str, "skew": float}
    """
    pt = (problem_type or "").lower()
    if pt != "regression":
        return {"strategy": "none", "reason": "not_regression", "skew": 0.0}

    y_arr = np.asarray(pd.Series(y).astype(float))
    finite = y_arr[np.isfinite(y_arr)]
    if finite.size == 0:
        return {"strategy": "none", "reason": "no_finite_values", "skew": 0.0}

    s = _safe_skew(finite)
    ymin = float(np.nanmin(finite))
    ymax = float(np.nanmax(finite))

    # Rules:
    # - Strong positive skew and strictly > -1 -> log1p
    # - Mixed sign values -> yeo-johnson if available
    # - Mild skew -> standardize (optional)
    if ymin >= 0 and s > 1.0:
        return {"strategy": "log1p", "reason": "positive_skew", "skew": s}
    if ymin < 0 < ymax:
        if PowerTransformer is not None:
            return {"strategy": "yeo-johnson", "reason": "mixed_signs", "skew": s}
        return {"strategy": "none", "reason": "mixed_signs_no_powertransformer", "skew": s}
    if abs(s) > 0.75:
        # still some skew – yeo-johnson if available
        if PowerTransformer is not None:
            return {"strategy": "yeo-johnson", "reason": "mild_skew", "skew": s}
    return {"strategy": "none", "reason": "no_transform_needed", "skew": s}


@dataclass
class TargetTransformer:
    """
    Stateless-ish wrapper with optional fitted components for regression targets.
    strategy: "auto" | "none" | "log1p" | "yeo-johnson" | "standardize"
    """
    strategy: str = "auto"

    # fitted objects
    _scaler: Optional[StandardScaler] = None
    _power: Any | None = None  # PowerTransformer
    _fitted_strategy: str = "none"

    def fit(self, y: Sequence[float], problem_type: str | None = None) -> "TargetTransformer":
        pt = (problem_type or "").lower()
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)

        chosen = self.strategy
        if self.strategy == "auto":
            suggestion = suggest_target_transform(y_arr.ravel(), problem_type=pt)
            chosen = str(suggestion["strategy"])

        self._fitted_strategy = chosen

        if pt != "regression" or chosen == "none":
            return self

        if chosen == "log1p":
            # nothing to "fit"
            return self

        if chosen == "yeo-johnson" and PowerTransformer is not None:
            self._power = PowerTransformer(method="yeo-johnson", standardize=False)
            self._power.fit(y_arr)
            return self

        if chosen == "standardize":
            self._scaler = StandardScaler()
            self._scaler.fit(y_arr)
            return self

        # fallback
        self._fitted_strategy = "none"
        return self

    def transform(self, y: Sequence[float]) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        s = self._fitted_strategy

        if s == "none":
            return y_arr.ravel()
        if s == "log1p":
            return np.log1p(np.maximum(y_arr, -0.999999)).ravel()
        if s == "yeo-johnson" and self._power is not None:
            return self._power.transform(y_arr).ravel()
        if s == "standardize" and self._scaler is not None:
            return self._scaler.transform(y_arr).ravel()
        # fallback
        return y_arr.ravel()

    def inverse_transform(self, y_t: Sequence[float]) -> np.ndarray:
        z = np.asarray(y_t, dtype=float).reshape(-1, 1)
        s = self._fitted_strategy

        if s == "none":
            return z.ravel()
        if s == "log1p":
            return (np.expm1(z)).ravel()
        if s == "yeo-johnson" and self._power is not None:
            return self._power.inverse_transform(z).ravel()
        if s == "standardize" and self._scaler is not None:
            return self._scaler.inverse_transform(z).ravel()
        return z.ravel()

    @property
    def fitted_strategy(self) -> str:
        return self._fitted_strategy


# =========================
# Class/sample weighting
# =========================

def compute_class_weights(y: Sequence[Any], method: str = "balanced") -> Optional[Dict[Any, float]]:
    """
    Return {class_label: weight} or None if unavailable.
    """
    y_arr = np.asarray(y)
    labels = np.unique(y_arr[~pd.isna(y_arr)])
    if labels.size == 0:
        return None
    if compute_class_weight is None:
        # simple inverse frequency
        vals, counts = np.unique(y_arr, return_counts=True)
        inv = counts.sum() / np.maximum(counts.astype(float), 1.0)
        inv = inv / inv.mean()
        return {v: float(w) for v, w in zip(vals, inv)}
    try:
        w = compute_class_weight(class_weight=method, classes=labels, y=y_arr)  # type: ignore[arg-type]
        return {lbl: float(ww) for lbl, ww in zip(labels, w)}
    except Exception:
        return None


def build_sample_weight(
    y: Sequence[Any],
    method: str = "balanced",
    *,
    max_weight: float = 25.0,
) -> Optional[np.ndarray]:
    """
    Build per-sample weights array (classification). Caps extreme weights.
    """
    cw = compute_class_weights(y, method=method)
    if not cw:
        return None
    y_arr = np.asarray(y)
    w = np.asarray([cw.get(val, 1.0) for val in y_arr], dtype=float)
    if np.isfinite(max_weight) and max_weight > 0:
        w = np.clip(w, 1.0 / max_weight, max_weight)
    # normalize to mean ~1
    m = np.nanmean(w) if w.size else 1.0
    if m and np.isfinite(m):
        w = w / m
    return w


# =========================
# Numeric scaler patch
# =========================

def patch_numeric_scaler(pre: AutoPrep, kind: str = "standard") -> AutoPrep:
    """
    Replace numeric scaler inside AutoPrep ColumnTransformer.

    kind:
      - "standard": StandardScaler(with_mean=True, with_std=True)
      - "robust": RobustScaler(quantile_range=(25,75))
      - "none": remove scaler step (keep imputer only)

    Returns the same `pre` object (mutated) for convenience.
    """
    ct = pre.pipeline
    # find 'num' transformer
    try:
        # sklearn stores transformers_ after fit, but we mutate the "transformers" definition instead
        new_transformers = []
        for name, trans, cols in ct.transformers:
            if name != "num":
                new_transformers.append((name, trans, cols))
                continue

            # trans is a Pipeline(imputer[, scaler])
            if hasattr(trans, "steps"):
                steps = list(trans.steps)
                # drop any scaler-like steps
                steps = [(n, s) for (n, s) in steps if n != "scaler"]
                if kind == "standard":
                    steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
                elif kind == "robust":
                    steps.append(("scaler", RobustScaler(quantile_range=(25.0, 75.0))))
                # "none": keep as-is without scaler
                from sklearn.pipeline import Pipeline

                trans = Pipeline(steps=steps)
            new_transformers.append(("num", trans, cols))
        ct.transformers = new_transformers  # type: ignore[attr-defined]
    except Exception:
        # ignore failures; return pre unchanged
        return pre
    return pre


__all__ = [
    "suggest_target_transform",
    "TargetTransformer",
    "compute_class_weights",
    "build_sample_weight",
    "patch_numeric_scaler",
    "replace_infinite_with_nan",
    "ensure_2d",
]
