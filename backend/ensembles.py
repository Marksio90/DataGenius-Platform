
# backend/ensembles.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

def weighted_blend(preds: List[np.ndarray], y_true: Optional[np.ndarray]=None, metric=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple non-negative weights with sum=1 (NNLS-like) using least squares on preds vs y_true.
    If y_true or metric is None, return equal weights.
    """
    k = len(preds)
    if k == 0:
        raise ValueError("No predictions to blend")
    if y_true is None or metric is None:
        w = np.ones(k) / k
        blend = np.average(np.stack(preds, axis=1), axis=1, weights=w)
        return blend, w
    # Stack predictions
    P = np.stack(preds, axis=1)  # (n_samples, k)
    # Constrained least squares: non-negative, sum=1 (projected gradient simple loop)
    w = np.ones(k) / k
    lr = 0.1
    for _ in range(200):
        grad = 2.0 * P.T @ (P @ w - y_true) / len(y_true)
        w = w - lr * grad
        w = np.clip(w, 0.0, None)
        s = w.sum()
        if s <= 1e-12:
            w = np.ones(k)/k
        else:
            w = w / s
    blend = P @ w
    return blend, w

def build_stacking(problem: str, base_estimators: List[Tuple[str, Any]], final_estimator: Any):
    from sklearn.ensemble import StackingRegressor, StackingClassifier
    if problem == "regression":
        return StackingRegressor(estimators=base_estimators, final_estimator=final_estimator, passthrough=False, n_jobs=-1)
    else:
        return StackingClassifier(estimators=base_estimators, final_estimator=final_estimator, passthrough=False, n_jobs=-1)
