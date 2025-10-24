
from __future__ import annotations
from typing import Optional
import numpy as np, pandas as pd
from functools import lru_cache

def safe_permutation_import():
    try:
        from sklearn.inspection import permutation_importance
        return permutation_importance
    except Exception:
        return None

def compute_permutation_importance(estimator, X: pd.DataFrame, y, problem: str="regression", n_repeats: int = 5, random_state: int = 42) -> Optional[pd.DataFrame]:
    perm = safe_permutation_import()
    if perm is None:
        return None
    try:
        scoring = "r2" if problem == "regression" else "accuracy"
        r = perm(estimator, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring)
        return pd.DataFrame({"feature": X.columns, "importance_mean": r.importances_mean, "importance_std": r.importances_std}).sort_values("importance_mean", ascending=False)
    except Exception:
        return None

@lru_cache(maxsize=16)
def _key(n, m, name): return (n, m, name)

def compute_shap_importance(estimator, X: pd.DataFrame) -> Optional[pd.DataFrame]:
    try:
        import shap  # type: ignore
    except Exception:
        return None
    try:
        Xs = X.sample(min(800, len(X)), random_state=42)
        _ = _key(len(Xs), Xs.shape[1], estimator.__class__.__name__)
        try:
            explainer = shap.TreeExplainer(estimator)
            sv = explainer.shap_values(Xs)
        except Exception:
            explainer = shap.KernelExplainer(estimator.predict, Xs.sample(min(200, len(Xs)), random_state=42))
            sv = explainer.shap_values(Xs.sample(min(500, len(Xs)), random_state=42))
        if isinstance(sv, list):
            vals = np.mean([np.abs(v).mean(axis=0) for v in sv], axis=0)
        else:
            vals = np.abs(sv).mean(axis=0)
        return pd.DataFrame({"feature": X.columns[:len(vals)], "importance_mean": vals}).sort_values("importance_mean", ascending=False)
    except Exception:
        return None
