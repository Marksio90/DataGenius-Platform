"""
explain_plus.py (Stage 8)
Docstring (PL): Explainability+: Unified FI + PDP/ICE + (gated) SHAP.
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance, partial_dependence

def unified_feature_importance(pipe, X: pd.DataFrame, y: pd.Series, problem_type: str, top_n: int = 20, random_state: int = 42) -> Dict[str, Any]:
    np.random.seed(random_state)
    cols = list(X.columns)
    model = pipe.named_steps.get("model", None)
    model_fi = {}
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        model_fi = {f"feature_{i}": float(v) for i,v in enumerate(raw)}
    elif hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        vals = np.mean(np.abs(coef), axis=0) if getattr(coef, "ndim", 1)>1 else np.abs(coef)
        model_fi = {f"feature_{i}": float(v) for i,v in enumerate(vals)}

    try:
        r = permutation_importance(pipe, X, y, n_repeats=5, random_state=random_state, n_jobs=-1, scoring=None)
        perm = {cols[i]: float(r.importances_mean[i]) for i in range(len(cols))}
    except Exception:
        perm = {}

    keys = list(set(list(model_fi.keys()) + list(perm.keys())))
    out = {}
    for k in keys:
        a = model_fi.get(k, 0.0)
        b = perm.get(k, 0.0)
        out[k] = float(0.5*a + 0.5*b)
    top = sorted(out.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return {"model_based": model_fi, "permutation": perm, "unified_top": top}

def pdp_ice(pipe, X: pd.DataFrame, feature: str, kind: str = "average", grid_resolution: int = 20) -> Dict[str, Any]:
    """
    kind: 'average' (PDP) lub 'individual' (ICE)
    """
    try:
        pd = partial_dependence(pipe, X, [feature], kind=("average" if kind=="average" else "individual"), grid_resolution=grid_resolution)
        return {"status":"OK", "values": pd["average" if kind=="average" else "individual"].tolist(), "grid": pd["grid_values"][0].tolist()}
    except Exception as e:
        return {"status":"ERR", "message": str(e)}