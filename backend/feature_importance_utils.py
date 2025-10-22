from __future__ import annotations
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np
import pandas as pd

# ===================== Permutation importance =====================

def safe_permutation_import():
    """Bezpieczny import sklearn.inspection.permutation_importance."""
    try:
        from sklearn.inspection import permutation_importance
        return permutation_importance
    except Exception:
        return None


def _infer_scoring(problem: str, y: pd.Series | np.ndarray, estimator) -> str:
    """
    Dobiera sensowny scoring:
      - regresja: r2
      - klasyfikacja: f1_weighted (bezpieczne dla niezbalansowanych i multiclass, nie wymaga predict_proba)
    """
    problem = (problem or "regression").lower()
    if problem.startswith("reg"):
        return "r2"
    try:
        n_classes = len(np.unique(pd.Series(y).dropna()))
    except Exception:
        n_classes = 2
    return "f1_weighted"


def compute_permutation_importance(
    estimator,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    problem: str = "regression",
    *,
    scoring: Optional[str] = None,
    n_repeats: int = 5,
    random_state: int = 42,
    max_samples: Optional[int] = None,
    n_jobs: int = 1
) -> Optional[pd.DataFrame]:
    """
    Zwraca DataFrame ['feature','importance_mean','importance_std'] posortowany malejąco.
    Stosuje opcjonalne próbkowanie (max_samples) dla przyspieszenia i bezpieczny dobór metryki.
    Zwraca pusty DataFrame przy błędzie zamiast None.
    """
    perm = safe_permutation_import()
    if perm is None or X is None or len(X) == 0:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])

    X = pd.DataFrame(X).copy()
    y_arr = np.asarray(y if isinstance(y, (pd.Series, np.ndarray)) else pd.Series(y).values)

    # Próbkowanie dla szybkości
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X), size=max_samples, replace=False)
        X = X.iloc[idx]
        y_arr = y_arr[idx]

    chosen_scoring = scoring or _infer_scoring(problem, y_arr, estimator)

    try:
        r = perm(
            estimator, X, y_arr,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=chosen_scoring,
            n_jobs=n_jobs
        )
        imp = (
            pd.DataFrame({
                "feature": X.columns,
                "importance_mean": r.importances_mean,
                "importance_std": r.importances_std
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        return imp
    except Exception:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])


# ========================= SHAP importance =========================

def _is_tree_model(est) -> bool:
    name = est.__class__.__name__.lower()
    return any(k in name for k in [
        "randomforest", "extratrees", "gradientboost", "histgradient",
        "xgb", "lgbm", "catboost", "decisiontree"
    ])


def _is_linear_model(est) -> bool:
    name = est.__class__.__name__.lower()
    return any(k in name for k in ["linear", "ridge", "lasso", "elasticnet", "logistic"])


def compute_shap_importance(
    estimator,
    X: pd.DataFrame,
    *,
    max_background: int = 200,
    max_samples: int = 800,
    random_state: int = 42
) -> Optional[pd.DataFrame]:
    """
    Liczy globalną ważność cech SHAP (średnia |SHAP|).
    Zwraca DataFrame ['feature','importance_mean'] posortowany malejąco.
    Zwraca None, jeśli SHAP nie jest dostępny lub wystąpił błąd krytyczny.
    """
    try:
        import shap  # type: ignore
    except Exception:
        return None

    if X is None or len(X) == 0:
        return None

    X = pd.DataFrame(X).copy()
    rng = np.random.RandomState(random_state)

    # Próbka danych (szybkość)
    if len(X) > max_samples:
        idx = rng.choice(len(X), size=max_samples, replace=False)
        Xs = X.iloc[idx].reset_index(drop=True)
    else:
        Xs = X.reset_index(drop=True)

    # Tło dla Kernel/LinearExplainer
    bsize = min(max_background, len(Xs))
    bg = Xs.iloc[rng.choice(len(Xs), size=bsize, replace=False)]

    try:
        est = estimator

        # Wybór explainera
        if _is_tree_model(est):
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(Xs)
        elif _is_linear_model(est):
            try:
                explainer = shap.LinearExplainer(est, bg)
                sv = explainer.shap_values(Xs)
            except Exception:
                # Fallback do KernelExplainer (wolniejszy)
                pred_fn = getattr(est, "predict", None)
                if pred_fn is None:
                    return None
                explainer = shap.KernelExplainer(pred_fn, bg)
                sv = explainer.shap_values(Xs)
        else:
            # KernelExplainer: preferuj predict_proba dla klasyfikacji, inaczej predict
            pred_fn = getattr(est, "predict_proba", None) or getattr(est, "predict", None)
            if pred_fn is None:
                return None
            explainer = shap.KernelExplainer(pred_fn, bg)
            sv = explainer.shap_values(Xs)

        # Multi-class zwraca listę macierzy; binarka/regresja – 2D
        if isinstance(sv, list):
            vals = np.mean([np.abs(v).mean(axis=0) for v in sv], axis=0)
        else:
            vals = np.abs(sv).mean(axis=0)

        vals = np.asarray(vals).ravel()
        k = min(len(vals), Xs.shape[1])

        df_imp = (
            pd.DataFrame({
                "feature": Xs.columns[:k],
                "importance_mean": vals[:k]
            })
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )
        return df_imp
    except Exception:
        return None