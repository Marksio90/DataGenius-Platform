# backend/explain_plus.py
"""
Model explainability helpers for TMIV – Advanced ML Platform.

Focus:
- SHAP (Tree/Model-agnostic) with safe fallbacks and compact outputs
- LIME (optional) per-instance explanations
- Permutation importance (scikit-learn)
- Simple PDP/ICE computation (model-agnostic)
- Lightweight PNG plot exporters to the cache artifacts directory

Design principles
-----------------
- Optional dependencies (shap, lime). If unavailable, functions degrade gracefully.
- Cache heavy computations (fingerprints over data & params).
- Avoid UI code here; return data structures / file paths. Frontend renders them.

Public API (selected)
---------------------
- shap_importance(estimator, X, feature_names, *, max_samples=2000) -> pd.DataFrame
- save_shap_summary_plot(estimator, X, feature_names, *, out_name=None, max_samples=2000) -> str | None
- lime_explain_instances(estimator, X, feature_names, *, class_names=None, sample_indices=None, num_features=10) -> list | None
- permutation_importance_df(estimator, X, y, *, scoring=None, n_repeats=5, random_state=42) -> pd.DataFrame
- compute_pdp_ice(estimator, X, feature_name, *, grid=None, kind="both", n_ice=50) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .cache_manager import cache_result, df_fingerprint, cached_path

# Optional deps
try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None  # type: ignore

try:
    from lime.lime_tabular import LimeTabularExplainer  # type: ignore
except Exception:  # pragma: no cover
    LimeTabularExplainer = None  # type: ignore

from sklearn.metrics import get_scorer
from sklearn.inspection import permutation_importance
from sklearn.utils import check_random_state


# =========================
# Helpers
# =========================

def _is_dataframe(X: Any) -> bool:
    return hasattr(X, "columns") and hasattr(X, "dtypes")


def _to_numpy(X: Any) -> np.ndarray:
    if _is_dataframe(X):
        return X.values
    return np.asarray(X)


def _feature_names_from(X: Any, feature_names: Sequence[str] | None) -> list[str]:
    if feature_names is not None:
        return list(map(str, feature_names))
    if _is_dataframe(X):
        return list(map(str, X.columns))
    return [f"x{i}" for i in range(_to_numpy(X).shape[1])]


def _safe_sample_rows(X: pd.DataFrame | np.ndarray, max_samples: int, random_state: int = 42):
    rs = check_random_state(random_state)
    if _is_dataframe(X):
        df = X  # type: ignore
        if len(df) <= max_samples:
            return df
        return df.sample(max_samples, random_state=rs)
    arr = _to_numpy(X)
    if arr.shape[0] <= max_samples:
        return arr
    idx = rs.choice(arr.shape[0], size=max_samples, replace=False)
    return arr[idx]


def _is_tree_model(estimator: Any) -> bool:
    name = estimator.__class__.__name__.lower()
    mod = getattr(estimator.__class__, "__module__", "")
    return any(
        kw in (name + " " + mod)
        for kw in [
            "xgboost", "lightgbm", "catboost",
            "randomforest", "extratrees", "gradientboosting", "histgradientboosting",
            "decisiontree",
        ]
    )


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


# =========================
# SHAP
# =========================

@cache_result(namespace="shap", ttl=1800)
def shap_importance(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None = None,
    *,
    max_samples: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute mean(|SHAP|) per feature. For multiclass, averages abs across classes.
    Returns DataFrame: columns = [feature, importance, importance_std].
    """
    if shap is None:
        return pd.DataFrame(columns=["feature", "importance", "importance_std"])

    Xs = _safe_sample_rows(X, max_samples=max_samples, random_state=random_state)
    fn = _feature_names_from(Xs, feature_names)

    # Build explainer
    try:
        if _is_tree_model(estimator):
            explainer = shap.TreeExplainer(estimator)
            shap_values = explainer(Xs)
            vals = np.array(shap_values.values)  # (n, n_features) or (n, n_classes, n_features)
        else:
            # Model-agnostic path; shap.Explainer auto-picks Kernel/Linear
            explainer = shap.Explainer(estimator, Xs)
            shap_values = explainer(Xs)
            vals = np.array(shap_values.values)
    except Exception:
        # As a last resort, attempt linear explainer if coef_ present
        try:
            explainer = shap.LinearExplainer(estimator, Xs)  # type: ignore[attr-defined]
            shap_values = explainer(Xs)
            vals = np.array(shap_values.values)
        except Exception:
            return pd.DataFrame(columns=["feature", "importance", "importance_std"])

    # Normalize shape
    # Binary/multiclass may produce (n, n_classes, n_features); take mean(abs) over classes
    if vals.ndim == 3:
        vals = np.mean(np.abs(vals), axis=1)  # (n, n_features)
    elif vals.ndim == 2:
        vals = np.abs(vals)  # (n, n_features)
    else:
        return pd.DataFrame(columns=["feature", "importance", "importance_std"])

    imp = vals.mean(axis=0)
    imp_std = vals.std(axis=0)

    df = pd.DataFrame({"feature": fn, "importance": imp, "importance_std": imp_std})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def _matplotlib_quiet_backend():
    import matplotlib
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    import matplotlib.pyplot as plt  # noqa: F401
    return matplotlib


@cache_result(namespace="shap_plot", ttl=1800)
def save_shap_summary_plot(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None = None,
    *,
    max_samples: int = 2000,
    out_name: str | None = None,
) -> str | None:
    """
    Save SHAP summary plot (beeswarm) as PNG under cache artifacts. Returns absolute path or None.
    """
    if shap is None:
        return None

    matplotlib = _matplotlib_quiet_backend()
    import matplotlib.pyplot as plt

    Xs = _safe_sample_rows(X, max_samples=max_samples)
    fn = _feature_names_from(Xs, feature_names)

    try:
        if _is_tree_model(estimator):
            explainer = shap.TreeExplainer(estimator)
        else:
            explainer = shap.Explainer(estimator, Xs)
        sv = explainer(Xs)
    except Exception:
        return None

    # Ensure feature names for plot
    try:
        if _is_dataframe(Xs):
            sv.feature_names = list(map(str, Xs.columns))  # type: ignore
        else:
            sv.feature_names = fn  # type: ignore
    except Exception:
        pass

    # Prepare output path
    base_name = out_name or f"shap_summary_{df_fingerprint(Xs if _is_dataframe(Xs) else pd.DataFrame(Xs, columns=fn))}.png"
    out_path = cached_path("explain", base_name)

    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(sv.values, _to_numpy(Xs), show=False, feature_names=fn)  # type: ignore[arg-type]
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150)
        plt.close()
        return str(out_path)
    except Exception:
        try:
            plt.close()
        except Exception:
            pass
        return None


# =========================
# LIME (optional)
# =========================

@cache_result(namespace="lime", ttl=1200)
def lime_explain_instances(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None = None,
    *,
    class_names: Sequence[str] | None = None,
    sample_indices: Sequence[int] | None = None,
    num_features: int = 10,
    mode: str | None = None,  # "classification" | "regression"
    random_state: int = 42,
) -> list[dict] | None:
    """
    Produce LIME explanations for selected instances.
    Returns list of dicts with keys: index, score, local_pred, exp ([(feature, weight), ...]).
    """
    if LimeTabularExplainer is None:
        return None

    Xnp = _to_numpy(X)
    fn = _feature_names_from(X, feature_names)

    # Mode detection
    if mode is None:
        # crude: presence of predict_proba suggests classification
        mode = "classification" if hasattr(estimator, "predict_proba") else "regression"

    rs = check_random_state(random_state)
    if sample_indices is None:
        k = min(3, Xnp.shape[0])
        sample_indices = list(rs.choice(Xnp.shape[0], size=k, replace=False))

    explainer = LimeTabularExplainer(
        training_data=Xnp,
        feature_names=fn,
        class_names=list(class_names) if class_names is not None else None,
        mode=mode,
        discretize_continuous=True,
        random_state=random_state,
        verbose=False,
    )

    results: list[dict] = []
    for idx in sample_indices:
        x0 = Xnp[idx]
        if mode == "classification":
            predict_fn = getattr(estimator, "predict_proba", estimator.predict)
        else:
            predict_fn = estimator.predict
        try:
            exp = explainer.explain_instance(
                data_row=x0,
                predict_fn=predict_fn,  # type: ignore[arg-type]
                num_features=num_features,
                top_labels=1 if mode == "classification" else None,
            )
            explanation = exp.as_list(label=exp.available_labels()[0] if mode == "classification" else None)
            local_pred = float(np.asarray(predict_fn(x0.reshape(1, -1))).reshape(-1)[0])  # type: ignore[call-arg]
            results.append(
                {
                    "index": int(idx),
                    "score": float(exp.score),
                    "local_pred": local_pred,
                    "exp": [(str(f), float(w)) for f, w in explanation],
                }
            )
        except Exception:
            # skip instance on failure
            continue

    return results


# =========================
# Permutation importance
# =========================

@cache_result(namespace="perm_imp", ttl=1200)
def permutation_importance_df(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    y: Sequence[Any],
    *,
    scoring: str | None = None,
    n_repeats: int = 5,
    random_state: int = 42,
    n_jobs: int | None = None,
) -> pd.DataFrame:
    """
    Compute permutation importance and return sorted DataFrame:
      [feature, importance_mean, importance_std]
    """
    fn = _feature_names_from(X, None)
    Xnp = _to_numpy(X)
    scorer = get_scorer(scoring) if scoring else None
    try:
        res = permutation_importance(
            estimator, Xnp, np.asarray(y), scoring=scorer, n_repeats=n_repeats, random_state=random_state, n_jobs=n_jobs
        )
        df = pd.DataFrame(
            {
                "feature": fn,
                "importance_mean": res.importances_mean,
                "importance_std": res.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_std"])


# =========================
# PDP / ICE
# =========================

@cache_result(namespace="pdp_ice", ttl=1800)
def compute_pdp_ice(
    estimator: Any,
    X: pd.DataFrame | np.ndarray,
    feature_name: str,
    *,
    grid: Sequence[float] | None = None,
    kind: str = "both",  # "pdp" | "ice" | "both"
    n_grid: int = 20,
    n_ice: int = 50,
    random_state: int = 42,
) -> dict:
    """
    Compute simple PDP (partial dependence) and ICE curves for a single feature.

    Returns dict with keys (depending on `kind`):
      - "grid": list[float]
      - "pdp": list[float]
      - "ice": list[list[float]]   # up to `n_ice` instances
      - "feature": str
    """
    if not _is_dataframe(X):
        # require DataFrame to reference a feature by name
        X = pd.DataFrame(_to_numpy(X), columns=_feature_names_from(X, None))
    df: pd.DataFrame = X  # type: ignore

    if feature_name not in df.columns:
        return {"feature": feature_name, "grid": [], "pdp": [], "ice": []}

    rs = check_random_state(random_state)
    col = df[feature_name]
    # Build grid
    if grid is None:
        if np.issubdtype(col.dtype, np.number):
            q = np.linspace(0.05, 0.95, n_grid)
            grid = np.quantile(col.dropna().values, q).tolist()
        else:
            # categorical-like: top unique values
            grid = col.dropna().value_counts().index[: min(n_grid, col.nunique())].tolist()

    # ICE sampling
    idx = df.index
    if len(idx) > n_ice:
        idx = rs.choice(df.index, size=n_ice, replace=False)

    def predict(arr: np.ndarray) -> np.ndarray:
        # Try proba for classifiers (prob of positive class), else predict
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(arr)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            if proba.ndim == 1:
                return proba
        return np.asarray(estimator.predict(arr)).reshape(-1)

    # Compute ICE
    ice_curves: list[list[float]] = []
    if kind in {"ice", "both"}:
        for rid in idx:
            row = df.loc[rid].copy()
            base = df.loc[[rid]].copy()
            vals: list[float] = []
            for v in grid:
                base.loc[rid, feature_name] = v
                vals.append(float(predict(base.values)[0]))
            ice_curves.append(vals)

    # Compute PDP as average ICE
    pdp_vals: list[float] = []
    if kind in {"pdp", "both"}:
        if not ice_curves:
            # build using a small random subset if ICE not computed
            sub = df.sample(min(n_ice, len(df)), random_state=rs)
            tmp_curves: list[list[float]] = []
            for _, row in sub.iterrows():
                base = row.to_frame().T
                vals = []
                for v in grid:
                    base.loc[:, feature_name] = v
                    vals.append(float(predict(base.values)[0]))
                tmp_curves.append(vals)
            pdp_vals = list(np.mean(np.asarray(tmp_curves), axis=0))
        else:
            pdp_vals = list(np.mean(np.asarray(ice_curves), axis=0))

    return {
        "feature": feature_name,
        "grid": list(map(lambda x: x.item() if hasattr(x, "item") else x, grid)),
        "pdp": pdp_vals,
        "ice": ice_curves,
    }


__all__ = [
    "shap_importance",
    "save_shap_summary_plot",
    "lime_explain_instances",
    "permutation_importance_df",
    "compute_pdp_ice",
]
