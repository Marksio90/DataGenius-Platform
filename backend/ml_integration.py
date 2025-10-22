# backend/ml_integration.py
"""
Training & evaluation integration for TMIV – Advanced ML Platform.

High-level goals
----------------
- Detect problem type (classification vs regression) from target.
- Build a **robust preprocessing** (via backend.auto_prep) and train **multiple models**.
- Evaluate on a hold-out split (+ optional light CV), produce a **leaderboard**,
  unify **feature importance**, and return everything needed by the UI/exports.
- Work even when optional libs (xgboost/lightgbm/catboost) are not installed.

Public API
----------
train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    *,
    plan: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int | None = 3,
    enable_ensembles: bool = True,
) -> dict

Returned payload (keys)
-----------------------
{
  "problem_type": "classification" | "regression",
  "target": str,
  "split": {"train_rows": int, "valid_rows": int, "stratified": bool},
  "leaderboard": pd.DataFrame,
  "best_model_name": str,
  "models": {name: fitted_estimator, ...},
  "preprocessor": AutoPrep,
  "feature_names": list[str],
  "feature_importance": pd.DataFrame,   # for the best model
  "y_encoder": LabelEncoder | None,     # for classification
  "metrics_by_model": dict[str, dict],  # detailed metrics (holdout)
  "cv_by_model": dict[str, dict],       # mean/std for CV if enabled
  "y_valid": np.ndarray,                # encoded for classification
  "y_mapping": dict | None              # class mapping if applicable
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from .auto_prep import AutoPrep, build_preprocessor, transform_fit
from .error_handler import safe_call
from .ensembles import (
    make_average_blender_classifier,
    make_average_blender_regressor,
    recommend_ensemble,
)
from .explain_plus import permutation_importance_df, shap_importance

# Optional heavy models
try:
    from xgboost import XGBClassifier, XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore
except Exception:  # pragma: no cover
    LGBMClassifier = LGBMRegressor = None  # type: ignore

try:
    from catboost import CatBoostClassifier, CatBoostRegressor  # type: ignore
except Exception:  # pragma: no cover
    CatBoostClassifier = CatBoostRegressor = None  # type: ignore

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline


# =========================
# Problem detection
# =========================


def detect_problem_type(y: pd.Series) -> str:
    """
    Heuristic detection of problem type from target series.
    - Numeric with many unique values -> regression
    - Boolean/binary or low-unique categorical -> classification
    """
    s = y
    if pd.api.types.is_bool_dtype(s):
        return "classification"
    nunique = int(s.nunique(dropna=True))
    total = int(s.notna().sum())
    # Strings/categories -> classification
    if pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s):
        return "classification"
    # Small unique set -> classification
    if nunique <= 10 and nunique / max(1, total) < 0.2:
        return "classification"
    # Numeric – check if integer-like with small unique
    if pd.api.types.is_integer_dtype(s) and nunique <= 20 and nunique / max(1, total) < 0.2:
        return "classification"
    # Otherwise regression
    return "regression"


# =========================
# Model zoo
# =========================


def _base_models_classification(n_jobs: int = -1) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    models["logreg"] = LogisticRegression(max_iter=2000, n_jobs=n_jobs)
    models["rf"] = RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=n_jobs, class_weight="balanced_subsample", random_state=42
    )
    if XGBClassifier is not None:
        models["xgb"] = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            n_jobs=n_jobs,
            random_state=42,
        )
    if LGBMClassifier is not None:
        models["lgbm"] = LGBMClassifier(
            n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, n_jobs=n_jobs, random_state=42
        )
    if CatBoostClassifier is not None:
        models["cat"] = CatBoostClassifier(
            iterations=500, learning_rate=0.05, depth=6, loss_function="Logloss", random_seed=42, verbose=False
        )
    return models


def _base_models_regression(n_jobs: int = -1) -> Dict[str, Any]:
    models: Dict[str, Any] = {}
    models["ridge"] = RidgeCV(alphas=(0.1, 1.0, 10.0))
    models["rf"] = RandomForestRegressor(n_estimators=500, max_depth=None, n_jobs=n_jobs, random_state=42)
    if XGBRegressor is not None:
        models["xgb"] = XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=n_jobs,
            random_state=42,
        )
    if LGBMRegressor is not None:
        models["lgbm"] = LGBMRegressor(
            n_estimators=600, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, n_jobs=n_jobs, random_state=42
        )
    if CatBoostRegressor is not None:
        models["cat"] = CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=6, loss_function="RMSE", random_seed=42, verbose=False
        )
    return models


# =========================
# Metrics
# =========================


def _eval_classification(est, X_val: np.ndarray, y_val: np.ndarray, *, proba_required: bool = False) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    y_pred = est.predict(X_val)
    metrics["accuracy"] = float(accuracy_score(y_val, y_pred))
    try:
        metrics["f1_weighted"] = float(f1_score(y_val, y_pred, average="weighted"))
        metrics["f1_macro"] = float(f1_score(y_val, y_pred, average="macro"))
    except Exception:
        pass

    # Probabilistic metrics
    auc = None
    aps = None
    ll = None
    try:
        proba = est.predict_proba(X_val)
        if proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 1):
            # force two-column
            proba = np.vstack([1 - proba, proba]).T
        n_classes = proba.shape[1]
        if n_classes == 2:
            auc = roc_auc_score(y_val, proba[:, 1])
            aps = average_precision_score(y_val, proba[:, 1])
        else:
            auc = roc_auc_score(y_val, proba, multi_class="ovr")
            aps = float(
                np.mean([average_precision_score((y_val == k).astype(int), proba[:, k]) for k in range(n_classes)])
            )
        ll = log_loss(y_val, proba, labels=np.unique(y_val))
    except Exception:
        # ignore if estimator has no proba
        if proba_required:
            raise

    if auc is not None:
        metrics["roc_auc"] = float(auc)
    if aps is not None:
        metrics["aps"] = float(aps)
    if ll is not None:
        metrics["logloss"] = float(ll)

    return metrics


def _eval_regression(est, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
    y_pred = np.asarray(est.predict(X_val)).reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))
    r2 = float(r2_score(y_val, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# =========================
# Feature importance (unified)
# =========================


def _coef_or_importance(est, feature_names: list[str]) -> Optional[pd.DataFrame]:
    # Tree-based
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(getattr(est, "feature_importances_")).reshape(-1)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        k = min(len(vals), len(feature_names))
        df = pd.DataFrame({"feature": feature_names[:k], "importance": vals[:k], "source": "model_attr"})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    # Linear-like
    if hasattr(est, "coef_"):
        coef = np.asarray(getattr(est, "coef_"))
        if coef.ndim == 1:
            vals = np.abs(coef)
        else:
            vals = np.abs(coef).mean(axis=0)
        k = min(len(vals), len(feature_names))
        df = pd.DataFrame({"feature": feature_names[:k], "importance": vals[:k], "source": "coef_abs"})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    return None


def _safe_shap(est, X, feature_names) -> Optional[pd.DataFrame]:
    try:
        df = shap_importance(est, X, feature_names)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.assign(source="shap")
            return df
    except Exception:
        pass
    return None


def _safe_permutation(est, X, y) -> Optional[pd.DataFrame]:
    try:
        df = permutation_importance_df(est, X, y, n_repeats=3)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df = df.rename(columns={"importance_mean": "importance"}).assign(source="permutation")
            return df[["feature", "importance", "importance_std", "source"]]
    except Exception:
        pass
    return None


def compute_feature_importance(est, X, y, feature_names) -> pd.DataFrame:
    """
    Try (1) model attrs, (2) SHAP, (3) permutation importance.
    """
    fi = _coef_or_importance(est, feature_names)
    if fi is not None and not fi.empty:
        return fi
    fi = _safe_shap(est, X, feature_names)
    if fi is not None and not fi.empty:
        return fi
    fi = _safe_permutation(est, X, y)
    if fi is not None and not fi.empty:
        return fi
    return pd.DataFrame(columns=["feature", "importance", "source"])


# =========================
# Leaderboard helpers
# =========================


def _select_primary_metric(problem_type: str, available_cols: Iterable[str]) -> str:
    cols = set(available_cols)
    if problem_type == "classification":
        for m in ("roc_auc", "f1_weighted", "accuracy", "aps"):
            if m in cols:
                return m
        return "accuracy"
    else:
        for m in ("r2", "rmse", "mae"):
            if m in cols:
                return m
        return "r2"


def _is_higher_better(metric: str, problem_type: str) -> bool:
    if problem_type == "classification":
        return metric not in {"logloss"}
    return metric in {"r2"}


def _build_leaderboard(metrics_by_model: Dict[str, Dict[str, float]], problem_type: str) -> pd.DataFrame:
    if not metrics_by_model:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(metrics_by_model, orient="index")
    df.index.name = "model"
    df.reset_index(inplace=True)
    primary = _select_primary_metric(problem_type, df.columns)
    df["primary_metric"] = df[primary] if primary in df.columns else np.nan
    higher_better = _is_higher_better(primary, problem_type)
    df.sort_values("primary_metric", ascending=not higher_better, inplace=True, na_position="last")
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# =========================
# Main API
# =========================


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    *,
    plan: Mapping[str, Any] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int | None = 3,
    enable_ensembles: bool = True,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Fit multiple models + evaluate, returning a rich result dict (see module docstring).
    """
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe.")

    # Detect problem type
    pt = (plan.get("problem_type") if plan else None) or detect_problem_type(df[target])

    # Preprocess
    pre: AutoPrep = build_preprocessor(df, target=target)
    X_all, y_series, feature_names = transform_fit(pre, df, target=target)
    if y_series is None:
        raise ValueError("Target column could not be extracted.")

    # Encode y for classification
    y_enc = None
    y_encoder: Optional[LabelEncoder] = None
    strat = False
    if pt == "classification":
        y_encoder = LabelEncoder().fit(y_series.astype(str))
        y_enc = y_encoder.transform(y_series.astype(str))
        strat = True
    else:
        y_enc = y_series.to_numpy()

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_enc,
        test_size=float(test_size),
        random_state=random_state,
        stratify=y_enc if strat else None,
    )

    # Build candidate models
    if pt == "classification":
        base_models = _base_models_classification(n_jobs=n_jobs)
    else:
        base_models = _base_models_regression(n_jobs=n_jobs)

    # Train & evaluate
    models_fitted: Dict[str, Any] = {}
    metrics_by_model: Dict[str, Dict[str, float]] = {}
    cv_by_model: Dict[str, Dict[str, float]] = {}

    for name, est in base_models.items():
        est_fitted = clone(est).fit(X_train, y_train)
        models_fitted[name] = est_fitted

        # Holdout metrics
        if pt == "classification":
            metrics = _eval_classification(est_fitted, X_val, y_val)
        else:
            metrics = _eval_regression(est_fitted, X_val, y_val)
        metrics_by_model[name] = metrics

        # Light CV (best-effort)
        if cv_folds and cv_folds >= 2:
            try:
                if pt == "classification":
                    # prefer roc_auc if possible
                    scoring = "roc_auc" if len(np.unique(y_train)) == 2 else "roc_auc_ovr"
                    # if fails (no proba), fall back to f1_weighted
                    try:
                        cv_obj = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                        scores = cross_val_score(clone(est), X_train, y_train, scoring=scoring, cv=cv_obj, n_jobs=n_jobs)
                    except Exception:
                        scoring = "f1_weighted"
                        cv_obj = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                        scores = cross_val_score(clone(est), X_train, y_train, scoring=scoring, cv=cv_obj, n_jobs=n_jobs)
                else:
                    scoring = "r2"
                    cv_obj = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                    scores = cross_val_score(clone(est), X_train, y_train, scoring=scoring, cv=cv_obj, n_jobs=n_jobs)
                cv_by_model[name] = {f"cv_{scoring}_mean": float(np.mean(scores)), f"cv_{scoring}_std": float(np.std(scores))}
            except Exception:
                # ignore CV failure
                pass

    # Optional ensembles
    if enable_ensembles and len(models_fitted) >= 2:
        rec = recommend_ensemble(rows=len(X_train), cols=len(feature_names), problem_type=pt, metrics=None)
        try:
            if pt == "classification" and rec.get("enable_blending"):
                blend = make_average_blender_classifier({k: base_models[k] for k in models_fitted.keys()})
                blend_name = "blend_avg"
                blend.fit(X_train, y_train)
                models_fitted[blend_name] = blend
                metrics_by_model[blend_name] = _eval_classification(blend, X_val, y_val)
            if pt == "regression" and rec.get("enable_blending"):
                blend = make_average_blender_regressor({k: base_models[k] for k in models_fitted.keys()})
                blend_name = "blend_avg"
                blend.fit(X_train, y_train)
                models_fitted[blend_name] = blend
                metrics_by_model[blend_name] = _eval_regression(blend, X_val, y_val)
        except Exception:
            # ensembles are optional; ignore failures
            pass

    # Leaderboard & best model
    leaderboard = _build_leaderboard(metrics_by_model, pt)
    if leaderboard.empty:
        raise RuntimeError("No metrics produced — training must have failed.")

    best_model_name = str(leaderboard.iloc[0]["model"])
    best_model = models_fitted[best_model_name]

    # Unify feature importance (best effort)
    fi_df = compute_feature_importance(best_model, X_val, y_val, feature_names)

    # Class distribution mapping (if classification)
    y_mapping = None
    if pt == "classification" and y_encoder is not None:
        y_mapping = {int(i): str(c) for i, c in enumerate(y_encoder.classes_)}

    return {
        "problem_type": pt,
        "target": target,
        "split": {"train_rows": int(len(X_train)), "valid_rows": int(len(X_val)), "stratified": bool(strat)},
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "models": models_fitted,
        "preprocessor": pre,
        "feature_names": feature_names,
        "feature_importance": fi_df,
        "y_encoder": y_encoder,
        "metrics_by_model": metrics_by_model,
        "cv_by_model": cv_by_model,
        "y_valid": y_val,
        "y_mapping": y_mapping,
    }


__all__ = ["train_and_evaluate", "detect_problem_type"]
