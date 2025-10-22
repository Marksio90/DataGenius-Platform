# backend/ensembles.py
"""
Ensembling utilities for TMIV – Advanced ML Platform.

Goals
-----
- Provide **stacking** and **blending** builders for both classification and regression.
- Keep dependencies minimal (scikit-learn + numpy).
- Be robust to estimators that don't implement `predict_proba` (fallbacks).
- Offer a tiny heuristic to decide **when** to enable ensembling.

Public API
----------
- recommend_ensemble(rows, cols, problem_type, metrics) -> dict
- build_stacking_classifier(base_models: dict, final_estimator=None, cv=5, passthrough=False, n_jobs=None)
- build_stacking_regressor(base_models: dict, final_estimator=None, cv=5, passthrough=False, n_jobs=None)
- make_average_blender_classifier(base_models: dict, weights: dict | None = None)
- make_average_blender_regressor(base_models: dict, weights: dict | None = None)

Notes
-----
- `base_models` is a mapping `{name: estimator}`; estimators will be cloned before fit.
- For classification blending, we average **probabilities** (or sigmoid(decision_function) fallback).
- For regression blending, we average predictions.
- Stacking uses scikit-learn `StackingClassifier/Regressor`.

Example
-------
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from backend.ensembles import (
        build_stacking_classifier, make_average_blender_classifier
    )

    base = {
        "lr": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200, n_jobs=-1),
    }
    stack = build_stacking_classifier(base, cv=5, passthrough=True)
    blend = make_average_blender_classifier(base)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    r2_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import LabelEncoder


# =========================
# Heuristics: when to ensemble
# =========================


def recommend_ensemble(
    rows: int,
    cols: int,
    problem_type: str,
    metrics: Mapping[str, float] | None = None,
) -> dict:
    """
    Lightweight rules of thumb to decide whether to try ensembling.

    Returns
    -------
    {
      "enable_stacking": bool,
      "enable_blending": bool,
      "cv": int,                 # suggested CV folds for stacking
      "meta": str,               # suggested meta-estimator
      "reason": str
    }
    """
    pt = (problem_type or "").lower()
    m = metrics or {}

    enable_stacking = False
    enable_blending = False
    reason_bits: List[str] = []

    # Dataset size
    if rows >= 3000 or cols >= 30:
        enable_stacking = True
        enable_blending = True
        reason_bits.append("big_dataset")

    # Performance signals: if primary metric not great, try ensembles
    if pt == "classification":
        auc = m.get("roc_auc")
        f1 = m.get("f1") or m.get("f1_weighted")
        if (auc is not None and auc < 0.85) or (f1 is not None and f1 < 0.8):
            enable_blending = True
            enable_stacking = True
            reason_bits.append("metric_room_for_improvement")
    elif pt in {"regression", "timeseries"}:
        r2 = m.get("r2")
        rmse = m.get("rmse")
        if (r2 is None or r2 < 0.75) or (rmse is not None and rmse > 0):
            enable_blending = True
            enable_stacking = True
            reason_bits.append("metric_room_for_improvement")

    # Small data – prefer blending (cheaper) over stacking
    if rows < 1200:
        enable_stacking = enable_stacking and (cols >= 20)  # optional
        enable_blending = True
        reason_bits.append("small_data_prefers_blend")

    # Defaults
    cv = 5 if rows >= 2000 else 3
    meta = "logreg" if pt == "classification" else "ridge"

    if not reason_bits:
        reason_bits.append("default")

    return {
        "enable_stacking": bool(enable_stacking),
        "enable_blending": bool(bool(enable_blending) or bool(enable_stacking)),
        "cv": int(cv),
        "meta": meta,
        "reason": ",".join(reason_bits),
    }


# =========================
# Blenders (average)
# =========================


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable
    z = np.clip(x, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def _proba_safe_binary(estimator: Any, X: np.ndarray) -> np.ndarray:
    """
    Return P(y=1) for binary classifiers.
    """
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
        if p.ndim == 1:
            return p
    if hasattr(estimator, "decision_function"):
        d = estimator.decision_function(X)
        d = np.asarray(d).reshape(-1)
        return _sigmoid(d)
    # Fallback: predicted label as hard 0/1
    yhat = estimator.predict(X)
    return (np.asarray(yhat).reshape(-1) == 1).astype(float)


def _proba_safe_multiclass(estimator: Any, X: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Return probability matrix of shape (n_samples, n_classes).
    """
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(X)
        if p.ndim == 1:
            # Somehow a vector returned; try to expand
            p = np.vstack([1 - p, p]).T
    elif hasattr(estimator, "decision_function"):
        d = estimator.decision_function(X)
        d = np.atleast_2d(d)
        # Softmax on decision scores
        d = d - d.max(axis=1, keepdims=True)
        expd = np.exp(d)
        p = expd / np.clip(expd.sum(axis=1, keepdims=True), 1e-9, None)
    else:
        # Hard labels -> one-hot
        yhat = np.asarray(estimator.predict(X)).reshape(-1)
        p = np.zeros((len(yhat), n_classes), dtype=float)
        for i, c in enumerate(yhat):
            p[i, int(c) if 0 <= int(c) < n_classes else 0] = 1.0
    # Ensure correct number of columns
    if p.shape[1] != n_classes:
        # pad/crop
        q = np.zeros((p.shape[0], n_classes), dtype=float)
        k = min(n_classes, p.shape[1])
        q[:, :k] = p[:, :k]
        p = q
    # Normalize rows
    s = np.clip(p.sum(axis=1, keepdims=True), 1e-9, None)
    return p / s


class AverageProbaBlenderClassifier(BaseEstimator, ClassifierMixin):
    """
    Simple probability-average blender for classification.

    Parameters
    ----------
    estimators : list[tuple[str, estimator]]
        Base estimators (will be cloned before fitting).
    weights : list[float] | None
        Optional weights per estimator; if None, equal weights are used.
    """

    def __init__(self, estimators: list[tuple[str, Any]], weights: list[float] | None = None):
        self.estimators = estimators
        self.weights = weights
        self.label_encoder_: Optional[LabelEncoder] = None
        self.fitted_estimators_: list[tuple[str, Any]] = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y):
        self.fitted_estimators_ = []
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        for name, est in self.estimators:
            self.fitted_estimators_.append((name, clone(est).fit(X, y_enc)))
        return self

    def predict_proba(self, X):
        if self.classes_ is None:
            raise RuntimeError("Model not fitted.")
        n_classes = len(self.classes_)
        # Collect probabilities
        probs = []
        for _, est in self.fitted_estimators_:
            if n_classes == 2:
                p1 = _proba_safe_binary(est, X)
                p = np.vstack([1 - p1, p1]).T
            else:
                p = _proba_safe_multiclass(est, X, n_classes)
            probs.append(p)
        if not probs:
            raise RuntimeError("No base estimators fitted.")

        P = np.stack(probs, axis=0)  # (n_estimators, n_samples, n_classes)
        if self.weights is None:
            w = np.ones((len(probs), 1, 1), dtype=float)
        else:
            w = np.asarray(self.weights, dtype=float).reshape(len(probs), 1, 1)
            w = w / np.clip(w.sum(), 1e-9, None)
        PW = P * w
        return PW.sum(axis=0)

    def predict(self, X):
        proba = self.predict_proba(X)
        y_idx = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(y_idx)  # type: ignore[call-arg]


class AverageBlenderRegressor(BaseEstimator, RegressorMixin):
    """
    Simple average (or weighted average) blender for regression.
    """

    def __init__(self, estimators: list[tuple[str, Any]], weights: list[float] | None = None):
        self.estimators = estimators
        self.weights = weights
        self.fitted_estimators_: list[tuple[str, Any]] = []

    def fit(self, X, y):
        self.fitted_estimators_ = [(name, clone(est).fit(X, y)) for name, est in self.estimators]
        return self

    def predict(self, X):
        preds = []
        for _, est in self.fitted_estimators_:
            p = np.asarray(est.predict(X)).reshape(-1)
            preds.append(p)
        if not preds:
            raise RuntimeError("No base estimators fitted.")
        M = np.stack(preds, axis=0)  # (n_estimators, n_samples)
        if self.weights is None:
            w = np.ones((len(preds), 1), dtype=float) / len(preds)
        else:
            w = np.asarray(self.weights, dtype=float).reshape(len(preds), 1)
            w = w / np.clip(w.sum(), 1e-9, None)
        return (M * w).sum(axis=0)


# =========================
# Builders
# =========================


def build_stacking_classifier(
    base_models: Mapping[str, Any],
    final_estimator: Any | None = None,
    *,
    cv: int = 5,
    passthrough: bool = False,
    n_jobs: int | None = None,
) -> StackingClassifier:
    """
    Build a StackingClassifier with sensible defaults.
    """
    estimators = [(k, clone(v)) for k, v in base_models.items()]
    meta = final_estimator or LogisticRegression(max_iter=2000, n_jobs=n_jobs if isinstance(n_jobs, int) else None)
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        passthrough=passthrough,
        cv=cv,
        n_jobs=n_jobs,
    )
    return model


def build_stacking_regressor(
    base_models: Mapping[str, Any],
    final_estimator: Any | None = None,
    *,
    cv: int = 5,
    passthrough: bool = False,
    n_jobs: int | None = None,
) -> StackingRegressor:
    """
    Build a StackingRegressor with sensible defaults.
    """
    estimators = [(k, clone(v)) for k, v in base_models.items()]
    meta = final_estimator or RidgeCV(alphas=(0.1, 1.0, 10.0))
    model = StackingRegressor(
        estimators=estimators,
        final_estimator=meta,
        passthrough=passthrough,
        cv=cv,
        n_jobs=n_jobs,
    )
    return model


def make_average_blender_classifier(
    base_models: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
) -> AverageProbaBlenderClassifier:
    """
    Create an AverageProbaBlenderClassifier from a dict of base models and optional weights.
    """
    ests = [(k, clone(v)) for k, v in base_models.items()]
    w_list = None
    if weights:
        w_list = [float(weights.get(k, 1.0)) for k in base_models.keys()]
    return AverageProbaBlenderClassifier(ests, weights=w_list)


def make_average_blender_regressor(
    base_models: Mapping[str, Any],
    weights: Mapping[str, float] | None = None,
) -> AverageBlenderRegressor:
    """
    Create an AverageBlenderRegressor from a dict of base models and optional weights.
    """
    ests = [(k, clone(v)) for k, v in base_models.items()]
    w_list = None
    if weights:
        w_list = [float(weights.get(k, 1.0)) for k in base_models.keys()]
    return AverageBlenderRegressor(ests, weights=w_list)


# =========================
# Quick evaluators (optional helpers)
# =========================


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    details: Dict[str, Any]


def evaluate_classifier(estimator: Any, X_val: np.ndarray, y_val: np.ndarray) -> EvalResult:
    """
    Compute a small set of classification metrics (macro/weighted where appropriate).
    """
    le = LabelEncoder().fit(y_val)
    y_true = le.transform(y_val)
    y_pred = estimator.predict(X_val)
    y_pred = le.transform(y_pred)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }
    # Probabilistic metrics when available
    try:
        proba = estimator.predict_proba(X_val)
        if proba.shape[1] == 2:
            auc = roc_auc_score(y_true, proba[:, 1])
            aps = average_precision_score(y_true, proba[:, 1])
        else:
            # one-vs-rest macro
            auc = roc_auc_score(y_true, proba, multi_class="ovr")
            aps = np.mean([
                average_precision_score((y_true == k).astype(int), proba[:, k])
                for k in range(proba.shape[1])
            ])
        metrics["roc_auc"] = float(auc)
        metrics["aps"] = float(aps)
    except Exception:
        pass

    return EvalResult(metrics=metrics, details={})


def evaluate_regressor(estimator: Any, X_val: np.ndarray, y_val: np.ndarray) -> EvalResult:
    """
    Compute a small set of regression metrics.
    """
    y_pred = np.asarray(estimator.predict(X_val)).reshape(-1)
    y_true = np.asarray(y_val).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return EvalResult(
        metrics={"rmse": rmse, "mae": mae, "r2": r2},
        details={},
    )


__all__ = [
    "recommend_ensemble",
    "build_stacking_classifier",
    "build_stacking_regressor",
    "make_average_blender_classifier",
    "make_average_blender_regressor",
    "evaluate_classifier",
    "evaluate_regressor",
    "AverageProbaBlenderClassifier",
    "AverageBlenderRegressor",
    "EvalResult",
]
