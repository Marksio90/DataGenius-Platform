# ml/utils.py
"""
ML Utils (skeleton → solid) – TMIV Advanced ML Platform.

Zestaw pomocników używanych w warstwie ML/Explain:
- Bez zależności od Streamlit/UI.
- Oparty na numpy/pandas/sklearn (soft – ale standardowe w projekcie).

Funkcje kluczowe:
- safe_predict_proba(model, X)              -> np.ndarray|None  (obsługuje decision_function)
- compute_classification_metrics(...)       -> dict
- compute_regression_metrics(...)           -> dict
- build_leaderboard(metrics_by_model, ...)  -> pd.DataFrame (z rankiem i kierunkiem metryk)
- feature_importance_from_model(model, ...) -> pd.DataFrame(feature, importance, rank)
- helpers: sigmoid, softmax, compress_binary_proba, get_feature_names

Uwaga:
- Dla Pipeline: zakładamy kroki ("pre", "est"), ale próbujemy też wykrywać „ostatni estimator”.
- FI:
  * drzewa: `feature_importances_`
  * modele liniowe: `abs(coef_)` (flatten + normalizacja)
  * gdy brak – zwracamy pusty DataFrame
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

# =========================
# Helpers: math & shapes
# =========================

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # stabilizacja numeryczna
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[neg])
    out[neg] = expx / (1.0 + expx)
    return out


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    sm = ex / np.sum(ex, axis=axis, keepdims=True)
    return sm


def compress_binary_proba(proba: np.ndarray) -> np.ndarray:
    """Dla binarnej (N,2) zwróć wektor (N,) prawd. klasy pozytywnej."""
    if proba is None:
        return None  # type: ignore[return-value]
    p = np.asarray(proba)
    if p.ndim == 2 and p.shape[1] == 2:
        return p[:, 1]
    return p


# =========================
# Feature names from preprocessors
# =========================

def _last_estimator(model: BaseEstimator) -> BaseEstimator:
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _preprocessor_from_pipeline(model: BaseEstimator):
    if isinstance(model, Pipeline):
        for name, step in model.steps:
            if name in {"pre", "preprocess", "preprocessor"}:
                return step
    return None


def get_feature_names(model: BaseEstimator, X: Optional[pd.DataFrame] = None) -> List[str]:
    """
    Spróbuj wyciągnąć nazwy cech po `ColumnTransformer`/OHE.
    Fallback: X.columns lub range.
    """
    pre = _preprocessor_from_pipeline(model)
    if pre is not None:
        try:  # sklearn >= 1.0
            names = pre.get_feature_names_out()
            return [str(n) for n in names]
        except Exception:
            pass
        # ColumnTransformer bez get_feature_names_out – spróbuj ręcznie
        try:
            # best-effort: spłaszcz nazwy
            names = []
            transformers = getattr(pre, "transformers_", None) or []
            for name, trans, cols in transformers:
                if hasattr(trans, "get_feature_names_out"):
                    try:
                        sub = list(map(str, trans.get_feature_names_out(cols)))
                    except Exception:
                        sub = list(map(str, cols if isinstance(cols, (list, tuple, np.ndarray)) else [cols]))
                else:
                    sub = list(map(str, cols if isinstance(cols, (list, tuple, np.ndarray)) else [cols]))
                names.extend(sub)
            if names:
                return names
        except Exception:
            pass

    # Fallbacky
    if X is not None and isinstance(X, pd.DataFrame):
        return [str(c) for c in X.columns]
    return []


# =========================
# Predict-proba (bezpiecznie)
# =========================

def safe_predict_proba(model: BaseEstimator, X) -> Optional[np.ndarray]:
    """
    Zwróć macierz prawdopodobieństw (N, C) lub wektor (N,) dla binarnej.
    - jeśli brak `predict_proba` → użyj `decision_function` i rzutuj (sigmoid/softmax).
    - jeśli brak obu → None.
    """
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)  # type: ignore[call-arg]
            return np.asarray(proba)
    except Exception:
        pass

    try:
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(X))  # type: ignore[call-arg]
            if scores.ndim == 1:
                p = sigmoid(scores)  # binarna
                return p
            else:
                return softmax(scores, axis=1)
    except Exception:
        pass

    return None


# =========================
# Metrics
# =========================

def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    *,
    average: str = "weighted",
) -> Dict[str, float]:
    """
    Zwróć słownik metryk klasyfikacyjnych.
    Obsługiwane:
      - accuracy, f1, f1_weighted
      - roc_auc (OVR jeśli multi) – wymaga y_proba / scores
      - logloss – wymaga y_proba
      - aps (Average Precision) – sensowny przede wszystkim binarnie
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        roc_auc_score,
        average_precision_score,
    )

    out: Dict[str, float] = {}
    # podstawowe
    try:
        out["accuracy"] = float(accuracy_score(y_true, y_pred))
    except Exception:
        out["accuracy"] = float("nan")

    try:
        out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    except Exception:
        out["f1_weighted"] = float("nan")

    try:
        # 'binary' może rzucać, gdy multiclass – wtedy 'macro'
        out["f1"] = float(f1_score(y_true, y_pred, average="binary"))
    except Exception:
        try:
            out["f1"] = float(f1_score(y_true, y_pred, average=average))
        except Exception:
            out["f1"] = float("nan")

    # prawdopodobieństwa / scores
    if y_proba is not None:
        p = np.asarray(y_proba)
        try:
            if p.ndim == 1:
                out["roc_auc"] = float(roc_auc_score(y_true, p))
            else:
                # multiclass
                out["roc_auc"] = float(roc_auc_score(y_true, p, multi_class="ovr"))
        except Exception:
            out["roc_auc"] = float("nan")

        try:
            if p.ndim == 1:
                out["aps"] = float(average_precision_score(y_true, p))
            else:
                # macro-avg AP na klasach (przybliżenie)
                aps_vals = []
                from sklearn.preprocessing import label_binarize
                y_bin = label_binarize(y_true, classes=np.unique(y_true))
                for j in range(p.shape[1]):
                    try:
                        aps_vals.append(float(average_precision_score(y_bin[:, j], p[:, j])))
                    except Exception:
                        aps_vals.append(np.nan)
                out["aps"] = float(np.nanmean(aps_vals))
        except Exception:
            out["aps"] = float("nan")

        try:
            out["logloss"] = float(log_loss(y_true, p if p.ndim == 2 else np.vstack([1 - p, p]).T))
        except Exception:
            out["logloss"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["logloss"] = float("nan")
        out["aps"] = float("nan")

    return out


def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Zwróć R2, RMSE, MAE."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    out: Dict[str, float] = {}
    try:
        out["r2"] = float(r2_score(y_true, y_pred))
    except Exception:
        out["r2"] = float("nan")
    try:
        out["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
    except Exception:
        out["rmse"] = float("nan")
    try:
        out["mae"] = float(mean_absolute_error(y_true, y_pred))
    except Exception:
        out["mae"] = float("nan")
    return out


# =========================
# Leaderboard
# =========================

DEFAULT_BIGGER_IS_BETTER = {
    "accuracy",
    "f1",
    "f1_weighted",
    "roc_auc",
    "roc_auc_ovr",
    "roc_auc_ovo",
    "r2",
    "aps",
    "score",
}
DEFAULT_LOWER_IS_BETTER = {"rmse", "mae", "logloss"}


def build_leaderboard(
    metrics_by_model: Mapping[str, Mapping[str, float]],
    *,
    primary_metric: str,
    bigger_is_better: Optional[Iterable[str]] = None,
    lower_is_better: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Zbuduj tabelę leaderboardu z rankiem wg `primary_metric`.
    `metrics_by_model`:
        {"model_name": {"metricA": 0.9, "metricB": 0.1, ...}, ...}
    """
    if not metrics_by_model:
        return pd.DataFrame(columns=["model", primary_metric, "rank"])

    rows: List[dict] = []
    for name, mm in metrics_by_model.items():
        row = {"model": name}
        row.update({k: (float(v) if v is not None else np.nan) for k, v in mm.items()})
        # kolumna pomocnicza
        row["primary_metric"] = row.get(primary_metric, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)

    # sort kierunek
    bigger = set(bigger_is_better or DEFAULT_BIGGER_IS_BETTER)
    lower = set(lower_is_better or DEFAULT_LOWER_IS_BETTER)
    if primary_metric in lower:
        asc = True
    else:
        asc = False

    df = df.sort_values(by=["primary_metric"], ascending=asc, na_position="last").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    # bardziej czytelna kolumna
    df = df.rename(columns={"primary_metric": primary_metric})
    return df


# =========================
# Feature Importance
# =========================

def feature_importance_from_model(
    model: BaseEstimator,
    *,
    X_fit: Optional[pd.DataFrame] = None,
    feature_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Wyciągnij FI z modelu lub pipeline'u:
      - drzewa: `feature_importances_`
      - liniowe: `abs(coef_)` (flatten)
    Nazwy cech: z preprocessora (get_feature_names_out) lub `feature_names` / `X_fit.columns`.

    Zwraca DataFrame: feature, importance, rank (malejąco po importance).
    Gdy brak FI → pusty DF (bez wyjątków).
    """
    est = _last_estimator(model)
    names: List[str] = list(feature_names or []) or get_feature_names(model, X_fit)

    # 1) Tree-based
    imp: Optional[np.ndarray] = None
    if hasattr(est, "feature_importances_"):
        try:
            imp = np.asarray(getattr(est, "feature_importances_"), dtype=float)
        except Exception:
            imp = None

    # 2) Linear models (coef_)
    if imp is None and hasattr(est, "coef_"):
        try:
            coef = np.asarray(getattr(est, "coef_"), dtype=float)
            if coef.ndim > 1:
                coef = np.mean(np.abs(coef), axis=0)  # multi-class → średnia po klasach
            else:
                coef = np.abs(coef)
            imp = coef
        except Exception:
            imp = None

    # brak FI
    if imp is None:
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    # dopasuj długość nazw
    if not names or len(names) != len(imp):
        names = [f"f{i}" for i in range(len(imp))]

    imp = np.where(np.isnan(imp), 0.0, imp)
    if normalize:
        s = float(np.sum(np.abs(imp)))
        if s > 0:
            imp = imp / s
        else:
            m = float(np.max(np.abs(imp))) if np.max(np.abs(imp)) > 0 else 1.0
            imp = imp / m

    df = pd.DataFrame({"feature": names, "importance": imp})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


__all__ = [
    "sigmoid",
    "softmax",
    "compress_binary_proba",
    "safe_predict_proba",
    "compute_classification_metrics",
    "compute_regression_metrics",
    "build_leaderboard",
    "feature_importance_from_model",
    "get_feature_names",
    "DEFAULT_BIGGER_IS_BETTER",
    "DEFAULT_LOWER_IS_BETTER",
]
