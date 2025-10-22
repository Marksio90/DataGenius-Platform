from __future__ import annotations
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Tuple, Optional

_EPS = 1e-9


def _as_1d(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        arr = np.asarray(x).squeeze()
    else:
        arr = np.asarray(x)
    return np.ravel(arr)


def _select_pos_proba(y_proba, proba_class: Optional[int]) -> np.ndarray:
    P = np.asarray(y_proba)
    if P.ndim == 1:
        p = P.astype(float)
    elif P.ndim == 2:
        c = 1 if proba_class is None and P.shape[1] == 2 else (proba_class or 0)
        c = int(np.clip(c, 0, P.shape[1] - 1))
        p = P[:, c].astype(float)
    else:
        raise ValueError("y_proba must be 1D vector or 2D probability matrix")
    # Bezpieczny zakres
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return p


def _to_binary_targets(y_true, proba_class: Optional[int], n_classes_in_proba: Optional[int]) -> np.ndarray:
    y = _as_1d(y_true)
    classes = np.unique(y)
    if classes.size <= 2:
        # spróbuj zmapować na {0,1}
        try:
            # jeśli to np. ['A','B'], weź drugą jako pozytyw
            pos = classes[1] if classes.size == 2 else classes[0]
            y_bin = (y == pos).astype(int) if not np.array_equal(classes, [0, 1]) else y.astype(int)
        except Exception:
            y_bin = (y == classes[-1]).astype(int)
        return y_bin

    # wieloklasa -> binarka względem wskazanej kolumny z proba
    pos_idx = 1 if (proba_class is None and (n_classes_in_proba or 0) == 2) else (proba_class or 0)
    pos_idx = int(np.clip(pos_idx, 0, classes.size - 1))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    y_idx = np.array([class_to_idx.get(val, -1) for val in y], dtype=int)
    return (y_idx == pos_idx).astype(int)


# ======================================================================
# API
# ======================================================================

def plot_roc_proba(
    y_true,
    y_proba,
    proba_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Zwraca (fpr, tpr, auc). Przyjmuje:
      - y_proba: wektor (n,) albo macierz (n, C),
      - dla (n, C) wybiera kolumnę `proba_class` (domyślnie 1 przy C=2, w innym wypadku 0).
    Gdy w y_true jest tylko jedna klasa → zwraca puste wektory i auc=nan.
    """
    try:
        from sklearn.metrics import roc_curve, auc
        p = _select_pos_proba(y_proba, proba_class)
        y_bin = _to_binary_targets(y_true, proba_class, p.ndim if p.ndim > 1 else None)

        # maska na NaN/inf
        yb = _as_1d(y_bin).astype(float)
        pp = _as_1d(p).astype(float)
        m = np.isfinite(yb) & np.isfinite(pp)
        yb, pp = yb[m], pp[m]

        if np.unique(yb).size < 2 or yb.size == 0:
            return np.array([]), np.array([]), float("nan")

        fpr, tpr, _ = roc_curve(yb.astype(int), pp)
        return fpr, tpr, float(auc(fpr, tpr))
    except Exception:
        return np.array([]), np.array([]), float("nan")


def plot_pr_curve(
    y_true,
    y_proba,
    proba_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zwraca (recall, precision). Obsługuje wektor/macierze proba analogicznie jak plot_roc_proba.
    Gdy w y_true jedna klasa → zwraca puste wektory.
    """
    try:
        from sklearn.metrics import precision_recall_curve
        p = _select_pos_proba(y_proba, proba_class)
        y_bin = _to_binary_targets(y_true, proba_class, p.ndim if p.ndim > 1 else None)

        yb = _as_1d(y_bin).astype(float)
        pp = _as_1d(p).astype(float)
        m = np.isfinite(yb) & np.isfinite(pp)
        yb, pp = yb[m], pp[m]

        if np.unique(yb).size < 2 or yb.size == 0:
            return np.array([]), np.array([])

        precision, recall, _ = precision_recall_curve(yb.astype(int), pp)
        return recall, precision
    except Exception:
        return np.array([]), np.array([])


def residuals_plot(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zwraca (y_pred_clean, residuals) z odfiltrowanymi NaN/inf.
    """
    yt = _as_1d(y_true).astype(float)
    yp = _as_1d(y_pred).astype(float)
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]
    res = yt - yp
    return yp, res


def calibration_data(
    y_true,
    y_proba,
    n_bins: int = 10,
    proba_class: Optional[int] = None,
    strategy: str = "quantile",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Zwraca (fraction_of_positives, mean_predicted_value).
    Bezpiecznie przetwarza proby (wektor/macierze) i przypadki jednoklasowe.
    """
    try:
        from sklearn.calibration import calibration_curve
        p = _select_pos_proba(y_proba, proba_class)
        y_bin = _to_binary_targets(y_true, proba_class, p.ndim if p.ndim > 1 else None)

        yb = _as_1d(y_bin).astype(int)
        pp = _as_1d(p).astype(float)
        m = np.isfinite(yb) & np.isfinite(pp)
        yb, pp = yb[m], pp[m]

        if np.unique(yb).size < 2 or yb.size == 0:
            return np.array([]), np.array([])

        # Najpierw 'quantile', a jak się nie uda – 'uniform'
        try:
            frac_pos, mean_pred = calibration_curve(yb, pp, n_bins=int(n_bins), strategy=strategy)
        except Exception:
            frac_pos, mean_pred = calibration_curve(yb, pp, n_bins=int(n_bins), strategy="uniform")
        return frac_pos, mean_pred
    except Exception:
        return np.array([]), np.array([])


__all__ = [
    "plot_roc_proba",
    "plot_pr_curve",
    "residuals_plot",
    "calibration_data",
]
