from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple


# =============================
# Helpers
# =============================

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    # stabilizacja numeryczna
    pos = x >= 0
    neg = ~pos
    z = np.zeros_like(x, dtype=float)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    z[neg] = ex / (1.0 + ex)
    return z


def _softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.nanmax(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    s = np.nansum(ez, axis=axis, keepdims=True)
    s[s == 0] = 1.0
    return ez / s


def _is_numeric_like(s: pd.Series, min_frac: float = 0.5) -> bool:
    cn = pd.to_numeric(s, errors="coerce")
    return cn.notna().mean() >= min_frac


def _numeric_quantile_grid(s: pd.Series, grid_size: int = 20) -> np.ndarray:
    cn = pd.to_numeric(s, errors="coerce")
    cn = cn.dropna()
    if cn.empty:
        return np.array([], dtype=float)
    qs = np.linspace(0, 1, max(2, grid_size))
    g = np.unique(np.nanquantile(cn, qs))
    # jeżeli wszystko stałe – zwróć jeden punkt
    return g


def _categorical_grid(s: pd.Series, max_levels: int = 20) -> np.ndarray:
    # najpopularniejsze kategorie (zachowaj kolejność malejącą)
    vc = s.astype("string").fillna("<NA>").value_counts(dropna=False)
    levels = vc.index[:max_levels].to_numpy()
    return levels.astype(object)


def _quantile_or_category_grid(s: pd.Series, grid_size: int = 20) -> np.ndarray:
    if _is_numeric_like(s):
        g = _numeric_quantile_grid(s, grid_size=grid_size)
        return g
    return _categorical_grid(s, max_levels=grid_size)


def _ensure_proba_matrix(p: np.ndarray) -> np.ndarray:
    """
    Ujednolica predykcje klasyfikacyjne do macierzy [n, C]:
      • 1D → binarne: [1-p, p]
      • dowolny 2D → clip+renorm, a jeśli wygląda jak logity → softmax
    """
    p = np.asarray(p)
    if p.ndim == 1:
        p = np.clip(p.astype(float), 0.0, 1.0)
        p = np.c_[1.0 - p, p]
    else:
        # jeśli wartości ewidentnie nie-probabilistyczne – użyj softmax
        if np.nanmin(p) < -1e-6 or np.nanmax(p) > 1 + 1e-6:
            p = _softmax(p, axis=1)
        p = np.nan_to_num(p, nan=1e-7, posinf=1-1e-7, neginf=1e-7)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        row_sum = p.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        p = p / row_sum
    return p


def _predict_values_or_proba(
    pipeline,
    X: pd.DataFrame,
    proba_class: Optional[int] = None,
) -> np.ndarray:
    """
    Zwróć:
      • dla klasyfikacji – wektor p(y==proba_class); jeśli proba_class=None:
        - binarne: klasa 1
        - wieloklasowe: klasa o najwyższym średnim prawdopodobieństwie
      • dla regresji – wektor wartości.
    """
    # 1) prefer predict_proba, jeśli dostępne
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)
        proba = _ensure_proba_matrix(proba)
        C = proba.shape[1]
        if proba_class is None:
            if C == 2:
                cls = 1
            else:
                cls = int(np.nanargmax(np.nanmean(proba, axis=0)))
        else:
            cls = int(np.clip(proba_class, 0, C - 1))
        return proba[:, cls]

    # 2) decision_function → przeskaluj do (0,1)
    if hasattr(pipeline, "decision_function"):
        df = pipeline.decision_function(X)
        df = np.asarray(df)
        if df.ndim == 1:          # binarne
            return _sigmoid(df)
        else:                      # wieloklasowe logity
            proba = _softmax(df, axis=1)
            if proba_class is None:
                cls = int(np.nanargmax(np.nanmean(proba, axis=0)))
            else:
                cls = int(np.clip(proba_class, 0, proba.shape[1] - 1))
            return proba[:, cls]

    # 3) fallback: predict (może być regresja albo już etykiety)
    pred = pipeline.predict(X)
    pred = np.asarray(pred)
    if pred.ndim == 1:
        return pred.astype(float, copy=False)
    # rzadko: pred ma kształt [n,C] – potraktuj jak proby/logity
    proba = _ensure_proba_matrix(pred)
    if proba_class is None:
        cls = int(np.nanargmax(np.nanmean(proba, axis=0)))
    else:
        cls = int(np.clip(proba_class, 0, proba.shape[1] - 1))
    return proba[:, cls]


# =============================
# PDP & ICE (1D)
# =============================

def compute_pdp_ice_data(
    pipeline,
    X: pd.DataFrame,
    feature: str,
    grid_size: int = 20,
    ice_samples: int = 50,
    proba_class: Optional[int] = None,
    random_state: int = 13
) -> Dict[str, Any]:
    """
    Oblicza PDP/ICE dla pojedynczej cechy.
    • Działa dla regresji i klasyfikacji (probabilities/decision_function/predict).
    • Obsługuje cechy kategoryczne (siatka = najpopularniejsze poziomy).

    Returns: {"grid": np.ndarray, "pdp": np.ndarray[len(grid)], "ice": List[np.ndarray]}
    """
    if X is None or feature not in X.columns:
        raise ValueError("X must be provided with the selected feature")

    # zbuduj siatkę wartości
    grid = _quantile_or_category_grid(X[feature], grid_size=grid_size)
    if grid.size == 0:
        raise ValueError(f"Feature '{feature}' has no valid values to build PDP grid.")

    # próbka do ICE
    n = len(X)
    if n == 0:
        return {"grid": grid, "pdp": np.zeros_like(grid, dtype=float), "ice": []}

    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=min(int(ice_samples), n), replace=False)

    ice_curves: List[np.ndarray] = []
    # generujemy replikę dla każdej próbki (rozmiar ~ ice_samples * len(grid))
    for i in idx:
        base = X.iloc[[i]].copy()
        repli = pd.concat([base] * len(grid), ignore_index=True)
        repli[feature] = grid
        preds_vec = _predict_values_or_proba(pipeline, repli, proba_class=proba_class)
        ice_curves.append(np.asarray(preds_vec, dtype=float).reshape(-1))

    # PDP = średnia po krzywych ICE
    if ice_curves:
        mat = np.vstack(ice_curves)  # [m, G]
        pdp = np.nanmean(mat, axis=0)
    else:
        pdp = np.zeros_like(grid, dtype=float)

    return {"grid": grid, "pdp": pdp, "ice": ice_curves}


# =============================
# PDP 2D (pair-wise)
# =============================

def compute_pdp2d_data(
    pipeline,
    X: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    grid_size: int = 30,
    proba_class: Optional[int] = None,
) -> Dict[str, Any]:
    """
    2D PDP (siatka po dwóch cechach). Obsługuje regresję i klasyfikację.
    Zwraca: {"x_grid": gx, "y_grid": gy, "Z": macierz [len(gy), len(gx)]}
    """
    if X is None or feature_x not in X.columns or feature_y not in X.columns:
        raise ValueError("X must contain the selected two features")

    gx = _quantile_or_category_grid(X[feature_x], grid_size=grid_size)
    gy = _quantile_or_category_grid(X[feature_y], grid_size=grid_size)
    if gx.size == 0 or gy.size == 0:
        raise ValueError("Cannot build grids for selected features (no valid values).")

    # Zbuduj punkty siatki przez produkt kartezjański (działa dla numeric i kategorii)
    rows: List[pd.DataFrame] = []
    for yi in gy:
        # wiersze dla stałego yi i wszystkich xi
        base = X.iloc[[0]].copy()
        base = pd.concat([base] * len(gx), ignore_index=True)
        base[feature_x] = gx
        base[feature_y] = yi
        rows.append(base)

    big = pd.concat(rows, ignore_index=True)

    preds_vec = _predict_values_or_proba(pipeline, big, proba_class=proba_class)
    Z = np.asarray(preds_vec, dtype=float).reshape(len(gy), len(gx))
    return {"x_grid": gx, "y_grid": gy, "Z": Z}
