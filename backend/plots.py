from __future__ import annotations
# -*- coding: utf-8 -*-

from backend.safe_utils import truthy_df_safe

# backend/plots.py

from typing import Optional, Dict, Sequence, Tuple
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


# ============================== HELPERY ==============================

_EPS = 1e-9

def _new_fig(size: Tuple[float, float]) -> plt.Figure:
    return plt.figure(figsize=size)

def _placeholder_fig(title: str, subtitle: str = "") -> plt.Figure:
    fig = _new_fig((6.5, 5))
    ax = plt.gca()
    ax.axis("off")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=14, weight="bold")
    if subtitle:
        ax.text(0.5, 0.45, subtitle, ha="center", va="center", fontsize=10)
    plt.tight_layout()
    return fig

def _as_1d(x) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.DataFrame)):
        arr = np.asarray(x).squeeze()
    else:
        arr = np.asarray(x)
    return np.ravel(arr)

def _as_binary_proba(proba: np.ndarray, proba_class: Optional[int] = None) -> np.ndarray:
    """
    Zwraca wektor prawdopodobieństw klasy pozytywnej.
    Akceptuje: (n,), (n,2) -> kolumna 1; (n,k) -> kolumna `proba_class` (domyślnie 0).
    """
    p = np.asarray(proba)
    if p.ndim == 1:
        out = p.astype(float)
    elif p.ndim == 2:
        if p.shape[1] == 2 and proba_class is None:
            out = p[:, 1].astype(float)
        else:
            c = int(np.clip(0 if proba_class is None else proba_class, 0, p.shape[1] - 1))
            out = p[:, c].astype(float)
    else:
        raise ValueError("Unsupported proba shape; expected 1D or 2D array")
    # Bezpieczne granice + finite
    out = np.clip(out, _EPS, 1.0 - _EPS)
    out[~np.isfinite(out)] = 0.5
    return out

def _binarize_targets(y_true, proba_class: Optional[int] = None, n_classes_hint: Optional[int] = None) -> np.ndarray:
    """
    Zamienia y na {0,1}. Dla wieloklasy bierze klasę o indeksie `proba_class` (domyślnie 0).
    Gdy etykiety nie są liczbowe, wybiera ostatnią etykietę jako pozytywną.
    """
    y = _as_1d(y_true)
    classes = np.unique(y)
    if classes.size <= 2:
        # spróbuj zmapować na {0,1}; jeśli to już 0/1, zostaw
        try:
            if set(np.unique(y)).issubset({0, 1}):
                return y.astype(int)
            # inny binarny format: pozytyw = druga unikalna
            pos = classes[1] if classes.size == 2 else classes[0]
            return (y == pos).astype(int)
        except Exception:
            return (y == classes[-1]).astype(int)
    # wieloklasa -> binarka vs wskazana klasa
    pos_idx = 1 if (proba_class is None and (n_classes_hint or 0) == 2) else (0 if proba_class is None else int(proba_class))
    try:
        # jeśli etykiety liczbowe i zawierają pos_idx
        if np.issubdtype(y.dtype, np.integer) and (pos_idx in classes):
            return (y == pos_idx).astype(int)
    except Exception:
        pass
    # fallback: ostatnia klasa jako pozytywna
    return (y == classes[-1]).astype(int)

def _filter_finite_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


# ============================== IMPORTANCE ==============================

def _pick_importance_column(imp: pd.DataFrame) -> str:
    """
    Ujednolica kolumnę ważności: obsługuje 'importance_mean', 'importance', 'shap_importance'.
    """
    for c in ("importance_mean", "importance", "shap_importance"):
        if c in imp.columns:
            return c
    if "importance" not in imp.columns:
        imp["importance"] = 0.0
    return "importance"

def plot_importance_bar(imp: pd.DataFrame, top: int = 20, title: str = "Feature importance") -> plt.Figure:
    if imp is None or len(imp) == 0:
        return _placeholder_fig(f"{title}", "(no data)")
    col = _pick_importance_column(imp)
    imp2 = imp.sort_values(col, ascending=False).head(top)
    fig = _new_fig((8, 5))
    plt.barh(list(imp2["feature"][::-1]), list(imp2[col][::-1]))
    plt.xlabel(col.replace("_", " ").title())
    plt.title(title)
    plt.tight_layout()
    return fig


# ============================== REGRESJA ==============================

def plot_true_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str = "True vs Predicted") -> plt.Figure:
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    y, yhat = _filter_finite_pair(y, yhat)
    if y.size == 0:
        return _placeholder_fig(title, "no finite data")
    lo = float(np.nanmin([y.min(), yhat.min()]))
    hi = float(np.nanmax([y.max(), yhat.max()]))
    fig = _new_fig((6, 5))
    plt.scatter(y, yhat, s=12, alpha=0.6)
    plt.plot([lo, hi], [lo, hi], ls="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_regression_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Residuals") -> plt.Figure:
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    y, yhat = _filter_finite_pair(y, yhat)
    res = y - yhat
    fig = _new_fig((6, 5))
    plt.scatter(yhat, res, alpha=0.5, s=12)
    plt.axhline(0.0, ls="--")
    plt.xlabel("Prediction")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_residuals_hist(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 30, title: str = "Residuals histogram") -> plt.Figure:
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    y, yhat = _filter_finite_pair(y, yhat)
    res = y - yhat
    fig = _new_fig((6, 5))
    plt.hist(res, bins=bins)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_residuals_qq(y_true: np.ndarray, y_pred: np.ndarray, title: str = "QQ Plot of residuals") -> plt.Figure:
    try:
        from scipy import stats
    except Exception:
        return _placeholder_fig(title, "scipy not available")
    y = np.asarray(y_true, dtype=float)
    yhat = np.asarray(y_pred, dtype=float)
    y, yhat = _filter_finite_pair(y, yhat)
    res = y - yhat
    fig = _new_fig((6, 5))
    stats.probplot(res, dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    return fig


# ============================== KLASYFIKACJA ==============================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Optional[Sequence] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix"
) -> plt.Figure:
    try:
        from sklearn.metrics import confusion_matrix
    except Exception:
        return _placeholder_fig(title, "sklearn missing confusion_matrix")

    y = np.asarray(y_true)
    p = np.asarray(y_pred)
    cm = confusion_matrix(y, p, labels=labels)

    if truthy_df_safe(normalize):
        with np.errstate(invalid="ignore", divide="ignore"):
            cmn = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1e-12, None)
        m = cmn
        fmt = ".2f"
    else:
        m = cm
        fmt = "d"

    fig = _new_fig((6.5, 5.5))
    ax = plt.gca()
    im = ax.imshow(m, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if labels is None and getattr(np, "unique", None) is not None:
        try:
            labels = np.unique(np.concatenate([y, p]))
        except Exception:
            labels = np.unique(y)
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, format(v, fmt), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_roc_pr_curves(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    title_prefix: str = "",
    proba_class: Optional[int] = None
) -> Dict[str, plt.Figure]:
    """
    ROC & PR dla binarnej lub wieloklasowej (wtedy binarka vs `proba_class`).
    """
    try:
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    except Exception:
        fig = _placeholder_fig(f"{title_prefix}ROC/PR", "sklearn metrics not available")
        return {"roc": fig, "pr": fig}

    p = _as_binary_proba(proba, proba_class)
    y = _binarize_targets(y_true, proba_class, proba.shape[1] if isinstance(proba, np.ndarray) and proba.ndim == 2 else None)
    y, p = _filter_finite_pair(y.astype(int), p)
    if np.unique(y).size < 2 or y.size == 0:
        fig = _placeholder_fig(f"{title_prefix}ROC/PR", "only one class present")
        return {"roc": fig, "pr": fig}

    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, p)
    ap = average_precision_score(y, p)

    fig1 = _new_fig((6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], ls="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"{title_prefix}ROC Curve".strip())
    plt.legend(loc="lower right")
    plt.tight_layout()

    fig2 = _new_fig((6, 5))
    plt.plot(rec, prec, lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix}Precision-Recall (AP={ap:.3f})".strip())
    plt.tight_layout()

    return {"roc": fig1, "pr": fig2}


def plot_calibration_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    *,
    proba_class: Optional[int] = None,
    strategy: str = "uniform"
) -> plt.Figure:
    try:
        from sklearn.calibration import calibration_curve
    except Exception:
        return _placeholder_fig(title, "sklearn missing calibration_curve")

    p = _as_binary_proba(proba, proba_class)
    y = _binarize_targets(y_true, proba_class, proba.shape[1] if isinstance(proba, np.ndarray) and proba.ndim == 2 else None)
    y, p = _filter_finite_pair(y.astype(int), p)
    if np.unique(y).size < 2 or y.size == 0:
        return _placeholder_fig(title, "only one class present")

    # preferuj 'quantile', a jak się wywali – 'uniform'
    try:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=int(n_bins), strategy=strategy)
    except Exception:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=int(n_bins), strategy="uniform")

    fig = _new_fig((6, 5))
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], ls="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_threshold_sweep(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    title: str = "Threshold sweep",
    proba_class: Optional[int] = None
) -> plt.Figure:
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score
    except Exception:
        return _placeholder_fig(title, "sklearn metrics not available")

    p = _as_binary_proba(proba, proba_class)
    y = _binarize_targets(y_true, proba_class, proba.shape[1] if isinstance(proba, np.ndarray) and proba.ndim == 2 else None)
    y, p = _filter_finite_pair(y.astype(int), p)
    if np.unique(y).size < 2 or y.size == 0:
        return _placeholder_fig(title, "only one class present")

    thrs = np.linspace(0.01, 0.99, 99)
    precs, recs, f1s = [], [], []
    for t in thrs:
        pred = (p >= t).astype(int)
        precs.append(float(precision_score(y, pred, zero_division=0)))
        recs.append(float(recall_score(y, pred, zero_division=0)))
        f1s.append(float(f1_score(y, pred, zero_division=0)))

    fig = _new_fig((7, 5))
    plt.plot(thrs, precs, label="Precision")
    plt.plot(thrs, recs, label="Recall")
    plt.plot(thrs, f1s, label="F1")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_score_hist_by_class(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    threshold: Optional[float] = None,
    title: str = "Score distribution by class",
    bins: int = 30,
    proba_class: Optional[int] = None
) -> plt.Figure:
    p = _as_binary_proba(proba, proba_class)
    y = _binarize_targets(y_true, proba_class, proba.shape[1] if isinstance(proba, np.ndarray) and proba.ndim == 2 else None)
    y, p = _filter_finite_pair(y.astype(int), p)
    if y.size == 0:
        return _placeholder_fig(title, "no finite data")

    fig = _new_fig((7, 5))
    plt.hist(p[y == 0], bins=bins, alpha=0.5, label="Class 0")
    plt.hist(p[y == 1], bins=bins, alpha=0.5, label="Class 1")
    if threshold is not None:
        plt.axvline(float(threshold), ls="--", label=f"thr={float(threshold):.2f}")
    plt.xlabel("Predicted probability")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    return fig


def plot_gain_lift_ks(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    title_prefix: str = "",
    proba_class: Optional[int] = None
) -> Dict[str, plt.Figure]:
    p = _as_binary_proba(proba, proba_class)
    y = _binarize_targets(y_true, proba_class, proba.shape[1] if isinstance(proba, np.ndarray) and proba.ndim == 2 else None)
    y, p = _filter_finite_pair(y.astype(int), p)
    if np.unique(y).size < 2 or y.size == 0:
        fig = _placeholder_fig(f"{title_prefix}Gain/Lift/KS", "only one class present")
        return {"gain": fig, "lift": fig, "ks": fig}

    order = np.argsort(-p)
    y_sorted = y[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = max(1, int(y.sum()))
    n = len(y)
    pct = np.arange(1, n + 1) / n

    gain = cum_pos / total_pos
    lift = gain / np.clip(pct, 1e-12, None)
    pos_cdf = cum_pos / total_pos
    neg_cdf = np.cumsum(1 - y_sorted) / max(1, (n - total_pos))
    ks = float(np.max(np.abs(pos_cdf - neg_cdf)))

    figs: Dict[str, plt.Figure] = {}

    # Gain
    fig_gain = _new_fig((6.5, 5))
    plt.plot(pct, gain, label="Cumulative gain")
    plt.plot([0, 1], [0, 1], ls="--", label="Baseline")
    plt.xlabel("Sample fraction")
    plt.ylabel("Gain")
    plt.title(f"{title_prefix}Cumulative Gain".strip())
    plt.legend()
    plt.tight_layout()
    figs["gain"] = fig_gain

    # Lift
    fig_lift = _new_fig((6.5, 5))
    plt.plot(pct, lift, label="Lift")
    plt.axhline(1.0, ls="--", label="Baseline")
    plt.xlabel("Sample fraction")
    plt.ylabel("Lift")
    plt.title(f"{title_prefix}Lift Curve".strip())
    plt.legend()
    plt.tight_layout()
    figs["lift"] = fig_lift

    # KS
    fig_ks = _new_fig((6.5, 5))
    plt.plot(pct, pos_cdf, label="Pos CDF")
    plt.plot(pct, neg_cdf, label="Neg CDF")
    plt.xlabel("Sample fraction")
    plt.ylabel("CDF")
    plt.title(f"{title_prefix}KS Curve (KS = {ks:.3f})".strip())
    plt.legend()
    plt.tight_layout()
    figs["ks"] = fig_ks

    return figs


__all__ = [
    "plot_importance_bar",
    "plot_true_vs_pred",
    "plot_regression_residuals",
    "plot_residuals_hist",
    "plot_residuals_qq",
    "plot_confusion_matrix",
    "plot_roc_pr_curves",
    "plot_calibration_curve",
    "plot_threshold_sweep",
    "plot_score_hist_by_class",
    "plot_gain_lift_ks",
]
