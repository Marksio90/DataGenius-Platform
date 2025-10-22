# backend/plots.py
"""
Plotting utilities for TMIV – Advanced ML Platform.

Scope (matplotlib-only, no seaborn):
- Classification: ROC, PR, Confusion Matrix, Calibration (reliability)
- Regression: residual histogram, y_pred vs y_true (with y=x), residuals vs y_pred
- Feature importance: horizontal bar chart (top-k)
- Model comparison: radar chart from leaderboard (normalized metrics)

All plotters:
- return ABSOLUTE file paths to saved PNGs under cache/artifacts/plots
- are defensive (work with binary & multiclass; missing metrics gracefully ignored)

Typical usage
-------------
    from backend.plots import (
        plot_classification_curves, plot_confusion_matrix,
        plot_calibration_curve, plot_regression_diagnostics,
        plot_feature_importance, plot_radar_leaderboard
    )

    paths = plot_classification_curves(y_true, y_proba, model_name="lgbm")
    cm = plot_confusion_matrix(y_true, y_pred, class_names=["no","yes"], model_name="lgbm")
    reg = plot_regression_diagnostics(y_true, y_pred, model_name="ridge")

Notes
-----
- Multiclass ROC/PR: we draw micro- and macro-average curves.
- Calibration: for multiclass we plot reliability of the argmax "positive" events (approx).
- Radar: normalize metrics to [0,1]; lower-is-better metrics (rmse/mae/logloss) are inverted.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
)
from sklearn.calibration import CalibrationDisplay

from .cache_manager import cached_path


# =========================
# Helpers
# =========================

def _mpl_agg() -> None:
    """Ensure a non-interactive backend is set."""
    import matplotlib
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass


def _save_fig(fig: Figure, name: str, *, subdir: str = "plots", dpi: int = 150) -> str:
    path = cached_path(subdir, f"{name}.png")
    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _as_2d_proba(proba: np.ndarray) -> np.ndarray:
    """Return (n_samples, n_classes) probabilities; accept binary vector as prob of class 1."""
    P = np.asarray(proba)
    if P.ndim == 1:
        P = P.reshape(-1)
        P = np.vstack([1.0 - P, P]).T  # binary
    return P


def _infer_class_names(y_true: Sequence[Any], n_classes: int) -> list[str]:
    uniq = np.unique(y_true)
    if len(uniq) == n_classes:
        return [str(u) for u in uniq]
    return [str(i) for i in range(n_classes)]


def _is_binary(y: Sequence[Any]) -> bool:
    return len(np.unique(y)) <= 2


# =========================
# Classification plots
# =========================

def plot_classification_curves(
    y_true: Sequence[int],
    y_proba: np.ndarray,
    *,
    model_name: str = "model",
    labels: Sequence[Any] | None = None,
) -> Dict[str, str]:
    """
    ROC & PR curves. For multiclass: draw micro- and macro-average curves.
    Returns dict: {"roc": path, "pr": path}
    """
    _mpl_agg()
    y_true = np.asarray(y_true)
    P = _as_2d_proba(y_proba)
    n_classes = P.shape[1]
    class_names = list(labels) if labels is not None else _infer_class_names(y_true, n_classes)

    paths: Dict[str, str] = {}

    # --- ROC ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)

    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, P[:, 1])
        auc = roc_auc_score(y_true, P[:, 1])
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")
    else:
        # binarize per-class
        from sklearn.preprocessing import label_binarize
        Y = label_binarize(y_true, classes=np.arange(n_classes))
        # per-class
        tprs = []
        aucs = []
        for k in range(n_classes):
            fpr, tpr, _ = roc_curve(Y[:, k], P[:, k])
            ax.plot(fpr, tpr, alpha=0.4, label=f"class {class_names[k]}")
            tprs.append(np.interp(np.linspace(0, 1, 200), fpr, tpr))
            aucs.append(roc_auc_score(Y[:, k], P[:, k]))
        # macro avg
        mean_tpr = np.mean(np.vstack(tprs), axis=0)
        ax.plot(np.linspace(0, 1, 200), mean_tpr, linewidth=2.0, label=f"macro avg (AUC={np.mean(aucs):.3f})")

        # micro avg
        Y_flat = Y.ravel()
        P_flat = P.ravel()
        fpr, tpr, _ = roc_curve(Y_flat, P_flat)
        ax.plot(fpr, tpr, linewidth=2.0, label="micro avg")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right", fontsize=9)
    paths["roc"] = _save_fig(fig, f"roc_{model_name}")

    # --- PR ---
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    if n_classes == 2:
        PrecisionRecallDisplay.from_predictions(y_true, P[:, 1], ax=ax, name=model_name)
    else:
        from sklearn.preprocessing import label_binarize
        Y = label_binarize(y_true, classes=np.arange(n_classes))
        aps = []
        for k in range(n_classes):
            pr, rc, _ = precision_recall_curve(Y[:, k], P[:, k])
            ax.plot(rc, pr, alpha=0.4, label=f"class {class_names[k]}")
            # AP approx
            aps.append(np.trapz(pr, rc))
        # macro
        ax.plot([0, 1], [np.mean(aps)] * 2, "k--", label=f"macro AP≈{np.mean(aps):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="lower left", fontsize=9)
    paths["pr"] = _save_fig(fig, f"pr_{model_name}")

    return paths


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    class_names: Sequence[str] | None = None,
    normalize: str | None = "true",
    model_name: str = "model",
) -> str:
    """
    Confusion matrix (normalized by default).
    """
    _mpl_agg()
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cn = list(class_names) if class_names is not None else [str(l) for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)  # type: ignore[arg-type]
    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Confusion Matrix" + (f" ({normalize})" if normalize else ""))
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(cn, rotation=45, ha="right")
    ax.set_yticklabels(cn)

    # annotate
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = cm[i, j]
            ax.text(j, i, format(val, fmt), ha="center", va="center", color="white" if val > thresh else "black")

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    return _save_fig(fig, f"cm_{model_name}")


def plot_calibration_curve(
    y_true: Sequence[int],
    y_proba: np.ndarray,
    *,
    model_name: str = "model",
    n_bins: int = 10,
) -> str:
    """
    Reliability diagram (calibration curve).
    Multiclass: treats the positive event as argmax class (approximate).
    """
    _mpl_agg()
    P = _as_2d_proba(y_proba)
    y_true = np.asarray(y_true)

    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.gca()

    if _is_binary(y_true):
        pos = P[:, 1]
        CalibrationDisplay.from_predictions(y_true, pos, n_bins=n_bins, name=model_name, ax=ax)
    else:
        # Approximation: consider correctness probability of predicted class
        y_hat = np.argmax(P, axis=1)
        correct = (y_hat == y_true).astype(int)
        conf = P[np.arange(len(P)), y_hat]
        CalibrationDisplay.from_predictions(correct, conf, n_bins=n_bins, name="argmax confidence", ax=ax)

    ax.set_title("Calibration / Reliability")
    return _save_fig(fig, f"calibration_{model_name}")


# =========================
# Regression plots
# =========================

def plot_regression_diagnostics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    model_name: str = "model",
) -> Dict[str, str]:
    """
    Residual histogram, y_pred vs y_true (with identity line), residuals vs y_pred.
    Returns dict: {"residual_hist", "pred_vs_true", "residuals_vs_pred"} -> path
    """
    _mpl_agg()
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    resid = y_true - y_pred

    out: Dict[str, str] = {}

    # Residual histogram
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.gca()
    ax.hist(resid, bins=40, edgecolor="black", alpha=0.75)
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residuals Histogram")
    ax.set_xlabel("y_true - y_pred")
    ax.set_ylabel("Count")
    out["residual_hist"] = _save_fig(fig, f"residual_hist_{model_name}")

    # y_pred vs y_true
    fig = plt.figure(figsize=(6.4, 6.0))
    ax = fig.gca()
    ax.scatter(y_true, y_pred, s=14, alpha=0.7)
    mn, mx = np.nanmin([y_true.min(), y_pred.min()]), np.nanmax([y_true.max(), y_pred.max()])
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    ax.set_title("Predicted vs True")
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    out["pred_vs_true"] = _save_fig(fig, f"pred_vs_true_{model_name}")

    # Residuals vs predictions
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.gca()
    ax.scatter(y_pred, resid, s=14, alpha=0.7)
    ax.axhline(0.0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Residuals vs Predicted")
    ax.set_xlabel("y_pred")
    ax.set_ylabel("Residual (y_true - y_pred)")
    out["residuals_vs_pred"] = _save_fig(fig, f"residuals_vs_pred_{model_name}")

    return out


# =========================
# Feature importance
# =========================

def plot_feature_importance(
    fi: pd.DataFrame | Sequence[Mapping[str, Any]] | Sequence[tuple[str, float]],
    *,
    top_k: int = 20,
    title: str = "Feature Importance (Top)",
    name: str = "fi",
) -> str:
    """
    Horizontal bar chart. Accepts:
    - DataFrame with columns 'feature' and one of {'importance', 'importance_mean'}
    - Iterable of dicts/tuples (feature, importance)
    """
    _mpl_agg()

    # Normalize to DataFrame
    def _to_df(x) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            cols = [c.lower() for c in x.columns]
            if "feature" in cols and "importance" in cols:
                return x.rename(columns={x.columns[cols.index("feature")]: "feature",
                                         x.columns[cols.index("importance")]: "importance"})[["feature", "importance"]]
            if "feature" in cols and "importance_mean" in cols:
                return x.rename(columns={x.columns[cols.index("feature")]: "feature",
                                         x.columns[cols.index("importance_mean")]: "importance"})[["feature", "importance"]]
        rows = []
        try:
            for item in x:
                if isinstance(item, Mapping):
                    f = item.get("feature")
                    v = item.get("importance", item.get("importance_mean", None))
                    if f is not None and v is not None:
                        rows.append((str(f), float(v)))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    rows.append((str(item[0]), float(item[1])))
        except Exception:
            pass
        return pd.DataFrame(rows, columns=["feature", "importance"])

    df = _to_df(fi)
    if df.empty:
        # Create an empty plot with message
        fig = plt.figure(figsize=(6.4, 4.0))
        ax = fig.gca()
        ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center")
        ax.axis("off")
        return _save_fig(fig, f"{name}_empty")

    df = df.dropna().sort_values("importance", ascending=True).tail(top_k)
    fig = plt.figure(figsize=(8.0, max(4.0, 0.35 * len(df) + 1.2)))
    ax = fig.gca()
    ax.barh(df["feature"], df["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    return _save_fig(fig, f"{name}_top{top_k}")


# =========================
# Radar chart (leaderboard)
# =========================

_LOWER_IS_BETTER = {"rmse", "mae", "logloss"}

def _normalize_metrics(df: pd.DataFrame, metric_cols: Sequence[str]) -> pd.DataFrame:
    """
    Normalize each metric column to [0,1] across models.
    Lower-better metrics are inverted.
    """
    out = df.copy()
    for m in metric_cols:
        if m not in out.columns:
            continue
        col = out[m].astype(float)
        if m.lower() in _LOWER_IS_BETTER:
            col = -col  # invert direction
        mn, mx = col.min(), col.max()
        if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
            out[m] = (col - mn) / (mx - mn)
        else:
            out[m] = 0.5  # degenerate case
    return out


def plot_radar_leaderboard(
    leaderboard: pd.DataFrame,
    *,
    metrics: Sequence[str] | None = None,
    name: str = "radar",
    title: str = "Model Comparison (Radar)",
) -> str:
    """
    Build a radar chart from leaderboard DataFrame.
    If `metrics` is None, tries reasonable defaults based on available columns.
    """
    _mpl_agg()
    if leaderboard is None or leaderboard.empty:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.gca()
        ax.text(0.5, 0.5, "No leaderboard available", ha="center", va="center")
        ax.axis("off")
        return _save_fig(fig, f"{name}_empty")

    cols = set(map(str, leaderboard.columns))
    default_cls = ["roc_auc", "f1_weighted", "accuracy", "aps"]
    default_reg = ["r2", "rmse", "mae"]
    # pick a set that exists
    m = [c for c in default_cls if c in cols] or [c for c in default_reg if c in cols]
    if not m:
        # fallback to all numeric except primary/aux columns
        candidates = [
            c for c in leaderboard.columns
            if c not in {"model", "rank", "primary_metric"} and np.issubdtype(leaderboard[c].dtype, np.number)
        ]
        m = candidates[:5] if candidates else []
    if metrics:
        m = [mm for mm in metrics if mm in cols] or m
    if not m:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.gca()
        ax.text(0.5, 0.5, "No comparable metrics", ha="center", va="center")
        ax.axis("off")
        return _save_fig(fig, f"{name}_nometrics")

    data = leaderboard[["model"] + m].copy()
    data_norm = _normalize_metrics(data, m)

    # Radar geometry
    labels = m
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close

    fig = plt.figure(figsize=(7.5, 7.0))
    ax = plt.subplot(111, polar=True)
    ax.set_title(title, y=1.08)

    # grid & tick labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_rlabel_position(0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])

    # plot each model (up to 7 for readability)
    for _, row in data_norm.head(7).iterrows():
        values = [float(row[c]) for c in labels]
        values += values[:1]
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.12, label=str(row["model"]))
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10), fontsize=9)

    return _save_fig(fig, f"{name}_{'_'.join(labels)}")


__all__ = [
    "plot_classification_curves",
    "plot_confusion_matrix",
    "plot_calibration_curve",
    "plot_regression_diagnostics",
    "plot_feature_importance",
    "plot_radar_leaderboard",
]
