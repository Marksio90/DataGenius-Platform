
# backend/plots.py
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Dict

def plot_importance_bar(imp: pd.DataFrame, top: int=20, title: str="Feature importance") -> plt.Figure:
    imp2 = imp.head(top)
    fig = plt.figure(figsize=(8, 5))
    plt.barh(imp2["feature"][::-1], imp2["importance_mean"][::-1])
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_regression_residuals(y_true: np.ndarray, y_pred: np.ndarray, title: str="Residuals") -> plt.Figure:
    res = y_true - y_pred
    fig = plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, res, alpha=0.5)
    plt.axhline(0.0)
    plt.xlabel("Prediction")
    plt.ylabel("Residual")
    plt.title(title)
    plt.tight_layout()
    return fig

def plot_roc_pr_curves(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, plt.Figure]:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)

    fig1 = plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.tight_layout()

    fig2 = plt.figure(figsize=(6, 5))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR (AP={ap:.3f})")
    plt.tight_layout()
    return {"roc": fig1, "pr": fig2}
