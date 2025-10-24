"""
plots.py
Docstring (PL): Wykresy Plotly dla klasyfikacji i regresji: ROC, PR, Confusion Matrix, Residuals, y_true vs y_pred.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

def roc_curve_fig(y_true, y_prob) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    return fig

def pr_curve_fig(y_true, y_prob) -> go.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="PR"))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
    return fig

def confusion_matrix_fig(y_true, y_pred) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=list(range(cm.shape[1])), y=list(range(cm.shape[0])), zauto=True))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
    return fig

def residuals_fig(y_true, y_pred) -> go.Figure:
    res = np.array(y_true) - np.array(y_pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=res, mode="markers", name="Residuals"))
    fig.update_layout(title="Residuals", xaxis_title="Index", yaxis_title="y_true - y_pred")
    return fig

def ytrue_ypred_fig(y_true, y_pred) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="y_pred vs y_true"))
    fig.add_trace(go.Scatter(x=y_true, y=y_true, mode="lines", name="Ideal", line=dict(dash="dash")))
    fig.update_layout(title="y_true vs y_pred", xaxis_title="y_true", yaxis_title="y_pred")
    return fig