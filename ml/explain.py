
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict

def fig_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
    return fig

def fig_pr(y_true, y_prob):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR"))
    fig.update_layout(title="Precision-Recall", xaxis_title="Recall", yaxis_title="Precision")
    return fig

def fig_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"]))
    fig.update_layout(title="Confusion Matrix")
    return fig
