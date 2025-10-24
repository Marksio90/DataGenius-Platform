"""
plots_plus.py
Docstring (PL): Dodatkowe wykresy: Calibration (reliability), Brier score, QQ-plot.
"""
from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np
import plotly.graph_objects as go
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from scipy import stats

def calibration_fig(y_true, y_prob, n_bins: int = 10) -> go.Figure:
    prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob
    prob_pos = np.clip(prob_pos, 1e-9, 1-1e-9)
    frac_pos, mean_pred = calibration_curve(y_true, prob_pos, n_bins=n_bins, strategy="quantile")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name="Reliability"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Ideal", line=dict(dash="dash")))
    fig.update_layout(title="Calibration Curve", xaxis_title="Mean predicted prob", yaxis_title="Fraction of positives")
    return fig

def brier_score(y_true, y_prob) -> float:
    prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] > 1 else y_prob
    prob_pos = np.clip(prob_pos, 1e-9, 1-1e-9)
    return float(brier_score_loss(y_true, prob_pos))

def qq_plot_fig(y_true, y_pred) -> go.Figure:
    # QQ-plot residuals vs normal
    res = np.array(y_true) - np.array(y_pred)
    (osm, osr), (slope, intercept, r) = stats.probplot(res, dist="norm")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="Residuals QQ"))
    line = slope*np.array(osm) + intercept
    fig.add_trace(go.Scatter(x=osm, y=line, mode="lines", name="Reference", line=dict(dash="dash")))
    fig.update_layout(title="QQ-plot (residuals vs normal)", xaxis_title="Theoretical quantiles", yaxis_title="Ordered values")
    return fig