"""
ui_compare.py
Docstring (PL): Widok porównania metryk i wykres radarowy (Plotly) na podstawie wyników treningu.
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
from typing import Dict, Any, List
from backend.metrics_helper import normalize_metrics, radar_series

def metrics_table(results: Dict[str, Dict[str, Any]]):
    """
    Docstring (PL): Renderuje prostą tabelę metryk modelowych.
    """
    import pandas as pd
    rows = []
    for name, info in results.items():
        row = {"model": name, "status": info.get("status", "-")}
        for k, v in info.get("metrics", {}).items():
            row[k] = v
        rows.append(row)
    if rows:
        df = pd.DataFrame(rows).set_index("model")
        st.dataframe(df)
    else:
        st.info("Brak metryk do porównania.")

def radar_plot(results: Dict[str, Dict[str, Any]]):
    """
    Docstring (PL): Radar znormalizowanych metryk (0..1). Mniej=lepiej (RMSE/MAE) odwrócone.
    """
    if not results:
        st.info("Brak wyników do wizualizacji.")
        return
    norm = normalize_metrics(results)
    models = list(results.keys())
    data = radar_series(norm, models)
    metrics = data["metrics"]
    fig = go.Figure()
    for m in models:
        fig.add_trace(go.Scatterpolar(
            r=data["series"][m] + [data["series"][m][0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=m
        ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])), showlegend=True, title="Porównanie modeli — radar (znormalizowane)")
    st.plotly_chart(fig, use_container_width=True)