"""
ui_panels_explain.py — Stage 8
"""
from __future__ import annotations
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any
from backend.plots import roc_curve_fig, pr_curve_fig, confusion_matrix_fig, residuals_fig, ytrue_ypred_fig
from backend.plots_plus import calibration_fig, brier_score, qq_plot_fig
from backend.explain_plus import pdp_ice
from backend.registry_utils import promote_pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from backend.async_ml_trainer import _split, _preprocessor

def page_results() -> None:
    st.header("🧠 Explainability++ — szczegółowe wykresy i rejestr")
    run = st.session_state.get("tmiv_train")
    if not run:
        st.info("Brak wyników. Uruchom trening w zakładce „Trening Modelu”."); return
    plan = run["plan"]; res = run["result"]
    best = res.get("best_model")
    if not best or res["results"].get(best, {}).get("status")!="ok":
        st.info("Brak najlepszego modelu z metrykami."); return

    info = res["results"][best]
    st.subheader(f"Najlepszy model: {best}")
    st.json(info["metrics"])

    # RZECZYWISTE y_true (przechowywane od Etapu 8)
    y_true = np.array(info.get("y_true")) if info.get("y_true") is not None else None
    y_pred = np.array(info.get("y_pred")) if info.get("y_pred") is not None else None
    y_prob = np.array(info.get("y_prob")) if info.get("y_prob") is not None else None

    if plan["problem_type"] == "classification" and y_true is not None and y_prob is not None:
        st.plotly_chart(roc_curve_fig(y_true, y_prob), use_container_width=True)
        st.plotly_chart(pr_curve_fig(y_true, y_prob), use_container_width=True)
        st.plotly_chart(confusion_matrix_fig(y_true, y_pred), use_container_width=True)
        st.plotly_chart(calibration_fig(y_true, y_prob), use_container_width=True)
        st.info(f"Brier score: {brier_score(y_true, y_prob):.4f}")

    if plan["problem_type"] == "regression" and y_true is not None and y_pred is not None:
        st.plotly_chart(residuals_fig(y_true, y_pred), use_container_width=True)
        st.plotly_chart(ytrue_ypred_fig(y_true, y_pred), use_container_width=True)
        st.plotly_chart(qq_plot_fig(y_true, y_pred), use_container_width=True)

    # PDP/ICE dla wybranej cechy (trenujemy szybki model pomocniczy na podzbiorze dla demonstracji)
    st.subheader("PDP / ICE (demo)")
    df = st.session_state.get("tmiv_df")
    if df is not None:
        feature = st.selectbox("Wybierz cechę", options=list(df.drop(columns=[plan["target"]]).columns))
        X_train, X_test, y_train, y_test = _split(df, plan["target"], plan["problem_type"])
        pre = _preprocessor(X_train)
        mdl = RandomForestClassifier(n_estimators=50) if plan["problem_type"]=="classification" else RandomForestRegressor(n_estimators=50)
        pipe = Pipeline([("prep", pre), ("model", mdl)])
        pipe.fit(X_train, y_train)
        pdp = pdp_ice(pipe, X_test, feature, kind="average")
        if pdp.get("status")=="OK":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pdp["grid"], y=pdp["values"], mode="lines", name="PDP"))
            fig.update_layout(title=f"PDP — {feature}", xaxis_title=feature, yaxis_title="Predykcja")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"PDP niedostępne: {pdp.get('message','-')}")

    # Promote to registry (joblib + opcjonalny ONNX)
    st.subheader("Model Registry — promote")
    name = st.text_input("Nazwa modelu", value=best)
    version = st.text_input("Wersja (semver)", value="2.0.0")
    exp_onnx = st.checkbox("Eksport ONNX", value=False)
    if st.button("📤 Promote to registry"):
        # trenowanie lekkiego pipeline'u od nowa do zapisu (w produkcji należałoby zapisać pipeline w run)
        X_train, X_test, y_train, y_test = _split(df, plan["target"], plan["problem_type"])
        pre = _preprocessor(X_train)
        mdl = RandomForestClassifier(n_estimators=50) if plan["problem_type"]=="classification" else RandomForestRegressor(n_estimators=50)
        pipe = Pipeline([("prep", pre), ("model", mdl)])
        pipe.fit(X_train, y_train)
        prom = promote_pipeline(name, version, pipe, X_train, info["metrics"], export_onnx=exp_onnx)
        st.success(f"Zarejestrowano: {prom['manifest']}")