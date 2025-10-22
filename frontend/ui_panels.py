# frontend/ui_panels.py
"""
Panels – gotowe panele wyników dla TMIV – Advanced ML Platform.

Zawartość:
- render_kpi_cards(metrics: dict, *, title=None)                -> karty z metrykami
- render_leaderboard_panel(leaderboard: pd.DataFrame)           -> tabela + CSV download
- render_feature_importance_panel(fi: pd.DataFrame, explain_service=None) -> tabela + wykres PNG (ExplainService)
- render_classification_panels(y_true, y_pred, y_proba, *, class_names=None, model_name="model", explain_service=None)
- render_regression_panels(y_true, y_pred, *, model_name="model", explain_service=None)

Uwaga:
- Wykresy generujemy przez ExplainService (PNG na dysku). Jeżeli ExplainService nie jest
  podany lub zgłosi błąd – panel pokaże część tabelaryczną/informacyjną bez wykresów.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import streamlit as st


# =========================
# KPI Cards
# =========================

def render_kpi_cards(metrics: Mapping[str, float] | None, *, title: str | None = None, columns: int = 4) -> None:
    """Wyświetl metryki jako karty (st.metric)."""
    if not metrics:
        st.info("Brak metryk do wyświetlenia.")
        return
    if title:
        st.subheader(title)

    items = [(k, metrics[k]) for k in sorted(metrics.keys())]
    cols = st.columns(max(1, int(columns)))
    for i, (name, val) in enumerate(items):
        with cols[i % len(cols)]:
            try:
                st.metric(label=name, value=f"{float(val):.4f}")
            except Exception:
                st.metric(label=name, value=str(val))


# =========================
# Leaderboard
# =========================

def render_leaderboard_panel(leaderboard: pd.DataFrame) -> None:
    """Tabela leaderboardu + przycisk pobrania CSV."""
    st.subheader("🏆 Leaderboard")
    if leaderboard is None or leaderboard.empty:
        st.info("Brak danych leaderboardu.")
        return
    st.dataframe(leaderboard.reset_index(drop=True), use_container_width=True)
    _download_csv_button(leaderboard, label="Pobierz leaderboard CSV")


# =========================
# Feature Importance
# =========================

def render_feature_importance_panel(
    fi: pd.DataFrame | None,
    *,
    explain_service=None,
    default_top_k: int = 20,
    title: str = "Ważność cech",
) -> None:
    st.subheader(f"🔍 {title}")
    if fi is None or len(fi) == 0:
        st.info("Brak danych FI.")
        return

    max_k = int(min(len(fi), 50))
    top_k = int(st.slider("Pokaż top-K cech", min_value=5, max_value=max_k, value=min(default_top_k, max_k), step=1))
    fi_top = fi.head(top_k).copy()

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(fi_top, use_container_width=True)
    with c2:
        if explain_service is not None:
            try:
                path = explain_service.plot_feature_importance(fi, top_k=top_k, name="fi_top")
                st.image(path, caption=f"Feature importance (top {top_k})")
            except Exception as e:
                st.warning(f"Nie udało się wygenerować wykresu FI: {e}")
        else:
            st.caption("Brak ExplainService – pomijam wykres.")


# =========================
# Klasyfikacja – krzywe / CM / kalibracja
# =========================

def render_classification_panels(
    y_true: np.ndarray,
    y_pred: np.ndarray | None,
    y_proba: np.ndarray | None,
    *,
    class_names: Sequence[str] | None = None,
    model_name: str = "model",
    explain_service=None,
) -> None:
    st.subheader("📈 Klasyfikacja – krzywe i macierz pomyłek")

    # Krzywe ROC/PR
    if explain_service is not None and y_proba is not None:
        try:
            charts = explain_service.plot_classification_curves(y_true, y_proba, model_name=model_name)
            for label, p in charts.items():
                st.image(p, caption=label, use_container_width=True)
        except Exception as e:
            st.warning(f"Nie udało się narysować krzywych ROC/PR: {e}")
    else:
        st.caption("Brak predict_proba lub ExplainService – pomijam ROC/PR.")

    # Kalibracja
    if explain_service is not None and y_proba is not None:
        try:
            cal = explain_service.plot_calibration_curve(y_true, y_proba, model_name=model_name, n_bins=10)
            st.image(cal, caption="Calibration curve", use_container_width=True)
        except Exception:
            pass

    # Macierz pomyłek
    if explain_service is not None and y_pred is not None:
        try:
            cm = explain_service.plot_confusion_matrix(y_true, y_pred, class_names=list(class_names or []), model_name=model_name)
            st.image(cm, caption="Confusion matrix", use_container_width=True)
        except Exception as e:
            st.warning(f"Nie udało się narysować macierzy pomyłek: {e}")
    elif y_pred is None:
        st.caption("Brak predykcji klasy – pomijam macierz pomyłek.")


# =========================
# Regresja – diagnostyka
# =========================

def render_regression_panels(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_name: str = "model",
    explain_service=None,
) -> None:
    st.subheader("📉 Regresja – diagnostyka")

    if explain_service is None:
        st.caption("Brak ExplainService – pomijam wykresy diagnostyczne.")
        return

    try:
        paths = explain_service.plot_regression_diagnostics(y_true, y_pred, model_name=model_name)
        for label, p in paths.items():
            st.image(p, caption=label, use_container_width=True)
    except Exception as e:
        st.warning(f"Nie udało się wygenerować wykresów diagnostycznych: {e}")


# =========================
# Helpers
# =========================

def _download_csv_button(df: pd.DataFrame, *, label: str = "Pobierz CSV") -> None:
    try:
        import io

        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            label=label,
            data=buf.getvalue().encode("utf-8"),
            file_name="tmiv_table.csv",
            mime="text/csv",
            use_container_width=False,
        )
    except Exception:
        pass
