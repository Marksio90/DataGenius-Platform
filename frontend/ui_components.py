# frontend/ui_components.py
"""
Komponenty UI dla TMIV – Advanced ML Platform.

Zawiera:
- render_sidebar() – pełny sidebar z nawigacją i statusem.
- sidebar() – kompatybilny wrapper (z Twojego skeletonu).
- load_or_upload_df(data_svc) – wczytywanie/wybór przykładowego zbioru.

Uwaga: `app.py` korzysta z render_sidebar() i load_or_upload_df().
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
import streamlit as st


def render_sidebar() -> str:
    """Sidebar z nawigacją i statusem aplikacji. Zwraca nazwę wybranej strony."""
    with st.sidebar:
        st.title("TMIV v2.0 Pro")
        st.caption("Advanced ML Platform")

        page = st.radio(
            "Nawigacja",
            options=[
                "📊 Analiza Danych",
                "🤖 Trening Modelu",
                "📈 Wyniki i Wizualizacje",
                "💡 Rekomendacje",
                "📚 Dokumentacja",
            ],
            index=0,
        )

        st.divider()
        st.caption("Stan aplikacji")
        st.write(
            {
                "df_loaded": st.session_state.get("df") is not None,
                "trained": st.session_state.get("train_result") is not None,
                "dataset": st.session_state.get("dataset_name") or "-",
            }
        )

    return page


def sidebar() -> str:
    """Kompatybilny alias do `render_sidebar()` (z Twojego szkicu)."""
    return render_sidebar()


def load_or_upload_df(data_svc) -> Tuple[pd.DataFrame | None, str | None]:
    """
    Panel wczytywania danych (CSV/XLSX/Parquet/JSON) lub użycia przykładowego datasetu.

    Zwraca:
        (df, name) albo (None, None), gdy nic nie wczytano.
    """
    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    with col1:
        uploaded = st.file_uploader(
            "Wczytaj plik",
            type=["csv", "xlsx", "parquet", "json"],
            help="Obsługiwane formaty: CSV, XLSX, Parquet, JSON.",
        )
    with col2:
        use_demo = st.button("Użyj przykładowego", type="primary")

    # Przykładowy dataset
    if use_demo:
        try:
            df = data_svc.load_example()
            st.session_state.df = df
            st.session_state.dataset_name = "avocado.csv"
            st.success(f"Wczytano przykładowy zbiór: avocado.csv (kształt: {df.shape})")
            return df, "avocado.csv"
        except Exception as e:
            st.error(f"Nie udało się wczytać przykładowych danych: {e}")
            return None, None

    # Upload pliku
    if uploaded is not None:
        try:
            df, name = data_svc.load_any(uploaded)
            st.session_state.df = df
            st.session_state.dataset_name = name
            st.success(f"Wczytano: {name} (kształt: {df.shape})")
            return df, name
        except Exception as e:
            st.error(f"Nie udało się wczytać pliku: {e}")
            return None, None

    # Jeśli mamy już DF w stanie – zwracamy go
    if st.session_state.get("df") is not None:
        return st.session_state.df, st.session_state.get("dataset_name")

    return None, None
