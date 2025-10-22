from __future__ import annotations

import streamlit as st
# ---- Streamlit page config (must be the first Streamlit command) ----
st.set_page_config(
    page_title="The Most Important Variables",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
# -*- coding: utf-8 -*-
"""
THE MOST IMPORTANT VARIABLES - Advanced ML Platform v2.0 Pro
Marksio AI Solutions
"""

from backend.safe_utils import truthy_df_safe

# === Standard libs
import os
import io
import time
import json
import asyncio
import shutil
import re
import warnings
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

warnings.filterwarnings("ignore")

# === Third-party
import numpy as np
import pandas as pd

# --- bootstrap session state (run first) ---
try:
    _ = st.session_state
    if "_once_flags" not in st.session_state or not isinstance(st.session_state.get("_once_flags"), (set, list, dict)):
        # używamy seta do sprawdzania członkostwa
        st.session_state["_once_flags"] = set()
    st.session_state.setdefault("show_health", False)
    st.session_state.setdefault("show_admin", False)
except Exception:
    # Streamlit założy session_state przy pierwszym dostępie – tu tylko zabezpieczenie
    pass

# guard: jednorazowy render narzędzi na rerun
if "_rendered_tools" not in st.session_state:
    st.session_state["_rendered_tools"] = set()
else:
    st.session_state["_rendered_tools"].clear()

from backend.ml_integration import compute_feature_importance
from backend.helpers.targeting import auto_select_target, stratified_sample_df
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# === Backend/Frontend modules (Twoja architektura)
from backend.utils import rate_limit, retry_on_failure, time_execution, memoize_in_session
from backend.error_handler import safe_execution
from config.settings import get_settings
from frontend.ui_components import UIComponents, render_advanced_overrides, render_ai_recommendations, render_training_controls
from backend.file_upload import FileUploadHandler
from backend.eda_integration import EDAAnalyzer, apply_ai_dataprep
from backend.ml_integration import MLModelTrainer, save_artifacts, fit_ensembles, evaluate_models_quick, train_multi_models
from backend.error_handler import health_monitor
from backend.monitoring import monitoring_dashboard, metrics_collector
from backend.reporting import generate_training_report
from backend import exporters

from backend.ai_integration import AIRecommender

# (opcjonalnie) kontener DI
try:
    from core.container import get_container
    USE_CONTAINER = True
except ImportError:
    USE_CONTAINER = False


# === Streamlit page config
# === CSS (estetyka wykresów Plotly w Streamlit)
st.markdown(
    """
<style>
.js-plotly-plot .plotly .svg-container {border: none !important;}
.stPlotlyChart > div {border: none !important; background: transparent !important;}
div[data-testid="stPlotlyChart"] {padding: 0 !important; border: none !important;}
.element-container div.stPlotlyChart > div {border: none !important; box-shadow: none !important;}
.plotly .gtitle {padding-bottom: 20px !important;}
.js-plotly-plot .plotly .gtitle {margin-bottom: 15px !important;}
</style>
""",
    unsafe_allow_html=True,
)


# === Helpers
def _fmt_float_safe(x, digits=2):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _ensure_session_defaults():
    ss = st.session_state
    ss.setdefault("training_in_progress", False)
    ss.setdefault("model_trained", False)
    ss.setdefault("training_started_at", None)
    ss.setdefault("training_finished_at", None)
    ss.setdefault("rate_limiter_cache", {})
    ss.setdefault("disable_rate_limit", True)
    ss.setdefault(
        "rate_limit_config",
        {
            "file_uploads": {"max": 1000, "window": 60},
            "training": {"max": 100, "window": 60},
        },
    )

def _pick_proba_like(results: Dict):
    """Zwraca tablicę prawdopodobieństw, jeśli jakikolwiek typowy klucz istnieje."""
    for k in ("probabilities", "y_proba", "proba", "pred_proba", "y_score"):
        v = results.get(k)
        if v is not None:
            return v
    return None

class MainApp:
    """Główna klasa aplikacji"""

    def __init__(self):
        self._init_container()
        self.settings = get_settings()
        self.ui = UIComponents()
        self.file_handler = FileUploadHandler()
        self.eda = EDAAnalyzer()
        self.ml_trainer = self._get_ml_trainer()
        self.health_monitor = health_monitor
        self.monitoring = monitoring_dashboard
        self._init_session_state()
        self.trainer = MLModelTrainer()

    # =========================================================================
    # MEMORY MANAGEMENT (dodane w v2.1)
    # =========================================================================

    MAX_SESSION_DF_SIZE_MB = 200  # Maksymalny rozmiar DataFrame w pamięci
    def _check_dataframe_memory(self, df: pd.DataFrame) -> Tuple[bool, float]:
        """Sprawdź czy DataFrame nie jest za duży"""
        size_mb = df.memory_usage(deep=True).sum() / (1024**2)
        return size_mb <= self.MAX_SESSION_DF_SIZE_MB, size_mb

    def _optimize_dataframe_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optymalizuj pamięć DataFrame"""
        df_optimized = df.copy()
        
        # Konwersja typów dla oszczędności pamięci
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        # Konwersja object -> category dla kolumn z małą liczbą unikalnych wartości
        for col in df_optimized.select_dtypes(include=['object']).columns:
            if df_optimized[col].nunique() / len(df_optimized) < 0.5:
                df_optimized[col] = df_optimized[col].astype('category')
        
        return df_optimized

    def _load_dataset_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bezpiecznie załaduj dataset z kontrolą pamięci"""
        is_ok, size_mb = self._check_dataframe_memory(df)
        
        if not truthy_df_safe(is_ok):
            st.warning(f"⚠️ Dataset duży ({size_mb:.1f}MB). Optymalizuję pamięć...")
            
            # Optymalizuj typy
            df = self._optimize_dataframe_memory(df)
            is_ok, size_mb = self._check_dataframe_memory(df)
            
            if not truthy_df_safe(is_ok):
                # Ostateczność - sampling
                sample_size = int(len(df) * (self.MAX_SESSION_DF_SIZE_MB / size_mb))
                st.warning(f"📉 Używam sampling do {sample_size:,} wierszy...")
                df = df.sample(n=sample_size, random_state=42)
                st.info(f"✓ Zmniejszono do {len(df):,} wierszy")
        
        return df

    # --- POMOCNIKI (w klasie MainApp) ---
    def _df_signature(self, df: pd.DataFrame):
        try:
            return int(pd.util.hash_pandas_object(df, index=True).sum())
        except Exception:
            return (df.shape, tuple(df.columns))

    def _basic_auto_clean(self, df: pd.DataFrame):
        """
        Delikatne auto-czyszczenie po wejściu na stronę Trening:
        - usuwa duplikaty,
        - próbuje skonwertować kolumny z datami,
        - BEZ docelowego kodowania kategorii (to zrobi pipeline).
        """
        steps = []
        dfc = df.copy()

        # 1) Duplikaty
        dups = int(dfc.duplicated().sum())
        if dups > 0:
            dfc = dfc.drop_duplicates()
            steps.append(f"🧹 Usunięto duplikaty: {dups}")

        # 2) Daty – heurystyka
        for c in dfc.columns:
            try:
                looks_like_date = ("date" in c.lower()) or pd.api.types.is_datetime64_any_dtype(dfc[c])
            except Exception:
                looks_like_date = False
            if looks_like_date or dfc[c].dtype == object:
                try:
                    parsed = pd.to_datetime(dfc[c], errors="ignore", utc=False)
                    if pd.api.types.is_datetime64_any_dtype(parsed):
                        dfc[c] = parsed
                        steps.append(f"📅 Skonwertowano '{c}' na datetime")
                except Exception:
                    pass

        steps.append(f"📊 Kształt po czyszczeniu: {df.shape} → {dfc.shape}")
        return dfc, "\n".join(steps)

    def _sanitize_dtypes_for_app(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizuje problematyczne typy:
        - kolumny category -> string (do UI/wykresów/EDA),
        - daty zostawia jako datetime64,
        """
        dfc = df.copy()
        # category -> string
        cat_cols = dfc.select_dtypes(include=["category"]).columns
        for c in cat_cols:
            dfc[c] = dfc[c].astype("string")
        # nullable string -> string
        str_cols = dfc.select_dtypes(include=["string"]).columns
        for c in str_cols:
            dfc[c] = dfc[c].astype("string")
        # krawędziowy przypadek CategoricalDtype
        from pandas.api.types import CategoricalDtype

        for c in dfc.columns:
            if isinstance(dfc[c].dtype, CategoricalDtype):
                dfc[c] = dfc[c].astype("string")
        return dfc

    def _init_container(self):
        if truthy_df_safe(USE_CONTAINER):
            try:
                self.container = get_container()
                self.platform_service = getattr(self.container, "platform_service", None)
            except Exception:
                self.container = None
                self.platform_service = None
        else:
            self.container = None
            self.platform_service = None

    def _get_ml_trainer(self):
        if getattr(self, "container", None) and hasattr(self.container, "ml_trainer"):
            return self.container.ml_trainer
        return MLModelTrainer()

    def _init_session_state(self):
        defaults = {
            "data_loaded": False,
            "model_trained": False,
            "analysis_complete": False,
            "current_dataset_hash": None,
            "df": None,
            "app_start_time": datetime.now(),
            "show_ai_recommendations": False,
            "ai_config_applied": False,
            # ▼▼ stan sidebaru
            "show_health": False,
            "show_admin": False,
        }
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    # === RUN / ROUTER / ERROR ===
    def run(self):
        try:
            self.ui.render_header()
            page = self._render_sidebar_with_monitoring()
            self._route_to_page(page)
            self._render_monitoring_if_enabled()
        except Exception as e:
            self._handle_critical_error(e)

    def _render_sidebar_with_monitoring(self) -> str:
        """
        Sidebar + narzędzia (bez infinite loop, bez zagnieżdżonych expanderów)
        """
        with st.sidebar:
            selected_page = self.ui.render_sidebar()

            st.markdown("---")
            st.caption("🛠️ Narzędzia")

            # Bezpieczne domyślne wartości
            st.session_state.setdefault('show_health', False)
            st.session_state.setdefault('show_admin', False)

            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 Health", key="btn_health_toggle", use_container_width=True):
                    st.session_state.show_health = not st.session_state.get('show_health', False)
            with col2:
                if st.button("🛠️ Admin", key="btn_admin_toggle", use_container_width=True):
                    st.session_state.show_admin = not st.session_state.get('show_admin', False)

            # HEALTH (expander tylko tutaj)
            if st.session_state.get('show_health', False):
                try:
                    with st.expander("📊 Health Dashboard", expanded=True):
                        self.health_monitor.render_health_dashboard()
                except Exception as e:
                    st.error(f"Health dashboard error: {e}")
                    st.info("Health dashboard chwilowo niedostępny.")

            # ADMIN (anty-nesting: preferujemy wrapper expander, inaczej fallback)
            if st.session_state.get('show_admin', False):
                try:
                    if hasattr(self.monitoring, "render_admin_expander"):
                        # Nowa wersja klasy ma wrappera – użyj go
                        self.monitoring.render_admin_expander(parent=st, expanded=True, title="🛠️ Admin Panel")
                    else:
                        # Starsza wersja: próbujemy z zewnętrznym expanderem…
                        try:
                            with st.expander("🛠️ Admin Panel", expanded=True):
                                # …i renderujemy TYLKO treść (bez własnych expanderów w środku)
                                self.monitoring.render_admin_panel(parent=st)
                        except Exception as e2:
                            # Jeśli wewnątrz i tak tworzony jest expander → renderujemy bez zewnętrznego
                            if "Expanders may not be nested" in str(e2):
                                self.monitoring.render_admin_panel(parent=st)
                            else:
                                raise
                except Exception as e:
                    st.error(f"Admin panel error: {e}")
                    st.info("Admin panel chwilowo niedostępny.")

        return selected_page


    def _route_to_page(self, page: str):
        page_handlers = {
            "📊 Analiza Danych": self.render_data_analysis_page,
            "🤖 Trening Modelu": self.render_model_training_page,
            "📈 Wyniki i Wizualizacje": self.render_results_page,
            "💡 Rekomendacje": self.render_recommendations_page,
            "📚 Dokumentacja": self.render_documentation_page,
        }
        handler = page_handlers.get(page)
        if truthy_df_safe(handler):
            handler()
        else:
            st.error(f"❌ Nieznana strona: {page}")

    def _render_monitoring_if_enabled(self):
        return

    def _handle_critical_error(self, error: Exception):
        st.error(f"❌ Krytyczny błąd: {str(error)}")
        st.info("🔄 Odśwież stronę")
        with st.expander("🐛 Szczegóły"):
            st.code(str(error))

    # ==================== ANALIZA DANYCH ====================
    def render_data_analysis_page(self):
        st.header("📊 Analiza i Wczytywanie Danych")
        self._render_data_upload_section()

        if st.session_state.data_loaded and st.session_state.df is not None:
            st.divider()
            self._render_eda_section()

    def _render_eda_section(self):
        df = st.session_state.df if hasattr(st.session_state, 'df') else None
        if df is None or len(df) == 0:
            st.info("Brak danych do analizy.")
            return

        tabs = st.tabs(["📊 Statystyki", "📈 Rozkłady", "🧩 Korelacje", "🏷️ Opisy kolumn (AI)"])

        # 📊 Statystyki
        with tabs[0]:
            try:
                self._render_statistics_tab(df)
            except Exception as e:
                st.warning(f"Statystyki niedostępne: {e}")

        # 📈 Rozkłady
        with tabs[1]:
            try:
                self._render_distributions_tab(df)
            except Exception as e:
                st.warning(f"Rozkłady niedostępne: {e}")

        # 🧩 Korelacje
        with tabs[2]:
            try:
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.shape[1] >= 2:
                    st.subheader("🔗 Korelacje")
                    corr = numeric_df.corr(numeric_only=True)
                    st.dataframe(corr.style.background_gradient(vmin=-1, vmax=1))
                else:
                    st.info("Za mało kolumn numerycznych do korelacji.")
            except Exception as e:
                st.warning(f"Korelacje niedostępne: {e}")

        # 🏷️ Opisy kolumn (AI)
        with tabs[3]:
            import json, inspect

            # ====== Pasek akcji ======
            a1, a2, a3, a4 = st.columns([1, 1, 1, 3])
            with a1:
                expand_all = st.checkbox(
                    "Rozwiń wszystko",
                    value=st.session_state.get("ai_desc_expand_all", False),
                    key="ai_desc_expand_all"
                )
            with a2:
                only_missing = st.checkbox(
                    "Tylko bez AI",
                    value=st.session_state.get("ai_desc_only_missing", False),
                    key="ai_desc_only_missing"
                )
            with a3:
                refresh_clicked = st.button("🔄 Odśwież opisy", key="ai_desc_force")
            with a4:
                filter_q = st.text_input(
                    "Filtruj kolumny",
                    value=st.session_state.get("ai_desc_filter", ""),
                    key="ai_desc_filter",
                    placeholder="np. price, date, id..."
                )

            # ====== Buduj / wczytaj opisy (z cache i/lub AI) ======
            df_ai = df.copy()
            df_ai.columns = df_ai.columns.map(str)

            if truthy_df_safe(refresh_clicked):
                # wyczyść cache sesji; jeżeli backend wspiera 'force', ominie też cache dyskowy
                st.session_state.pop("ai_column_descriptions", None)

            desc = st.session_state.get("ai_column_descriptions")

            if not isinstance(desc, dict) or refresh_clicked:
                try:
                    gen_fn = getattr(self.eda, "generate_column_descriptions", None)
                    if gen_fn is None:
                        raise RuntimeError("Brak metody generate_column_descriptions w self.eda")

                    # Spróbuj z 'force' (jeśli wspierane), inaczej bez
                    try:
                        # najpierw sprawdzamy sygnaturę
                        sig = None
                        try:
                            sig = inspect.signature(gen_fn)
                        except Exception:
                            pass

                        if truthy_df_safe(sig) and ("force" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())):
                            desc = gen_fn(df_ai, force=bool(refresh_clicked))
                        else:
                            # metoda nie wspiera 'force' -> fallback
                            desc = gen_fn(df_ai)
                    except TypeError:
                        # gdyby mimo wszystko poleciał TypeError, wołamy bez 'force'
                        desc = gen_fn(df_ai)

                    if not isinstance(desc, dict):
                        desc = {}
                    # Spłaszcz ewentualny blok {"descriptions": {...}}
                    if "descriptions" in desc and isinstance(desc["descriptions"], dict):
                        desc = desc["descriptions"]

                    st.session_state["ai_column_descriptions"] = desc
                except Exception as e:
                    st.warning(f"Nie udało się uzyskać opisów AI: {e}")
                    desc = {}

            # Jednorazowy fallback (symulacja) do rozpoznania, gdzie AI faktycznie zadziałało
            try:
                sim_desc = self.eda._generate_with_simulation(df_ai)
            except Exception:
                sim_desc = {}

            # ====== Filtrowanie kolumn ======
            filtered_cols = [c for c in df.columns if (not filter_q) or (str(filter_q).lower() in str(c).lower())]

            # ====== Render ======
            matched_ai = 0
            for col in filtered_cols:
                col_s = str(col)
                ai_text = desc.get(col_s)
                sim_text = sim_desc.get(col_s)

                # finalny opis (AI jeśli jest, inaczej symulacja)
                final_text = ai_text or sim_text or "Opis niedostępny."
                # heurystyka źródła: jeśli AI istnieje i różni się od symulacji → AI
                is_ai = (ai_text is not None) and (sim_text is None or str(ai_text).strip() != str(sim_text).strip())

                if truthy_df_safe(only_missing) and is_ai:
                    # w trybie "Tylko bez AI" pomijamy kolumny, które mają AI
                    continue

                if truthy_df_safe(is_ai):
                    matched_ai += 1

                badge = "🤖" if is_ai else "🧮"
                with st.expander(f"📊 {col}  {badge}", expanded=bool(st.session_state.get("ai_desc_expand_all", False))):
                    tag = "🤖 AI" if is_ai else "🧮 Fallback"
                    st.markdown(f"*Źródło:* **{tag}**")
                    st.write(f"**Opis:** {final_text}")

                    # Typ + szybkie staty
                    st.write(f"**Typ:** {df[col].dtype}")
                    if pd.api.types.is_numeric_dtype(df[col]):
                        col_num = pd.to_numeric(df[col], errors='coerce')
                        st.write(
                            f"**Min:** {_fmt_float_safe(col_num.min(), 3)}, "
                            f"**Max:** {_fmt_float_safe(col_num.max(), 3)}, "
                            f"**Średnia:** {_fmt_float_safe(col_num.mean(), 3)}"
                        )
                    else:
                        st.write(f"**Unikalne:** {df[col].nunique()}")

            # Podsumowanie
            st.caption(
                f"🔎 AI opisy dopasowane: **{matched_ai}/{len(filtered_cols)}** "
                f"(na {len(df.columns)} kolumn łącznie)"
            )

            # ====== Eksport ======
            e1, e2 = st.columns(2)
            with e1:
                st.download_button(
                    "⬇️ Eksport JSON",
                    data=json.dumps(
                        {str(c): (desc.get(str(c)) or sim_desc.get(str(c)) or "") for c in df.columns},
                        ensure_ascii=False, indent=2
                    ).encode("utf-8"),
                    file_name="column_descriptions.json",
                    mime="application/json",
                    key="ai_desc_export_json",
                    use_container_width=True
                )
            with e2:
                export_df = pd.DataFrame({
                    "column": [str(c) for c in df.columns],
                    "description": [desc.get(str(c)) or sim_desc.get(str(c)) or "" for c in df.columns],
                    "source": [
                        "AI" if (desc.get(str(c)) and (
                            (sim_desc.get(str(c)) is None) or
                            (str(desc.get(str(c))).strip() != str(sim_desc.get(str(c))).strip())
                        )) else "fallback"
                        for c in df.columns
                    ],
                })
                st.download_button(
                    "⬇️ Eksport CSV",
                    data=export_df.to_csv(index=False).encode("utf-8"),
                    file_name="column_descriptions.csv",
                    mime="text/csv",
                    key="ai_desc_export_csv",
                    use_container_width=True
                )

    def _render_data_upload_section(self):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🔄 Wczytaj swoje dane")
            uploaded_file = st.file_uploader("Wybierz plik CSV lub JSON", type=["csv", "json"])
            if truthy_df_safe(uploaded_file):
                self._handle_file_upload(uploaded_file)
        with col2:
            st.subheader("🎲 Przykładowe dane")
            sample_datasets = {
                "🥑 Avocado Prices": "avocado",
                "🌸 Iris Classification": "iris",
                "🍷 Wine Quality": "wine_quality",
                "💎 Diamonds": "diamonds",
            }
            selected_sample = st.selectbox("Wybierz zbiór", list(sample_datasets.keys()))
            if st.button("📥 Załaduj przykładowe dane", type="primary"):
                self._handle_sample_dataset(sample_datasets[selected_sample])

    def _handle_file_upload(self, uploaded_file):
        try:
            df = self.file_handler.load_file(uploaded_file)
            self._load_dataset(df)
            st.success(f"✅ Wczytano dane! Kształt: {df.shape}")
        except Exception as e:
            st.error(f"❌ Błąd: {str(e)}")

    def _handle_sample_dataset(self, dataset_name: str):
        df = self.file_handler.load_sample_dataset(dataset_name)
        self._load_dataset(df)
        st.success(f"✅ Załadowano! Kształt: {df.shape}")

    def _reset_model_state(self):
        keys_to_reset = [
            "model_trained",
            "analysis_complete",
            "model_results",
            "target_column",
            "problem_type",
            "df_clean",
        ]
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]

    def _load_dataset(self, df: pd.DataFrame):
        import hashlib

        new_hash = hashlib.md5(str(df.values).encode()).hexdigest()
        if st.session_state.current_dataset_hash != new_hash:
            self._reset_model_state()
            st.session_state.current_dataset_hash = new_hash

        df = self._sanitize_dtypes_for_app(df)
        df = self._load_dataset_safe(df) 
        st.session_state.df = df
        st.session_state.data_loaded = True
    def _execute_auto_clean(self):
        df = st.session_state.df
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)

        def update_progress(step: float):
            progress_bar.progress(step)

        def update_status(msg: str):
            status_placeholder.info(msg)

        try:
            df_clean = self.eda.auto_clean(df, progress_cb=update_progress, log_cb=update_status)
            st.session_state.df_clean = df_clean
            progress_bar.progress(1.0)
            status_placeholder.success("✅ Czyszczenie zakończone!")
            time.sleep(1)
            progress_placeholder.empty()
            status_placeholder.empty()  # rerun removed
        except Exception as e:
            st.error(f"❌ Błąd: {e}")
            progress_placeholder.empty()
            status_placeholder.empty()
    def _render_statistics_tab(self, df: pd.DataFrame):
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.subheader("📊 Statystyki")
            extended_stats = numeric_df.describe()
            # uwaga: .skew()/.kurtosis() mogą być deprecated; fallback:
            try:
                extended_stats.loc["skewness"] = numeric_df.skew(numeric_only=True)
            except Exception:
                pass
            try:
                extended_stats.loc["kurtosis"] = numeric_df.kurtosis(numeric_only=True)
            except Exception:
                pass
            st.dataframe(extended_stats.round(4), use_container_width=True)

            st.subheader("⚠️ Outliers (IQR)")
            outliers_info = self._analyze_outliers(numeric_df)
            st.dataframe(outliers_info, use_container_width=True)
        else:
            st.info("Brak kolumn numerycznych")

    def _analyze_outliers(self, numeric_df: pd.DataFrame) -> pd.DataFrame:
        outliers_info = []
        for col in numeric_df.columns:
            Q1, Q3 = numeric_df[col].quantile(0.25), numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)]
            outliers_pct = (len(outliers) / len(numeric_df)) * 100 if len(numeric_df) else 0.0
            outliers_info.append(
                {
                    "Kolumna": col,
                    "Liczba": len(outliers),
                    "Procent": f"{_fmt_float_safe(outliers_pct, 2)}%",
                    "Zakres": f"{_fmt_float_safe(lower, 3)} - {_fmt_float_safe(upper, 3)}",
                }
            )
        return pd.DataFrame(outliers_info)

    def _render_heatmap_tab(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.subheader("🔥 Macierz korelacji")
            corr_method = st.selectbox("Metoda:", ["pearson", "spearman", "kendall"])
            corr_matrix = df[numeric_cols].corr(method=corr_method)
            fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                aspect="auto",
                title=f"Korelacja ({corr_method})",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
            fig.update_layout(height=600, margin=dict(t=80))
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🎯 Najsilniejsze korelacje")
            strong = self._find_strong_correlations(corr_matrix)
            if not strong.empty:
                st.dataframe(strong, use_container_width=True)
        else:
            st.info("Potrzeba min. 2 kolumn numerycznych")

    def _find_strong_correlations(self, corr_matrix: pd.DataFrame) -> pd.DataFrame:
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if not pd.isna(val):
                    corr_pairs.append(
                        {
                            "Zmienna 1": corr_matrix.columns[i],
                            "Zmienna 2": corr_matrix.columns[j],
                            "Korelacja": val,
                            "Siła": abs(val),
                        }
                    )
        if truthy_df_safe(corr_pairs):
            df_corr = pd.DataFrame(corr_pairs).sort_values("Siła", ascending=False)
            return df_corr.head(10)[["Zmienna 1", "Zmienna 2", "Korelacja"]]
        return pd.DataFrame()

    def _render_distributions_tab(self, df: pd.DataFrame):
        st.subheader("📈 Rozkłady")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected = st.selectbox("Kolumna:", numeric_cols)
            col_data = df[selected].dropna()

            fig = make_subplots(rows=2, cols=2, subplot_titles=["Histogram", "Box", "Violin", "Q-Q"])
            fig.add_trace(go.Histogram(x=col_data, nbinsx=30, showlegend=False), row=1, col=1)
            fig.add_trace(go.Box(y=col_data, showlegend=False), row=1, col=2)
            fig.add_trace(go.Violin(y=col_data, showlegend=False), row=2, col=1)

            theoretical_q = stats.norm.ppf(np.linspace(0.01, 0.99, len(col_data)))
            sample_q = np.sort(col_data)
            fig.add_trace(
                go.Scatter(x=theoretical_q, y=sample_q, mode="markers", showlegend=False), row=2, col=2
            )
            fig.update_layout(title=f"Analiza: {selected}", height=800, margin=dict(t=100))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 🧪 Test normalności")
            shapiro_stat, shapiro_p = stats.shapiro(col_data.sample(min(5000, len(col_data))))
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Statystyka", f"{_fmt_float_safe(shapiro_stat, 6)}")
            with c2:
                st.metric("p-value", f"{_fmt_float_safe(shapiro_p, 6)}")
            if shapiro_p > 0.05:
                st.success("✅ Rozkład prawdopodobnie normalny")
            else:
                st.warning("⚠️ Rozkład prawdopodobnie nie-normalny")

    def _render_ai_insights_tab(self, df: pd.DataFrame):
        st.subheader("🎯 AI Insights")
        with st.spinner("🤖 AI analizuje..."):
            eda_results = {}
            try:
                eda_results = self.eda.generate_comprehensive_eda_report(df)
            except Exception as e:
                st.info(f"Moduł AI Insights niedostępny: {e}")
                eda_results = {}

        if eda_results.get("data_quality_issues"):
            st.markdown("#### ⚠️ Problemy z jakością")
            for issue in eda_results["data_quality_issues"]:
                severity_map = {"high": "error", "medium": "warning", "low": "info"}
                severity = issue.get("severity", "low")
                getattr(st, severity_map.get(severity, "info"))(
                    f"**{issue.get('type','Issue')}**: {issue.get('description','')}\n\n💡 {issue.get('recommendation','')}"
                )
        else:
            st.success("✅ Brak problemów z jakością!")

        if eda_results.get("recommendations"):
            st.markdown("#### 💡 Rekomendacje")
            for i, rec in enumerate(eda_results["recommendations"], 1):
                st.success(f"**{i}.** {rec}")

    # === Ultra Add-ons (przeniesione z luzem pisanych bloków do metod klasy)
    def _render_ultra_targeting_profit(self):
        st.markdown("### 🎯 Targeting & Profit (Ultra Add-ons)")
        try:
            from pathlib import Path as _Path
            import json as _json

            out_dir = _Path("artifacts") / "ultra_export"

            meta_ultra = {}
            if "bundle" in globals():
                try:
                    meta_ultra = (globals().get("bundle") or {}).get("ultra", {}) or {}
                except Exception:
                    meta_ultra = {}
            if not truthy_df_safe(meta_ultra):
                mp = out_dir / "meta.json"
                if mp.exists():
                    meta_ultra = _json.loads(mp.read_text(encoding="utf-8"))

            c1, c2, c3 = st.columns(3)
            with c1:
                value_tp = st.number_input(
                    "Value per TP (revenue)",
                    min_value=0.0,
                    max_value=1e9,
                    value=float(meta_ultra.get("value_tp", 10.0)),
                    step=1.0,
                    key="ultra_value_tp_t",
                )
            with c2:
                cost_fp = st.number_input(
                    "Cost FP",
                    min_value=0.0,
                    max_value=1e9,
                    value=float(meta_ultra.get("cost_fp", 1.0)),
                    step=1.0,
                    key="ultra_cost_fp_t",
                )
            with c3:
                cost_fn = st.number_input(
                    "Cost FN",
                    min_value=0.0,
                    max_value=1e9,
                    value=float(meta_ultra.get("cost_fn", 5.0)),
                    step=1.0,
                    key="ultra_cost_fn_t",
                )

            evalp = meta_ultra.get("eval") or {}
            y_true = evalp.get("y_true")
            proba = evalp.get("proba")
            if y_true is not None and proba is not None:
                import numpy as _np
                import pandas as _pd

                y = _np.asarray(y_true).astype(int)
                p = _np.asarray(proba).astype(float)
                k_pct = st.slider("Top-K % (targeting)", 1, 50, 10, 1, key="ultra_topk_t")
                topn = max(1, int(len(p) * k_pct / 100))
                order = _np.argsort(-p)
                idx = order[:topn]
                tp = int((y[idx] == 1).sum())
                fp = int((y[idx] == 0).sum())
                fn = int((y[order[topn:]] == 1).sum())
                profit = tp * float(value_tp) - fp * float(cost_fp) - fn * float(cost_fn)
                st.metric("Top-K Profit", f"{profit:,.0f}")

                st.download_button(
                    "⬇️ Pobierz topK indeksy",
                    data=_pd.DataFrame({"row_index": idx, "score": p[idx]}).to_csv(index=False).encode("utf-8"),
                    file_name=f"top_{k_pct}pct_indices.csv",
                    mime="text/csv",
                )
            else:
                st.caption("Brak proba/y_true — uruchom trening klasyfikacji, aby zobaczyć targeting.")
        except Exception as e:
            st.warning(f"Targeting niedostępny: {e}")

    def _render_ultra_pdp_ice(self):
        st.markdown("### 🔍 PDP / ICE Explorer (Ultra Add-ons)")
        try:
            import numpy as _np
            import pandas as _pd
            import json as _json
            from pathlib import Path as _Path

            out_dir = _Path("artifacts") / "ultra_export"

            meta_ultra = {}
            if "bundle" in globals():
                try:
                    meta_ultra = (globals().get("bundle") or {}).get("ultra", {}) or {}
                except Exception:
                    meta_ultra = {}
            if not meta_ultra and (out_dir / "meta.json").exists():
                meta_ultra = _json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

            Xte_json = meta_ultra.get("Xte_sample_json")
            if not truthy_df_safe(Xte_json):
                st.caption("Brak Xte_sample_json w meta — najpierw uruchom pipeline Ultra.")
                return

            Xte_s = _pd.read_json(Xte_json, orient="split")
            num_cols = Xte_s.select_dtypes(include=[_np.number]).columns.tolist()
            if not truthy_df_safe(num_cols):
                st.info("Brak cech numerycznych do PDP/ICE.")
                return

            col = st.selectbox("Cechy numeryczne", options=num_cols, key="ultra_pdp_col_t")
            n_grid = st.slider("Liczba punktów siatki", 10, 80, 20, 5, key="ultra_pdp_grid_t")
            if st.button("Rysuj PDP/ICE", key="ultra_pdp_btn_t"):
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import joblib

                model_path = out_dir / "model.joblib"
                model = joblib.load(model_path)
                base = Xte_s.copy()
                qs = base[col].quantile([0.05, 0.95]).values
                grid = _np.linspace(qs[0], qs[1], n_grid)
                problem = meta_ultra.get("problem")

                # PDP
                pdp_vals = []
                for v in grid:
                    Xg = base.copy()
                    Xg[col] = v
                    if problem == "classification":
                        yp = model.predict_proba(Xg)[:, 1]
                    else:
                        yp = model.predict(Xg)
                    pdp_vals.append(float(_np.mean(yp)))
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(grid, pdp_vals)
                ax.set_xlabel(col)
                ax.set_ylabel("Mean prediction")
                ax.set_title(f"PDP: {col}")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                # ICE
                fig = plt.figure()
                ax = fig.add_subplot(111)
                samp = base.sample(min(30, len(base)), random_state=42)
                for _, row in samp.iterrows():
                    vals = []
                    for v in grid:
                        r = row.copy()
                        r[col] = v
                        if problem == "classification":
                            val = float(model.predict_proba(_pd.DataFrame([r]))[:, 1][0])
                        else:
                            val = float(model.predict(_pd.DataFrame([r]))[0])
                        vals.append(val)
                    ax.plot(grid, vals, alpha=0.3)
                ax.set_xlabel(col)
                ax.set_ylabel("Prediction")
                ax.set_title(f"ICE: {col}")
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        except Exception as e:
            st.warning(f"PDP/ICE niedostępne: {e}")

    def _render_ultra_report(self):
        st.markdown("### 📄 Raport z wyników (Ultra Add-ons)")
        try:
            from backend.export_everything import build_report
            from pathlib import Path as _Path
            import json as _json

            out_dir = _Path("artifacts") / "ultra_export"
            if st.button("Generuj raport (HTML/PDF)", key="ultra_report_btn_t"):
                meta_ultra = {}
                if "bundle" in globals():
                    try:
                        meta_ultra = (globals().get("bundle") or {}).get("ultra", {}) or {}
                    except Exception:
                        meta_ultra = {}
                if not meta_ultra and (out_dir / "meta.json").exists():
                    meta_ultra = _json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))

                rpath = build_report(meta_ultra, out_dir)
                st.success("Raport gotowy — poniżej przyciski pobierania")
                st.download_button(
                    "⬇️ Pobierz report.html",
                    data=open(rpath, "rb").read(),
                    file_name="report.html",
                    mime="text/html",
                )
                pdfp = out_dir / "report.pdf"
                if pdfp.exists():
                    st.download_button(
                        "⬇️ Pobierz report.pdf",
                        data=open(pdfp, "rb").read(),
                        file_name="report.pdf",
                        mime="application/pdf",
                    )
                else:
                    st.caption("(PDF opcjonalny — wymaga wkhtmltopdf/pdfkit)")
        except Exception as e:
            st.error(f"Błąd generowania raportu: {e}")

    # ==================== TRENING MODELU — STRONA ====================

    def render_model_training_page(self):
        st.header("🤖 Trening Modelu")

        # 0) Dane wejściowe
        if not st.session_state.get('data_loaded') or st.session_state.get('df') is None:
            st.warning("Najpierw wczytaj dane w sekcji **📊 Analiza Danych**.")
            return

        # 1) Delikatny AUTO-CLEAN po wejściu (jednorazowo per dataset)
        df_raw = st.session_state.df
        sig = self._df_signature(df_raw)
        if st.session_state.get("auto_clean_key") != sig:
            df_clean, report = self._basic_auto_clean(df_raw)
            st.session_state.df_clean = df_clean
            st.session_state.clean_report = report
            st.session_state.auto_clean_key = sig

        if st.session_state.get("clean_report"):
            with st.expander("🧹 Raport czyszczenia (auto)", expanded=False):
                st.text(st.session_state.clean_report)

        if "df_clean" not in st.session_state or st.session_state.df_clean is None:
            st.error("Brak danych po auto-czyszczeniu.")
            return

        df = st.session_state.df_clean
        # 3) Wybór targetu
        target_column = self._render_target_selection(df)
        if not truthy_df_safe(target_column):
            return
        st.divider()
        # 4) AI plan (POPRAWIONE)
        final_config = self._render_ai_training_plan(df, target_column)
        if not isinstance(final_config, dict):
            final_config = {'enable_ensemble': False, 'train_size': 0.8, 'cv_folds': 5}
        st.session_state['ai_plan'] = final_config
        st.divider()
        # 6) Start treningu
        self._render_training_button(df, target_column, final_config)
        
        if st.session_state.get('model_trained') and 'model_results' in st.session_state:
            st.divider()
            st.success("✅ Model wytrenowany! Zobacz 📈 Wyniki i Wizualizacje")
            self._render_quick_results_preview()

    # ==================== POD-METODY UŻYWANE NA STRONIE ====================

    def _render_algorithms_info(self):
        """
        Sekcja informacyjna o wspieranych algorytmach ML
        (zależnie od typu problemu: regresja / klasyfikacja)
        """
        st.markdown("### 🤖 Dostępne algorytmy i silniki")
        st.info(
            "Poniżej znajdziesz zestawienie modeli używanych w procesie automatycznego treningu.\n\n"
            "- **Regresja**: Linear, Ridge, Lasso, ElasticNet, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting,\n"
            "  XGBRegressor, LGBMRegressor, CatBoostRegressor\n"
            "- **Klasyfikacja**: Logistic, RidgeClassifier, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting,\n"
            "  XGBClassifier, LGBMClassifier, CatBoostClassifier\n\n"
            "Silniki opcjonalne (**XGB / LGBM / CatBoost**) są automatycznie wykrywane – "
            "jeśli biblioteka jest zainstalowana, model zostanie włączony do puli treningowej."
        )

        st.markdown("#### 🔬 Tryby treningu")
        st.write(
            "- 🏎️ **fast_small** — szybki trening na ograniczonym zbiorze, mało modeli, brak tuningu\n"
            "- ⚖️ **balanced** — standardowy zestaw modeli z 5-fold CV\n"
            "- 🎯 **accurate / advanced** — dokładniejszy trening, pełny zbiór, tuning hiperparametrów\n"
            "- 🤝 **ensemble** — łączenie najlepszych modeli (Voting / Stacking / Blending)"
        )

        st.markdown("#### ⚙️ Tuning i optymalizacja")
        st.write(
            "- **RandomizedSearchCV** dla szybkiego tuningu\n"
            "- **GridSearchCV** dla dokładnego tuningu\n"
            "- **TOP-K** najlepszych modeli wybieranych wg metryki AI-driven\n"
            "- **Benchmark czasu** i pomiar inferencji po treningu"
        )

        st.markdown("#### 📦 Eksporty i raporty")
        st.write(
            "- Artefakty: `model.joblib`, `columns.json`, `metrics.csv`, `plan.json`, `recs.json`\n"
            "- Raport PDF z metrykami i wykresami (ROC, PR, residua, importances)\n"
            "- SHAP i Permutation Importance dla interpretowalności"
        )

        st.success("✅ Wszystkie modele są trenowane i oceniane automatycznie – aplikacja wybiera najlepszy zestaw według typu problemu.")


    def _render_target_selection(self, df: pd.DataFrame) -> Optional[str]:
        st.subheader("🎯 Krok 1: Wybór zmiennej docelowej")
        col1, col2 = st.columns([2, 1])

        with col1:
            preferred = st.session_state.get("preferred_target")

            # AI-sugestia z trenera (best effort)
            try:
                suggested_ai = self.ml_trainer.suggest_target_column(df)
            except Exception:
                suggested_ai = None

            # Auto-wybór na bazie heurystyk + preferred
            auto_choice, reason = auto_select_target(df, preferred or suggested_ai)

            if truthy_df_safe(preferred) and preferred not in df.columns and auto_choice:
                st.info(f"Wybrana kolumna celu '{preferred}' nie istnieje. Używam: '{auto_choice}'.")

            suggested = auto_choice or suggested_ai or (df.columns[0] if len(df.columns) else None)

            target_options = list(df.columns)
            default_idx = target_options.index(suggested) if (suggested in target_options) else 0

            target_column = st.selectbox(
                "Zmienna docelowa (target)",
                target_options,
                index=default_idx,
                help=f"💡 AI sugeruje: {suggested}" if suggested else None,
            )

        with col2:
            try:
                target_info = self._analyze_target_column(df, target_column)
                st.info(f"🎯 **AI sugeruje:** {suggested}")
                st.metric("Unikalne", target_info.get("unique_count", 0))
                st.metric("Typ", target_info.get("problem_type", "—"))
            except Exception as e:
                st.caption(f"Nie udało się przeanalizować celu: {e}")

        # zapamiętujemy wybór użytkownika
        st.session_state["preferred_target"] = target_column
        return target_column
    
    def _analyze_target_column(self, df: pd.DataFrame, target_column: str) -> Dict:
        unique_count = df[target_column].nunique()
        if df[target_column].dtype in ['object', 'category'] or unique_count <= 20:
            problem_type = "Klasyfikacja"
        else:
            problem_type = "Regresja"
        return {
            'unique_count': unique_count,
            'problem_type': problem_type,
            'dtype': str(df[target_column].dtype),
            'null_count': int(df[target_column].isnull().sum())
        }

    def _render_ai_training_plan(self, df: pd.DataFrame, target_column: str) -> Dict:
        st.subheader("🤖 AI Plan Treningu")

        with st.spinner("🔮 Analiza..."):
            if (
                "ai_training_plan" not in st.session_state
                or st.session_state.get("last_target") != target_column
                or st.session_state.get("ai_plan_hash") != self._df_signature(df)
            ):
                ai_plan = self._generate_ai_plan(df, target_column)
                st.session_state.ai_training_plan = ai_plan
                st.session_state.last_target = target_column
                st.session_state.ai_plan_hash = self._df_signature(df)
            else:
                ai_plan = st.session_state.ai_training_plan

        st.success("✅ Plan wygenerowany!")

        # ==== PODSTAWOWE KARTY (jak dotąd) ====
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("#### 📊 Podział danych")
            train_size = st.slider("Train size", 0.6, 0.9, ai_plan.get('train_size', 0.8), 0.05)
            st.caption(ai_plan.get('train_size_reason', ""))

        with col2:
            st.markdown("#### 🎯 Strategia")
            algorithm_strategy = st.selectbox(
                "Strategia treningu",
                ["fast_small", "balanced", "accurate", "advanced"],
                index=max(0, ["fast_small","balanced","accurate","advanced"].index(ai_plan.get('recommended_strategy','balanced')))
            )
            st.caption(ai_plan.get('strategy_reason', ""))

        with col3:
            st.markdown("#### ⚡ Optymalizacje")
            enable_tuning = st.checkbox("Hyperparameter Tuning", value=ai_plan.get('enable_hyperparameter_tuning', False))
            ensemble_mode = st.checkbox("Ensemble mode", value=ai_plan.get('enable_ensemble', False))
            st.caption("Tuning i łączenie modeli (stacking/blending).")

        col4, col5 = st.columns(2)
        with col4:
            st.markdown("#### 🔄 Walidacja")
            cv_folds = st.selectbox("CV folds", [3, 5, 10], index=[3, 5, 10].index(ai_plan.get('cv_folds', 5)))
            st.caption("Standardowe K-fold / w zależności od typu problemu.")

        with col5:
            st.markdown("#### 📈 Metryka")
            _metrics_list = ["roc_auc", "accuracy", "f1", "f1_weighted", "precision", "recall", "rmse", "mae", "r2"]
            _rec_metric = ai_plan.get("recommended_metric", "roc_auc")
            if _rec_metric not in _metrics_list:
                _rec_metric = "f1"
            optimization_metric = st.selectbox(
                "Metryka optymalizacji",
                _metrics_list,
                index=_metrics_list.index(_rec_metric)
            )
            st.caption("Dobierz pod typ problemu i priorytety biznesowe.")

        st.divider()

        # ==== NOWE OPCJE – trafne i praktyczne ====

        # 1) Balans klas, próbkowanie, próg decyzji
        st.markdown("### ⚖️ Balans & Próbkowanie")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stratify = st.checkbox("Stratyfikacja", value=True)
        with c2:
            class_weight = st.selectbox("Wagi klas", ["none", "balanced"])
        with c3:
            sampling = st.selectbox("Próbkowanie", ["none", "undersample", "oversample", "smote"])
        with c4:
            sampling_ratio = st.slider("Udział mniejszości (jeśli próbkowanie)", 0.05, 0.9, 0.5, 0.05)

        c5, c6 = st.columns(2)
        with c5:
            threshold_opt = st.selectbox("Optymalizacja progu", ["none", "f1", "youden_j"])
        with c6:
            profit_opt = st.checkbox("Uwzględnij koszty/korzyści (TP/FP/FN)", value=False)

        if truthy_df_safe(profit_opt):
            p1, p2, p3 = st.columns(3)
            with p1:
                value_tp = st.number_input("Value per TP", min_value=0.0, value=10.0, step=1.0)
            with p2:
                cost_fp = st.number_input("Cost FP", min_value=0.0, value=1.0, step=1.0)
            with p3:
                cost_fn = st.number_input("Cost FN", min_value=0.0, value=5.0, step=1.0)

        st.divider()

        # 2) Walidacja – zaawansowana (typy CV)
        st.markdown("### 🧪 Walidacja (zaawansowana)")
        v1, v2, v3 = st.columns(3)
        with v1:
            cv_type = st.selectbox("Typ CV", ["KFold", "StratifiedKFold", "GroupKFold", "TimeSeriesSplit"], index=1 if stratify else 0)
        with v2:
            shuffle = st.checkbox("Shuffle (jeśli dotyczy)", value=True)
        with v3:
            random_state = st.number_input("Random state", value=42, step=1)

        group_col = None
        time_series = (cv_type == "TimeSeriesSplit")
        group_kfold = (cv_type == "GroupKFold")

        if truthy_df_safe(time_series):
            st.info("Wykryto TimeSeriesSplit – stratyfikacja będzie ignorowana, a shuffle=FALSE wymuszony.")
            shuffle = False
            # podpowiedź kolumny czasu
            dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in str(c).lower()]
            if truthy_df_safe(dt_cols):
                st.caption(f"⏱️ Kolumny czasowe wykryte: {', '.join(map(str, dt_cols[:5]))}")
        if truthy_df_safe(group_kfold):
            possible_groups = [c for c in df.columns if df[c].nunique() < len(df) * 0.5]
            group_col = st.selectbox("Kolumna grupy (GroupKFold)", options=possible_groups or ["(brak)"])

        st.divider()

        # 3) Przetwarzanie cech
        st.markdown("### 🧩 Przetwarzanie cech")
        f1, f2, f3, f4 = st.columns(4)
        with f1:
            num_imputer = st.selectbox("Imputacja num.", ["mean", "median"])
        with f2:
            cat_imputer = st.selectbox("Imputacja kat.", ["most_frequent", "constant"])
        with f3:
            scaling = st.selectbox("Skalowanie", ["none", "standard", "minmax", "robust"])
        with f4:
            encoding = st.selectbox("Kodowanie kat.", ["one_hot", "target", "ordinal"])

        f5, f6, f7, f8 = st.columns(4)
        with f5:
            outliers = st.selectbox("Outliery", ["none", "winsorize_iqr", "winsorize_zscore"])
        with f6:
            variance_threshold = st.slider("Min. wariancja (selekcja)", 0.0, 0.05, 0.0, 0.005)
        with f7:
            feat_select = st.selectbox("Selekcja cech", ["none", "mutual_info", "kbest"])
        with f8:
            top_k = st.number_input("Top-K (jeśli selekcja)", min_value=1, value=min(50, max(1, df.shape[1] // 2)))

        st.divider()

        # 4) Rodziny modeli
        st.markdown("### 🧠 Rodziny modeli")
        fam_all = ["linear", "tree", "random_forest", "gbm", "xgboost", "lightgbm", "catboost", "svm", "knn", "naive_bayes", "mlp"]
        model_families = st.multiselect("Wybierz rodziny modeli", fam_all, default=["tree", "random_forest", "gbm", "xgboost", "lightgbm"])

        st.divider()

        # 5) Budżet i ustawienia techniczne
        st.markdown("### 🛠️ Budżet / Techniczne")
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            use_full = st.checkbox("Pełny dataset", value=len(df) <= 15000)
        with t2:
            n_jobs = st.number_input("Równoległość (n_jobs)", min_value=-1, value=-1, step=1, help="-1 = wszystkie rdzenie")
        with t3:
            max_train_time = st.number_input("Limit czasu treningu [s] / model", min_value=0, value=0, step=10, help="0 = bez limitu")
        with t4:
            hpo_n_trials = st.number_input("HPO: liczba prób (n_trials)", min_value=0, value=30 if enable_tuning else 0, step=5)

        t5, t6 = st.columns(2)
        with t5:
            early_stopping = st.checkbox("Early stopping (jeśli wspierane)", value=True)
        with t6:
            low_memory = st.checkbox("Tryb oszczędny (downcast float32)", value=False)

        st.divider()

        # Przyciski
        col_btn1, col_btn2 = st.columns([1, 1])
        apply_ai = False
        with col_btn1:
            if st.button("✨ Zastosuj cały plan AI", type="primary", use_container_width=True):
                apply_ai = True
        with col_btn2:
            st.caption("Możesz też skorzystać z ustawień powyżej (ręcznie).")

        # === Zwracamy konfigurację ===
        if truthy_df_safe(apply_ai):
            # rekomendacja AI (minimalny zestaw) – pozostawiamy jak dotąd
            return ai_plan

        cfg = {
            # dotychczasowe
            'train_size': float(train_size),
            'cv_folds': int(cv_folds),
            'recommended_strategy': algorithm_strategy,
            'recommended_metric': optimization_metric,
            'enable_hyperparameter_tuning': bool(enable_tuning),
            'enable_ensemble': bool(ensemble_mode),
            'use_full_dataset': bool(use_full),

            # nowe – balans/próg
            'stratify': bool(stratify) and not time_series,
            'class_weight': class_weight,
            'sampling': sampling,
            'sampling_ratio': float(sampling_ratio),
            'threshold_opt': threshold_opt,
            'profit_opt': bool(profit_opt),
            'value_tp': float(value_tp) if profit_opt else None,
            'cost_fp': float(cost_fp) if profit_opt else None,
            'cost_fn': float(cost_fn) if profit_opt else None,

            # walidacja zaawans.
            'cv_type': cv_type,
            'shuffle': bool(shuffle),
            'random_state': int(random_state),
            'time_series': bool(time_series),
            'group_column': str(group_col) if truthy_df_safe(group_kfold) and group_col not in (None, "(brak)") else None,

            # przetwarzanie cech
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'scaling': scaling,
            'encoding': encoding,
            'outliers': outliers,
            'variance_threshold': float(variance_threshold),
            'feature_selection': feat_select,
            'top_k_features': int(top_k),

            # rodziny modeli
            'model_families': model_families,

            # budżet/tech
            'n_jobs': int(n_jobs),
            'max_train_time': int(max_train_time),
            'hpo_n_trials': int(hpo_n_trials),
            'early_stopping': bool(early_stopping),
            'low_memory': bool(low_memory),
        }

        return cfg

    def _generate_ai_plan(self, df: pd.DataFrame, target_column: str) -> Dict:
        """
        Generuje rozszerzony plan AI na podstawie wielkości danych, typu problemu
        i właściwości celu/cech. Zwraca słownik gotowy do podania do UI.
        """
        import numpy as np
        n_samples, n_features = df.shape

        # --- Typ problemu
        problem_type = self.ml_trainer.detect_problem_type(df, target_column)  # "Klasyfikacja" | "Regresja"
        is_classification = str(problem_type).lower().startswith("klasyf")
        target_analysis = self._analyze_target_column(df, target_column)

        # --- Statystyki wejściowe
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        frac_num = (len(num_cols) / max(1, n_features))
        high_dim = n_features > n_samples

        # --- Czas/daty
        date_like_cols = [
            c for c in df.columns
            if "date" in str(c).lower() or "time" in str(c).lower()
            or pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        looks_like_timeseries = len(date_like_cols) > 0

        # --- Dystrybucja klas (dla klasyfikacji)
        minority_ratio = None
        n_classes = None
        if truthy_df_safe(is_classification):
            vc = df[target_column].value_counts(dropna=False)
            n_classes = len(vc)
            if len(vc) >= 2:
                maj = float(vc.iloc[0])
                mino = float(vc.iloc[-1])
                minority_ratio = min(mino, maj) / max(maj, mino) if max(maj, mino) > 0 else 1.0
            else:
                minority_ratio = 1.0

        plan: Dict[str, Any] = {}

        # ============ 1) Train size ============
        if n_samples < 1000:
            plan['train_size'] = 0.85
            plan['train_size_reason'] = "Mały zbiór — więcej danych do treningu."
        elif n_samples < 10000:
            plan['train_size'] = 0.80
            plan['train_size_reason'] = "Standardowy podział 80/20."
        else:
            plan['train_size'] = 0.75
            plan['train_size_reason'] = "Duży zbiór — więcej danych na walidację."

        # ============ 2) CV podstawowe ============
        if n_samples < 500:
            plan['cv_folds'] = 3
            plan['cv_reason'] = "Mało danych — 3-fold CV."
        elif n_samples < 5000:
            plan['cv_folds'] = 5
            plan['cv_reason'] = "Balans — 5-fold CV."
        else:
            plan['cv_folds'] = 10
            plan['cv_reason'] = "Dużo danych — 10-fold CV."

        # ============ 3) Strategia algorytmów ============
        # Dopasowane do UI: ["fast_small","balanced","accurate","advanced"]
        if n_samples < 1000:
            plan['recommended_strategy'] = 'fast_small'
            plan['strategy_reason'] = "Mało próbek — szybkie, lekkie modele."
        elif high_dim or (n_features > n_samples * 0.8):
            plan['recommended_strategy'] = 'advanced'
            plan['strategy_reason'] = "Wysoka wymiarowość — metody zaawansowane."
        elif n_samples > 30000:
            plan['recommended_strategy'] = 'accurate'
            plan['strategy_reason'] = "Więcej danych — stawiamy na dokładność."
        else:
            plan['recommended_strategy'] = 'balanced'
            plan['strategy_reason'] = "Zbalansowany kompromis prędkość/dokładność."

        # ============ 4) Metryka ============
        if truthy_df_safe(is_classification):
            if (n_classes == 2) or (target_analysis.get('unique_count') == 2):
                # Clas. binarna
                if minority_ratio is not None and minority_ratio < 0.5:
                    plan['recommended_metric'] = 'roc_auc'
                    plan['metric_reason'] = "Binarna + niezbalansowana — ROC AUC lepiej oddaje ranking."
                else:
                    plan['recommended_metric'] = 'f1'
                    plan['metric_reason'] = "Binarna — F1 jako kompromis precision/recall."
            else:
                plan['recommended_metric'] = 'f1_weighted'
                plan['metric_reason'] = "Wiele klas — F1 (waga)."
        else:
            plan['recommended_metric'] = 'r2'
            plan['metric_reason'] = "Regresja — R²."

        # ============ 5) Tuning/Ensemble ============
        plan['enable_hyperparameter_tuning'] = bool(n_samples > 2000 and n_features < 300)
        plan['tuning_reason'] = "Tuning ON (rozmiar/cechy OK)" if plan['enable_hyperparameter_tuning'] else "Tuning OFF — zbyt kosztowny."

        plan['enable_ensemble'] = bool(n_samples > 5000)
        plan['ensemble_reason'] = "Ensemble ON (dużo danych)" if plan['enable_ensemble'] else "Ensemble OFF (opcjonalne)."

        # ============ 6) Balans klas / sampling / próg ============
        if truthy_df_safe(is_classification):
            # Stratyfikacja: nie dla TimeSeries
            plan['stratify'] = not looks_like_timeseries
            plan['stratify_reason'] = "Stratyfikacja utrzymuje proporcje klas." if plan['stratify'] else "SPLIT czasowy — bez stratyfikacji."

            # Wagi klas
            if (minority_ratio is not None) and (minority_ratio < 0.6):
                plan['class_weight'] = 'balanced'
                plan['class_weight_reason'] = "Niezbalansowane — włącz wagi klas."
            else:
                plan['class_weight'] = 'none'
                plan['class_weight_reason'] = "Proporcje ok — wagi niekonieczne."

            # Sampling
            if (minority_ratio is not None) and (minority_ratio < 0.4):
                plan['sampling'] = 'smote' if len(num_cols) > 0 and n_samples >= 1000 else 'oversample'
                plan['sampling_reason'] = "Silna nierównowaga — SMOTE/oversampling."
                plan['sampling_ratio'] = 0.5
            else:
                plan['sampling'] = 'none'
                plan['sampling_reason'] = "Brak silnej nierównowagi — sampling zbędny."
                plan['sampling_ratio'] = 0.0

            # Optymalizacja progu
            if plan['recommended_metric'] == 'roc_auc':
                plan['threshold_opt'] = 'youden_j'
                plan['threshold_reason'] = "Maksymalizacja TPR−FPR (Youden J) po krzywej ROC."
            elif plan['recommended_metric'] in ('f1', 'f1_weighted'):
                plan['threshold_opt'] = 'f1'
                plan['threshold_reason'] = "Dobór progu pod maksymalizację F1."
            else:
                plan['threshold_opt'] = 'none'
                plan['threshold_reason'] = "Próg nieoptymalizowany pod wybraną metrykę."
        else:
            plan['stratify'] = False
            plan['class_weight'] = 'none'
            plan['sampling'] = 'none'
            plan['sampling_ratio'] = 0.0
            plan['threshold_opt'] = 'none'
            plan['stratify_reason'] = "Regresja — brak klas do stratyfikacji."
            plan['class_weight_reason'] = "Regresja — brak wag klas."
            plan['sampling_reason'] = "Regresja — sampling klas nie dotyczy."
            plan['threshold_reason'] = "Regresja — próg nie dotyczy."

        # ============ 7) Walidacja zaawansowana ============
        if truthy_df_safe(looks_like_timeseries):
            plan['cv_type'] = 'TimeSeriesSplit'
            plan['shuffle'] = False
            plan['cv_type_reason'] = "Dane czasowe — TimeSeriesSplit + bez shuffle."
        else:
            if truthy_df_safe(is_classification):
                plan['cv_type'] = 'StratifiedKFold'
                plan['cv_type_reason'] = "Klasyfikacja — użyj StratifiedKFold."
            else:
                plan['cv_type'] = 'KFold'
                plan['cv_type_reason'] = "Regresja — zwykły KFold."
            plan['shuffle'] = True

        plan['random_state'] = 42
        # GroupKFold – wykrywanie grup jest domenowe, więc domyślnie None
        plan['group_column'] = None

        # ============ 8) Przetwarzanie cech ============
        plan['num_imputer'] = 'median'
        plan['cat_imputer'] = 'most_frequent'

        # Skalowanie
        if frac_num > 0.5 and not looks_like_timeseries:
            plan['scaling'] = 'standard'
            plan['scaling_reason'] = "Wiele cech numerycznych — standard scaler pomaga modelom liniowym/SVM/MLP."
        else:
            plan['scaling'] = 'none'
            plan['scaling_reason'] = "Przewaga drzew/cech kat. — skalowanie opcjonalne."

        # Kodowanie kat.
        high_card_cols = [c for c in cat_cols if df[c].nunique(dropna=True) > 50]
        if len(high_card_cols) > 0 and n_samples > 3000:
            plan['encoding'] = 'target'
            plan['encoding_reason'] = "Wysoka krotność kategorii — target encoding."
        else:
            plan['encoding'] = 'one_hot'
            plan['encoding_reason'] = "Niska/umiarkowana krotność — one-hot."

        # Outliery
        if (frac_num > 0.5) and (n_samples > 1000):
            plan['outliers'] = 'winsorize_iqr'
            plan['outliers_reason'] = "Spore i numeryczne — przytnij ogony (IQR)."
        else:
            plan['outliers'] = 'none'
            plan['outliers_reason'] = "Mało numerycznych / mało próbek — bez winsoryzacji."

        # Selekcja cech
        if high_dim or (n_features > 300):
            plan['feature_selection'] = 'kbest'
            plan['feature_selection_reason'] = "Dużo cech — k-best/MI ograniczy wymiar."
            plan['variance_threshold'] = 0.001
            plan['top_k_features'] = int(min(50, max(10, n_features // 2)))
        else:
            plan['feature_selection'] = 'none'
            plan['feature_selection_reason'] = "Wymiar umiarkowany — selekcja opcjonalna."
            plan['variance_threshold'] = 0.0
            plan['top_k_features'] = int(min(50, max(10, n_features // 2)))

        # ============ 9) Rodziny modeli ============
        if truthy_df_safe(is_classification):
            families = ["tree", "random_forest", "gbm", "xgboost", "lightgbm", "linear"]
            if n_samples < 20000:
                families += ["svm", "knn"]
            if len(cat_cols) > 0:
                families += ["naive_bayes"]
            plan['model_families'] = sorted(list(dict.fromkeys(families)))
            plan['model_families_reason'] = "Drzewa/boostingi + liniówka; SVM/KNN dla mniejszych zbiorów, NB przy kategoriach."
        else:
            families = ["linear", "tree", "random_forest", "gbm", "xgboost", "lightgbm"]
            plan['model_families'] = families
            plan['model_families_reason'] = "Regresja: liniówka + drzewa/boostingi sprawdzają się najlepiej."

        # ============ 10) Budżet / techniczne ============
        plan['use_full_dataset'] = bool(n_samples <= 15000)
        plan['n_jobs'] = -1
        if plan['enable_hyperparameter_tuning']:
            # rozmiar budżetu zależny od danych
            if n_samples < 5000:
                plan['hpo_n_trials'] = 30
                plan['budget_reason'] = "Mały/średni zbiór — 30 prób HPO."
            elif n_samples < 30000:
                plan['hpo_n_trials'] = 50
                plan['budget_reason'] = "Większy zbiór — 50 prób HPO."
            else:
                plan['hpo_n_trials'] = 80
                plan['budget_reason'] = "Bardzo duży zbiór — 80 prób HPO."
        else:
            plan['hpo_n_trials'] = 0
            plan['budget_reason'] = "Tuning wyłączony."

        # limit czasu / model (sekundy) – 0 = brak limitu
        if n_samples < 10000:
            plan['max_train_time'] = 0
        else:
            plan['max_train_time'] = 180  # 3 min / model jako miękki bezpiecznik

        plan['early_stopping'] = True
        plan['low_memory'] = bool(high_dim or (n_features > 400))

        # ============ 11) Uwagi specjalne ============
        plan['special_considerations'] = []
        if target_analysis.get('null_count', 0) > 0:
            plan['special_considerations'].append(f"⚠️ Target ma {target_analysis['null_count']} NaN.")
        if truthy_df_safe(high_dim):
            plan['special_considerations'].append("⚠️ Więcej cech niż próbek — ryzyko overfittingu.")
        if n_samples > 50000:
            plan['special_considerations'].append("💡 Duży dataset — trening może potrwać dłużej.")
        if truthy_df_safe(looks_like_timeseries):
            plan['special_considerations'].append("⏱️ Wykryto kolumny czasowe — rozważ cechy lag/rolling.")

        return plan

    def _families_all(self, problem_type: str):
        """Pełna lista rodzin modeli rozpoznawana przez UI/Trainer."""
        base = [
            "linear", "tree", "random_forest", "gbm",
            "xgboost", "lightgbm", "catboost",
            "svm", "knn", "naive_bayes", "mlp"
        ]
        return base

    def _class_imbalance_ratio(self, y: pd.Series) -> float:
        """Prosta miara niezbalansowania klas (max/min)."""
        vc = y.value_counts(dropna=True)
        if len(vc) < 2 or vc.min() == 0:
            return 1.0
        return float(vc.max()) / float(vc.min())

    def _auto_select_model_families(self, df: pd.DataFrame, target_column: str, problem_type: str):
        """Heurystyczny dobór rodzin modeli zależny od zbioru."""
        rows, cols = df.shape
        num_cols = df.drop(columns=[target_column], errors="ignore").select_dtypes(include=[np.number]).shape[1]
        cat_cols = df.drop(columns=[target_column], errors="ignore").select_dtypes(include=["object", "category", "bool"]).shape[1]

        families = []

        if str(problem_type).lower().startswith("klasyf") or "class" in str(problem_type).lower():
            # Bazowe, bezpieczne:
            families += ["linear", "tree", "random_forest", "gbm"]

            # Boostingi:
            if rows <= 100_000:
                families += ["xgboost"]
            families += ["lightgbm"]  # LGBM jest zwykle najszybsze na dużych

            # Kategoryczne → CatBoost:
            if cat_cols > 0:
                families += ["catboost"]

            # Małe zbiory → dorzuć 'cięższe' klasyki:
            if rows < 2_000:
                families += ["svm", "knn", "naive_bayes", "mlp"]

            # Niezbalansowane → faworyzuj lasy/boostingi, wytnij KNN/SVM/NB:
            try:
                imb = self._class_imbalance_ratio(df[target_column].dropna())
                if imb > 3:
                    families = [f for f in families if f not in ("knn", "svm", "naive_bayes")]
                    families = list(dict.fromkeys(
                        families + ["random_forest", "gbm", "lightgbm", "xgboost"]
                    ))
            except Exception:
                pass

            # Dużo cech → unikaj KNN/SVM/MLP:
            if cols > 200:
                families = [f for f in families if f not in ("knn", "svm", "mlp")]

        else:  # REGRESJA
            families += ["linear", "tree", "random_forest", "gbm", "lightgbm"]
            if rows <= 100_000:
                families += ["xgboost"]
            if cat_cols > 0:
                families += ["catboost"]

            if rows < 3_000:
                families += ["svm", "mlp"]

            if cols > 200:
                families = [f for f in families if f not in ("knn", "svm", "mlp")]  # (knn nie ma w regresji w tej liście)

        # Dedup z zachowaniem kolejności:
        families = list(dict.fromkeys(families))
        # Na wszelki wypadek: odfiltruj poza listą znanych:
        known = set(self._families_all(problem_type))
        families = [f for f in families if f in known]
        return families

    def _render_final_configuration(self, df: pd.DataFrame, target_column: str, ai_plan: Dict) -> Dict:
        # Jeśli konfiguracja została już zastosowana wcześniej – zwróć ją (Twoja logika)
        if st.session_state.get('ai_config_applied', False):
            return st.session_state.get('final_config', ai_plan)

        st.session_state.setdefault('final_config', {})

        problem_type = self.ml_trainer.detect_problem_type(df, target_column)
        applied = bool(st.session_state.get('ai_plan_applied', False))  # ustawiane przy „Zastosuj cały plan”

        with st.expander("🔧 Dostosuj konfigurację (opcjonalnie)", expanded=False):
            st.markdown("### ⚙️ Zaawansowane ustawienia")
            col1, col2, col3 = st.columns(3)

            # ====== Podział / CV ======
            with col1:
                st.markdown("**📊 Podział**")
                train_size = st.slider(
                    "Train size", 0.6, 0.9,
                    float(ai_plan.get('train_size', 0.8)), 0.05,
                    key="cfg_train_size"
                )
                cv_folds = st.selectbox(
                    "CV folds",
                    [3, 5, 10],
                    index=[3, 5, 10].index(int(ai_plan.get('cv_folds', 5))),
                    key="cfg_cv_folds"
                )

            # ====== Strategia / Metryka ======
            with col2:
                st.markdown("**🎯 Algorytmy**")
                strategy_choices = ["all", "fast_small", "balanced", "accurate", "advanced", "ensemble"]
                strategy_default = ai_plan.get('recommended_strategy', 'balanced')
                if strategy_default not in strategy_choices:
                    # mapowanie starych aliasów
                    alias_map = {"fast": "fast_small", "accurate": "accurate", "advanced": "advanced", "all": "all"}
                    strategy_default = alias_map.get(str(strategy_default), "balanced")

                algorithm_strategy = st.selectbox(
                    "Strategia",
                    strategy_choices,
                    index=strategy_choices.index(strategy_default),
                    key="cfg_strategy"
                )

                if problem_type == "Klasyfikacja":
                    metric_options = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted", "roc_auc", "pr_auc"]
                    default_metric = ai_plan.get('recommended_metric', "f1_weighted")
                    if default_metric not in metric_options:
                        # tolerancja starych nazw
                        if default_metric in ("f1", "f1_macro", "f1_micro"):
                            default_metric = "f1_weighted"
                        elif default_metric in ("auc", "roc_auc_ovr", "roc_auc_ovo"):
                            default_metric = "roc_auc"
                        else:
                            default_metric = "f1_weighted"
                else:
                    metric_options = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]
                    default_metric = ai_plan.get('recommended_metric', "r2")
                    if default_metric not in metric_options:
                        default_metric = "r2"

                optimization_metric = st.selectbox(
                    "Metryka",
                    metric_options,
                    index=metric_options.index(default_metric),
                    key="cfg_metric"
                )

            # ====== Optymalizacje ======
            with col3:
                st.markdown("**⚡ Optymalizacje**")
                enable_tuning = st.checkbox(
                    "Hyperparameter tuning",
                    value=bool(ai_plan.get('enable_hyperparameter_tuning', False)),
                    key="cfg_hpo"
                )
                ensemble_mode = st.checkbox(
                    "Ensemble mode",
                    value=bool(ai_plan.get('enable_ensemble', False)),
                    key="cfg_ens"
                )
                use_full = st.checkbox(
                    "Pełny dataset",
                    value=bool(len(df) <= 15000),
                    key="cfg_full_data"
                )

            # ====== Rodziny modeli ======
            st.markdown("---")
            st.markdown("**🧠 Rodziny modeli**")

            # auto-wyliczenie rodzin (za każdym razem aktualne wobec danych/typu problemu)
            auto_families = self._auto_select_model_families(df, target_column, problem_type)

            # pełna lista opcji (dopasowana do problemu, jeśli Twoja implementacja to wspiera)
            families_options = self._families_all(problem_type)

            # Czy użyć automatu (domyślnie: tak, jeśli zastosowano plan AI; zapamiętywane między rerunami):
            auto_lock_default = applied or st.session_state.get("use_auto_families_default", True)
            use_auto_families = st.checkbox(
                "Użyj automatycznego doboru (z Planu AI)",
                value=auto_lock_default,
                help="Gdy włączone – lista jest uzupełniana i blokowana przez AI.",
                key="cfg_use_auto_families"
            )

            # Jeżeli automat aktywny i użytkownik nie robił override, ustaw stan *przed* renderem widgetu:
            if truthy_df_safe(use_auto_families) and not st.session_state.get("model_families_user_override", False):
                st.session_state['model_families'] = list(auto_families)

            # domyślna wartość w multiselect (gdyby nie było stanu)
            default_families = (
                st.session_state.get('model_families')
                or (st.session_state.get('final_config', {}).get('model_families'))
                or auto_families
                or families_options
            )

            # multiselect; w trybie auto jest disabled, ale nadal pokazuje stan
            model_families = st.multiselect(
                "Wybierz rodziny modeli",
                options=families_options,
                default=default_families,
                key="model_families",
                disabled=use_auto_families
            )

            # jeśli auto – narzuć automat i skasuj flagę override
            if truthy_df_safe(use_auto_families):
                model_families = list(auto_families)
                st.session_state["model_families_user_override"] = False
            else:
                # użytkownik może ręcznie nadpisywać
                st.session_state["model_families_user_override"] = True

            # zapamiętaj preferencję auto/manual na kolejne reruny
            st.session_state["use_auto_families_default"] = bool(use_auto_families)

            # zsynchronizuj z final_config (żeby inne miejsca mogły tego użyć)
            st.session_state['final_config']['model_families'] = list(model_families)

            # ====== Zwracana konfiguracja ======
            return {
                'train_size': float(train_size),
                'cv_folds': int(cv_folds),
                'recommended_strategy': str(algorithm_strategy),
                'recommended_metric': str(optimization_metric),
                'enable_hyperparameter_tuning': bool(enable_tuning),
                'enable_ensemble': bool(ensemble_mode),
                'use_full_dataset': bool(use_full),
                'model_families': list(model_families),
            }

        # --- helpers ---
        def _v(key, default=None):
            return ai_plan.get(key, default)

        def _safe_idx(options, value, default_idx=0):
            try:
                return options.index(value)
            except Exception:
                return default_idx

        # --- problem type / metryki ---
        problem_type = self.ml_trainer.detect_problem_type(df, target_column)
        is_clf = str(problem_type).lower().startswith("klasyf")

        # Pełniejsze listy (obsługują różne warianty z planu i trenera)
        metrics_clf = ["roc_auc", "accuracy", "f1", "f1_weighted", "precision", "recall"]
        metrics_reg = ["r2", "rmse", "mae", "neg_root_mean_squared_error", "neg_mean_absolute_error"]

        metric_opts = metrics_clf if is_clf else metrics_reg
        rec_metric = _v('recommended_metric', "roc_auc" if is_clf else "r2")
        metric_idx = _safe_idx(metric_opts, rec_metric, 0)

        # Strategie – uwzględniamy stare aliasy, ale pokazujemy docelowe
        strategy_opts = ["fast_small", "balanced", "accurate", "advanced"]
        rec_strategy = _v('recommended_strategy', 'balanced')
        if rec_strategy in ("all", "fast"):          # aliasy ze starszej wersji
            rec_strategy = "balanced" if rec_strategy == "all" else "fast_small"
        strat_idx = _safe_idx(strategy_opts, rec_strategy, 1)

        # CV
        cv_type_opts = ["KFold", "StratifiedKFold", "GroupKFold", "TimeSeriesSplit"]
        rec_cv_type = _v('cv_type', "StratifiedKFold" if is_clf else "KFold")
        cv_type_idx = _safe_idx(cv_type_opts, rec_cv_type, 1 if is_clf else 0)

        with st.expander("🔧 Dostosuj konfigurację (opcjonalnie)", expanded=False):
            st.markdown("### ⚙️ Zaawansowane ustawienia")

            # === PODSTAWOWE ===
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**📊 Podział**")
                train_size = st.slider("Train size", 0.60, 0.90, float(_v('train_size', 0.80)), 0.05)
                cv_folds = st.selectbox("CV folds", [3, 5, 10], index=_safe_idx([3, 5, 10], int(_v('cv_folds', 5))))
            with c2:
                st.markdown("**🎯 Strategia**")
                algorithm_strategy = st.selectbox("Strategia", strategy_opts, index=strat_idx)
                optimization_metric = st.selectbox("Metryka", metric_opts, index=metric_idx)
            with c3:
                st.markdown("**⚡ Optymalizacje**")
                enable_tuning = st.checkbox("Hyperparameter Tuning", value=bool(_v('enable_hyperparameter_tuning', False)))
                ensemble_mode = st.checkbox("Ensemble", value=bool(_v('enable_ensemble', False)))
                use_full = st.checkbox("Pełny dataset", value=bool(_v('use_full_dataset', len(df) <= 15000)))

            st.divider()

            # === WALIDACJA (zaawansowana) ===
            st.markdown("### 🧪 Walidacja")
            v1, v2, v3 = st.columns(3)
            with v1:
                cv_type = st.selectbox("Typ CV", cv_type_opts, index=cv_type_idx)
            with v2:
                shuffle = st.checkbox("Shuffle (jeśli dotyczy)", value=bool(_v('shuffle', True)))
            with v3:
                random_state = st.number_input("Random state", value=int(_v('random_state', 42)), step=1)

            group_column = None
            if cv_type == "GroupKFold":
                poss_groups = [c for c in df.columns if df[c].nunique(dropna=True) < len(df) * 0.5]
                group_column = st.selectbox("Kolumna grupy", options=(poss_groups or ["(brak)"]))
                if group_column == "(brak)":
                    group_column = None

            if cv_type == "TimeSeriesSplit":
                shuffle = False  # TS: shuffle off
                st.caption("⏱️ TimeSeriesSplit — `shuffle` wyłączony, stratyfikacja ignorowana.")

            st.divider()

            # === BALANS / SAMPLING / PRÓG ===
            st.markdown("### ⚖️ Balans & próbkowanie")
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                stratify = st.checkbox("Stratyfikacja", value=bool(_v('stratify', True and cv_type != "TimeSeriesSplit")))
            with b2:
                class_weight = st.selectbox("Wagi klas", ["none", "balanced"], index=_safe_idx(["none", "balanced"], _v('class_weight', 'none')))
            with b3:
                sampling = st.selectbox("Próbkowanie", ["none", "undersample", "oversample", "smote"],
                                        index=_safe_idx(["none","undersample","oversample","smote"], _v('sampling', 'none')))
            with b4:
                sampling_ratio = st.slider("Udział mniejszości (jeśli próbkowanie)", 0.05, 0.90, float(_v('sampling_ratio', 0.5)), 0.05)

            b5, b6 = st.columns(2)
            with b5:
                threshold_opt = st.selectbox("Optymalizacja progu", ["none", "f1", "youden_j"],
                                            index=_safe_idx(["none","f1","youden_j"], _v('threshold_opt', 'none')))
            with b6:
                profit_opt = st.checkbox("Uwzględnij koszty/korzyści (TP/FP/FN)", value=bool(_v('profit_opt', False)))

            value_tp = cost_fp = cost_fn = None
            if truthy_df_safe(profit_opt):
                p1, p2, p3 = st.columns(3)
                with p1:
                    value_tp = st.number_input("Value per TP", min_value=0.0, value=float(_v('value_tp', 10.0)), step=1.0)
                with p2:
                    cost_fp = st.number_input("Cost FP", min_value=0.0, value=float(_v('cost_fp', 1.0)), step=1.0)
                with p3:
                    cost_fn = st.number_input("Cost FN", min_value=0.0, value=float(_v('cost_fn', 5.0)), step=1.0)

            st.divider()

            # === PRZETWARZANIE CECH ===
            st.markdown("### 🧩 Przetwarzanie cech")
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                num_imputer = st.selectbox("Imputacja num.", ["mean", "median"], index=_safe_idx(["mean","median"], _v('num_imputer','median')))
            with f2:
                cat_imputer = st.selectbox("Imputacja kat.", ["most_frequent", "constant"], index=_safe_idx(["most_frequent","constant"], _v('cat_imputer','most_frequent')))
            with f3:
                scaling = st.selectbox("Skalowanie", ["none", "standard", "minmax", "robust"], index=_safe_idx(["none","standard","minmax","robust"], _v('scaling','none')))
            with f4:
                encoding = st.selectbox("Kodowanie kat.", ["one_hot", "target", "ordinal"], index=_safe_idx(["one_hot","target","ordinal"], _v('encoding','one_hot')))

            f5, f6, f7, f8 = st.columns(4)
            with f5:
                outliers = st.selectbox("Outliery", ["none", "winsorize_iqr", "winsorize_zscore"], index=_safe_idx(["none","winsorize_iqr","winsorize_zscore"], _v('outliers','none')))
            with f6:
                variance_threshold = st.slider("Min. wariancja (selekcja)", 0.0, 0.05, float(_v('variance_threshold', 0.0)), 0.005)
            with f7:
                feature_selection = st.selectbox("Selekcja cech", ["none", "mutual_info", "kbest"], index=_safe_idx(["none","mutual_info","kbest"], _v('feature_selection','none')))
            with f8:
                top_k_features = st.number_input("Top-K (jeśli selekcja)", min_value=1, value=int(_v('top_k_features', min(50, max(1, df.shape[1] // 2)))))

            st.divider()

            # === RODZINY MODELI ===
            st.markdown("### 🧠 Rodziny modeli")
            fam_all = ["linear", "tree", "random_forest", "gbm", "xgboost", "lightgbm", "catboost", "svm", "knn", "naive_bayes", "mlp"]
            model_families = st.multiselect(
                "Wybierz rodziny modeli",
                fam_all,
                default=_v('model_families', ["tree", "random_forest", "gbm", "xgboost", "lightgbm"])
            )

            st.divider()

            # === BUDŻET / TECHNICZNE ===
            st.markdown("### 🛠️ Budżet / Techniczne")
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                n_jobs = st.number_input("Równoległość (n_jobs)", min_value=-1, value=int(_v('n_jobs', -1)), step=1)
            with t2:
                max_train_time = st.number_input("Limit czasu [s]/model", min_value=0, value=int(_v('max_train_time', 0)), step=10)
            with t3:
                hpo_n_trials = st.number_input("HPO: liczba prób (n_trials)", min_value=0, value=int(_v('hpo_n_trials', 0)), step=5)
            with t4:
                early_stopping = st.checkbox("Early stopping", value=bool(_v('early_stopping', True)))

            low_memory = st.checkbox("Tryb oszczędny (downcast float32)", value=bool(_v('low_memory', False)))

        # Mapowanie aliasów metryk na nazwy „techniczne” (jeśli trainer tak oczekuje)
        metric_choice = optimization_metric
        metric_map_out = {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "precision_weighted": "precision",  # ujednolicenie ewentualnych starych nazw
        }
        metric_out = metric_map_out.get(metric_choice, metric_choice)

        final_cfg = {
            # podstawowe
            'train_size': float(train_size),
            'cv_folds': int(cv_folds),
            'recommended_strategy': algorithm_strategy,
            'recommended_metric': metric_out,
            'enable_hyperparameter_tuning': bool(enable_tuning),
            'enable_ensemble': bool(ensemble_mode),
            'use_full_dataset': bool(use_full),

            # walidacja
            'cv_type': cv_type,
            'shuffle': bool(shuffle),
            'random_state': int(random_state),
            'group_column': group_column,

            # balans / sampling / próg
            'stratify': bool(stratify) and cv_type != "TimeSeriesSplit",
            'class_weight': class_weight,
            'sampling': sampling,
            'sampling_ratio': float(sampling_ratio),
            'threshold_opt': threshold_opt,
            'profit_opt': bool(profit_opt),
            'value_tp': float(value_tp) if profit_opt else None,
            'cost_fp': float(cost_fp) if profit_opt else None,
            'cost_fn': float(cost_fn) if profit_opt else None,

            # przetwarzanie cech
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'scaling': scaling,
            'encoding': encoding,
            'outliers': outliers,
            'variance_threshold': float(variance_threshold),
            'feature_selection': feature_selection,
            'top_k_features': int(top_k_features),

            # modele
            'model_families': model_families,

            # budżet / techniczne
            'n_jobs': int(n_jobs),
            'max_train_time': int(max_train_time),
            'hpo_n_trials': int(hpo_n_trials),
            'early_stopping': bool(early_stopping),
            'low_memory': bool(low_memory),
        }

        return final_cfg

    def _render_training_button(self, df: pd.DataFrame, target_column: str, config: Dict):
        """
        Przycisk startu + szacowanie czasu. Obsługuje blokadę, gdy trening trwa.
        """
        import datetime as _dt

        st.markdown("### 🚀 Gotowy do treningu?")

        # Bezpieczne domyślne
        st.session_state.setdefault('training_in_progress', False)
        st.session_state.setdefault('model_trained', False)

        # Szacowanie czasu z uwzględnieniem rozszerzonego configu
        estimated_time = self._estimate_training_time(df, config)

        # Blok przycisku gdy trwa trening
        btn_disabled = bool(st.session_state.get('training_in_progress', False))

        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button(
                "🎯 ROZPOCZNIJ TRENING MODELU",
                type="primary",
                use_container_width=True,
                disabled=btn_disabled
            ):
                self._execute_training(df, target_column, config)

        with col2:
            st.info(f"⏱️ **Szacowany czas:**\n{estimated_time}")

    def _estimate_training_time(self, df: pd.DataFrame, config: Dict) -> str:
        """
        Szacuje czas treningu biorąc pod uwagę:
        - rozmiar danych i cech,
        - CV folds / typ CV,
        - strategię algorytmów / liczbę rodzin modeli,
        - ensemble,
        - HPO (n_trials),
        - równoległość (n_jobs),
        - max_train_time (cięcie do limitu).
        """
        import math

        n = len(df) if config.get('use_full_dataset', False) else min(8000, len(df))
        p = max(1, df.shape[1] - 1)

        # bazowy koszt ~ liniowo w n, logarytmicznie w p (łagodniej dla wysokich wymiarów)
        base = (n / 1000.0) * (1.0 + math.log10(max(10, p))) * 10.0

        # CV
        cv_folds = int(config.get('cv_folds', 5) or 5)
        base *= (1.0 + 0.15 * (cv_folds - 1))  # każdy dodatkowy fold ~+15%

        # typ strategii / rodziny modeli
        strategy = str(config.get('recommended_strategy', 'balanced') or 'balanced')
        strat_mult = {
            'fast_small': 0.6, 'fast': 0.6, 'balanced': 1.0,
            'accurate': 1.6, 'advanced': 2.2, 'all': 1.4, 'ensemble': 1.8
        }.get(strategy, 1.0)
        base *= strat_mult

        fam = config.get('model_families') or []
        if isinstance(fam, (list, tuple)) and len(fam) > 0:
            base *= min(2.5, 0.6 + 0.15 * len(fam))  # więcej rodzin = dłużej (ale z limitem)

        # Ensemble
        if config.get('enable_ensemble', False):
            base *= 1.5

        # HPO
        if config.get('enable_hyperparameter_tuning', False):
            trials = int(config.get('hpo_n_trials', 30) or 30)
            base *= min(3.5, 1.0 + trials / 40.0)

        # n_jobs (równoległość)
        n_jobs = int(config.get('n_jobs', -1) or -1)
        if n_jobs == -1:
            base *= 0.7
        elif n_jobs > 1:
            base *= max(0.4, 1.0 - 0.08 * (n_jobs - 1))

        # max_train_time (limit na model) – miękkie cięcie
        max_per_model = int(config.get('max_train_time', 0) or 0)
        if max_per_model > 0:
            base = min(base, max_per_model * 1.2)

        # Format
        if base < 60:
            return f"{int(base)} sek"
        elif base < 3600:
            return f"{int(base // 60)} min"
        else:
            return f"{_fmt_float_safe(base / 3600.0, 1)} godz"

    def _maybe_sample(self, df_in: pd.DataFrame, ycol: str, use_full: bool, n_max: int = 8000, random_state: int = 42):
        """Stratyfikowany sampling dla klasyfikacji; losowy dla regresji. Zwraca (df_out, msg)."""
        import pandas as pd
        if use_full or len(df_in) <= n_max:
            return df_in, None
        try:
            # Jeżeli masz helper w projekcie – użyje go:
            from backend.helpers.targeting import stratified_sample_df  # noqa
            out, msg = stratified_sample_df(df_in, ycol, n_max=n_max, random_state=random_state)
            return out, msg
        except Exception:
            y = df_in[ycol]
            # regresja -> zwykły sample
            if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
                out = df_in.sample(n=n_max, random_state=random_state)
                return out, f"Regresja: próbka {len(out)} z {len(df_in)} (fallback)."
            # klasyfikacja -> ręczna stratyfikacja
            frac = n_max / len(df_in)
            parts = []
            for _, g in df_in.groupby(ycol):
                k = max(1, int(round(len(g) * frac)))
                parts.append(g.sample(n=min(k, len(g)), random_state=random_state))
            out = pd.concat(parts).sample(frac=1.0, random_state=random_state).head(n_max)
            return out, f"Klasyfikacja: stratyfikowana próbka {len(out)} z {len(df_in)} (fallback)."


    def _clean_df_for_training(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Minimalny, bezpieczny cleaning:
        - jeżeli target wygląda na one-hot (np. 'type_organic'): usuń bazę 'type' i siostry 'type_*' poza targetem
        - usuń stałe kolumny (poza targetem)
        - sparsuj kolumny dat do year/month/dow i usuń oryginał
        - usuń duplikaty kolumn (identyczne wektory)
        """
        import re
        import pandas as pd

        df2 = df.copy()

        # 1) leak: one-hot target -> usuń bazę i siostry
        base = None
        if target_column in df2.columns:
            m = re.match(r"^(.+?)_([^_]+)$", str(target_column))
            if truthy_df_safe(m):
                vals = pd.Series(df2[target_column]).dropna().unique()
                if set(pd.Series(vals).astype(str)) <= {"0", "1"} or set(vals) <= {0, 1, 0.0, 1.0}:
                    base = m.group(1)
        if truthy_df_safe(base):
            drop_cols = []
            if base in df2.columns:
                drop_cols.append(base)
            drop_cols += [c for c in df2.columns if c.startswith(base + "_") and c != target_column]
            if truthy_df_safe(drop_cols):
                df2 = df2.drop(columns=[c for c in drop_cols if c in df2.columns], errors="ignore")

        # 2) stałe kolumny (poza targetem)
        const_cols = [c for c in df2.columns if c != target_column and df2[c].nunique(dropna=True) <= 1]
        if truthy_df_safe(const_cols):
            df2 = df2.drop(columns=const_cols, errors="ignore")

        # 3) daty -> cechy numeryczne
        def _parseable_datetime(s: pd.Series) -> bool:
            if s.dtype.name.startswith("datetime"):
                return True
            sample = s.dropna().astype(str).head(200)
            if len(sample) == 0:
                return False
            ok = 0
            for v in sample:
                try:
                    pd.to_datetime(v); ok += 1
                except Exception:
                    pass
            return ok / len(sample) >= 0.9

        for c in list(df2.columns):
            if c == target_column:
                continue
            try:
                if _parseable_datetime(df2[c]):
                    s = pd.to_datetime(df2[c], errors="coerce")
                    df2[c + "_year"]  = s.dt.year
                    df2[c + "_month"] = s.dt.month
                    df2[c + "_dow"]   = s.dt.dayofweek
                    df2 = df2.drop(columns=[c], errors="ignore")
            except Exception:
                pass

        # 4) duplikaty kolumn
        seen = {}
        dup = []
        for c in df2.columns:
            try:
                h = pd.util.hash_pandas_object(df2[c], index=False).sum()
                if h in seen and df2[c].equals(df2[seen[h]]):
                    if c != target_column:
                        dup.append(c)
                else:
                    seen[h] = c
            except Exception:
                continue
        if truthy_df_safe(dup):
            df2 = df2.drop(columns=dup, errors="ignore")

        return df2

        # === [NOWE] Regulator prędkości – dobór szybkich modeli i limitów ===
    def _apply_speed_governor(self, df_in: pd.DataFrame, ycol: str, problem_type: str, config: Dict):
        """
        Zwraca (df_out, overrides, msg):
        - df_out: ewentualnie dodatkowo zredukowana próbka dla bardzo dużych danych
        - overrides: narzucane ustawienia do train_model (jeśli wspierane przez trenera)
        - msg: krótka informacja diagnostyczna (lub None)
        """
        rows, cols = df_in.shape
        heavy = (rows > 12000) or (cols > 80)
        # aktualny tryb (jeśli już gdzieś ustawiasz) – zachowaj jeśli szybki
        strategy_in = (config.get('recommended_strategy') or
                    config.get('algorithm_selection') or
                    'balanced').lower()

        # Zestawy szybkich rodzin modeli
        fast_cls = [
            "LogisticRegression", "LinearSVC", "SGDClassifier",
            "DecisionTreeClassifier", "RandomForestClassifier", "ExtraTreesClassifier",
            "HistGradientBoostingClassifier", "LGBMClassifier", "XGBClassifier"
        ]
        fast_reg = [
            "Ridge", "ElasticNet", "SGDRegressor",
            "DecisionTreeRegressor", "RandomForestRegressor", "ExtraTreesRegressor",
            "HistGradientBoostingRegressor", "LGBMRegressor", "XGBRegressor"
        ]

        # „Kill list” na długie treningi (tylko informacyjnie; użyjemy pozytywnej listy fast*)
        slow_families = [
            "GaussianProcessClassifier", "GaussianProcessRegressor",
            "SVC", "SVR", "KNeighborsClassifier", "KNeighborsRegressor",
            "MLPClassifier", "MLPRegressor"
        ]

        # Domyślna strategia: na ciężkich danych wymuś fast_small
        strategy_out = strategy_in
        if truthy_df_safe(heavy) and strategy_in not in ("fast_small", "fast", "ultra_fast"):
            strategy_out = "fast_small"

        # Dodatkowy sampling dla bardzo dużych danych (np. do 5000)
        msg = None
        df_out = df_in
        if truthy_df_safe(heavy):
            cap = 5000 if problem_type.lower().startswith("klasyf") or "class" in problem_type.lower() else 8000
            # użyj dostępnego helpera _maybe_sample (dodałeś go wcześniej)
            try:
                df_out, msg = self._maybe_sample(
                    df_in, ycol, use_full=False, n_max=cap,
                    random_state=int(config.get('random_state', 42) or 42)
                )
            except Exception:
                pass

        # Narzuć szybkie ustawienia (jeśli Twój trainer je obsługuje – wstrzykniemy przez safe_kwargs)
        overrides = {
            "recommended_strategy": strategy_out,   # jeśli trainer czyta 'algorithm_selection' – podmienimy niżej
            "model_families": fast_cls if problem_type.lower().startswith("klasyf") or "class" in problem_type.lower() else fast_reg,
            "enable_hyperparameter_tuning": False,  # tuning off na dużych zbiorach
            "hpo_n_trials": 0,
            "early_stopping": True,
            "n_jobs": -1,
            "max_train_time": 180,                 # jeśli trainer wspiera; inaczej zostanie zignorowane przez safe_kwargs
            "use_full_dataset": False,             # pilnujmy, by nie wrócił pełny zbiór przypadkiem
            # sampling/stratify zostawiamy jak było, bo już próbkujemy wyżej
        }
        return df_out, overrides, msg


    # (Masz już to w poprzedniej iteracji – zostawiam tylko dla spójności.) 
    def _maybe_sample(self, df_in: pd.DataFrame, ycol: str, use_full: bool, n_max: int = 8000, random_state: int = 42):
        """Stratyfikowany sampling dla klasyfikacji; losowy dla regresji. Zwraca (df_out, msg)."""
        import pandas as pd
        if use_full or len(df_in) <= n_max:
            return df_in, None
        try:
            from backend.helpers.targeting import stratified_sample_df  # jeśli masz helpera
            out, msg = stratified_sample_df(df_in, ycol, n_max=n_max, random_state=random_state)
            return out, msg
        except Exception:
            y = df_in[ycol]
            if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
                out = df_in.sample(n=n_max, random_state=random_state)
                return out, f"Regresja: próbka {len(out)} z {len(df_in)} (fallback)."
            frac = n_max / len(df_in)
            parts = []
            for _, g in df_in.groupby(ycol):
                k = max(1, int(round(len(g) * frac)))
                parts.append(g.sample(n=min(k, len(g)), random_state=random_state))
            out = pd.concat(parts).sample(frac=1.0, random_state=random_state).head(n_max)
            return out, f"Klasyfikacja: stratyfikowana próbka {len(out)} z {len(df_in)} (fallback)."

    def _execute_training(self, df: pd.DataFrame, target_column: str, config: Dict):
        """
        Wykonaj trening z progress tracking, bezpiecznym stanem i szerszą konfiguracją.
        """
        import inspect, time, traceback
        import numpy as np
        import pandas as pd

        progress_container = st.container()
        with progress_container:
            st.markdown("### 🤖 Trening...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            st.session_state.training_in_progress = True
            try:
                status_text.text("🔄 Przygotowanie...")
                progress_bar.progress(10)

                problem_type = self.ml_trainer.detect_problem_type(df, target_column)

                # === [DOPISANE] sampling bazowy (zostawiam Twoją wcześniejszą logikę – tylko ją używam)
                use_full = bool(config.get('use_full_dataset', False)) or (len(df) <= 15000)
                df_to_train, base_msg = self._maybe_sample(
                    df, target_column, use_full,
                    n_max=8000,
                    random_state=int(config.get('random_state', 42) or 42)
                )
                if truthy_df_safe(base_msg):
                    st.info(f"{base_msg}")
                elif len(df_to_train) < len(df):
                    st.info(f"Duży zbiór danych ({len(df)}). Używam próbki {len(df_to_train)} dla szybszego treningu…")

                # minimalny cleaning (Twoja istniejąca metoda – zostawiam)
                df_to_train = self._clean_df_for_training(df_to_train, target_column)

                # === [DOPISANE] dodatkowy „governor” dla bardzo dużych zestawów
                df_to_train, overrides, speed_msg = self._apply_speed_governor(
                    df_to_train, target_column, problem_type, config
                )
                if truthy_df_safe(speed_msg):
                    st.info(speed_msg)
                # === [KONIEC DOPISKU] ===

                # Zbierz parametry zgodnie z rozszerzonym configiem (Twoje)
                train_kwargs = dict(
                    df=df,  # (oryginalny wpis – NIE USUWAM)
                    target_column=target_column,
                    train_size=config.get('train_size', 0.8),
                    cv_folds=int(config.get('cv_folds', 5) or 5),
                    cv_type=config.get('cv_type', None),
                    shuffle=bool(config.get('shuffle', True)),
                    random_state=int(config.get('random_state', 42) or 42),
                    group_column=config.get('group_column', None),

                    stratify=bool(config.get('stratify', True)),
                    class_weight=config.get('class_weight', 'none'),
                    sampling=config.get('sampling', 'none'),
                    sampling_ratio=float(config.get('sampling_ratio', 0.0) or 0.0),

                    use_full_dataset=bool(config.get('use_full_dataset', False)),
                    algorithm_selection=config.get('recommended_strategy', 'balanced'),
                    model_families=config.get('model_families', None),

                    optimization_metric=config.get('recommended_metric', None),
                    enable_hyperparameter_tuning=bool(config.get('enable_hyperparameter_tuning', False)),
                    hpo_n_trials=int(config.get('hpo_n_trials', 0) or 0),
                    n_jobs=int(config.get('n_jobs', -1) or -1),
                    max_train_time=int(config.get('max_train_time', 0) or 0),
                    early_stopping=bool(config.get('early_stopping', True)),
                    enable_ensemble=bool(config.get('enable_ensemble', False)),

                    num_imputer=config.get('num_imputer', 'median'),
                    cat_imputer=config.get('cat_imputer', 'most_frequent'),
                    scaling=config.get('scaling', 'none'),
                    encoding=config.get('encoding', 'one_hot'),
                    outliers=config.get('outliers', 'none'),
                    variance_threshold=float(config.get('variance_threshold', 0.0) or 0.0),
                    feature_selection=config.get('feature_selection', 'none'),
                    top_k_features=int(config.get('top_k_features', 0) or 0),

                    threshold_opt=config.get('threshold_opt', 'none'),
                    profit_opt=bool(config.get('profit_opt', False)),
                    value_tp=config.get('value_tp', None),
                    cost_fp=config.get('cost_fp', None),
                    cost_fn=config.get('cost_fn', None),
                )

                # Przefiltruj kwargs po sygnaturze trenera, żeby nie wysadzić nieobsługiwanymi parametrami (Twoje)
                sig = inspect.signature(self.ml_trainer.train_model)
                supported = set(sig.parameters.keys())
                safe_kwargs = {k: v for k, v in train_kwargs.items() if k in supported}

                # === [DOPISANE] wstrzyknij overrides + podmień df na df_to_train
                if 'df' in supported:
                    safe_kwargs['df'] = df_to_train
                if truthy_df_safe(overrides):
                    # mapowanie aliasu strategii
                    if 'algorithm_selection' in supported:
                        safe_kwargs['algorithm_selection'] = overrides.get('recommended_strategy',
                                                                        safe_kwargs.get('algorithm_selection'))
                    # lista rodzin modeli – jeśli wspierasz
                    if 'model_families' in supported and overrides.get('model_families'):
                        safe_kwargs['model_families'] = overrides['model_families']
                    # reszta flag (jeśli trener obsługuje)
                    for k in ('enable_hyperparameter_tuning', 'hpo_n_trials', 'n_jobs',
                            'max_train_time', 'early_stopping', 'use_full_dataset'):
                        if k in supported and k in overrides:
                            safe_kwargs[k] = overrides[k]
                # === [KONIEC DOPISKU] ===

                status_text.text("🎯 Trenowanie...")
                progress_bar.progress(35)

                results = self.ml_trainer.train_model(**safe_kwargs)

                progress_bar.progress(90)
                status_text.text("💾 Zapisywanie...")

                st.session_state.model_results = results
                st.session_state.target_column = target_column
                st.session_state.problem_type = problem_type
                st.session_state.model_trained = True
                st.session_state.analysis_complete = True
                st.session_state.training_in_progress = False

                progress_bar.progress(100)
                status_text.text("🎉 Zakończono!")

                # Wypisz najlepsze – korzystamy z tego co zwróci trener (Twoje)
                best_model = results.get('best_model', 'Unknown')
                if str(problem_type).lower().startswith("regres"):
                    score = results.get('r2') or results.get('rmse') or results.get('mae')
                    lbl = "R²" if 'r2' in results else ("RMSE" if 'rmse' in results else "MAE")
                    st.success(f"🏆 **{best_model}** | {lbl} = {_fmt_float_safe(score, 4)}")
                else:
                    # priorytet według metryki optymalizacji (Twoje)
                    pref = (config.get('recommended_metric') or '').lower()
                    candidates = {
                        'roc_auc': results.get('roc_auc'),
                        'f1': results.get('f1') or results.get('f1_weighted'),
                        'accuracy': results.get('accuracy'),
                        'precision': results.get('precision'),
                        'recall': results.get('recall'),
                        'pr_auc': results.get('pr_auc'),
                    }
                    score = candidates.get(pref) or results.get('roc_auc') or results.get('f1') or results.get('accuracy')
                    lbl = next((k.upper() for k, v in candidates.items() if v == score), "SCORE")
                    st.success(f"🏆 **{best_model}** | {lbl} = {_fmt_float_safe(score, 4)}")

                st.balloons()
                time.sleep(1.2)
                progress_bar.empty()
                status_text.empty()  # rerun removed

            except Exception as e:
                st.session_state.training_in_progress = False
                st.error(f"❌ Błąd treningu: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                with st.expander("🐛 Szczegóły błędu"):
                    st.code(traceback.format_exc())

    def _render_quick_results_preview(self):
        """
        Kompaktowy podgląd wyników (regresja / klasyfikacja) – prezentuje to,
        co najczęściej się przydaje i co zwrócił trainer.
        """
        results = st.session_state.model_results
        problem_type = st.session_state.problem_type

        st.markdown("#### 📊 Podgląd")

        if str(problem_type).lower().startswith("regres"):
            metrics = {
                "R²": results.get('r2'),
                "MAE": results.get('mae'),
                "RMSE": results.get('rmse')
            }
        else:
            metrics = {
                "ROC AUC": results.get('roc_auc'),
                "F1": results.get('f1') or results.get('f1_weighted'),
                "Accuracy": results.get('accuracy'),
                "Precision": results.get('precision'),
                "Recall": results.get('recall'),
            }

        # karty metryk
        shown = [(k, v) for k, v in metrics.items() if v is not None]
        if not truthy_df_safe(shown):
            st.caption("Brak zebranych metryk do podglądu.")
            return

        cols = st.columns(len(shown))
        for (name, val), col in zip(shown, cols):
            with col:
                st.metric(name, f"{_fmt_float_safe(val, 4)}")

# ==================== WYNIKI I WIZUALIZACJE ====================

    def render_results_page(self):
        """Strona wyników i wizualizacji"""
        if not st.session_state.get('model_trained', False):
            st.warning("⚠️ Najpierw wytrenuj model w sekcji 'Trening Modelu'")
            return

        st.header("📈 Wyniki i Zaawansowane Wizualizacje")
        results = st.session_state.model_results
        problem_type = st.session_state.get('problem_type', 'Regresja')

        self._render_feature_importance(results)
        st.divider()
        self._render_advanced_visualizations(results, problem_type)

    def _render_feature_importance(self, results: Dict):
        st.subheader("🎯 Analiza najważniejszych cech")

        if 'feature_importance' not in results or not isinstance(results['feature_importance'], pd.DataFrame):
            st.info("Brak danych o ważności cech")
            return

        importance_df = results['feature_importance'].copy()

        col1, col2 = st.columns(2)
        with col1:
            top_n = st.slider("Liczba cech:", 5, min(50, len(importance_df)), 15)
        with col2:
            chart_type = st.selectbox("Typ wykresu:", ["Bar poziomy", "Bar pionowy", "Waterfall"])

        top_features = importance_df.head(top_n)

        if chart_type == "Bar poziomy":
            fig = px.bar(top_features, x='importance', y='feature', orientation='h',
                         title=f"Top {top_n} najważniejszych cech",
                         color='importance', color_continuous_scale='viridis',
                         text='importance_pct')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        elif chart_type == "Bar pionowy":
            fig = px.bar(top_features, x='feature', y='importance',
                         title=f"Top {top_n} najważniejszych cech",
                         color='importance', color_continuous_scale='viridis',
                         text='importance_pct')
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_xaxes(tickangle=45)
        else:
            fig = go.Figure(go.Waterfall(
                name="Feature Importance", orientation="v",
                measure=["relative"] * len(top_features),
                x=top_features['feature'], y=top_features['importance'],
                text=top_features['importance_pct'].round(1).astype(str) + '%',
                textposition="outside",
                connector={"line": {"color": "rgb(63, 63, 63)"}}
            ))
            fig.update_layout(title=f"Waterfall - Top {top_n} cech")

        fig.update_layout(height=max(400, top_n * 25), showlegend=False, margin=dict(t=100, b=50))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔍 Interpretacja:")
        interpretation_df = top_features.copy()
        interpretation_df['Poziom'] = interpretation_df['importance_pct'].apply(
            lambda x: "🔥 Krytyczna" if x > 20 else "⚡ Wysoka" if x > 10 else "📊 Średnia" if x > 5 else "📈 Niska"
        )
        interpretation_df['Kumulatywny'] = interpretation_df['cumulative_importance'].round(1).astype(str) + '%'

        display_cols = ['feature', 'importance', 'importance_pct', 'Poziom', 'Kumulatywny']
        st.dataframe(
            interpretation_df[display_cols].rename(columns={
                'feature': 'Cecha', 'importance': 'Ważność', 'importance_pct': 'Ważność %'
            }),
            use_container_width=True, hide_index=True
        )

        cumsum_80 = (interpretation_df['cumulative_importance'] <= 80).sum()
        st.info(f"💡 Pierwsze {cumsum_80} cech wyjaśnia 80% ważności modelu.")

    def _render_advanced_visualizations(self, results: Dict, problem_type: str):
        st.subheader("🎨 Zaawansowane wizualizacje")

        with st.spinner("🎨 Generowanie..."):
            visualizations = self.ml_trainer.create_advanced_visualizations(results, problem_type)

        viz_tabs = st.tabs([
            "🏆 Porównanie", "🎯 Feature Analysis", "📊 Predykcje",
            "📈 Learning Curves", "🔍 Dodatkowe"
        ])

        with viz_tabs[0]:
            if 'model_comparison' in visualizations:
                st.plotly_chart(visualizations['model_comparison'], use_container_width=True)

        with viz_tabs[1]:
            if 'feature_importance_advanced' in visualizations:
                st.plotly_chart(visualizations['feature_importance_advanced'], use_container_width=True)

        with viz_tabs[2]:
            if 'predictions_plot' in visualizations:
                st.plotly_chart(visualizations['predictions_plot'], use_container_width=True)
            elif 'confusion_matrix' in visualizations:
                st.plotly_chart(visualizations['confusion_matrix'], use_container_width=True)

        with viz_tabs[3]:
            if 'learning_curves' in visualizations:
                st.plotly_chart(visualizations['learning_curves'], use_container_width=True)
            else:
                self._render_learning_curves_fallback(results, problem_type)

        with viz_tabs[4]:
            if problem_type == "Regresja" and 'predictions' in results:
                self._render_residual_analysis(results)
            elif str(problem_type).lower().startswith("klasy"):
                self._render_additional_classification(results)
            else:
                st.caption("ℹ️ Brak dodatkowych wizualizacji dla tego trybu.")

    def _render_residual_analysis(self, results: Dict):
        y_pred = np.asarray(results['predictions'])
        y_actual = np.asarray(results['actual'])
        residuals = y_actual - y_pred

        fig = make_subplots(rows=2, cols=2,
            subplot_titles=['Residuals vs Fitted', 'Histogram', 'Q-Q plot', 'Scale-Location'])

        fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'), row=1, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

        fig.add_trace(go.Histogram(x=residuals, name='Hist', showlegend=False), row=1, col=2)

        (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm", plot=None)
        fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Q-Q', showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=osm, y=slope * osm + intercept, mode='lines',
                                 name='Q-Q line', showlegend=False), row=2, col=1)

        sqrt_residuals = np.sqrt(np.abs(residuals))
        fig.add_trace(go.Scatter(x=y_pred, y=sqrt_residuals, mode='markers',
                                 name='Scale-Location', showlegend=False), row=2, col=2)

        fig.update_layout(height=800, title="🔍 Analiza residuals", margin=dict(t=100))
        st.plotly_chart(fig, use_container_width=True)

    def _render_learning_curves_fallback(self, results: Dict, problem_type: str):
        """Proxy krzywych uczenia, gdy trainer nie zwrócił 'learning_curves'."""
        try:
            import pandas as pd  # noqa: F401
            from sklearn.model_selection import learning_curve, StratifiedKFold, KFold
            from sklearn.linear_model import LogisticRegression, RidgeCV
            from sklearn.pipeline import make_pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.impute import SimpleImputer
        except Exception:
            st.caption("ℹ️ Krzywe uczenia wymagają scikit-learn.")
            return

        df = st.session_state.get("df_clean") or st.session_state.get("df")
        target = st.session_state.get("target_column")
        if df is None or not target or target not in df.columns:
            st.caption("ℹ️ Brak danych/targetu do policzenia krzywych uczenia.")
            return

        # Bezpieczeństwo i szybkość
        df_local = df.copy()
        if len(df_local) > 5000:
            df_local = df_local.sample(5000, random_state=42)

        X = df_local.drop(columns=[target])
        y = df_local[target]

        # Prosty, odporny preprocesor
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        pre = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                                    OneHotEncoder(handle_unknown="ignore")), cat_cols),
            ],
            remainder="drop",
            sparse_threshold=0.3,
        )

        # Domyślny estimator (gdy trener nie udostępnił gotowego)
        if str(problem_type).lower().startswith("regres"):
            estimator = RidgeCV(alphas=(0.1, 1.0, 10.0))
            scoring = "r2"
            cv = KFold(n_splits=min(3, max(2, len(df_local)//5)), shuffle=True, random_state=42)
        else:
            estimator = LogisticRegression(max_iter=1000, n_jobs=None)
            # spróbuj dobrać rozsądny scoring
            n_classes = pd.Series(y).nunique()
            scoring = "roc_auc_ovr" if n_classes > 2 else "roc_auc"
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        pipe = make_pipeline(pre, estimator)

        try:
            import numpy as np
            import plotly.graph_objects as go

            train_sizes, train_scores, val_scores = learning_curve(
                pipe, X, y,
                cv=cv, scoring=scoring,
                n_jobs=1,  # stabilniej pod Streamlit
                train_sizes=np.linspace(0.1, 1.0, 5),
                shuffle=True, random_state=42
            )
            train_mean = train_scores.mean(axis=1)
            val_mean = val_scores.mean(axis=1)

            fig = go.Figure()
            fig.add_scatter(x=train_sizes, y=train_mean, mode="lines+markers", name="Train")
            fig.add_scatter(x=train_sizes, y=val_mean, mode="lines+markers", name="CV")
            fig.update_layout(
                title="📈 Learning Curves (fallback)",
                xaxis_title="Liczba próbek w treningu",
                yaxis_title=("R²" if scoring == "r2" else scoring.upper()),
                height=450,
                margin=dict(t=80)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("ℹ️ Wykres policzony lokalnie na uproszczonym estimatorze.")
        except Exception as e:
            st.caption(f"ℹ️ Nie udało się policzyć krzywych uczenia: {e}")

    def _render_additional_classification(self, results: Dict):
        """Dodatkowe wizualizacje dla klasyfikacji: balans klas + (opcjonalnie) ROC/PR."""
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

        y_true = results.get("actual")
        y_pred = results.get("predictions")
        if y_true is None:
            st.caption("ℹ️ Brak 'actual' w wynikach – nie mogę narysować dodatkowych wykresów.")
            return

        # Balans klas
        vc = pd.Series(y_true).value_counts().sort_index()
        st.markdown("##### ⚖️ Balans klas")
        st.plotly_chart(px.bar(vc, labels={'index': 'Klasa', 'value': 'Liczność'},
                            title="Rozkład klas (y_true)"), use_container_width=True)

        # Spróbuj znaleźć prawdopodobieństwa
        proba = _pick_proba_like(results)
        if proba is None:
            st.caption("ℹ️ Brak prawdopodobieństw w wynikach – ROC/PR pominięte.\n"
                    "Dodaj w trainerze zwracanie proba pod kluczem 'probabilities' / 'y_proba'.")
            return

        proba = np.asarray(proba)
        y_true = np.asarray(y_true)

        try:
            if proba.ndim == 1 or proba.shape[1] == 1:
                # binarka: proba klasy 1
                p1 = proba.ravel()
                fpr, tpr, _ = roc_curve(y_true, p1)
                roc_auc = auc(fpr, tpr)
                pr_p, pr_r, _ = precision_recall_curve(y_true, p1)
                ap = average_precision_score(y_true, p1)

                fig_roc = go.Figure()
                fig_roc.add_scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={roc_auc:.3f})")
                fig_roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash"))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=380)
                st.plotly_chart(fig_roc, use_container_width=True)

                fig_pr = go.Figure()
                fig_pr.add_scatter(x=pr_r, y=pr_p, mode="lines", name=f"PR (AP={ap:.3f})")
                fig_pr.update_layout(title="Precision–Recall Curve", xaxis_title="Recall", yaxis_title="Precision", height=380)
                st.plotly_chart(fig_pr, use_container_width=True)
            else:
                # multiclass – pokazujemy mikro-średnią PR jako szybki podgląd
                y_bin = pd.get_dummies(y_true)
                pr_p, pr_r, _ = precision_recall_curve(y_bin.values.ravel(), proba.ravel())
                ap = average_precision_score(y_bin.values, proba, average="micro")

                fig_pr = go.Figure()
                fig_pr.add_scatter(x=pr_r, y=pr_p, mode="lines", name=f"PR micro (AP={ap:.3f})")
                fig_pr.update_layout(title="Precision–Recall (micro-avg)", xaxis_title="Recall", yaxis_title="Precision", height=380)
                st.plotly_chart(fig_pr, use_container_width=True)

        except Exception as e:
            st.caption(f"ℹ️ Nie udało się zbudować ROC/PR: {e}")

# ==================== REKOMENDACJE ====================

    def render_recommendations_page(self):
        if not st.session_state.get('model_trained', False):
            st.warning("⚠️ Najpierw wytrenuj model")
            return

        st.header("💡 Zaawansowane Rekomendacje i Wnioski")

        results = st.session_state.model_results
        target_column = st.session_state.target_column
        problem_type = st.session_state.problem_type

        with st.spinner("🤖 Generowanie rekomendacji..."):
            recommendations = self._generate_recommendations(results, target_column, problem_type)

        rec_tabs = st.tabs([
            "🎯 Kluczowe Wnioski", "📈 Rekomendacje", "⚠️ Ograniczenia",
            "🔮 Następne Kroki", "💼 Wdrożenie"
        ])

        with rec_tabs[0]:
            st.markdown("### 🎯 Najważniejsze wnioski")
            st.success(recommendations['key_insights'])
            self._render_business_metrics(results, problem_type)

        with rec_tabs[1]:
            st.markdown("### 📈 Konkretne rekomendacje")
            for i, rec in enumerate(recommendations['action_items'], 1):
                st.info(f"**{i}.** {rec}")

        with rec_tabs[2]:
            st.markdown("### ⚠️ Ograniczenia i ryzyka")
            st.warning(recommendations['limitations'])

        with rec_tabs[3]:
            st.markdown("### 🔮 Następne kroki")
            for i, step in enumerate(recommendations['next_steps'], 1):
                st.write(f"**{i}.** {step}")

        with rec_tabs[4]:
            st.markdown("### 💼 Wdrożenie")
            self._render_implementation_guide()

    def _generate_recommendations(self, results: Dict, target_column: str, problem_type: str) -> Dict:
        try:
            from backend.ai_integration import AIDescriptionGenerator
            ai_gen = AIDescriptionGenerator()
            return ai_gen.generate_recommendations(results, target_column, problem_type)
        except Exception:
            return self._fallback_recommendations(results, target_column, problem_type)

    def _fallback_recommendations(self, results: Dict, target_column: str, problem_type: str) -> Dict:
        if problem_type == "Regresja":
            r2 = results.get('r2', 0)
            performance = "doskonałą" if r2 > 0.9 else "bardzo dobrą" if r2 > 0.8 else "dobrą"
        else:
            acc = results.get('accuracy', 0)
            performance = "doskonałą" if acc > 0.95 else "bardzo dobrą" if acc > 0.9 else "dobrą"

        return {
            'key_insights': f"Model osiągnął {performance} wydajność dla {target_column}.",
            'action_items': [
                "Monitoruj metryki w produkcji",
                "Zbieraj feedback użytkowników",
                "Regularnie aktualizuj model",
                "Ustaw alerty dla spadków wydajności",
                "A/B testy przed wdrożeniem"
            ],
            'limitations': "Model bazuje na danych historycznych.",
            'next_steps': [
                "Walidacja na świeżych danych",
                "Integracja z systemami",
                "Setup monitoringu",
                "Dokumentacja",
                "Plan retrainingu"
            ]
        }

    def _render_business_metrics(self, results: Dict, problem_type: str):
        st.markdown("#### 💰 Biznes")
        col1, col2, col3 = st.columns(3)

        if problem_type == "Regresja":
            with col1:
                r2 = results.get('r2', 0)
                st.metric("Przewidywalność", f"{_fmt_float_safe(r2*100, 1)}%")
            with col2:
                mae = results.get('mae', 0)
                st.metric("Średni błąd", f"{_fmt_float_safe(mae, 2)}")
            with col3:
                confidence = "Wysoka" if r2 > 0.8 else "Średnia" if r2 > 0.6 else "Niska"
                st.metric("Pewność", confidence)
        else:
            with col1:
                acc = results.get('accuracy', 0)
                st.metric("Dokładność", f"{_fmt_float_safe(acc*100, 1)}%")
            with col2:
                f1 = results.get('f1_weighted') or results.get('f1', 0)
                st.metric("F1-Score", f"{_fmt_float_safe(f1, 3)}")
            with col3:
                value = "Bardzo wysoka" if acc > 0.9 else "Wysoka" if acc > 0.8 else "Umiarkowana"
                st.metric("Wartość", value)

    def _render_implementation_guide(self):
        steps = [
            {'step': '1. Walidacja', 'desc': 'Test na świeżych danych', 'time': '1-2 tyg', 'res': 'Biznes + DS'},
            {'step': '2. Integracja', 'desc': 'Integracja systemów', 'time': '2-4 tyg', 'res': 'Dev + DevOps'},
            {'step': '3. Monitoring', 'desc': 'Alerty i dashboardy', 'time': '1 tyg', 'res': 'MLOps + DE'},
            {'step': '4. Szkolenie', 'desc': 'Szkolenie użytkowników', 'time': '1 tyg', 'res': 'Team szkoleniowy'},
            {'step': '5. Optymalizacja', 'desc': 'Feedback i iteracje', 'time': 'Ciągły', 'res': 'Cały team'}
        ]

        for s in steps:
            with st.expander(f"📋 {s['step']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Opis:** {s['desc']}")
                with col2:
                    st.write(f"**Timeline:** {s['time']}")
                with col3:
                    st.write(f"**Zasoby:** {s['res']}")


# ==================== DOKUMENTACJA ====================

    def render_documentation_page(self):
        st.header("📚 Dokumentacja i Pomoc")

        tab1, tab2, tab3, tab4 = st.tabs(["🚀 Quick Start", "📖 Przewodnik", "❓ FAQ", "🔧 Tech"])

        with tab1:
            st.markdown("""
            ## 🚀 Szybki Start

            ### Krok 1: Wczytaj dane
            📊 Analiza Danych → CSV/JSON lub przykładowy dataset

            ### Krok 2: Oczyść (opcjonalnie)
            🧹 Auto-clean → AI automatycznie oczyści

            ### Krok 3: Trenuj
            🤖 Trening Modelu → Wybierz target → AI Plan → Zastosuj plan → ROZPOCZNIJ TRENING

            ### Krok 4: Wyniki
            📈 Wyniki i Wizualizacje + 💡 Rekomendacje
            """)

        with tab2:
            st.markdown("""
            ## 📖 Szczegółowy Przewodnik

            ### Analiza Danych
            - Upload: CSV, JSON (max 200MB)
            - Auto-clean: duplikaty, outliers, NaN, daty
            - EDA: 5 zakładek (Info, Stats, Heatmapa, Rozkłady, AI)

            ### Trening
            - AI Plan: inteligentny system dobiera konfigurację
            - 18+ algorytmów regresji, 20+ klasyfikacji
            - XGBoost, LightGBM, CatBoost (opcjonalne)
            - Hyperparameter tuning (auto)
            - Ensemble mode

            ### Wyniki
            - Feature Importance wizualizacje
            - Porównanie wszystkich modeli
            - Learning curves
            - Residual analysis (regresja)

            ### Rekomendacje
            - AI Insights biznesowe
            - Plan wdrożenia
            - Analiza ryzyka
            """)

        with tab3:
            st.markdown("""
            ## ❓ FAQ

            **Q: Jakie pliki?**
            A: CSV (różne separatory), JSON. Max 200MB.

            **Q: Co to AI Plan?**
            A: AI analizuje dane i dobiera automatycznie train/test split, strategię algorytmów, metrykę, tuning, ensemble.

            **Q: Mogę dostosować?**
            A: Tak! "🔧 Dostosuj ręcznie" → wszystkie parametry.

            **Q: Gdzie wyniki?**
            A: Po treningu: krótkie podsumowanie + pełne w 📈 Wyniki; rekomendacje w 💡 Rekomendacje

            **Q: Eksport?**
            A: Sidebar → 💾 Eksport: JSON, CSV, HTML, TXT, PKL, ZIP (wszystko)
            """)

        with tab4:
            st.markdown("""
            ## 🔧 Tech

            ### Architektura
            Frontend (Streamlit) → Core Services → Backend (ML, EDA, AI, Upload, Cache, Monitoring)

            ### Komponenty
            - MLModelTrainer: 40+ algorytmów
            - EDAAnalyzer: Auto-clean + reports
            - AIDescriptionGenerator: AI opisy
            - SmartCache: 3-level (session/memory/disk)
            - HealthMonitor: monitoring

            ### Bezpieczeństwo
            - Rate limiting
            - Data validation
            - Credential Manager
            - Smart error handling

            ### Performance
            - Async training
            - Smart caching
            - Auto sampling (>15k rows)
            - Progress tracking
            """)


# ==================== MAIN ====================

def main():
    """Entry point"""
    try:
        app = MainApp()
        app.run()
    except Exception as e:
        st.error(f"❌ Krytyczny błąd: {str(e)}")
        st.info("🔄 Odśwież stronę")
        with st.expander("🐛 Szczegóły"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

# ========== POMOCNICZE FLOW (poza klasą — auto-hook kontrolowany przez session_state) ==========

def _render_training_ai_flow(df: pd.DataFrame, target: str):
    if not st.session_state.get('model_trained', False) and not st.session_state.get('training_in_progress', False):
        st.markdown("## 🧪 Training (AI-Driven)")

        # 0) Recommender + kontrolki
        if "ai_recommender" not in st.session_state:
            st.session_state.ai_recommender = AIRecommender()
        apply, train = render_training_controls()

        # 1) Zbuduj/odśwież plan i rekomendacje
        if apply or st.session_state.get("ai_plan") is None:
            plan = st.session_state.ai_recommender.build_dataprep_plan(df, target)
            recs = st.session_state.ai_recommender.build_training_recommendations(df, target)
            st.session_state.ai_plan = plan
            st.session_state.ai_recs = recs

        # 2) Pokaż plan + zaawansowane
        if st.session_state.get("ai_plan") and st.session_state.get("ai_recs"):
            render_ai_recommendations(st.session_state.ai_plan, st.session_state.ai_recs)
            st.session_state.ai_recs, _ = render_advanced_overrides(st.session_state.ai_recs)
            st.session_state['ai_speed_mode'] = st.session_state.ai_recs.get('speed_mode', 'balanced')

            # 2a) AUTO DATA PREP przy pierwszym wejściu
            if 'df_prepared' not in st.session_state:
                with st.status("🔧 Automatyczne przygotowanie danych (AI)...", expanded=True) as s2:
                    dfp, log = apply_ai_dataprep(df, target, st.session_state.ai_plan)
                    st.session_state['df_prepared'] = dfp
                    st.session_state['prep_log'] = log
                    for ev in log:
                        st.write(f"• **{ev.get('name','Step')}** — {ev.get('detail','')}")
                    s2.update(label="Przygotowanie zakończone", state="complete")

        # 3) Trening — tylko jeśli kliknięto
        if truthy_df_safe(train) and st.session_state.get("ai_plan") and st.session_state.get("ai_recs"):
            with st.status("Uruchamiam pipeline treningowy...", expanded=True) as status:
                # Krok 1
                st.write("➡️ **Krok 1:** AI Data Prep")
                dfp = st.session_state.get('df_prepared')
                if dfp is None:
                    dfp, log = apply_ai_dataprep(df, target, st.session_state.ai_plan)
                    st.session_state['df_prepared'] = dfp
                    st.session_state['prep_log'] = log
                else:
                    log = st.session_state.get('prep_log', [])
                for ev in log:
                    st.write(f"- {ev['name']}: {ev['detail']}")

                # Krok 2: CV i wybór modelu
                st.write("➡️ **Krok 2:** Cross-Validation i wybór modelu")
                def _progress(msg):
                    st.write(msg)

                ensembles = {}
                best_model = None
                results_df, best = train_multi_models(dfp, target, st.session_state.ai_recs, progress_cb=_progress)

                # Krok 3: Refit i Ensembles
                st.write("➡️ **Krok 3:** Refit najlepszego modelu na całości")
                st.write("➡️ **Krok 3b:** Ensembles (Voting/Stacking)")
                if truthy_df_safe(best):
                    try:
                        from backend.ml_integration import _lazy_imports
                        glb = _lazy_imports()
                        Model = glb.get(best.get('model'))
                        if Model is not None:
                            X = dfp.drop(columns=[target])
                            y = dfp[target]
                            best_model = Model(**(best.get('params', {}) or {}))
                            best_model.fit(X, y)
                            st.success(f"Model {best['model']} zrefitowany na całości danych.")
                        else:
                            st.warning(f"Nie znaleziono klasy modelu: {best.get('model')}")

                        # Ensembles z top modeli
                        try:
                            top_models = results_df.to_dict(orient="records") if (results_df is not None and not results_df.empty) else []
                            ensembles = fit_ensembles(
                                X, y, top_models,
                                st.session_state.get('ai_speed_mode', 'balanced'),
                                st.session_state.ai_recs.get('problem', 'regression')
                            ) or {}
                            for en_name in ensembles.keys():
                                st.success(f"Zbudowano ensemble: {en_name}")

                            # Szybki 3-fold check dla ensembles
                            if truthy_df_safe(ensembles):
                                try:
                                    qdf = evaluate_models_quick(
                                        X, y, ensembles,
                                        st.session_state.ai_recs.get('problem', 'regression'),
                                        folds=3
                                    )
                                    if truthy_df_safe(qdf):
                                        st.write("**Szybki 3-fold check (ensembles):**")
                                        st.dataframe(qdf, use_container_width=True)
                                except Exception as ee2:
                                    st.info(f"Brak szybkiej oceny ensemble: {ee2}")
                        except Exception as ee:
                            st.info(f"Nie zbudowano ensemble: {ee}")
                    except Exception as e:
                        st.warning(f"Nie udało się zrefitować: {e}")
                else:
                    st.info("Brak najlepszego modelu (best == None).")

                # Krok 4: Artefakty + eksporty + raport
                st.write("➡️ **Krok 4:** Zapis artefaktów")
                artifacts_dir = save_artifacts(
                    best_model, dfp, target,
                    st.session_state.ai_plan,
                    st.session_state.ai_recs,
                    results_df if results_df is not None else None,
                    ensembles if ensembles else None
                )

                try:
                    if best_model is not None:
                        Xs = dfp.drop(columns=[target]).iloc[:5]
                        onnx_path = os.path.join(artifacts_dir, 'model.onnx')
                        pmml_path = os.path.join(artifacts_dir, 'model.pmml')
                        onnx_ok = exporters.export_onnx(best_model, Xs, onnx_path)
                        pmml_ok = exporters.export_pmml(best_model, Xs, pmml_path)
                        if onnx_ok: st.caption('Wyeksportowano ONNX: model.onnx')
                        if pmml_ok: st.caption('Wyeksportowano PMML: model.pmml')
                    else:
                        st.caption('Pominięto eksporty (brak best_model).')
                except Exception as e:
                    st.info(f'Eksporty nieudane: {e}')

                try:
                    rep_path = generate_training_report(
                        artifacts_dir,
                        results_df if results_df is not None else None,
                        best if best is not None else None,
                        st.session_state.ai_plan,
                        st.session_state.ai_recs
                    )
                    st.caption(f'Raport zapisany: {rep_path}')
                except Exception as e:
                    st.info(f'Nie udało się wygenerować raportu: {e}')

                st.info(f"Artefakty zapisane w: {artifacts_dir}")

                # Krok 5: Podsumowanie
                st.write("➡️ **Krok 5:** Podsumowanie")
                if results_df is not None and not results_df.empty and best:
                    st.dataframe(results_df, use_container_width=True)
                    st.success(f"🏆 Najlepszy: {best['model']} — {best['metric']}={_fmt_float_safe(best['cv_mean'], 5)}")
                elif results_df is not None and not results_df.empty:
                    st.dataframe(results_df, use_container_width=True)
                    st.warning("Wyniki są dostępne, ale brak obiektu 'best'.")
                else:
                    st.warning("Brak wyników — żaden z modeli nie został przetrenowany.")

                status.update(label="Zakończono", state="complete")


# AI Training Flow (auto-hook): best-effort rendering when data and target are known
try:
    if st.session_state.get('auto_ai_training_flow', True):
        df = st.session_state.get('uploaded_df') or st.session_state.get('df')
        target = st.session_state.get('target') or st.session_state.get('target_column') or st.session_state.get('y_col')
        active_tab = st.session_state.get('active_tab') or st.session_state.get('current_tab') or st.session_state.get('current_page')
        if df is not None and target and (active_tab in ('Training','Szkolenie','Model Training', None)):
            _render_training_ai_flow(df, target)
except Exception:
    pass


# (opcjonalne) demo inferencji
import io
def _render_inference_demo(model, df_prepared: pd.DataFrame, target: str):
    st.markdown("### 🔍 Szybki test wytrenowanego modelu (biznesowo)")
    if model is None or df_prepared is None:
        st.info("Brak modelu lub danych.")
        return
    X = df_prepared.drop(columns=[target])
    # 1) Predict on a few sample rows
    st.write("**Podgląd predykcji (5 losowych wierszy):**")
    sample = X.sample(n=min(5, len(X)), random_state=42) if len(X)>0 else X.head(5)
    if not sample.empty:
        preds = model.predict(sample)
        st.dataframe(pd.DataFrame({"index": sample.index, "prediction": preds}), use_container_width=True)
    # 2) Upload CSV for batch predictions
    up = st.file_uploader("Wgraj CSV do predykcji (kolumny zgodne z danymi po przygotowaniu)", type=["csv"], key="pred_csv")
    if up is not None:
        try:
            dfu = pd.read_csv(up)
            preds = model.predict(dfu)
            out = dfu.copy()
            out['prediction'] = preds
            st.success("Predykcje wykonane. Poniżej wynik:")
            st.dataframe(out.head(50), use_container_width=True)
            buf = io.BytesIO()
            out.to_csv(buf, index=False)
            st.download_button("Pobierz wyniki CSV", data=buf.getvalue(), file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.warning(f"Błąd predykcji: {e}")
    st.markdown("#### Jak użyć modelu poza aplikacją (przykład Python)")
    st.code("""
import pandas as pd
from joblib import load
from backend.runtime_preprocessor import apply_plan

# wczytaj dane surowe
df_raw = pd.read_csv("your_data.csv")

# wczytaj plan data-prep
df_plan = pd.read_json("artifacts/last_run/plan.json")  # lub konkretna ścieżka z timestampu

# przygotuj dane jak w aplikacji
df_prepared = apply_plan(df_raw, target="{target}", plan=df_plan)
X = df_prepared.drop(columns=["{target}"])

# wczytaj model
model = load("artifacts/last_run/model.joblib")

# predykcja
preds = model.predict(X)
print(preds[:10])
""".replace("{target}", "TARGET_NAME"))