# -*- coding: utf-8 -*-
"""
THE MOST IMPORTANT VARIABLES - Advanced ML Platform v2.0 Pro
Marksio AI Solutions
"""

from __future__ import annotations

# === Standard libs
import os
import io
import time
import json
import asyncio
import shutil
import warnings
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

warnings.filterwarnings("ignore")

# === Third-party
import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# === Backend/Frontend modules (Twoja architektura)
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
st.set_page_config(
    page_title="The Most Important Variables",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        if USE_CONTAINER:
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
        Sidebar + leniwe włączanie narzędzi (bez snapshotów).
        """
        with st.sidebar:
            selected_page = self.ui.render_sidebar()

            st.markdown("---")
            st.caption("🛠️ Narzędzia")

            st.checkbox("📊 Health Dashboard", key="show_health")
            st.checkbox("🛠️ Admin Panel", key="show_admin")

            if st.session_state.get("show_health"):
                with st.expander("Health Dashboard", expanded=True):
                    try:
                        with st.spinner("Ładowanie…"):
                            self.health_monitor.render_health_dashboard()
                    except Exception:
                        st.info("Health dashboard chwilowo niedostępny.")

            if st.session_state.get("show_admin"):
                with st.expander("Admin Panel", expanded=True):
                    try:
                        with st.spinner("Ładowanie…"):
                            self.monitoring.render_admin_panel()
                    except Exception:
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
        if handler:
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

    def _render_data_upload_section(self):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("🔄 Wczytaj swoje dane")
            uploaded_file = st.file_uploader("Wybierz plik CSV lub JSON", type=["csv", "json"])
            if uploaded_file:
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
            st.rerun()
        except Exception as e:
            st.error(f"❌ Błąd: {str(e)}")

    def _handle_sample_dataset(self, dataset_name: str):
        df = self.file_handler.load_sample_dataset(dataset_name)
        self._load_dataset(df)
        st.success(f"✅ Załadowano! Kształt: {df.shape}")
        st.rerun()

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
            status_placeholder.empty()
            st.rerun()
        except Exception as e:
            st.error(f"❌ Błąd: {e}")
            progress_placeholder.empty()
            status_placeholder.empty()

    def _render_eda_section(self):
        st.subheader("🔍 Eksploracyjna Analiza Danych")
        df = st.session_state.get("df_clean", st.session_state.df)

        # sanity dla dtype (np. CategoricalDtype)
        df = self._sanitize_dtypes_for_app(df)

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            [
                "📋 Podstawowe Info",
                "📊 Statystyki",
                "🔥 Heatmapa",
                "📈 Rozkłady",
                "🎯 AI Insights",
            ]
        )

        with tab1:
            self._render_basic_info_tab(df)

        with tab2:
            # Twoja karta statystyk
            self._render_statistics_tab(df)
            # — Ultra Add-ons — w obrębie karty Statystyki
            st.markdown("---")
            self._render_ultra_targeting_profit()
            st.markdown("---")
            self._render_ultra_pdp_ice()
            st.markdown("---")
            self._render_ultra_report()

        with tab3:
            self._render_heatmap_tab(df)

        with tab4:
            self._render_distributions_tab(df)

        with tab5:
            self._render_ai_insights_tab(df)

    def _render_basic_info_tab(self, df: pd.DataFrame):
        st.markdown("#### 🧾 Informacje ogólne")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Wierszy", f"{df.shape[0]:,}")
            st.metric("Kolumn", df.shape[1])
        with col2:
            st.metric("Brakujące", int(df.isnull().sum().sum()))
            st.metric("Numeryczne", len(df.select_dtypes(include=[np.number]).columns))

        st.subheader("🏷️ Typy danych")
        dtypes_info = df.dtypes.value_counts()
        fig = px.pie(
            values=dtypes_info.values,
            names=[str(d) for d in dtypes_info.index],
            title="Rozkład typów danych",
            hole=0.3,
        )
        fig.update_layout(margin=dict(t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🏷️ Opisy kolumn (AI)")
        try:
            column_descriptions = self.eda.generate_column_descriptions(df)
        except Exception:
            column_descriptions = {}

        if column_descriptions:
            for col, desc in column_descriptions.items():
                with st.expander(f"📊 {col}"):
                    st.write(f"**Opis:** {desc}")
                    st.write(f"**Typ:** {df[col].dtype}")
                    if pd.api.types.is_numeric_dtype(df[col]):
                        stats_col = df[col].describe()
                        st.write(
                            f"**Min:** {_fmt_float_safe(stats_col.get('min', np.nan), 3)}, "
                            f"**Max:** {_fmt_float_safe(stats_col.get('max', np.nan), 3)}, "
                            f"**Średnia:** {_fmt_float_safe(stats_col.get('mean', np.nan), 3)}"
                        )
                    else:
                        mode_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                        st.write(
                            f"**Unikalne:** {df[col].nunique()}, "
                            f"**Najczęstsza:** {mode_val}"
                        )
        else:
            st.caption("Brak opisów AI lub moduł niedostępny.")

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
        if corr_pairs:
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
            if not meta_ultra:
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
            if not Xte_json:
                st.caption("Brak Xte_sample_json w meta — najpierw uruchom pipeline Ultra.")
                return

            Xte_s = _pd.read_json(Xte_json, orient="split")
            num_cols = Xte_s.select_dtypes(include=[_np.number]).columns.tolist()
            if not num_cols:
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

        # 2) Info o algorytmach
        self._render_algorithms_info()
        st.divider()

        # 3) Wybór targetu
        target_column = self._render_target_selection(df)
        if not target_column:
            return
        st.divider()

        # 4) AI plan (POPRAWIONE)
        ai_plan = self._render_ai_training_plan(df, target_column)
        if not isinstance(ai_plan, dict):
            ai_plan = {'enable_ensemble': False, 'train_size': 0.8, 'cv_folds': 5}
        st.session_state['ai_plan'] = ai_plan
        st.divider()

        # 5) Finalna konfiguracja (AI / ręczna)
        final_config = self._render_final_configuration(df, target_column, ai_plan)
        st.divider()

        # 6) Start treningu + szybki podgląd
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
            suggested = self.ml_trainer.suggest_target_column(df)
            target_options = list(df.columns)
            default_idx = target_options.index(suggested) if suggested in target_options else 0
            target_column = st.selectbox(
                "Zmienna docelowa (target)",
                target_options,
                index=default_idx,
                help=f"💡 AI sugeruje: {suggested}"
            )

        with col2:
            target_info = self._analyze_target_column(df, target_column)
            st.info(f"🎯 **AI sugeruje:** {suggested}")
            st.metric("Unikalne", target_info['unique_count'])
            st.metric("Typ", target_info['problem_type'])

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
        with st.expander("🤖 AI Plan Treningu (kliknij aby rozwinąć)", expanded=False):
            st.markdown("### 🧠 AI analizuje dane...")
            with st.spinner("🔮 Analiza..."):
                if 'ai_training_plan' not in st.session_state or st.session_state.get('last_target') != target_column:
                    ai_plan = self._generate_ai_plan(df, target_column)
                    st.session_state.ai_training_plan = ai_plan
                    st.session_state.last_target = target_column
                else:
                    ai_plan = st.session_state.ai_training_plan

            st.success("✅ Plan wygenerowany!")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### 📊 Podział danych")
                st.info(f"**Train:** {_fmt_float_safe(ai_plan['train_size'], 2)}")
                st.caption(ai_plan['train_size_reason'])
                st.markdown("#### 🔄 Walidacja")
                st.info(f"**CV folds:** {ai_plan['cv_folds']}")
                st.caption(ai_plan['cv_reason'])

            with col2:
                st.markdown("#### 🎯 Strategia")
                st.info(f"**Algorytmy:** {ai_plan['recommended_strategy']}")
                st.caption(ai_plan['strategy_reason'])
                st.markdown("#### 📈 Metryka")
                st.info(f"**Metryka:** {ai_plan['recommended_metric']}")
                st.caption(ai_plan['metric_reason'])

            with col3:
                st.markdown("#### ⚡ Optymalizacje")
                st.info(f"**Tuning:** {'✅' if ai_plan['enable_hyperparameter_tuning'] else '❌'}")
                st.caption(ai_plan['tuning_reason'])
                st.info(f"**Ensemble:** {'✅' if ai_plan['enable_ensemble'] else '❌'}")
                st.caption(ai_plan['ensemble_reason'])

            if ai_plan.get('special_considerations'):
                st.markdown("#### 💡 Uwagi:")
                for consideration in ai_plan['special_considerations']:
                    st.warning(f"• {consideration}")

            st.divider()
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                if st.button("✨ Zastosuj cały plan AI", type="primary", use_container_width=True):
                    st.session_state.ai_config_applied = True
                    st.session_state.final_config = ai_plan
                    st.success("✅ Plan zastosowany!")
                    st.rerun()
            with col_btn2:
                if st.button("🔧 Dostosuj ręcznie", use_container_width=True):
                    st.session_state.ai_config_applied = False
                    st.info("👇 Dostosuj poniżej")

        return ai_plan

    def _generate_ai_plan(self, df: pd.DataFrame, target_column: str) -> Dict:
        n_samples, n_features = df.shape
        problem_type = self.ml_trainer.detect_problem_type(df, target_column)
        target_analysis = self._analyze_target_column(df, target_column)

        plan: Dict[str, Any] = {}

        # Train size
        if n_samples < 1000:
            plan['train_size'] = 0.85
            plan['train_size_reason'] = "Mały zbiór — więcej danych do treningu."
        elif n_samples < 10000:
            plan['train_size'] = 0.8
            plan['train_size_reason'] = "Standardowy podział 80/20."
        else:
            plan['train_size'] = 0.75
            plan['train_size_reason'] = "Duży zbiór — więcej danych na walidację."

        # CV
        if n_samples < 500:
            plan['cv_folds'] = 3
            plan['cv_reason'] = "Mało danych — 3-fold CV."
        elif n_samples < 5000:
            plan['cv_folds'] = 5
            plan['cv_reason'] = "Balans — 5-fold CV."
        else:
            plan['cv_folds'] = 10
            plan['cv_reason'] = "Dużo danych — 10-fold CV."

        # Strategy
        if n_samples < 1000:
            plan['recommended_strategy'] = 'fast'
            plan['strategy_reason'] = "Szybkie algorytmy sprawdzą się najlepiej."
        elif n_features > n_samples * 0.8:
            plan['recommended_strategy'] = 'advanced'
            plan['strategy_reason'] = "Wysoka wymiarowość — metody zaawansowane."
        else:
            plan['recommended_strategy'] = 'all'
            plan['strategy_reason'] = "Sprawdź szeroki wachlarz algorytmów."

        # Metric
        if problem_type == "Klasyfikacja":
            if target_analysis['unique_count'] == 2:
                plan['recommended_metric'] = 'roc_auc'
                plan['metric_reason'] = "Klasyfikacja binarna — ROC AUC."
            else:
                plan['recommended_metric'] = 'f1_weighted'
                plan['metric_reason'] = "Wiele klas — F1 (waga)."
        else:
            plan['recommended_metric'] = 'r2'
            plan['metric_reason'] = "Regresja — R²."

        plan['enable_hyperparameter_tuning'] = bool(n_samples > 2000 and n_features < 100)
        plan['tuning_reason'] = "Tuning ON (rozmiar/cechy OK)" if plan['enable_hyperparameter_tuning'] else "Tuning OFF — zbyt kosztowny."

        plan['enable_ensemble'] = bool(n_samples > 5000)
        plan['ensemble_reason'] = "Ensemble ON (dużo danych)" if plan['enable_ensemble'] else "Ensemble OFF (opcjonalne)."

        plan['special_considerations'] = []
        if target_analysis['null_count'] > 0:
            plan['special_considerations'].append(f"⚠️ Target ma {target_analysis['null_count']} NaN.")
        if n_features > n_samples:
            plan['special_considerations'].append("⚠️ Więcej cech niż próbek — ryzyko overfittingu.")
        if n_samples > 50000:
            plan['special_considerations'].append("💡 Duży dataset — trening może potrwać dłużej.")

        return plan

    def _render_final_configuration(self, df: pd.DataFrame, target_column: str, ai_plan: Dict) -> Dict:
        if st.session_state.get('ai_config_applied', False):
            return st.session_state.get('final_config', ai_plan)

        with st.expander("🔧 Dostosuj konfigurację (opcjonalnie)", expanded=False):
            st.markdown("### ⚙️ Zaawansowane ustawienia")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**📊 Podział**")
                train_size = st.slider("Train size", 0.6, 0.9, ai_plan.get('train_size', 0.8), 0.05)
                cv_folds = st.selectbox(
                    "CV folds",
                    [3, 5, 10],
                    index=[3, 5, 10].index(ai_plan.get('cv_folds', 5))
                )

            with col2:
                st.markdown("**🎯 Algorytmy**")
                algorithm_strategy = st.selectbox(
                    "Strategia",
                    ["all", "fast", "accurate", "ensemble", "advanced"],
                    index=["all", "fast", "accurate", "ensemble", "advanced"].index(
                        ai_plan.get('recommended_strategy', 'all')
                    )
                )

                problem_type = self.ml_trainer.detect_problem_type(df, target_column)
                if problem_type == "Klasyfikacja":
                    metric_options = ["accuracy", "f1_weighted", "precision_weighted", "roc_auc"]
                else:
                    metric_options = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]

                optimization_metric = st.selectbox(
                    "Metryka",
                    metric_options,
                    index=metric_options.index(ai_plan.get('recommended_metric', metric_options[0]))
                )

            with col3:
                st.markdown("**⚡ Optymalizacje**")
                enable_tuning = st.checkbox(
                    "Hyperparameter tuning",
                    value=ai_plan.get('enable_hyperparameter_tuning', False)
                )
                ensemble_mode = st.checkbox(
                    "Ensemble mode",
                    value=ai_plan.get('enable_ensemble', False)
                )
                use_full = st.checkbox("Pełny dataset", value=len(df) <= 15000)

            return {
                'train_size': train_size,
                'cv_folds': cv_folds,
                'recommended_strategy': algorithm_strategy,
                'recommended_metric': optimization_metric,
                'enable_hyperparameter_tuning': enable_tuning,
                'enable_ensemble': ensemble_mode,
                'use_full_dataset': use_full
            }

    def _render_training_button(self, df: pd.DataFrame, target_column: str, config: Dict):
        if not st.session_state.get('model_trained', False) and not st.session_state.get('training_in_progress', False):
            st.markdown("### 🚀 Gotowy do treningu?")
            estimated_time = self._estimate_training_time(df, config)

            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🎯 ROZPOCZNIJ TRENING MODELU", type="primary", use_container_width=True):
                    self._execute_training(df, target_column, config)
            with col2:
                st.info(f"⏱️ **Czas:**\n{estimated_time}")

    def _estimate_training_time(self, df: pd.DataFrame, config: Dict) -> str:
        n_samples = len(df) if config.get('use_full_dataset', False) else min(8000, len(df))
        n_features = len(df.columns) - 1
        base_time = (n_samples / 1000) * (n_features / 10) * 15

        multipliers = {"all": 3.0, "fast": 0.5, "accurate": 4.0, "ensemble": 3.5, "advanced": 5.0}
        multiplier = multipliers.get(config.get('recommended_strategy', 'all'), 1.0)

        if config.get('enable_hyperparameter_tuning', False):
            multiplier *= 3.0

        total = base_time * multiplier
        if total < 60:
            return f"{int(total)} sek"
        elif total < 3600:
            return f"{int(total / 60)} min"
        else:
            return f"{_fmt_float_safe(total / 3600, 1)} godz"

    def _execute_training(self, df: pd.DataFrame, target_column: str, config: Dict):
        """Wykonaj trening z progress tracking."""
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🤖 Trening...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                status_text.text("🔄 Przygotowanie...")
                progress_bar.progress(10)

                problem_type = self.ml_trainer.detect_problem_type(df, target_column)

                status_text.text("🎯 Trenowanie...")
                progress_bar.progress(30)

                results = self.ml_trainer.train_model(
                    df=df,
                    target_column=target_column,
                    train_size=config.get('train_size', 0.8),
                    cv_folds=config.get('cv_folds', 5),
                    random_state=42,
                    use_full_dataset=config.get('use_full_dataset', False),
                    algorithm_selection=config.get('recommended_strategy', 'all'),
                    optimization_metric=config.get('recommended_metric'),
                    enable_hyperparameter_tuning=config.get('enable_hyperparameter_tuning', False)
                )

                progress_bar.progress(90)
                status_text.text("💾 Zapisywanie...")

                st.session_state.model_results = results
                st.session_state.target_column = target_column
                st.session_state.problem_type = problem_type
                st.session_state.model_trained = True
                st.session_state.analysis_complete = True

                progress_bar.progress(100)
                status_text.text("🎉 Zakończono!")

                best_model = results.get('best_model', 'Unknown')
                if problem_type == "Regresja":
                    score = results.get('r2', 0)
                    st.success(f"🏆 **{best_model}** | R² = {_fmt_float_safe(score, 4)}")
                else:
                    score = results.get('accuracy', 0)
                    st.success(f"🏆 **{best_model}** | Accuracy = {_fmt_float_safe(score, 4)}")

                st.balloons()
                time.sleep(2)
                progress_bar.empty()
                status_text.empty()
                st.rerun()

            except Exception as e:
                st.error(f"❌ Błąd treningu: {str(e)}")
                progress_bar.empty()
                status_text.empty()
                with st.expander("🐛 Szczegóły błędu"):
                    st.code(traceback.format_exc())


    def _render_quick_results_preview(self):
        results = st.session_state.model_results
        problem_type = st.session_state.problem_type

        st.markdown("#### 📊 Podgląd")
        if problem_type == "Regresja":
            metrics = {"R²": results.get('r2'), "MAE": results.get('mae'), "RMSE": results.get('rmse')}
        else:
            metrics = {
                "Accuracy": results.get('accuracy'),
                "F1": results.get('f1_weighted') or results.get('f1'),
                "Precision": results.get('precision')
            }

        cols = st.columns(len(metrics))
        for (name, val), col in zip(metrics.items(), cols):
            with col:
                st.metric(name, f"{_fmt_float_safe(val, 4)}" if val is not None else "—")


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

        with viz_tabs[4]:
            if problem_type == "Regresja" and 'predictions' in results:
                self._render_residual_analysis(results)

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
        if train and st.session_state.get("ai_plan") and st.session_state.get("ai_recs"):
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
                if best:
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
                            if ensembles:
                                try:
                                    qdf = evaluate_models_quick(
                                        X, y, ensembles,
                                        st.session_state.ai_recs.get('problem', 'regression'),
                                        folds=3
                                    )
                                    if qdf is not None and not qdf.empty:
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
