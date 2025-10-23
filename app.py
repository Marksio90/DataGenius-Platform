"""
TMIV - The Most Important Variables
Advanced ML Platform v2.0 Pro

Główna aplikacja Streamlit.
"""

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.ai_integration import get_ai_integration
from backend.auto_prep import auto_preprocess
from backend.cache_manager import get_cache_manager
from backend.dtype_sanitizer import sanitize_dataframe
from backend.eda_integration import perform_eda
from backend.error_handler import handle_errors
from backend.explain_plus import aggregate_feature_importance
from backend.export_everything import ArtifactExporter
from backend.export_explain_pdf import ExplainabilityPDFExporter
from backend.file_upload import load_file, validate_dataframe
from backend.ml_integration import MLIntegration
from backend.monitoring import display_monitoring_panel
from backend.telemetry import init_telemetry, record_telemetry_event
from backend.utils_target import detect_target_column, detect_timeseries_signals
from config.settings import get_settings
from frontend.ui_compare import (
    render_model_comparison_table,
    render_model_ranking,
    render_radar_comparison,
    render_training_time_comparison,
)
from frontend.ui_components import (
    render_download_button,
    render_file_uploader,
    render_info_box,
    render_sidebar,
    render_strategy_selector,
    render_target_selector,
    render_warning_box,
)
from frontend.ui_panels import (
    render_classification_results,
    render_feature_importance_panel,
    render_model_details_expander,
    render_regression_results,
)

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Konfiguracja strony
st.set_page_config(
    page_title="TMIV - Advanced ML Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Settings
settings = get_settings()

# Inicjalizacja telemetrii
init_telemetry()


def initialize_session_state():
    """Inicjalizuje session state."""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'eda_complete' not in st.session_state:
        st.session_state.eda_complete = False
    if 'eda_results' not in st.session_state:
        st.session_state.eda_results = None
    if 'training_complete' not in st.session_state:
        st.session_state.training_complete = False
    if 'ml_results' not in st.session_state:
        st.session_state.ml_results = None
    if 'plots' not in st.session_state:
        st.session_state.plots = {}


@handle_errors(show_in_ui=True)
def page_data_analysis():
    """Strona: Analiza Danych."""
    st.title("📊 Analiza Danych")

    # File uploader
    uploaded_file = render_file_uploader()

    if uploaded_file is not None:
        # Wczytaj plik
        with st.spinner("Wczytywanie danych..."):
            df, metadata = load_file(uploaded_file)
            record_telemetry_event("data_loaded", rows=len(df), columns=len(df.columns))

        st.success(f"✅ Wczytano {metadata['n_rows']} wierszy, {metadata['n_columns']} kolumn")

        # Walidacja
        is_valid, warnings = validate_dataframe(df)
        if warnings:
            render_warning_box(warnings)

        # Sanityzacja
        with st.spinner("Sanityzacja danych..."):
            df_clean, sanitize_report = sanitize_dataframe(df, aggressive=False)
            st.session_state.df = df_clean

        st.info(f"Sanityzacja: {sanitize_report['final_shape']}")

        # Podgląd danych
        with st.expander("👀 Podgląd Danych", expanded=True):
            st.dataframe(df_clean.head(20), use_container_width=True)

        # Podstawowe statystyki
        st.subheader("📈 Podstawowe Statystyki")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wiersze", len(df_clean))
        with col2:
            st.metric("Kolumny", len(df_clean.columns))
        with col3:
            numeric_cols = df_clean.select_dtypes(include=['number']).columns
            st.metric("Numeryczne", len(numeric_cols))
        with col4:
            categorical_cols = df_clean.select_dtypes(include=['object']).columns
            st.metric("Kategoryczne", len(categorical_cols))

        # EDA
        st.subheader("🔍 Analiza Eksploracyjna (EDA)")

        if st.button("🚀 Uruchom EDA", type="primary", use_container_width=True):
            with st.spinner("Wykonywanie analizy EDA..."):
                # Sprawdź czy użyć profilingu
                use_profiling = st.checkbox(
                    "Włącz zaawansowany profiling (może zająć kilka minut)",
                    value=False
                )

                eda_results = perform_eda(df_clean, include_profiling=use_profiling)
                st.session_state.eda_results = eda_results
                st.session_state.eda_complete = True
                st.session_state.data_loaded = True

                record_telemetry_event("eda_completed")

            st.success("✅ Analiza EDA zakończona!")
            st.rerun()

        # Wyświetl wyniki EDA jeśli dostępne
        if st.session_state.eda_complete and st.session_state.eda_results:
            eda_results = st.session_state.eda_results

            # Statystyki numeryczne
            if eda_results['numeric_stats']:
                with st.expander("📊 Statystyki Kolumn Numerycznych", expanded=False):
                    df_numeric_stats = pd.DataFrame(eda_results['numeric_stats'])
                    st.dataframe(df_numeric_stats, use_container_width=True)

            # Statystyki kategoryczne
            if eda_results['categorical_stats']:
                with st.expander("📝 Statystyki Kolumn Kategorycznych", expanded=False):
                    for cat_stat in eda_results['categorical_stats']:
                        st.markdown(f"**{cat_stat['column']}**")
                        st.text(f"Unikalne wartości: {cat_stat['n_unique']}")
                        st.text(f"Najczęstsza: {cat_stat['most_common']} ({cat_stat['most_common_pct']:.1f}%)")
                        if cat_stat.get('top_categories'):
                            df_top = pd.DataFrame(cat_stat['top_categories'])
                            st.dataframe(df_top, use_container_width=True)
                        st.markdown("---")

            # Korelacje
            if eda_results['correlation_matrix'] is not None:
                with st.expander("🔗 Macierz Korelacji", expanded=False):
                    st.dataframe(eda_results['correlation_matrix'], use_container_width=True)

                    # Wysokie korelacje
                    high_corr = eda_results.get('high_correlations', [])
                    if high_corr:
                        st.markdown("**Wysokie korelacje (|r| ≥ 0.7):**")
                        for col1, col2, corr in high_corr[:10]:
                            st.text(f"{col1} ↔ {col2}: {corr:.3f}")

            # Opisy kolumn (AI)
            st.subheader("🤖 Opisy Kolumn (AI)")

            if st.button("✨ Generuj Opisy", use_container_width=True):
                ai = get_ai_integration()
                cache_manager = get_cache_manager()

                with st.spinner("Generowanie opisów kolumn..."):
                    df_hash = cache_manager.get_dataframe_hash(df_clean)
                    descriptions = ai.describe_columns(df_clean, df_hash=df_hash)

                    if descriptions:
                        st.session_state.column_descriptions = descriptions

                st.rerun()

            # Wyświetl opisy jeśli dostępne
            if 'column_descriptions' in st.session_state:
                descriptions = st.session_state.column_descriptions
                with st.expander("📖 Opisy Kolumn", expanded=True):
                    for col, desc in descriptions.items():
                        st.markdown(f"**{col}**")
                        st.text(desc)
                        st.markdown("---")

    else:
        render_info_box(
            "ℹ️ Jak zacząć?",
            """
            1. Wczytaj plik z danymi (CSV, Excel, Parquet, JSON)
            2. Przejrzyj podstawowe statystyki
            3. Uruchom analizę EDA
            4. Przejdź do treningu modelu
            """,
            expanded=True
        )


@handle_errors(show_in_ui=True)
def page_model_training():
    """Strona: Trening Modelu."""
    st.title("🤖 Trening Modelu")

    if not st.session_state.data_loaded or st.session_state.df is None:
        st.warning("⚠️ Najpierw wczytaj dane w sekcji 'Analiza Danych'")
        return

    df = st.session_state.df

    st.info(f"📊 Dataset: {len(df)} wierszy, {len(df.columns)} kolumn")

    # Detekcja target
    st.subheader("🎯 Wybór Target")

    auto_target = detect_target_column(df)
    target_col = render_target_selector(df.columns.tolist(), auto_detected=auto_target)

    # Sprawdź sygnały time series
    ts_signals = detect_timeseries_signals(df)
    if ts_signals['is_likely_timeseries']:
        render_warning_box(ts_signals['warnings'])

    # Strategia treningu
    strategy, use_tuning, use_ensemble = render_strategy_selector()

    # Przycisk treningu
    if st.button("🚀 Rozpocznij Trening", type="primary", use_container_width=True):
        with st.spinner("Trening w toku... To może zająć kilka minut."):
            # Progress placeholder
            progress_placeholder = st.empty()

            # MLIntegration
            ml = MLIntegration()

            try:
                # Uruchom pipeline
                pipeline_results = ml.run_full_pipeline(
                    df=df,
                    target_col=target_col,
                    strategy=strategy,
                    use_tuning=use_tuning,
                    use_ensemble=use_ensemble
                )

                st.session_state.ml_results = pipeline_results
                st.session_state.training_complete = True

                record_telemetry_event(
                    "training_completed",
                    n_models=pipeline_results['n_models_trained']
                )

                progress_placeholder.success("✅ Trening zakończony pomyślnie!")

            except Exception as e:
                progress_placeholder.error(f"❌ Błąd treningu: {e}")
                logger.error(f"Training error: {e}")
                return

        st.rerun()

    # Wyświetl wyniki jeśli dostępne
    if st.session_state.training_complete and st.session_state.ml_results:
        st.success("✅ Trening zakończony!")

        ml_results = st.session_state.ml_results

        # Podsumowanie
        st.subheader("📋 Podsumowanie")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modele wytrenowane", ml_results['n_models_trained'])
        with col2:
            st.metric("Najlepszy model", ml_results['best_model_name'])
        with col3:
            st.metric("Features", ml_results['n_features'])


@handle_errors(show_in_ui=True)
def page_results_visualization():
    """Strona: Wyniki i Wizualizacje."""
    st.title("📈 Wyniki i Wizualizacje")

    if not st.session_state.training_complete or st.session_state.ml_results is None:
        st.warning("⚠️ Najpierw wytrenuj modele w sekcji 'Trening Modelu'")
        return

    ml_results = st.session_state.ml_results
    results = ml_results['models']
    problem_type = ml_results['problem_type']
    X_test = ml_results['X_test']
    y_test = ml_results['y_test']
    feature_names = ml_results['feature_names']
    class_names = ml_results.get('class_names')

    # Ranking modeli
    if ml_results.get('model_ranking'):
        render_model_ranking(ml_results['model_ranking'])
        st.markdown("---")

    # Porównanie modeli
    st.subheader("📊 Porównanie Modeli")

    # Tabela
    metrics_to_show = ['accuracy', 'f1'] if 'classification' in problem_type else ['r2', 'rmse']
    render_model_comparison_table(results, metrics_to_show)

    # Radar chart
    radar_img = render_radar_comparison(
        results,
        metrics_to_show,
        metric_names={'accuracy': 'Accuracy', 'f1': 'F1', 'r2': 'R²', 'rmse': 'RMSE'}
    )
    if radar_img:
        st.session_state.plots['radar_comparison'] = radar_img

    # Training time comparison
    time_img = render_training_time_comparison(results)
    if time_img:
        st.session_state.plots['training_time'] = time_img

    st.markdown("---")

    # Szczegóły najlepszego modelu
    st.subheader("🏆 Najlepszy Model")

    best_model_name = ml_results['best_model_name']
    best_model = ml_results['best_model']
    best_result = next((r for r in results if r['model_name'] == best_model_name), None)

    if best_result:
        # Wyniki klasyfikacji/regresji
        if 'classification' in problem_type:
            plots = render_classification_results(
                best_model_name,
                best_model,
                X_test,
                y_test,
                best_result['test_metrics'],
                class_names
            )
            st.session_state.plots.update(plots)
        else:
            plots = render_regression_results(
                best_model_name,
                best_model,
                X_test,
                y_test,
                best_result['test_metrics']
            )
            st.session_state.plots.update(plots)

    st.markdown("---")

    # Feature Importance
    st.subheader("🔍 Feature Importance")

    with st.spinner("Agregowanie feature importance..."):
        df_fi = aggregate_feature_importance(results, feature_names)

        if not df_fi.empty:
            fi_img = render_feature_importance_panel(
                df_fi['feature'].tolist(),
                df_fi['importance'].values,
                top_n=20
            )
            if fi_img:
                st.session_state.plots['feature_importance'] = fi_img

    st.markdown("---")

    # Szczegóły wszystkich modeli
    with st.expander("📋 Szczegóły Wszystkich Modeli", expanded=False):
        for result in results:
            render_model_details_expander(result['model_name'], result)


@handle_errors(show_in_ui=True)
def page_recommendations():
    """Strona: Rekomendacje."""
    st.title("💡 Rekomendacje")

    if not st.session_state.training_complete or st.session_state.ml_results is None:
        st.warning("⚠️ Najpierw wytrenuj modele w sekcji 'Trening Modelu'")
        return

    ml_results = st.session_state.ml_results
    eda_results = st.session_state.eda_results or {}

    st.subheader("🤖 Rekomendacje Biznesowe (AI)")

    if st.button("✨ Generuj Rekomendacje", use_container_width=True):
        ai = get_ai_integration()

        with st.spinner("Generowanie rekomendacji..."):
            # Przygotuj podsumowania
            eda_summary = eda_results.get('basic_stats', {})
            ml_summary = {
                'best_model_name': ml_results['best_model_name'],
                'n_models_trained': ml_results['n_models_trained'],
                'best_metric': 'N/A'  # TODO: dodaj najlepszą metrykę
            }

            recommendations = ai.generate_business_recommendations(eda_summary, ml_summary)

            if recommendations:
                st.session_state.recommendations = recommendations

        st.rerun()

    # Wyświetl rekomendacje jeśli dostępne
    if 'recommendations' in st.session_state:
        st.markdown(st.session_state.recommendations)

    st.markdown("---")

    # Eksport artefaktów
    st.subheader("📦 Eksport Artefaktów")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("📥 Eksportuj ZIP", use_container_width=True):
            with st.spinner("Tworzenie archiwum ZIP..."):
                exporter = ArtifactExporter()
                zip_path = exporter.export_to_zip(
                    ml_results['models'],
                    ml_results,
                    plots=st.session_state.plots
                )

            with open(zip_path, 'rb') as f:
                render_download_button(
                    f.read(),
                    zip_path.name,
                    "📥 Pobierz ZIP",
                    "application/zip"
                )

    with col2:
        if st.button("📄 Eksportuj PDF", use_container_width=True):
            with st.spinner("Generowanie raportu PDF..."):
                pdf_exporter = ExplainabilityPDFExporter()

                # Feature importance
                feature_importance = None
                if ml_results.get('feature_names'):
                    from backend.explain_plus import aggregate_feature_importance
                    df_fi = aggregate_feature_importance(
                        ml_results['models'],
                        ml_results['feature_names']
                    )
                    if not df_fi.empty:
                        feature_importance = df_fi.to_dict('records')

                pdf_path = pdf_exporter.export_report(
                    ml_results,
                    feature_importance=feature_importance,
                    plots=st.session_state.plots
                )

            with open(pdf_path, 'rb') as f:
                render_download_button(
                    f.read(),
                    pdf_path.name,
                    "📄 Pobierz PDF",
                    "application/pdf"
                )


def page_documentation():
    """Strona: Dokumentacja."""
    st.title("📚 Dokumentacja")

    st.markdown("""
    ## TMIV - The Most Important Variables
    ### Advanced ML Platform v2.0 Pro

    ---

    ### 🚀 Szybki Start

    1. **Wczytaj dane**: Przejdź do sekcji "Analiza Danych" i wczytaj plik CSV/Excel/Parquet/JSON
    2. **Analiza EDA**: Uruchom analizę eksploracyjną danych
    3. **Trening modelu**: Wybierz target i strategię treningu
    4. **Wyniki**: Przejrzyj wyniki, wizualizacje i feature importance
    5. **Eksport**: Pobierz artefakty (ZIP) i raport (PDF)

    ---

    ### 📊 Wspierane Formaty Danych

    - **CSV** (.csv) - separator automatycznie wykrywany
    - **Excel** (.xlsx, .xls) - pierwszy arkusz
    - **Parquet** (.parquet) - format kolumnowy
    - **JSON** (.json) - format JSON

    **Wymagania:**
    - Minimum 10 wierszy
    - Co najmniej 2 kolumny
    - Kolumna target z więcej niż 1 unikalną wartością

    ---

    ### 🎯 Detekcja Target

    Aplikacja automatycznie wykrywa kolumnę target na podstawie:
    - Nazwy kolumny (target, label, class, y, outcome, etc.)
    - Typu danych (binarne, kategoryczne małowartościowe)
    - Pozycji (ostatnia kolumna jako fallback)

    Możesz także wybrać target ręcznie.

    ---

    ### ⚙️ Strategie Treningu

    **fast_small** - Szybkie modele podstawowe
    - Dla małych zbiorów (<500 próbek)
    - 3-5 modeli
    - Czas: ~1-2 minuty

    **balanced** (zalecane) - Zbalansowany kompromis
    - Dla średnich zbiorów (500-10k próbek)
    - 5-7 modeli
    - Czas: ~3-5 minut

    **accurate** - Rozszerzony zestaw
    - Dla większych zbiorów (>10k próbek)
    - 7-10 modeli
    - Czas: ~5-10 minut

    **advanced** - Wszystkie modele + opcje zaawansowane
    - Dla dużych zbiorów (>10k próbek)
    - 10+ modeli + tuning/ensemble
    - Czas: ~10-30 minut

    ---

    ### 📈 Metryki

    **Klasyfikacja binarna:**
    - ROC AUC
    - Accuracy
    - Precision / Recall
    - F1 Score
    - Average Precision

    **Klasyfikacja wieloklasowa:**
    - Accuracy
    - F1 Weighted/Macro/Micro

    **Regresja:**
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² (Coefficient of Determination)

    ---

    ### 🔍 Feature Importance

    Aplikacja agreguje feature importance z wielu modeli używając:
    - Feature importances (tree-based models)
    - Coefficients (linear models)
    - Permutation importance (opcjonalnie)

    ---

    ### 🤖 Integracja AI

    **Bez kluczy API:**
    - Deterministyczne fallbacki
    - Heurystyczne opisy kolumn
    - Podstawowe rekomendacje

    **Z kluczami API (OpenAI/Anthropic):**
    - Inteligentne opisy kolumn
    - Kontekstualne rekomendacje biznesowe
    - Interpretacja wyników

    ---

    ### 📦 Eksport

    **ZIP Archive:**
    - Modele (.joblib)
    - Metryki (JSON, CSV)
    - Wykresy (PNG)
    - Konfiguracja (JSON)
    - README z instrukcjami

    **PDF Report:**
    - Executive summary
    - Opis danych
    - Wyniki modeli
    - Feature importance
    - Wizualizacje
    - Wnioski i rekomendacje

    ---

    ### ⚠️ Ograniczenia

    - Maksymalny rozmiar pliku: 200 MB
    - Brak wsparcia dla deep learning
    - Time series: podstawowe wsparcie (ostrzeżenia)
    - NLP: wymaga preprocessingu zewnętrznego

    ---

    ### 🛠️ Rozwiązywanie Problemów

    **Problem:** Błąd wczytywania CSV
    **Rozwiązanie:** Sprawdź encoding (UTF-8 zalecane), separator, format dat

    **Problem:** Błąd treningu modelu
    **Rozwiązanie:** Sprawdź braki danych, typy kolumn, rozmiar datasetu

    **Problem:** Brak opisów AI
    **Rozwiązanie:** Dodaj klucze API lub użyj fallbacku deterministycznego

    **Problem:** Wolny trening
    **Rozwiązanie:** Użyj strategii "fast_small" lub zmniejsz dataset

    ---

    ### 📞 Wsparcie

    - Dokumentacja: `docs/README.md`
    - GitHub: [Issues](https://github.com/your-repo/issues)
    - Email: support@tmiv.ai

    ---

    ### 🔐 Bezpieczeństwo

    - Klucze API są szyfrowane w sesji
    - Brak logowania danych użytkownika
    - Sanityzacja promptów AI
    - Walidacja inputów

    ---

    ### 📝 Wersja

    **TMIV v{version}**
    Data wydania: 2024-12-19
    Python: 3.11+
    Streamlit: 1.37+
    """.format(version=settings.app_version))


def main():
    """Główna funkcja aplikacji."""
    # Inicjalizacja
    initialize_session_state()

    # Sidebar
    config = render_sidebar()

    # Nagłówek
    st.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <h1>🤖 TMIV - Advanced ML Platform v2.0 Pro</h1>
            <p style='font-size: 1.2rem; color: #666;'>The Most Important Variables</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Nawigacja
    tabs = st.tabs([
        "📊 Analiza Danych",
        "🤖 Trening Modelu",
        "📈 Wyniki i Wizualizacje",
        "💡 Rekomendacje",
        "📚 Dokumentacja"
    ])

    with tabs[0]:
        page_data_analysis()

    with tabs[1]:
        page_model_training()

    with tabs[2]:
        page_results_visualization()

    with tabs[3]:
        page_recommendations()

    with tabs[4]:
        page_documentation()

    # Monitoring (ukryty panel)
    with st.sidebar.expander("🔧 Monitoring & Admin", expanded=False):
        display_monitoring_panel()


if __name__ == "__main__":
    main()