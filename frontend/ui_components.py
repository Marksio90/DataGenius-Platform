"""
UI Components — update (Stage 6): zapisujemy flagi do session_state, żeby panele miały do nich dostęp.
"""
from __future__ import annotations
import streamlit as st
from config.settings import settings

def sidebar() -> dict:
    st.sidebar.title("TMIV – Ustawienia")
    with st.sidebar.expander("🔐 Klucze (lokalne)", expanded=False):
        st.caption("Klucze są trzymane lokalnie (.env/secrets). Nigdy nie logujemy wartości.")
        openai = st.text_input("OpenAI API Key", type="password", value=settings.openai_api_key or "")
        anthropic = st.text_input("Anthropic API Key", type="password", value=settings.anthropic_api_key or "")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Strategia mocy")
    strategy = st.sidebar.radio(
        "Wybierz profil:",
        options=["fast_small", "balanced", "accurate", "advanced"],
        index=["fast_small", "balanced", "accurate", "advanced"].index(settings.strategy_default),
        help="Profil wpływa na dobór modeli, ensembling i ewentualne HPO."
    )

    st.sidebar.subheader("🧪 Funkcje (gated)")
    use_shap = st.sidebar.toggle("SHAP Explainability", value=settings.use_shap)
    use_optuna = st.sidebar.toggle("Optuna HPO", value=settings.use_optuna)
    use_onnx = st.sidebar.toggle("ONNX Export", value=settings.use_onnx)
    use_polars = st.sidebar.toggle("Polars LTS (EDA)", value=settings.use_polars)
    use_gx = st.sidebar.toggle("Great Expectations", value=settings.use_gx)
    use_otel = st.sidebar.toggle("OpenTelemetry", value=settings.use_opentelemetry)
    use_sbom = st.sidebar.toggle("SBOM & Licencje", value=settings.use_sbom)
    rag_explain = st.sidebar.toggle("RAG Explainability", value=settings.rag_explain)

    st.sidebar.subheader("⏱️ Limity / Budżety")
    max_train_time = st.sidebar.number_input("Max train time [s]", min_value=30, max_value=3600, value=settings.max_train_time_sec, step=30)
    max_hpo_trials = st.sidebar.number_input("Max HPO trials", min_value=1, max_value=200, value=settings.max_hpo_trials, step=1)
    max_models_parallel = st.sidebar.number_input("Max models parallel", min_value=1, max_value=8, value=settings.max_models_parallel, step=1)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧰 Narzędzia")
    reset_cache = st.sidebar.button("Wyczyść cache")
    admin_mode = st.sidebar.toggle("Admin / Monitoring", value=False)

    cfg = {
        "openai_api_key": openai.strip() or None,
        "anthropic_api_key": anthropic.strip() or None,
        "strategy": strategy,
        "flags": {
            "USE_SHAP": use_shap,
            "USE_OPTUNA": use_optuna,
            "USE_ONNX": use_onnx,
            "USE_POLARS": use_polars,
            "USE_GX": use_gx,
            "USE_OPENTELEMETRY": use_otel,
            "USE_SBOM": use_sbom,
            "RAG_EXPLAIN": rag_explain,
        },
        "limits": {
            "MAX_TRAIN_TIME_SEC": int(max_train_time),
            "MAX_HPO_TRIALS": int(max_hpo_trials),
            "MAX_MODELS_PARALLEL": int(max_models_parallel),
        },
        "reset_cache": reset_cache,
        "admin_mode": admin_mode,
    }
    # zapisz do session_state dla innych paneli
    st.session_state["ui_flags"] = cfg["flags"]
    st.session_state["ui_limits"] = cfg["limits"]
    return cfg