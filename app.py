"""
TMIV – Advanced ML Platform v2.0 Pro (Stage 8: Final polish)
"""
from __future__ import annotations
import platform, streamlit as st, subprocess, sys, os, json
from config.settings import settings
from frontend.ui_components import sidebar
from frontend.ui_panels import page_data as page_data_base, page_training as page_training_base, page_results as page_results_base, page_recommendations, page_docs
from frontend.ui_panels_explain import page_results as page_results_explain
from frontend.ui_compare import metrics_table, radar_plot

st.set_page_config(page_title="TMIV – Advanced ML Platform v2.0 Pro", layout="wide")
ui_cfg = sidebar()

st.title("TMIV – Advanced ML Platform v2.0 Pro")
st.caption("Stage 8: y_true, Explainability++ (PDP/ICE/Calibration), Registry promote, Admin telemetry")

PAGES = ["📊 Analiza Danych", "🤖 Trening Modelu", "📈 Wyniki (porównanie)", "🧠 Explainability++", "💡 Rekomendacje", "📚 Dokumentacja"]
page = st.selectbox("Nawigacja", PAGES, index=0)

with st.expander("ℹ️ Stan aplikacji / Runtime info", expanded=True):
    st.json({
        "df_loaded": "tmiv_df" in st.session_state,
        "trained": "tmiv_train" in st.session_state,
        "dataset": "w pamięci" if "tmiv_df" in st.session_state else "-",
        "strategy": ui_cfg["strategy"],
        "flags": ui_cfg["flags"],
        "limits": ui_cfg["limits"],
        "providers": settings.provider_status(),
        "versions": {"python": platform.python_version(), "streamlit": st.__version__}
    })

if ui_cfg["admin_mode"]:
    st.subheader("🛡️ Admin / Monitoring")
    # Telemetry JSONL preview
    log_path = "artifacts/logs/tmiv_telemetry.jsonl"
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            last = f.readlines()[-30:]
        st.text_area("Telemetry (ostatnie wpisy)", "".join(last), height=200)
    else:
        st.info("Brak logów telemetry na razie.")
    # SBOM & Licenses
    if ui_cfg["flags"].get("USE_SBOM", False):
        if st.button("📜 Build SBOM + Licencje"):
            subprocess.run([sys.executable, "scripts/build_sbom.py"])
            subprocess.run([sys.executable, "scripts/gen_licenses.py"])
            st.success("SBOM + licencje wygenerowane do artifacts/sbom/")

if page == "📊 Analiza Danych":
    page_data_base()
elif page == "🤖 Trening Modelu":
    page_training_base(ui_cfg["strategy"], ui_cfg["limits"])
elif page == "📈 Wyniki (porównanie)":
    run = st.session_state.get("tmiv_train")
    if run:
        st.subheader("Porównanie modeli")
        metrics_table(run["result"]["results"])
        radar_plot(run["result"]["results"])
    page_results_base()
elif page == "🧠 Explainability++":
    page_results_explain()
elif page == "💡 Rekomendacje":
    page_recommendations()
elif page == "📚 Dokumentacja":
    page_docs()

st.markdown("---")
st.caption("TMIV v2.0 Pro — Stage 8")