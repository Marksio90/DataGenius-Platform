# app.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import streamlit as st

from core.container import get_container
from frontend.ui_components import render_sidebar, load_or_upload_df
from frontend.ui_docs import render_docs

st.set_page_config(
    page_title="TMIV – Advanced ML Platform v2.0 Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Załaduj style (jeśli są)
try:
    st.markdown(Path("frontend/styles.css").read_text(encoding="utf-8"), unsafe_allow_html=True)
except Exception:
    pass

# DI & telemetry
C = get_container()
tele = C.resolve("telemetry")
tele.start_heartbeat(30)

data_svc = C.resolve("data")
eda_svc = C.resolve("eda")
ml_svc = C.resolve("ml")
explain_svc = C.resolve("explain")
export_svc = C.resolve("export")
insights_svc = C.resolve("insights")
db_svc = C.resolve_or_none("db")

# Stan
st.session_state.setdefault("df", None)
st.session_state.setdefault("dataset_name", None)
st.session_state.setdefault("train_result", None)

# Sidebar / routing
page = render_sidebar()

# =========================
# 📊 Analiza Danych
# =========================
if page == "📊 Analiza Danych":
    st.title("📊 Analiza Danych")
    df, name = load_or_upload_df(data_svc)
    if df is None:
        st.success("Szkielet gotowy — wgraj plik lub użyj przykładowego zestawu, aby zacząć.")
        st.stop()

    st.subheader("Podsumowanie")
    ov = eda_svc.overview(df)
    c1, c2, c3 = st.columns(3)
    c1.metric("Wiersze", ov["shape"]["rows"])
    c2.metric("Kolumny", ov["shape"]["cols"])
    miss_pct = round(np.mean([v["pct"] for v in ov["missing"].values()]) if ov["missing"] else 0.0, 2)
    c3.metric("Braki ogółem (%)", miss_pct)

    with st.expander("Typy i braki (JSON)", expanded=False):
        st.json({"dtypes": ov["dtypes"], "missing": ov["missing"]})

    st.subheader("Korelacje (numeryczne)")
    corr = eda_svc.correlations(df)
    if corr is None or corr.empty:
        st.caption("Brak kolumn numerycznych lub stałe wariancje.")
    else:
        st.dataframe(corr)

    st.subheader("Profil EDA (HTML, opcjonalnie)")
    if st.button("Zbuduj raport profilujący"):
        with st.spinner("Generuję profil..."):
            html = eda_svc.profile_html(df)
        if html:
            st.success("Profil zapisany.")
            st.code(html)
            st.markdown(f"[Otwórz raport]({Path(html).as_uri()})")
        else:
            st.warning("Profiling niedostępny (brak pakietu) lub błąd generowania.")

# =========================
# 🤖 Trening Modelu
# =========================
elif page == "🤖 Trening Modelu":
    st.title("🤖 Trening Modelu")
    if st.session_state.df is None:
        st.info("Najpierw wczytaj dane w zakładce **📊 Analiza Danych**.")
        st.stop()

    df = st.session_state.df
    cols = list(df.columns)
    target = st.selectbox("Wybierz kolumnę docelową (target)", options=cols, index=min(0, len(cols)-1))
    strategy = st.selectbox("Strategia", ["balanced", "fast_small", "accurate"], index=0)
    test_size = st.slider("Wielkość walidacji (test_size)", 0.05, 0.5, 0.2, 0.01)
    random_state = st.number_input("random_state", min_value=0, value=42, step=1)

    if st.button("Trenuj modele", type="primary"):
        with tele.start_span("training", {"target": target, "rows": len(df)}):
            with st.spinner("Trenuję…"):
                plan = ml_svc.build_training_plan(df, target, strategy=strategy, random_state=int(random_state))
                result = ml_svc.train_and_evaluate(
                    df, target, plan=plan, test_size=float(test_size), random_state=int(random_state)
                )
                st.session_state.train_result = result
                if db_svc:
                    try:
                        fp = data_svc.fingerprint(df)
                        db_svc.ensure_run_meta(
                            run_id=f"run-{fp}",
                            dataset_name=st.session_state.dataset_name or "dataset",
                            fingerprint=fp,
                            problem_type=result["problem_type"],
                        )
                    except Exception:
                        pass

    tr = st.session_state.train_result
    if tr:
        st.subheader("Leaderboard")
        st.dataframe(tr["leaderboard"])
        st.subheader("Ważność cech (best model)")
        st.dataframe(tr["feature_importance"].head(30))
        st.info(f"Najlepszy model: **{tr['best_model_name']}**; typ: **{tr['problem_type']}**")

# =========================
# 📈 Wyniki i Wizualizacje
# =========================
elif page == "📈 Wyniki i Wizualizacje":
    st.title("📈 Wyniki i Wizualizacje")
    tr = st.session_state.train_result
    if not tr:
        st.info("Najpierw wytrenuj modele w zakładce **🤖 Trening Modelu**.")
        st.stop()

    target = tr["target"]
    df = st.session_state.df
    best_name = tr["best_model_name"]
    best_model = tr["models"][best_name]
    charts = {}

    if tr["problem_type"] == "classification":
        st.subheader("Krzywe ROC/PR · Kalibracja · Macierz")
        X = df.drop(columns=[target])
        try:
            proba = best_model.predict_proba(X)
        except Exception:
            proba = None
        y_true = tr["y_valid"]
        if proba is not None:
            paths = explain_svc.plot_classification_curves(y_true, proba, model_name=best_name)
            charts.update(paths)
            for label, p in paths.items():
                st.image(p, caption=label)
            try:
                cal = explain_svc.plot_calibration_curve(y_true, proba, model_name=best_name, n_bins=10)
                charts["calibration"] = cal
                st.image(cal, caption="Calibration")
            except Exception:
                pass
        try:
            pred = best_model.predict(X)
            cm = explain_svc.plot_confusion_matrix(y_true, pred, model_name=best_name)
            charts["cm"] = cm
            st.image(cm, caption="Confusion matrix")
        except Exception:
            pass
    else:
        st.subheader("Diagnostyka regresji")
        X = df.drop(columns=[target])
        y_true = df[target].to_numpy()
        pred = best_model.predict(X)
        paths = explain_svc.plot_regression_diagnostics(y_true, pred, model_name=best_name)
        charts.update(paths)
        for label, p in paths.items():
            st.image(p, caption=label)

    st.subheader("Ważność cech (wykres)")
    try:
        fi_plot = explain_svc.plot_feature_importance(tr["feature_importance"], top_k=20, name="fi_best")
        charts["feature_importance"] = fi_plot
        st.image(fi_plot, caption="Feature importance (top 20)")
    except Exception:
        st.caption("Nie udało się narysować FI.")

    st.subheader("Eksporty")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📄 PDF Explainability"):
            pdf = explain_svc.build_pdf_report(
                "tmiv_report.pdf",
                metrics=tr["metrics_by_model"].get(best_name, {}),
                feature_importance=tr["feature_importance"],
                charts=charts,
                title="TMIV – Explainability",
                model_name=best_name,
                dataset_name=st.session_state.dataset_name or "dataset",
                problem_type=tr["problem_type"],
                run_id=f"run-{data_svc.fingerprint(df)}",
            )
            st.success(f"PDF zapisany: {pdf}")
            st.code(pdf)
    with c2:
        if st.button("🧰 ZIP – pełny eksport artefaktów"):
            zip_path = explain_svc.export_everything(
                run_id=f"run-{data_svc.fingerprint(df)}",
                problem_type=tr["problem_type"],
                metrics=tr["metrics_by_model"].get(best_name, {}),
                dataset_name=st.session_state.dataset_name or "dataset",
                dataset_fingerprint=data_svc.fingerprint(df),
                leaderboard=tr["leaderboard"],
                feature_importance=tr["feature_importance"],
                models={best_name: best_model},
                plots=charts,
                notes="Export from Streamlit UI",
            )
            st.success(f"ZIP zapisany: {zip_path}")
            st.code(zip_path)

# =========================
# 💡 Rekomendacje
# =========================
elif page == "💡 Rekomendacje":
    st.title("💡 Rekomendacje")
    tr = st.session_state.train_result
    if not tr:
        st.info("Najpierw wytrenuj modele, aby zobaczyć rekomendacje.")
        st.stop()

    ctx = {
        "problem_type": tr["problem_type"],
        "metrics": tr["metrics_by_model"].get(tr["best_model_name"], {}),
        "feature_importance": tr["feature_importance"],
        "rows": len(st.session_state.df),
        "cols": st.session_state.df.shape[1],
    }
    recs = insights_svc.generate_recommendations(ctx)
    for r in recs:
        st.markdown(f"- {r}")

# =========================
# 📚 Dokumentacja
# =========================
elif page == "📚 Dokumentacja":
    st.title("📚 Dokumentacja")
    render_docs()

else:
    st.title("TMIV – Advanced ML Platform v2.0 Pro")
    st.success("Szkielet projektu gotowy. Dodamy funkcje krok po kroku.")
    st.write("➡️ Zacznij od wrzucenia danych w zakładce **Analiza Danych**.")
