"""
ui_panels.py — Stage 6 (gated features wired)
"""
from __future__ import annotations
import streamlit as st
from typing import Dict, Any
from backend import file_upload, dtype_sanitizer, runtime_preprocessor, eda_integration, training_plan, async_ml_trainer
from backend.ai_integration import deterministic_recommendations
from backend.export_explain_pdf import build_pdf
from backend.export_everything import export_zip
from backend.hpo_optuna import hpo_rf
from backend.onnx_export import export_onnx
from backend.validation import validate_pandera, validate_gx
from backend.feature_store import save_baseline, load_baseline
from backend.drift_detection import drift_report
import os
from datetime import datetime

def page_data() -> None:
    st.header("📊 Analiza Danych — Upload, EDA i Walidacja")
    up = st.file_uploader("Wgraj plik danych (CSV/XLSX/Parquet/JSON)", type=["csv","xlsx","parquet","pq","json"])
    df = None; rep = {}
    if st.button("Użyj próbki data/avocado.csv"):
        df, rep = file_upload.load_from_path("data/avocado.csv")
    elif up is not None:
        df, rep = file_upload.load_from_bytes(up.name, up.getvalue())
    else:
        st.info("Załaduj plik, albo kliknij „Użyj próbki data/avocado.csv”.")
        return

    if rep.get("code") != "OK":
        st.error(f"{rep.get('code')}: {rep.get('message')}")
        if "details" in rep: st.exception(rep["details"])
        return

    df2, rep_types = dtype_sanitizer.sanitize_dtypes(df)
    df3, meta = runtime_preprocessor.preprocess_runtime(df2)

    st.success(f"Wczytano {rep['rows']} wierszy, {rep['cols']} kolumn.")
    with st.expander("👀 Podgląd (pierwsze 50 wierszy)", expanded=False):
        st.dataframe(df3.head(50))
    with st.expander("🧾 Raport zmian typów", expanded=False):
        st.json(rep_types)
    with st.expander("🆔 Fingerprint i zmiany nazw", expanded=True):
        st.json(meta)

    # Walidacja (Pandera)
    vres = validate_pandera(df3)
    if vres["status"] == "OK":
        st.success("Walidacja (pandera): OK")
    else:
        st.warning(f"Walidacja (pandera): {vres.get('message','-')}")

    # EDA (Polars LTS jeśli flaga włączona)
    use_polars = st.session_state.get("ui_flags", {}).get("USE_POLARS", False)
    eda = eda_integration.quick_eda(df3, use_polars=use_polars)
    st.subheader("📈 Szybkie EDA")
    st.json({k: v for k, v in eda.items() if k in ["rows","cols","numeric_cols","categorical_cols","datetime_cols"]})

    # Zapis do sesji
    st.session_state["tmiv_df"] = df3
    st.session_state["tmiv_fingerprint"] = meta.get("fingerprint")
    st.session_state["tmiv_eda"] = eda

    # GE (gated)
    if st.session_state.get("ui_flags", {}).get("USE_GX", False):
        gres = validate_gx(df3)
        if gres["status"] == "OK":
            st.success("Great Expectations: OK")
        else:
            st.info(f"Great Expectations: {gres.get('message','-')}")

def page_training(strategy: str, limits: Dict[str, int]) -> None:
    st.header("🤖 Trening Modelu — plan, HPO (gated) i uruchomienie")
    # zachowujemy flagi z sidebaru w session_state (przekazane z app.py)
    use_optuna = st.session_state.get("ui_flags", {}).get("USE_OPTUNA", False)
    df = st.session_state.get("tmiv_df")
    if df is None:
        st.info("Najpierw wczytaj dane na stronie „Analiza Danych”."); return
    plan = training_plan.build_training_plan(df, strategy=strategy)
    if plan.get("status") != "OK":
        st.error(f"{plan.get('status')}: {plan.get('message')}"); return
    st.success(f"Zidentyfikowany target: **{plan['target']}**, problem: **{plan['problem_type']}**")
    st.json(plan["validation"])

    if use_optuna:
        if st.button("🔎 Optuna HPO (krótki budżet)"):
            from config.settings import settings
            hpo = hpo_rf(df, plan["target"], plan["problem_type"], n_trials=min(10, settings.max_hpo_trials), timeout_sec= settings.max_train_time_sec)
            if hpo.get("status") == "OK":
                st.success(f"HPO OK. Najlepsze parametry: {hpo['best_params']} • wynik: {hpo['best_value']}")
                st.session_state["tmiv_hpo"] = hpo
            else:
                st.info(hpo.get("message","HPO niedostępne."))

    if st.button("🚀 Start treningu"):
        res = async_ml_trainer.train_async(
            df=df,
            target=plan["target"],
            problem_type=plan["problem_type"],
            strategy=strategy,
            max_parallel=int(limits["MAX_MODELS_PARALLEL"]),
            max_time_sec=int(limits["MAX_TRAIN_TIME_SEC"]),
            random_state=42,
        )
        st.session_state["tmiv_train"] = {"plan": plan, "result": res}
        st.success(f"Trening zakończony. Czas: {res['elapsed_sec']} s.")

def page_results() -> None:
    st.header("📈 Wyniki i Wizualizacje — eksport, ONNX, Drift")
    run = st.session_state.get("tmiv_train")
    if not run:
        st.info("Brak wyników. Uruchom trening w zakładce „Trening Modelu”."); return
    plan = run["plan"]; res = run["result"]

    # Eksport ZIP/PDF
    if st.button("📦 Eksportuj ZIP artefaktów"):
        payload = {
            "plan": plan,
            "results": res["results"],
            "eda": st.session_state.get("tmiv_eda", {}),
            "config": {
                "strategy": plan.get("strategy"),
                "problem_type": plan.get("problem_type"),
                "target": plan.get("target"),
                "fingerprint": st.session_state.get("tmiv_fingerprint"),
            },
        }
        out = "artifacts/exports/tmiv_export.zip"
        from backend.export_everything import export_zip
        export_zip(out, payload)
        st.success(f"Zapisano ZIP: {out}")

    if st.button("🧾 Generuj PDF Explainability"):
        best = res.get("best_model")
        best_metrics = res["results"].get(best, {}).get("metrics", {}) if best else {}
        from backend.ai_integration import deterministic_recommendations
        rec = deterministic_recommendations(plan["problem_type"], best or "-", best_metrics)
        context = {
            "fingerprint": st.session_state.get("tmiv_fingerprint"),
            "strategy": plan.get("strategy"),
            "problem_type": plan.get("problem_type"),
            "target": plan.get("target"),
            "validation": plan.get("validation"),
            "results": res["results"],
            "recommendations": rec,
            "seed": 42,
            "generated_at": datetime.utcnow().isoformat()+"Z",
        }
        out_pdf = "artifacts/reports/tmiv_report.pdf"
        from backend.export_explain_pdf import build_pdf
        build_pdf(out_pdf, context)
        st.success(f"Wygenerowano PDF: {out_pdf}")

    # ONNX (gated)
    if st.session_state.get("ui_flags", {}).get("USE_ONNX", False):
        st.markdown("—")
        st.subheader("ONNX (gated)")
        if st.button("⬇️ Eksportuj model do ONNX (przykładowy model RF)"):
            # Trenuj ponownie mały RF z pipeline aby mieć konkretny obiekt do eksportu
            import numpy as np
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            import pandas as pd
            from backend.async_ml_trainer import _split, _preprocessor

            X_train, X_test, y_train, y_test = _split(st.session_state["tmiv_df"], plan["target"], plan["problem_type"])
            pre = _preprocessor(X_train)
            mdl = RandomForestClassifier(n_estimators=50) if plan["problem_type"]=="classification" else RandomForestRegressor(n_estimators=50)
            pipe = Pipeline([("prep", pre), ("model", mdl)])
            pipe.fit(X_train, y_train)
            from backend.onnx_export import export_onnx
            res_ox = export_onnx(pipe, X_train if hasattr(X_train,"shape") else st.session_state["tmiv_df"].drop(columns=[plan["target"]]), "artifacts/exports/model.onnx")
            if res_ox.get("status")=="OK":
                st.success("ONNX zapisany: artifacts/exports/model.onnx")
            else:
                st.info(res_ox.get("message","ONNX niedostępne."))

    # Baseline + Drift
    st.markdown("—")
    st.subheader("Baseline & Drift")
    if st.button("💾 Zapisz baseline (EDA + fingerprint)"):
        from backend.feature_store import save_baseline
        payload = {
            "fingerprint": st.session_state.get("tmiv_fingerprint"),
            "eda": st.session_state.get("tmiv_eda", {}),
        }
        path = save_baseline(payload, name="baseline")
        st.success(f"Zapisano baseline: {path}")
    if st.button("🧭 Porównaj z baseline (drift)"):
        from backend.feature_store import load_baseline
        base = load_baseline("baseline")
        if not base:
            st.info("Brak baseline — zapisz najpierw.")
        else:
            # Użyj histogramów z EDA jako przybliżenia lub porównaj surowe dane (dla uproszczenia: surowe liczby, jeśli dostępne)
            import pandas as pd
            from backend.drift_detection import drift_report
            df_cur = st.session_state.get("tmiv_df")
            # Bez oryginalnej ramki baseline – porównanie heurystyczne niemożliwe. W Stage 6 wykorzystujemy aktualną kopię jako zastępnik.
            rep = {"note":"Brak surowych danych baseline; Stage 6 porównuje bieżące dane do siebie (diagnostyka).",
                   "now": drift_report(df_cur, df_cur)}
            st.json(rep)

def page_recommendations() -> None:
    st.header("💡 Rekomendacje biznesowe")
    run = st.session_state.get("tmiv_train")
    if not run:
        st.info("Uruchom trening, aby zobaczyć rekomendacje."); return
    plan = run["plan"]; res = run["result"]
    best = res.get("best_model")
    best_metrics = res["results"].get(best, {}).get("metrics", {}) if best else {}
    rec = deterministic_recommendations(plan["problem_type"], best or "-", best_metrics)
    for r in rec:
        st.markdown(f"- {r}")

def page_docs() -> None:
    st.header("📚 Dokumentacja")
    st.markdown("Zajrzyj do `docs/README.md` po instrukcje i plan dalszego rozwoju.")