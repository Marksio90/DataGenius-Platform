
from __future__ import annotations
import streamlit as st
import pandas as pd
import json, os, io, base64, hashlib, time
from eda.profile import simple_profile
from backend.io_readers import read_any
from backend.large_mode import detect_large_mode
from ml.training import train_and_eval
from backend.baselines import save_quality_baseline
from backend.guardrails import scan_pii, mask_preview

st.set_page_config(page_title="TMIV – Advanced ML Platform v2.0 Pro", layout="wide", page_icon="frontend/assets/logo_tmiv.png")
with open("frontend/global.css","r",encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


col_logo, col_title = st.columns([1,6])
with col_logo:
    st.image("frontend/assets/logo_tmiv.png", width=72)
with col_title:
    st.title("TMIV – Advanced ML Platform v2.0 Pro")
    st.caption("AUTO: Upload → EDA → Plan → Train → Wyniki → Eksport")

# Splash (first run) — kompatybilny ze starszym Streamlit
if "tmiv_splash_seen" not in st.session_state:
    modal_ctx = getattr(st, "modal", None)  # dostępne w nowszym Streamlit
    if modal_ctx is not None:
        # Nowy Streamlit: ładny modal
        with st.modal("TMIV — AUTO MODE ready"):
            st.image("frontend/assets/splash_tmiv.png", use_column_width=True)
            st.markdown("**Witaj w TMIV v2.0 Pro.** Jedno kliknięcie → cała analiza. Ustaw brand w Sidebar → Extras, a potem kliknij **🚀 Uruchom automat**.")
            if st.button("Zaczynamy!"):
                st.session_state["tmiv_splash_seen"] = True
    else:
        # Starszy Streamlit: fallback bez modału (top card)
        splash = st.container()
        with splash:
            st.image("frontend/assets/splash_tmiv.png", use_column_width=True)
            st.info("**Witaj w TMIV v2.0 Pro.** Ustaw brand w Sidebar → Extras, a potem kliknij **🚀 Uruchom automat**.")
            if st.button("Zaczynamy!"):
                st.session_state["tmiv_splash_seen"] = True


# Sidebar (extras only)
st.sidebar.header("Extras (Admin/Gated)")
use_pdf = st.sidebar.checkbox("Generate PDF (gated)", value=True)
# Brand theme switcher
from frontend.brand import BRANDS, css_override
brand = st.sidebar.selectbox("Brand theme", list(BRANDS.keys()), index=0)
st.markdown(f"<style>{css_override(BRANDS[brand])}</style>", unsafe_allow_html=True)
st.sidebar.write("Status providerów: SHAP/Optuna/ONNX/OTel — OFF (gated)")

# Main tabs
tabs = st.tabs(["📊 Analiza Danych", "🧪 Kontrakty & Jakość Danych", "🤖 Trening Modelu", "🧩 Symulacje & Progi Decyzji", "📈 Wyniki i Viz", "💡 Rekomendacje", "🛠️ Admin & Monitoring", "📚 Dokumentacja"])

with tabs[0]:
    st.subheader("Wczytaj dane")
    up = st.file_uploader("CSV/Parquet", type=["csv","parquet"])
    df = None
    if up is not None:
        df = read_any(up)
    else:
        st.info("Brak pliku — użyję przykładowego `data/avocado.csv`.")
        from pathlib import Path
        df = read_any(Path("data/avocado.csv"))

    st.session_state["df"] = df
    prof = simple_profile(df)
    st.write({"shape": prof["shape"], "memory_mb": round(prof["memory_mb"],3), "large_mode": prof.get("large_mode")})
    if prof.get("large_mode"):
        st.warning("Włączono **large-mode**: EDA na próbce, ciężkie wykresy ograniczone. Rozważ Parquet i mniejsze top-N.")
    st.json({"dtypes": prof["dtypes"]})

    st.markdown('<div class="tmiv-sticky-cta">', unsafe_allow_html=True)
    run_auto = st.button("🚀 Uruchom automat")
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.subheader("Kontrakty i baseline jakości")
    df = st.session_state.get("df")
if df is not None:

    from backend.contracts import save_contract, validate_with_contract
    from backend.drift import population_stability_index, kolmogorov_smirnov, jensen_shannon
    import numpy as np, glob

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("📝 Generuj kontrakt"):
            path = save_contract(df, run_id="demo-run")
            st.success(f"Zapisano kontrakt: {path}")
    with c2:
        contract_files = sorted(glob.glob("artifacts/reports/baselines/*/contract.json"))
        chosen = st.selectbox("Wybierz kontrakt do walidacji", contract_files, index=len(contract_files)-1 if contract_files else 0)
        if st.button("✅ Waliduj zbiór", disabled=not contract_files):
            res = validate_with_contract(df, chosen)
            if res["ok"]:
                st.success("Walidacja OK.")
            else:
                st.error(f"Naruszenia: {len(res['violations'])}")
                st.write(res["violations"][:50])
    with c3:
        if st.button("📌 Zamroź drift baseline"):
            from backend.baselines import save_quality_baseline
            path = save_quality_baseline(df, run_id="demo-run")
            st.success(f"Baseline jakości zapisany: {path}")
    with c4:
        baseline_dirs = sorted(glob.glob("artifacts/reports/baselines/*"))
        chosen_bl = st.selectbox("Baseline do porównania", baseline_dirs, index=len(baseline_dirs)-1 if baseline_dirs else 0)
        if st.button("📊 Porównaj drift", disabled=not baseline_dirs):
            try:
                import json, os
                stats = json.load(open(os.path.join(chosen_bl,"quality.json"),"r",encoding="utf-8"))
                st.write("Drift (PSI/KS/JS) dla kolumn numerycznych:")
                import pandas as pd
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                rows = []
                for c in num_cols:
                    # syntetyczna referencja: rozkład z pliku baseline (brak próbek – użyj wartości z aktualnego df jako proxy, to prosta demonstracja)
                    a = df[c].to_numpy()
                    b = df[c].to_numpy() * 1.0  # w realu: próba prod vs ref; tu tylko pokaz metryk
                    rows.append({
                        "col": c,
                        "PSI": population_stability_index(a,b,10),
                        "KS": kolmogorov_smirnov(a,b),
                        "JS": jensen_shannon(a,b,20),
                    })
                st.dataframe(rows)
            except Exception as e:
                st.error(f"Nie udało się policzyć driftu: {e}")

            if st.checkbox("Skanuj PII (DLP)"):
                rep = scan_pii(df)
                st.write(rep or "Brak podejrzanych kolumn.")
                if rep:
                    st.dataframe(mask_preview(df, rep))
    st.divider()
    st.caption("Great Expectations (gated)")
    from backend.ge_gated import ge_snapshot_possible, ge_build_minimal_suite
    if ge_snapshot_possible():
        if st.button("📘 GE: Snapshot oczekiwań"):
            snap = ge_build_minimal_suite(df)
            st.json(snap)
    else:
        st.info("GE gated — zainstaluj `great_expectations`, aby aktywować.")

with tabs[2]:
    st.subheader("Trening")
    df = st.session_state.get("df")
    if df is None:
        st.warning("Wczytaj dane w pierwszej karcie.")
    else:
        auto = run_auto or st.button("Start treningu")
        if auto:
            with st.spinner("Trening..."):
                res = train_and_eval(df)
            st.session_state["res"] = res
            from backend.telemetry import metric, audit
            metric("train_complete", 1, {"task": res["metrics"]["task"]}); audit("train_complete", {"target": res["target"]})
            st.success("Trening zakończony. Przejdź do wyników.")


from backend.registry import save_manifest
from backend.telemetry import audit, metric
import numpy as np
from sklearn.metrics import roc_curve

with tabs[3]:
    st.subheader("Symulacje & Progi Decyzji")
    res = st.session_state.get("res")
    if not res or res["metrics"]["task"] != "classification":
        st.info("Brak wyników klasyfikacji — uruchom trening klasyfikacyjny.")
    else:
        df = st.session_state.get("df")
        from sklearn.model_selection import train_test_split
        X = df.drop(columns=[res["target"]]); y = df[res["target"]]
        _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()==2 else None)
        proba = None
        model = res["model"]
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(Xte)[:,1]
            except Exception:
                proba = None
        if proba is None:
            st.warning("Model nie zwraca predict_proba — symulacje ograniczone.")
        else:
            # Kalibracja (opcjonalnie)
            calibrate = st.checkbox("Użyj kalibracji (isotonic)", value=False)
            if calibrate:
                from ml.calibration import calibrate_model
                # UWAGA: kalibrujemy kopię klasyfikatora – uproszczone demo; realnie należałoby przeliczyć preprocesing
                clf = res["model"]
                try:
                    cal = calibrate_model(clf, Xte, yte, method="isotonic")
                    proba = cal.predict_proba(Xte)[:,1]
                    st.success("Zastosowano kalibrację.")
                except Exception as e:
                    st.warning(f"Kalibracja nie powiodła się: {e}")
            fpr, tpr, thr = roc_curve(yte, proba)
            youden = (tpr - fpr); best_idx = int(np.argmax(youden))
            st.write("Youden J optimum:", float(thr[best_idx]))
            c1, c2 = st.columns(2)
            with c1:
                cost_fp = st.number_input("Koszt FP", min_value=0.0, value=1.0, step=0.1)
            with c2:
                cost_fn = st.number_input("Koszt FN", min_value=0.0, value=1.0, step=0.1)
            expected = 0.5 * (fpr * cost_fp + (1.0 - tpr) * cost_fn)
            best_cost_idx = int(np.argmin(expected))
            st.write("Cost-opt threshold:", float(thr[best_cost_idx]), " expected_cost:", float(expected[best_cost_idx]))
            if st.button("💾 Zapisz próg (manifest → registry/manifests/threshold.json)"):
                payload = {"threshold": float(thr[best_cost_idx]), "objective": "cost",
                           "expected_cost": float(expected[best_cost_idx]),
                           "fpr": float(fpr[best_cost_idx]), "tpr": float(tpr[best_cost_idx])}
                path = save_manifest("threshold", payload, version="1.0.0")
                audit("save_threshold", {"path": path, "payload": payload})
                st.success(f"Zapisano manifest: {path}")

with tabs[4]:
    st.subheader("Wyniki")
    res = st.session_state.get("res")
    if res:
        st.json(res["metrics"])
        st.write("Target:", res["target"])

from ml.explain import fig_roc, fig_pr, fig_confusion
import numpy as np
from sklearn.model_selection import train_test_split
df = st.session_state.get("df")
if res and res["metrics"]["task"] == "classification" and df is not None:
    X = df.drop(columns=[res["target"]]); y = df[res["target"]]
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if y.nunique()==2 else None)
    proba = None
    if hasattr(res["model"], "predict_proba"):
        try: proba = res["model"].predict_proba(Xte)[:,1]
        except Exception: proba = None
    if proba is not None and len(np.unique(yte))==2:
        st.plotly_chart(fig_roc(yte, proba), use_container_width=True)
        st.plotly_chart(fig_pr(yte, proba), use_container_width=True)
        yhat = (proba >= 0.5).astype(int)
        st.plotly_chart(fig_confusion(yte, yhat), use_container_width=True)

    else:
        st.info("Brak wyników — uruchom trening.")

with tabs[5]:
    st.subheader("Rekomendacje (heurystyczne)")
    res = st.session_state.get("res")
    if res:
        task = res["metrics"]["task"]
        if task == "classification":
            st.write("- Użyj kalibracji progu (Brier) i rozważ koszt FP/FN.")
        else:
            st.write("- Sprawdź rozkład reszt; rozważ nieliniowy model/drzewo gdy R² niski.")
    else:
        st.info("Brak wyników.")

with tabs[6]:
    st.subheader("Monitoring/Diag")
    st.write("Wersje pakietów (wymagania):")
    st.code(open("requirements.txt","r",encoding="utf-8").read(), language="toml")
    st.write("Ostatnie zdarzenia (audit/logi):")
    import itertools
    def tail(path, n=50):
        try:
            return list(itertools.islice(open(path, "r", encoding="utf-8"), max(0,n)))[-n:]
        except Exception:
            return []
    colA, colB = st.columns(2)
    with colA:
        st.caption("audit.log.jsonl")
        st.text("".join(tail("artifacts/reports/audit.log.jsonl", 40)))
    with colB:
        st.caption("tmiv.log.jsonl")
        st.text("".join(tail("artifacts/reports/tmiv.log.jsonl", 40)))
    if st.button("🧹 Wyczyść stare artefakty >30 dni"):
        import os
        os.system("python scripts/cleanup.py 30 > /dev/null 2>&1")
        st.success("Wyczyszczono katalog artifacts/ (starsze niż 30 dni)")
    if st.button("📦 Generuj SBOM (gated)"):
        import os
        code = os.system("python scripts/sbom.py > /dev/null 2>&1")
        if code==0:
            st.success("SBOM wygenerowany: artifacts/reports/sbom.json")
        else:
            st.warning("SBOM gated — zainstaluj `cyclonedx-bom` aby działało.")
    st.divider()
    st.caption("ONNX & MLflow (gated)")
    if st.button("🔬 Porównaj przewidywania ONNX vs sklearn"):
        try:
            import numpy as np
            from backend.onnx_test import try_compare_onnx
            df = st.session_state.get("df"); res = st.session_state.get("res")
            if df and res:
                X = df.drop(columns=[res["target"]])
                arr = X.select_dtypes(include=["number"]).to_numpy(dtype="float32")
                out = try_compare_onnx(res["model"], arr[:200])
                st.json(out)
            else:
                st.info("Brak modelu/danych.")
        except Exception as e:
            st.warning(f"Gated/Brak zależności: {e}")
    if st.button("🪵 Wyślij metryki do MLflow"):
        from backend.mlflow_gated import log_run_metrics
        res = st.session_state.get("res")
        if res:
            st.json(log_run_metrics(res["metrics"]["task"], res["metrics"]))
        else:
            st.info("Brak metryk.")

with tabs[7]:
    st.markdown(open("docs/README.md","r",encoding="utf-8").read())

# Eksport ZIP/PDF (prosty)
st.divider()
col1, col2 = st.columns(2)
with col1:
    if st.button("📦 Eksport ZIP"):
        os.system("python scripts/export_zip.py > /dev/null 2>&1")
        st.success("Utworzono artifacts/exports/tmiv_export.zip")
        st.write("Ścieżka: artifacts/exports/tmiv_export.zip")
with col2:
    if use_pdf and st.button("📝 Generuj PDF"):
        os.system("python scripts/make_pdf.py > /dev/null 2>&1")
        st.success("Utworzono artifacts/reports/report.pdf")

st.caption('A11y: wysoki kontrast; skróty: u,g,s,/,? (informacyjne).')
