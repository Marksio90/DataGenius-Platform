# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import os, json, zipfile
from io import BytesIO
import streamlit as st
import pandas as pd

__all__ = [
    "UIComponents",
    "render_sidebar_clean",
    "render_ai_recommendations",
    "render_training_controls",
    "render_advanced_overrides",
    "render_ai_config_simple",
    "resolve_problem_type",
    "_render_session_info",
    "_render_export_section",
    "ensure_default_navigation",
    "inject_soft_theme",
]

def inject_soft_theme():
    """Subtelne kolory, łagodne promienie, lepsza typografia."""
    st.markdown(
        """
        <style>
        :root {
            --soft-bg: #17181c;
            --soft-card: #20222a;
            --soft-accent: #7aa2f7; /* niebieski pastel */
            --soft-accent-2: #9ece6a; /* zieleń pastel */
            --soft-accent-3: #f7768e; /* róż/coral */
            --soft-text: #e8eaed;
            --soft-muted: #aab2bf;
            --soft-border: #2a2d36;
            --radius: 14px;
        }
        .stApp { background-color: var(--soft-bg); }
        .block-container { padding-top: 2rem; }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1b1d23 0%, #181a20 100%);
            border-right: 1px solid var(--soft-border);
        }
        /* Panele/containter */
        .stExpander, .stCheckbox, .stRadio, .stDownloadButton, .stButton, .stTextInput, .stNumberInput, .stSelectbox {
            border-radius: var(--radius) !important;
        }
        /* Przyciski */
        button[kind="primary"] {
            background: var(--soft-accent) !important;
            color: #0c0f14 !important;
            border-radius: var(--radius) !important;
            border: 0 !important;
        }
        .stButton>button {
            border-radius: var(--radius) !important;
        }
        /* Inputy */
        .stTextInput>div>div>input, .stPassword>div>div>input {
            border-radius: 10px !important;
        }
        /* Tagi/statusy */
        .tmiv-badge { 
            display: inline-block; padding: 2px 8px; border-radius: 999px; 
            font-size: 12px; line-height: 18px; border: 1px solid var(--soft-border); color: var(--soft-muted);
        }
        .tmiv-badge.ok { background: rgba(158,206,106,0.15); color: #b9f399; border-color: rgba(158,206,106,0.4); }
        .tmiv-badge.warn { background: rgba(247,118,142,0.12); color: #fcb0be; border-color: rgba(247,118,142,0.4); }
        .tmiv-note { font-size: 12px; color: var(--soft-muted); }
        /* Sekcja tytułowa sidebara */
        .tmiv-version { color: var(--soft-muted); font-size: 12px; margin-bottom: 8px; }
        </style>
        """
    , unsafe_allow_html=True)

# Ensure a sensible default page early to avoid "Nieznana strona: None"
def ensure_default_navigation():
    if "nav_page" not in st.session_state and "page" not in st.session_state:
        st.session_state["nav_page"] = "📊 Analiza Danych"
        st.session_state["page"] = "📊 Analiza Danych"
        st.session_state["page_slug"] = "analysis"
        st.session_state["page_label"] = "📊 Analiza Danych"

# === SecureKeyManager =====================================================
try:
    from cryptography.fernet import Fernet
except Exception:
    Fernet = None  # type: ignore

class SecureKeyManager:
    def __init__(self):
        try:
            if 'fernet_key' not in st.session_state:
                st.session_state['fernet_key'] = Fernet.generate_key() if Fernet else None
            self._fernet = Fernet(st.session_state['fernet_key']) if Fernet and st.session_state.get('fernet_key') else None
            st.session_state.setdefault('secure_storage', {})
            self._store = st.session_state['secure_storage']
        except Exception:
            self._fernet = None
            self._store = {}

    def get_key(self, name: str) -> Optional[str]:
        try:
            token = self._store.get(name)
            if not token:
                return None
            if self._fernet is None:
                return token if isinstance(token, str) else token.decode('utf-8', errors='ignore')
            return self._fernet.decrypt(token).decode('utf-8')
        except Exception:
            return None

    def set_key(self, name: str, value: str) -> bool:
        try:
            value = (value or '').strip()
            if not value:
                return False
            if self._fernet is None:
                self._store[name] = value
            else:
                self._store[name] = self._fernet.encrypt(value.encode('utf-8'))
            return True
        except Exception:
            return False

    def delete_key(self, name: str) -> bool:
        try:
            if name in self._store:
                del self._store[name]
            return True
        except Exception:
            return False

# === Problem type resolver ===============================================
def resolve_problem_type(df: pd.DataFrame, target: str) -> str:
    app_ref = st.session_state.get("app")
    trainer = None
    if app_ref is not None:
        trainer = getattr(app_ref, "ml_trainer", None) or getattr(app_ref, "trainer", None)
    if trainer is None:
        try:
            from backend.ml_integration import MLModelTrainer
            trainer = MLModelTrainer()
        except Exception:
            trainer = None
    pt = st.session_state.get("problem_type")
    if not pt and trainer is not None:
        try:
            pt = trainer.detect_problem_type(df, target)
        except Exception:
            pt = None
    if not pt:
        try:
            s = df[target]
            nunq = s.nunique(dropna=True)
            pt = "classification" if s.dtype.kind in ("O","U","S") or nunq <= max(2, int((len(s) or 1) ** 0.5)) else "regression"
        except Exception:
            pt = "regression"
    st.session_state["problem_type"] = pt
    return pt

# === AI config (safe, single header) =====================================
def render_ai_config_simple(providers: Optional[Dict[str, Dict[str, Any]]] = None, *, show_title: bool = False):
    if show_title:
        st.markdown("### 🤖 Konfiguracja AI")

    secure_mgr = SecureKeyManager()
    try:
        import keyring as system_keyring  # type: ignore
    except Exception:
        system_keyring = None  # type: ignore
    try:
        from backend.security_manager import CredentialManager
        credential_manager = CredentialManager()
    except Exception:
        credential_manager = None

    def _truthy(x) -> bool:
        try:
            return bool(x) and str(x).lower() not in {"none","nan","null",""}
        except Exception:
            return False

    def _has_key_session(provider: str) -> bool:
        try:
            return _truthy(secure_mgr.get_key(provider))
        except Exception:
            return False

    def _has_key_external(provider: str) -> bool:
        # Keyring / ENV (read-only indicator; nie barwimy nim statusu głównego)
        try:
            if system_keyring is not None:
                if _truthy(system_keyring.get_password("TMIV", provider)):
                    return True
        except Exception:
            pass
        try:
            if credential_manager and hasattr(credential_manager, "get_api_key"):
                val = credential_manager.get_api_key(provider)  # może zwrócić z ENV
                return bool(val and val.strip())
        except Exception:
            pass
        return False

    def _store_key(provider: str, api_key: str) -> bool:
        api_key = str(api_key or "").strip()
        if provider == "openai" and not api_key.startswith("sk-"):
            st.error("❌ OpenAI klucz musi zaczynać się od 'sk-'"); return False
        if provider == "anthropic" and not api_key.startswith("sk-ant-"):
            st.error("❌ Anthropic klucz musi zaczynać się od 'sk-ant-'"); return False
        try:
            secure_mgr.set_key(provider, api_key)
            try:
                import keyring as _kr  # type: ignore
                _kr.set_password("TMIV", provider, api_key)  # best-effort
            except Exception:
                pass
            try:
                st.toast("🔒 Klucz zapisany", icon="✅")
            except Exception:
                st.success("🔒 Klucz zapisany")
            return True
        except Exception:
            st.error("❌ Nie udało się zapisać klucza API.")
            return False

    if providers is None:
        providers = {
            "openai": {"label": "OpenAI API key", "placeholder": "sk-..."},
            "anthropic": {"label": "Anthropic API key", "placeholder": "sk-ant-..."},
        }

    for prov, meta in providers.items():
        st.write(f"**{meta.get('label', prov)}**")
        c = st.container()
        with c:
            col_in, col_btn, col_status = st.columns([5,2,2])
            with col_in:
                val = st.text_input("", type="password",
                                    placeholder=meta.get("placeholder",""), key=f"__input_{prov}")
            with col_btn:
                if st.button("Zapisz", key=f"btn_save_{prov}", use_container_width=True):
                    _store_key(prov, val)
            with col_status:
                ok_session = _has_key_session(prov)
                st.write("Status:")
                st.markdown(f'<span class="tmiv-badge {"ok" if ok_session else ""}">{"✅" if ok_session else "⚪"}</span>', unsafe_allow_html=True)
            # Uwaga o zewnętrznym źródle
            if not _truthy(val):
                ext = _has_key_external(prov)
                if ext and not ok_session:
                    st.caption("🔎 Wykryto klucz w systemie (keyring/ENV). Zapisz tutaj, aby używać w tej sesji.")

        st.markdown("---")

# === Session info (opcjonalne) ===========================================
def _render_session_info():
    import platform, sys
    st.subheader("ℹ️ Informacje o sesji")
    df = st.session_state.get("df")
    rows = int(df.shape[0]) if isinstance(df, pd.DataFrame) else 0
    cols = int(df.shape[1]) if isinstance(df, pd.DataFrame) else 0
    target = st.session_state.get("target") or "—"
    problem_type = st.session_state.get("problem_type") or "—"
    model_trained = bool(st.session_state.get("model_trained") or st.session_state.get("trained_model"))
    analysis_state = st.session_state.get("analysis_state") or ("in_progress" if st.session_state.get("analysis_running") else "idle")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dane", "✅" if df is not None else "❌"); st.caption(f"{rows} × {cols}")
    with c2:
        st.metric("Target", target); st.caption(f"Typ: {problem_type}")
    with c3:
        st.metric("Model", "Wytrenowany" if model_trained else "Brak"); st.caption("⏳" if not model_trained else "✅")
    with c4:
        st.metric("Analiza", "w toku" if analysis_state == "in_progress" else "idle"); st.caption("🔄" if analysis_state == "in_progress" else "⏸️")
    try:
        from backend.security_manager import CredentialManager
        cm = CredentialManager()
        openai_ok = False; anthropic_ok = False
        # Celowo nie ufamy starym zmiennym sesji
        if cm.get_api_key("openai"): openai_ok = True
        if cm.get_api_key("anthropic"): anthropic_ok = True
    except Exception:
        openai_ok = False; anthropic_ok = False
    st.markdown(f"**OpenAI klucz:** {'🔒 OK' if openai_ok else '⚪ brak'} &nbsp;|&nbsp; **Anthropic klucz:** {'🔒 OK' if anthropic_ok else '⚪ brak'}")
    st.divider()
    st.caption("System")
    try:
        import streamlit, pandas, sklearn
        st.write(f"Python: `{sys.version.split()[0]}` · Platforma: `{platform.system()} {platform.release()}` · "
                 f"Streamlit: `{streamlit.__version__}` · pandas: `{pandas.__version__}` · scikit-learn: `{sklearn.__version__}`")
    except Exception:
        st.write(f"Python: `{sys.version.split()[0]}` · Platforma: `{platform.system()} {platform.release()}`")

# === Export section + PDF ================================================
def _render_export_section():
    import datetime
    st.subheader("⬇️ Eksport / Pobieranie")
    df = st.session_state.get("df")
    model = st.session_state.get("trained_model") or st.session_state.get("model")
    metrics = st.session_state.get("metrics") or st.session_state.get("last_metrics") or {}
    problem_type = st.session_state.get("problem_type"); target = st.session_state.get("target")

    if isinstance(df, pd.DataFrame):
        try:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button("📄 Pobierz dane (CSV)", data=csv_bytes, file_name="data.csv", mime="text/csv", use_container_width=True)
        except Exception as e:
            st.warning(f"Nie udało się przygotować CSV: {e}")

    snapshot = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "problem_type": problem_type, "target": target, "metrics": metrics,
        "df_shape": (int(df.shape[0]), int(df.shape[1])) if isinstance(df, pd.DataFrame) else None,
    }
    snap_bytes = json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("🧠 Pobierz snapshot sesji (JSON)", data=snap_bytes, file_name="session_snapshot.json",
                       mime="application/json", use_container_width=True)

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if isinstance(df, pd.DataFrame):
            try:
                zf.writestr("data_preview.csv", df.head(1000).to_csv(index=False).encode("utf-8"))
            except Exception: pass
        try: zf.writestr("metrics.json", json.dumps(metrics, ensure_ascii=False, indent=2))
        except Exception: pass
        try:
            if model is not None:
                zf.writestr("model_info.txt", f"{type(model).__name__}")
        except Exception: pass
        try: zf.writestr("session.json", json.dumps(snapshot, ensure_ascii=False, indent=2))
        except Exception: pass
    st.download_button("🗜️ Pobierz pakiet wyników (ZIP)", data=buf.getvalue(), file_name="results_bundle.zip",
                       mime="application/zip", use_container_width=True)

    st.markdown("—"); st.caption("Jeśli chcesz pełny PDF z wykresami, kliknij poniżej:")
    if st.button("📄 Generuj raport PDF", use_container_width=True):
        pdf_bytes = _generate_pdf_report(snapshot=snapshot)
        if pdf_bytes:
            st.download_button("📥 Pobierz raport PDF", data=pdf_bytes, file_name="report.pdf",
                               mime="application/pdf", use_container_width=True)
        else:
            st.warning("Nie udało się wygenerować PDF. Zainstaluj opcjonalną zależność: `reportlab`.")

def _generate_pdf_report(snapshot: dict):
    from io import BytesIO
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
    except Exception:
        return None
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=A4); width, height = A4
    c.setFont("Helvetica-Bold", 16); c.drawString(2*cm, height-2*cm, "THE MOST IMPORTANT VARIABLES — Raport")
    c.setFont("Helvetica", 10)
    import datetime as _dt
    c.drawString(2*cm, height-2.6*cm, f"Data: {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(2*cm, height-3.1*cm, f"Typ: {snapshot.get('problem_type')}, Target: {snapshot.get('target')}")
    shape = snapshot.get("df_shape") or (0,0)
    c.drawString(2*cm, height-3.6*cm, f"Dane: {shape[0]} wierszy × {shape[1]} kolumn")
    y = height - 4.6*cm
    c.setFont("Helvetica-Bold", 12); c.drawString(2*cm, y, "Metryki:"); y -= 0.6*cm
    c.setFont("Helvetica", 10)
    metrics = snapshot.get("metrics") or {}
    if metrics:
        for k, v in list(metrics.items())[:30]:
            c.drawString(2.2*cm, y, f"• {k}: {v}"); y -= 0.5*cm
            if y < 3*cm: c.showPage(); y = height - 2*cm
    else:
        c.drawString(2.2*cm, y, "Brak metryk.")
    try:
        charts = st.session_state.get("export_charts_png") or st.session_state.get("last_figs_png") or []
        for img_bytes in charts[:6]:
            if y < 8*cm: c.showPage(); y = height - 2*cm
            img = ImageReader(BytesIO(img_bytes))
            c.drawImage(img, 2*cm, y-6*cm, width=width-4*cm, height=6*cm, preserveAspectRatio=True, anchor='sw')
            y -= 6.5*cm
    except Exception: pass
    c.showPage(); c.save(); return buf.getvalue()

# === AI Recommendations / Training / Advanced ============================
def render_ai_recommendations(context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    st.markdown("### 💡 Rekomendacje AI")
    problem_type = (context or {}).get("problem_type") or st.session_state.get("problem_type") or "classification"
    if problem_type == "regression":
        metrics = ["rmse","mae","r2"]
        models = [{"name": n} for n in ["LinearRegression","RandomForestRegressor","XGBoostRegressor","LightGBMRegressor","CatBoostRegressor"]]
    else:
        metrics = ["f1_weighted","roc_auc_ovr","accuracy"]
        models = [{"name": n} for n in ["LogisticRegression","RandomForestClassifier","XGBoostClassifier","LightGBMClassifier","CatBoostClassifier","SVC"]]
    recs: Dict[str, Any] = {"cv_folds":5,"metrics":metrics,"models":models,"problem_type":problem_type,"random_state":42,"test_size":0.2,"n_jobs":-1}
    c1, c2 = st.columns(2)
    with c1: st.write("**Typ problemu:**", problem_type); st.write("**Metryki:**", ", ".join(metrics))
    with c2: st.write("**CV folds:**", recs["cv_folds"]); st.write("**Modele:**", ", ".join([m["name"] for m in models]))
    st.caption("Podpowiedzi są startowe — możesz je zmodyfikować w 'Zaawansowane'.")
    return recs

def render_training_controls(recs: Dict[str, Any] | None = None) -> Tuple[bool, bool]:
    st.markdown("### 🤖 Trening — sterowanie")
    recs = dict(recs or {})
    c1, c2, c3 = st.columns(3)
    with c1: test_size = st.slider("Test size", 0.05, 0.4, float(recs.get("test_size", 0.2)), 0.05)
    with c2: cv_folds = st.number_input("CV folds", 2, 20, int(recs.get("cv_folds", 5)))
    with c3: random_state = st.number_input("Random state", 0, 9999, int(recs.get("random_state", 42)))
    n_jobs = st.select_slider("Równoległość (n_jobs)", options=[-1,1,2,4,8,16], value=int(recs.get("n_jobs", -1)))
    apply_recs = st.checkbox("Zastosuj rekomendacje AI", value=True, key="__apply_recs_train")
    train_clicked = st.button("🚀 Rozpocznij trening", use_container_width=True)
    st.session_state["train_params"] = {"test_size":float(test_size),"cv_folds":int(cv_folds),"random_state":int(random_state),"n_jobs":int(n_jobs),"apply_recs":bool(apply_recs)}
    return bool(apply_recs), bool(train_clicked)

def render_advanced_overrides(recs: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    st.markdown("### Zaawansowane (opcjonalne)")
    with st.expander("Pokaż / ukryj zaawansowane", expanded=False):
        enable = st.checkbox("Chcę ręcznie nadpisać rekomendacje", value=False, key="__adv_overrides")
        if not enable:
            st.info("AI ustawi wszystko automatycznie. Możesz włączyć nadpisywanie powyżej.")
            return recs, False
        mod = dict(recs or {})
        cv_default = int(mod.get("cv_folds", 5) or 5); metrics_default = list(mod.get("metrics", []))
        models_list = mod.get("models", recs.get("models", []) if recs else [])
        model_names = [m.get("name") if isinstance(m, dict) else str(m) for m in models_list]
        selected_default = model_names[:]
        c1, c2 = st.columns(2)
        with c1:
            mod["cv_folds"] = st.number_input("CV folds", min_value=2, max_value=20, value=cv_default)
            metrics_all = ["f1_weighted","roc_auc_ovr","accuracy","precision_weighted","recall_weighted","rmse","mae","r2","mape","smape"]
            mod["metrics"] = st.multiselect("Metryki (priorytet pierwszej)", metrics_all, default=metrics_default)
        with c2:
            selected = st.multiselect("Modele do trenowania", model_names, default=selected_default)
            mod["models"] = [m for m in models_list if (m.get("name") if isinstance(m, dict) else str(m)) in selected]
        st.success("Zastosowano nadpisania (tymczasowe w tej sesji).")
        return mod, True

# === Sidebar (soft theme + navigation; version on TOP) ===================
    ensure_auto_ai_col_desc(force=False)
    # Sekcje (etykieta, slug)
    sections: List[Tuple[str, str]] = [
        ("📊 Analiza Danych", "analysis"),
        ("🤖 Trening Modelu", "training"),
        ("📈 Wyniki i Wizualizacje", "results"),
        ("💡 Rekomendacje", "recommendations"),
        ("📚 Dokumentacja", "docs"),
    ]
    labels = [s[0] for s in sections]
    slug_by_label = {lbl: slug for lbl, slug in sections}
    label_by_slug = {slug: lbl for lbl, slug in sections}

    # Ustal indeks startowy z istniejącego stanu
    current_label = st.session_state.get("page_label") or st.session_state.get("nav_page")
    current_slug = st.session_state.get("page_slug")
    if not current_label and current_slug and current_slug in label_by_slug:
        current_label = label_by_slug[current_slug]
    start_index = labels.index(current_label) if current_label in labels else 0

    with st.sidebar:
        # WERSJA NA GÓRZE
        st.markdown('<div class="tmiv-version">🛠️ Wersja: 2.1.0 · Marksio AI Solutions</div>', unsafe_allow_html=True)

        st.header("🤖 Konfiguracja AI")
        try:
            # ważne: show_title=False => brak drugiego nagłówka
            render_ai_config_simple(show_title=False)
        except Exception:
            st.caption("Moduł konfiguracji AI niedostępny.")

        st.divider()
        st.subheader("📂 Wybierz sekcję")
        chosen = st.radio("",
                          labels,
                          index=start_index,
                          key="nav_page",
                          label_visibility="collapsed",
                          help="Wybierz główną sekcję aplikacji.")
        # Zapisz w sesji
        st.session_state["page_label"] = chosen
        st.session_state["page_slug"] = slug_by_label.get(chosen, "analysis")
        st.session_state["page"] = chosen

        st.divider()
        st.subheader("⚡ Szybkie akcje")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔁 Przeładuj", key="sb_reload", use_container_width=True, help="Przeładuj interfejs aplikacji."):
                st.rerun()
        with c2:
            reset_clicked = st.button("🧹 Resetuj sesję", key="sb_reset", use_container_width=True, help="Wyczyść stan sesji.")
        remove_df = st.checkbox("Usuń dane przy resecie", value=False, key="sb_remove_df", help="Zaznacz, aby skasować wczytane dane.")
        if reset_clicked:
            keys_to_keep = {"secure_storage", "fernet_key"}  # zachowaj szyfrowane klucze
            df_obj = None if remove_df else st.session_state.get("df")
            preserved = {k: st.session_state[k] for k in keys_to_keep if k in st.session_state}
            st.session_state.clear(); st.session_state.update(preserved)
            if df_obj is not None: st.session_state["df"] = df_obj
            try:
                st.toast("Sesja została zresetowana.", icon="🧹")
            except Exception:
                st.success("Sesja została zresetowana.")
            st.rerun()

        # WERSJA na dole usuwamy (przeniesiona na górę)
    return chosen

# === UIComponents class (shims + statics) ================================
class UIComponents:
    _render_session_info = staticmethod(_render_session_info)
    _render_export_section = staticmethod(_render_export_section)

    def render_header(self, title: str = "THE MOST IMPORTANT VARIABLES - Advanced ML Platform v2.0", subtitle: str = "Zaawansowana platforma AI/ML"):
        st.title("🎯 " + str(title)); st.caption("🚀 " + str(subtitle))

    def render_topbar(self, *args, **kwargs):
        return self.render_header(*args, **kwargs)

    def render_footer(self):
        st.markdown("---"); st.caption("🛠️ Wersja: 2.1.0 · Marksio AI Solutions · Advanced ML Platform")

    def render_session_info(self):
        return _render_session_info()

    def render_export_section(self):
        return _render_export_section()

    def render_sidebar(self) -> str:
        return render_sidebar_clean()

    def render_sidebar_clean(self) -> str:
        return render_sidebar_clean()

    def __getattr__(self, name: str):
        def _noop(*args, **kwargs):
            return None
        return _noop


# === Auto refresh of AI column descriptions ===============================
def _df_signature(df: pd.DataFrame) -> str:
    try:
        import hashlib
        sig = hashlib.sha256()
        sig.update("|".join(map(str, df.columns)).encode("utf-8"))
        sig.update("|".join(map(str, df.dtypes.astype(str))).encode("utf-8"))
        # sample up to 200 rows for stability
        preview = df.head(200).to_json(orient="split", index=False)
        sig.update(preview.encode("utf-8"))
        return sig.hexdigest()
    except Exception:
        try:
            return str((tuple(df.columns), tuple(map(str, df.dtypes)), int(df.shape[0])))
        except Exception:
            return "df_sig_fallback"

def _compute_ai_col_desc(df: pd.DataFrame) -> Dict[str, str]:
    """Try backend AI recommender; fallback to heuristics. Returns {col: description}."""
    desc: Dict[str, str] = {}
    # 1) Backend integration if available
    try:
        from backend.ai_integration import AIRecommender  # type: ignore
        ai = AIRecommender()
        # prefer describe_columns, fallback to explain_columns
        if hasattr(ai, "describe_columns"):
            maybe = ai.describe_columns(df)  # expected dict
            if isinstance(maybe, dict) and maybe:
                return {str(k): str(v) for k, v in maybe.items()}
        if hasattr(ai, "explain_columns"):
            maybe = ai.explain_columns(df)
            if isinstance(maybe, dict) and maybe:
                return {str(k): str(v) for k, v in maybe.items()}
    except Exception:
        pass
    # 2) Heuristic fallback
    try:
        for c in df.columns:
            s = df[c]
            nnull = int(s.isna().sum())
            nunq = int(s.nunique(dropna=True))
            dtype = str(s.dtype)
            ratio_missing = f"{(nnull/len(df))*100:.1f}%" if len(df) else "0%"
            if s.dtype.kind in ("i","u","f"):
                base = "Cecha numeryczna"
                if nunq <= max(2, int(len(s) ** 0.5)):
                    base += " (niewiele unikatów — możliwa kategoria zakodowana liczbowo)"
            elif s.dtype.kind in ("b",):
                base = "Cecha binarna (True/False)"
            elif s.dtype.kind in ("O","U","S"):
                base = "Cecha kategoryczna/tekstowa"
            elif "datetime" in dtype:
                base = "Cecha czasowa / data"
            else:
                base = f"Typ {dtype}"
            desc[c] = f"{base}. Unikatów: {nunq}. Braki: {ratio_missing}."
    except Exception:
        pass
    return desc

def ensure_auto_ai_col_desc(force: bool = False) -> None:
    df = st.session_state.get("df")
    if not isinstance(df, pd.DataFrame) or df is None or getattr(df, "empty", False):
        return
    sig = _df_signature(df)
    if force or st.session_state.get("_ai_desc_sig") != sig:
        try:
            desc = _compute_ai_col_desc(df)
            if isinstance(desc, dict) and desc:
                st.session_state["ai_col_desc"] = desc
                st.session_state["_ai_desc_sig"] = sig
                st.session_state["ai_desc_count"] = len(desc)
                try:
                    st.toast("🔁 Zaktualizowano opisy kolumn (AI)", icon="✨")
                except Exception:
                    pass
        except Exception:
            # nie przerywamy aplikacji
            pass


# convenience binding
UIComponents.ensure_auto_ai_col_desc = staticmethod(ensure_auto_ai_col_desc)
