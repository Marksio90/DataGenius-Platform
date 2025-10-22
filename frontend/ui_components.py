import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import pickle
import time
from backend.safe_utils import truthy_df_safe

def rerun_once(key='_rerun_once'):
    import streamlit as st
    if not st.session_state.get(key, False):
        st.session_state[key] = True
        st.rerun()


PLAN_HELP = {
    "Train size": "Jaki procent danych trafia do zbioru treningowego. Więcej dla treningu = mniej dla walidacji/testu.",
    "Strategia treningu": "Zestaw predefiniowanych wyborów modeli i ustawień dla małych/średnich/dużych zbiorów.",
    "Optymalizacje": "Dodatkowe techniki jak stacking/blending, które mogą poprawić wyniki kosztem czasu.",
    "CV folds": "Liczba foldów w walidacji krzyżowej. Więcej = stabilniejsza ocena, ale wolniej.",
    "Metryka optymalizacji": "Kryterium wyboru najlepszego modelu. Np. F1 (ważona) dla niezbalansowanych klas.",
    "Wagi klas": "Włącza wagi klas, żeby karać bardziej błędy wobec mniejszości klasy.",
    "Próbkowanie": "Metody równoważenia danych (undersampling/oversampling).",
    "Udział mniejszości (jeśli próbkowanie)": "Docelowy udział klasy mniejszościowej po próbkowaniu.",
    "Optymalizacja progu": "Dostraja próg decyzji (np. >0.5) pod wybraną metrykę.",
    "Typ CV": "Rodzaj walidacji (np. StratifiedKFold dla klasyfikacji).",
    "Random state": "Ziarno losowości dla replikowalności wyników.",
    "Imputacja num.": "Jak uzupełniać braki w kolumnach liczbowych (np. średnią).",
    "Imputacja kat.": "Jak uzupełniać braki w kolumnach kategorycznych (np. najczęstsza).",
    "Skalowanie": "Standaryzacja cech liczbowych. Często pomaga modelom liniowym.",
    "Kodowanie kat.": "Sposób zamiany kategorii na liczby (np. One-Hot).",
    "Outliery": "Obsługa obserwacji odstających (wykrywanie/ograniczanie wpływu).",
    "Min. wariancja (selekcja)": "Filtr usuwa cechy o bardzo niskiej zmienności.",
    "Selekcja cech": "Włącza selekcję cech (np. top-K wg ważności).",
    "Top-K (jeśli selekcja)": "Ile najlepszych cech zostawić przy włączonej selekcji.",
    "Rodziny modeli": "Grupy modeli do rozważenia (drzewa, GBM, XGBoost, LightGBM itp.).",
    "Równoległość (n_jobs)": "Ile rdzeni CPU użyć. -1 = wszystkie dostępne.",
    "Limit czasu treningu [s] / model": "Twardy limit czasu na pojedynczy model.",
    "HPO: liczba prób (n_trials)": "Liczba prób w strojenia hiperparametrów (im więcej tym lepiej, ale wolniej).",
}

# ✅ DODANE: Bezpieczne zarządzanie kluczami API
from cryptography.fernet import Fernet

# ✅ DODANE: opcjonalny systemowy keyring (produkcyjny storage)
try:
    import keyring as system_keyring  # type: ignore
except Exception:  # pragma: no cover
    system_keyring = None  # type: ignore


# --- Anti-dup helpers ---
if "_once_flags" not in st.session_state:
    st.session_state["_once_flags"] = set()


def _once_per_run(flag: str) -> bool:
    """Zwraca True tylko przy pierwszym wywołaniu w danym rerunie."""
    if flag in st.session_state["_once_flags"]:
        return False
    st.session_state["_once_flags"].add(flag)
    return True


def _fmt_float_safe(x, digits=2):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def _to_serializable(obj: Any) -> Any:
    """
    Odporna konwersja obiektów (np. numpy/pandas) na typy JSON-owalne.
    """
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


# ✅ Cache funkcja poza klasą (czytelniej i stabilniej z dekoratorem)
@st.cache_data(ttl=60, show_spinner=False)
def _get_dataset_metrics_cached(_df: pd.DataFrame) -> dict:
    """Oblicz metryki datasetu (cached dla performance)"""
    try:
        numeric_cols = len(_df.select_dtypes(include=[np.number]).columns)
        memory_mb = float(_df.memory_usage(deep=True).sum() / (1024**2))
        missing_total = _df.isnull().sum().sum()
        total_cells = _df.shape[0] * _df.shape[1]
        completeness = 100 - float((missing_total / max(total_cells, 1)) * 100)

        return {
            'rows': int(len(_df)),
            'cols': int(len(_df.columns)),
            'numeric_cols': numeric_cols,
            'total_cols': int(len(_df.columns)),
            'memory_mb': memory_mb,
            'completeness': completeness
        }
    except Exception:
        return {
            'rows': 0, 'cols': 0, 'numeric_cols': 0,
            'total_cols': 0, 'memory_mb': 0.0, 'completeness': 0.0
        }



try:
    from backend.security_manager import credential_manager  # type: ignore
except ImportError:
    # Fallback jeśli security_manager nie jest dostępny
    credential_manager = None


class UIComponents:
    """
    Klasa zawierająca wszystkie komponenty interfejsu użytkownika
    """

    def __init__(self):
        pass
    # --- Compatibility wrapper for older callers ---
    def render_ai_config(self):
        """Public alias kept for backward compatibility (calls _render_ai_config_simple)."""
        try:
            return self._render_ai_config_simple()
        except AttributeError:
            # if somehow missing, create a minimal inline renderer
            import streamlit as st
            st.warning("⚠️ Brak modułu konfiguracji AI. Zaktualizuj UIComponents.")
            return None

    def render_header(self):
        """Renderuje nagłówek aplikacji"""

        # Custom CSS (ulepszone style + poprawka dla wykresów)
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
        }
        .header-title {
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .header-subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            text-align: center;
            margin: 0.5rem 0;
        }
        .metric-card h3 {
            color: white !important;
            font-size: 2rem;
            margin: 0;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
        }
        .metric-card p {
            color: white !important;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            font-size: 0.9rem;
            text-shadow: 1px 1px 1px rgba(0,0,0,0.2);
        }
        .feature-card {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        }
        .success-message {
            background: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #28a745;
        }
        .warning-message {
            background: #fff3cd;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
        }

        /* NOWE - Poprawki dla wykresów Plotly */
        .stPlotlyChart {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 10px 0;
            padding: 10px;
        }

        /* Zapobiega nakładaniu się wykresów */
        .element-container {
            margin-bottom: 20px;
        }

        /* Responsive dla wykresów */
        @media (max-width: 768px) {
            .stPlotlyChart {
                margin: 5px 0;
                padding: 5px;
            }
        }

        /* Poprawka dla sidebar */
        section[data-testid="stSidebar"] { padding-top: 1rem; }

        /* Lepsze spacing między sekcjami */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* Poprawki dla expanderów */
        .streamlit-expanderHeader {
            font-weight: bold;
            font-size: 1.1em;
        }

        .streamlit-expanderContent {
            padding: 15px;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            background: #fafafa;
        }
        </style>
        """, unsafe_allow_html=True)

        # Header główny
        st.markdown("""
        <div class="main-header">
            <div class="header-title">🎯 The Most Important Variables</div>
            <div class="header-subtitle">
                🚀 Zaawansowana platforma AI/ML do inteligentnej analizy najważniejszych cech w danych
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Container dla metryk
        metrics_container = st.container()
        with metrics_container:
            if _once_per_run("header_metrics"):
                self._render_dataset_metrics()

    def _render_dataset_metrics(self):
        """Renderuje ulepszone metryki datasetu (z cache)"""
        if "df" in st.session_state and st.session_state.get('data_loaded', False):
            df = st.session_state["df"]

            # ✅ Użyj cached metrics zamiast obliczać za każdym razem
            metrics = _get_dataset_metrics_cached(df)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                rows_formatted = f"{metrics['rows']:,}"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 {rows_formatted}</h3>
                    <p>Wierszy danych</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔢 {metrics['numeric_cols']}/{metrics['total_cols']}</h3>
                    <p>Kolumn numerycznych</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>✨ {_fmt_float_safe(metrics['completeness'], 1)}%</h3>
                    <p>Kompletność danych</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💾 {_fmt_float_safe(metrics['memory_mb'], 1)}MB</h3>
                    <p>Rozmiar w pamięci</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Puste metryki gdy brak danych
            col1, col2, col3, col4 = st.columns(4)

            metrics = [
                ("📊", "Wierszy danych"),
                ("🔢", "Kolumn numerycznych"),
                ("✨", "Kompletność danych"),
                ("💾", "Rozmiar w pamięci")
            ]

            for col, (icon, label) in zip([col1, col2, col3, col4], metrics):
                with col:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{icon} --</h3>
                        <p>{label}</p>
                    </div>
                    """, unsafe_allow_html=True)

    def render_sidebar(self) -> str:
        """Renderuje sidebar z nawigacją"""

        with st.sidebar:
            # ✅ Bezpieczna konfiguracja AI (keyring + Fernet fallback)
            self._render_ai_config_simple()

            # Status aplikacji
            self._render_status_indicator()

            st.divider()

            # Menu nawigacyjne
            pages = [
                "📊 Analiza Danych",
                "🤖 Trening Modelu",
                "📈 Wyniki i Wizualizacje",
                "💡 Rekomendacje",
                "📚 Dokumentacja"
            ]

            selected_page = st.radio(
                "Wybierz sekcję:",
                pages,
                index=0
            )

            st.divider()

            # Informacje o sesji
            self._render_session_info()

            # Szybkie akcje
            st.markdown("### ⚡ Szybkie akcje")

            if st.button("🔄 Resetuj sesję", key="btn_reset_session", use_container_width=True, help="Wyczyść wszystkie dane i rozpocznij od nowa"):
                self._reset_session()

            # NAPRAWIONY EKSPORT
            self._render_export_section()

            # Footer sidebar - ZAKTUALIZOWANY BRANDING
            st.markdown("---")
            st.markdown("**🛠️ Wersja:** 2.1.0")
            st.markdown("**🚀 Made by Marksio AI Solutions**")
            st.markdown("**⭐ Advanced ML Platform**")

        return selected_page

    def render_ai_config(self):
        """Public alias for backward compatibility."""
        return self._render_ai_config_simple()

    def _render_ai_config_simple(self):
        """BEZPIECZNA konfiguracja kluczy AI z keyring + szyfrowany fallback."""
        import os, streamlit as st
        try:
            import keyring as system_keyring  # type: ignore
        except Exception:
            system_keyring = None  # type: ignore
    
        secure_mgr = SecureKeyManager()
        st.markdown("### 🤖 Konfiguracja AI")
    
        def _has_key(provider: str) -> bool:
            try:
                if secure_mgr.get_key(provider):
                    return True
            except Exception:
                pass
            if system_keyring is not None:
                try:
                    if system_keyring.get_password("TMIV", provider):
                        return True
                except Exception:
                    pass
            if os.getenv(f"{provider}".upper() + "_API_KEY"):
                return True
            return bool(st.session_state.get(f"{provider}_api_key"))
    
        def _store_key(provider: str, api_key: str) -> bool:
            api_key = str(api_key or '').strip()
            if provider == 'openai' and not api_key.startswith('sk-'):
                st.error("❌ OpenAI klucz musi zaczynać się od 'sk-'")
                return False
            if provider == 'anthropic' and not api_key.startswith('sk-ant-'):
                st.error("❌ Anthropic klucz musi zaczynać się od 'sk-ant-'")
                return False
            try:
                if system_keyring is not None:
                    system_keyring.set_password("TMIV", provider, api_key)
                    st.success("🔒 Zapisano w systemowym keyringu")
                    return True
            except Exception:
                st.warning("⚠️ Nie udało się zapisać w keyringu – użyję szyfrowanej sesji")
            if secure_mgr.set_key(provider, api_key):
                st.warning("⚠️ Zapisano w sesji (szyfrowane). Rozważ użycie systemowego keyringu.")
                return True
            return False
    
        def _delete_key(provider: str) -> bool:
            ok = True
            try:
                if system_keyring is not None:
                    try:
                        system_keyring.delete_password("TMIV", provider)
                    except Exception:
                        pass
            except Exception:
                ok = False
            ok = secure_mgr.delete_key(provider) and ok
            return ok
    
        with st.expander("🔑 Klucze API", expanded=False):
            for provider, label in [("openai","OpenAI"), ("anthropic","Anthropic")]:
                c1, c2, c3 = st.columns([3,1,1])
                with c1:
                    placeholder = "sk-..." if provider=="openai" else "sk-ant-..."
                    api_key = st.text_input(f"{label} API Key", type="password", placeholder=placeholder, key=f"{provider}_input")
                with c2:
                    if st.button("Zapisz", key=f"{provider}_save"):
                        if _store_key(provider, api_key):
                            st.success(f"✅ {label} – zapisano")
                        else:
                            st.error(f"❌ {label} – nie zapisano")
                with c3:
                    if st.button("Usuń", key=f"{provider}_del"):
                        if _delete_key(provider):
                            st.success(f"🗑️ {label} – usunięto")
                        else:
                            st.error(f"❌ {label} – nie usunięto")
                st.caption(f"Status: {'🟢' if _has_key(provider) else '🔴'}")
    
        st.info("🔒 Preferowany storage: systemowy keyring. Fallback: szyfrowana sesja (Fernet).")

    def _render_status_indicator(self):
        """Renderuje wskaźnik statusu aplikacji"""
        st.markdown("### 📊 Status aplikacji")

        # Status wczytania danych
        if st.session_state.get('data_loaded', False):
            st.success("✅ Dane wczytane")
        else:
            st.error("❌ Brak danych")

        # Status treningu modelu
        if st.session_state.get('model_trained', False):
            st.success("✅ Model wytrenowany")
        else:
            st.warning("⏳ Model nie wytrenowany")

        # Status analizy
        if st.session_state.get('analysis_complete', False):
            st.success("✅ Analiza zakończona")
        else:
            st.info("🔄 Analiza w toku")

    def _render_export_section(self):
        """Shim: brak właściwej implementacji (_render_export_section). Zapobiega awarii UI."""
        import streamlit as st
        st.info('⚙️ _render_export_section – placeholder. Zaktualizuj moduł UI, aby włączyć pełną funkcję.')

    def _render_session_info(self):
        """Shim: brak właściwej implementacji (_render_session_info). Zapobiega awarii UI."""
        import streamlit as st
        st.info('⚙️ _render_session_info – placeholder. Zaktualizuj moduł UI, aby włączyć pełną funkcję.')

class SecureKeyManager:
    """
    Minimalny menedżer bezpiecznych kluczy używany przez UI:
    - przechowuje klucze zaszyfrowane w st.session_state (Fernet),
    - nie koliduje z systemowym keyringiem (który obsługujemy osobno).
    """
    def __init__(self):
        try:
            import streamlit as st
            from cryptography.fernet import Fernet
            # Klucz szyfrujący sesji
            if 'fernet_key' not in st.session_state:
                st.session_state['fernet_key'] = Fernet.generate_key()
            self._fernet = Fernet(st.session_state['fernet_key'])
            # Namespace w sesji
            st.session_state.setdefault('secure_storage', {})
            self._store = st.session_state['secure_storage']
        except Exception:
            # Skrajny fallback: brak szyfrowania (niezalecane, ale lepsze niż crash)
            self._fernet = None
            self._store = {}

    def get_key(self, name: str):
        try:
            token = self._store.get(name)
            if not token:
                return None
            if self._fernet is None:
                return token
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

    def _render_ai_config_simple(self):
        """BEZPIECZNA konfiguracja AI - KOMPLETNA IMPLEMENTACJA"""
        st.markdown("### 🤖 Konfiguracja AI")

        secure_mgr = SecureKeyManager()

        # --- helpery bezpieczeństwa z realnym keyringiem + fallback ---
        def _has_key(provider: str) -> bool:
            """Sprawdź czy mamy klucz dla dostawcy."""
            # 1) Session (szyfrowany)
            try:
                dec = secure_mgr.get_key(provider)
                if truthy_df_safe(dec):
                    return True
            except Exception:
                pass

            # 2) System keyring
            if system_keyring is not None:
                try:
                    val = system_keyring.get_password("TMIV", provider)
                    if truthy_df_safe(val):
                        return True
                except Exception:
                    pass

            # 3) CredentialManager (ENV / session / keyring)
            try:
                if truthy_df_safe(credential_manager) and hasattr(credential_manager, "get_api_key"):
                    val = credential_manager.get_api_key(provider)
                    return bool(val and val.strip())
            except Exception:
                pass

            # 4) Legacy session backup
            return bool(st.session_state.get(f"{provider}_api_key"))

        def _store_key(provider: str, api_key: str) -> bool:
            """Bezpiecznie zapisz klucz API (keyring → Fernet/session)."""
            try:
                # Walidacja prefiksu (+ sanity trim)
                api_key = str(api_key or "").strip()
                if provider == "openai" and not api_key.startswith("sk-"):
                    st.error("❌ OpenAI klucz musi zaczynać się od 'sk-'")
                    return False
                if provider == "anthropic" and not api_key.startswith("sk-ant-"):
                    st.error("❌ Anthropic klucz musi zaczynać się od 'sk-ant-'")
                    return False

                # 1) System keyring (jeśli dostępny)
                if system_keyring is not None:
                    try:
                        system_keyring.set_password("TMIV", provider, api_key)
                        # lokalny backup w sesji (szyfrowany)
                        secure_mgr.store_key(provider, api_key)
                        return True
                    except Exception:
                        pass

                # 2) Szyfrowana sesja (fallback)
                if secure_mgr.store_key(provider, api_key):
                    # (opcjonalny) legacy session backup, jeśli ktoś go czyta poza UI
                    st.session_state[f"{provider}_api_key"] = api_key
                    st.warning("⚠️ Zapisano w sesji (szyfrowane). Rozważ użycie systemowego keyringu.")
                    return True

                return False

            except Exception as e:
                st.error(f"❌ Nie udało się zapisać klucza: {e}")
                # Ostateczny fallback – zwykła sesja (niezalecane)
                try:
                    st.session_state[f"{provider}_api_key"] = api_key
                    st.warning("⚠️ Klucz zapisany tylko w sesji ze względu na błąd keyringu")
                    return True
                except Exception:
                    return False

        def _remove_key(provider: str) -> bool:
            """Bezpiecznie usuń klucz API z keyringu i sesji."""
            ok = False
            # 1) System keyring
            if system_keyring is not None:
                try:
                    system_keyring.delete_password("TMIV", provider)
                    ok = True
                except Exception:
                    pass
            # 2) Szyfrowana sesja
            try:
                if f"{provider}_api_key_encrypted" in st.session_state:
                    del st.session_state[f"{provider}_api_key_encrypted"]
                    ok = True
            except Exception:
                pass
            # 3) Legacy session
            try:
                if f"{provider}_api_key" in st.session_state:
                    del st.session_state[f"{provider}_api_key"]
                    ok = True
            except Exception:
                pass
            return ok

        # --- UI: status + przycisk wejścia do panelu ---
        has_openai = _has_key("openai")
        has_anthropic = _has_key("anthropic")

        if has_openai or has_anthropic:
            providers = []
            if truthy_df_safe(has_openai):
                providers.append("OpenAI GPT")
            if truthy_df_safe(has_anthropic):
                providers.append("Anthropic Claude")

            provider_text = " + ".join(providers)
            st.success(f"🔒 Zabezpieczone: {provider_text}")

            if st.button("🔧 Zarządzaj kluczami", key="btn_keys"):
                st.session_state.show_secure_config = True
                rerun_once()
        else:
            st.info("🎭 Tryb: Symulacja AI")
            if st.button("🚀 Dodaj bezpieczne API", key="btn_add_keys"):
                st.session_state.show_secure_config = True
                rerun_once()

        # --- Panel zarządzania kluczami ---
        if st.session_state.get("show_secure_config", False):
            st.markdown("---")
            with st.container():
                st.markdown("### 🔒 Bezpieczne zarządzanie API")

                # Status obecnych kluczy
                col_status1, col_status2 = st.columns(2)
                with col_status1:
                    if truthy_df_safe(has_openai):
                        st.success("✅ OpenAI: Skonfigurowany")
                    else:
                        st.error("❌ OpenAI: Brak klucza")
                with col_status2:
                    if truthy_df_safe(has_anthropic):
                        st.success("✅ Anthropic: Skonfigurowany")
                    else:
                        st.error("❌ Anthropic: Brak klucza")

                # Wybór dostawcy
                provider = st.selectbox(
                    "🎯 Wybierz dostawcę:",
                    ["openai", "anthropic"],
                    key="provider_select",
                    help="Wybierz którego dostawcę chcesz skonfigurować"
                )

                # Input dla klucza
                placeholder = "sk-..." if provider == "openai" else "sk-ant-..."
                api_key = st.text_input(
                    f"🔑 Klucz {provider.upper()}",
                    type="password",
                    placeholder=placeholder,
                    help="Klucz będzie zaszyfrowany i bezpiecznie przechowany",
                    key="api_key_input",
                )

                # Przyciski akcji
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("💾 Zapisz", key="btn_save_key", use_container_width=True):
                        if not truthy_df_safe(api_key):
                            st.warning("⚠️ Wpisz klucz przed zapisem")
                        else:
                            if _store_key(provider, api_key):
                                st.success("✅ Klucz zapisany pomyślnie!")
                                time.sleep(1)  # Krótka pauza
                                st.session_state.show_secure_config = False
                                rerun_once()

                with col2:
                    current_has_key = _has_key(provider)
                    if st.button(
                        "🗑️ Usuń",
                        key="btn_remove_key",
                        use_container_width=True,
                        disabled=not current_has_key
                    ):
                        if _remove_key(provider):
                            st.success(f"✅ Klucz {provider} usunięty!")
                            time.sleep(1)
                            st.session_state.show_secure_config = False
                            rerun_once()

                with col3:
                    if st.button("❌ Zamknij", key="btn_cancel_key", use_container_width=True):
                        st.session_state.show_secure_config = False
                        rerun_once()

                st.info("🔒 Preferowany storage: systemowy keyring. Fallback: szyfrowana sesja (Fernet).")

    def _render_status_indicator(self):
        """Renderuje wskaźnik statusu aplikacji"""
        st.markdown("### 📊 Status aplikacji")

        # Status wczytania danych
        if st.session_state.get('data_loaded', False):
            st.success("✅ Dane wczytane")
        else:
            st.error("❌ Brak danych")

        # Status treningu modelu
        if st.session_state.get('model_trained', False):
            st.success("✅ Model wytrenowany")
        else:
            st.warning("⏳ Model nie wytrenowany")

        # Status analizy
        if st.session_state.get('analysis_complete', False):
            st.success("✅ Analiza zakończona")
        else:
            st.info("🔄 Analiza w toku")

    def _render_session_info(self):
        """Renderuje informacje o sesji"""
        st.subheader("🗂️ Informacje o sesji")

        df = st.session_state.get("df", None)

        # Domyślne wartości gdy danych nie ma
        rows = cols = 0
        memory_usage_mb = 0.0
        shape_txt = "brak"

        if isinstance(df, pd.DataFrame):
            rows, cols = df.shape
            try:
                memory_usage_mb = float(df.memory_usage(deep=True).sum()) / (1024 ** 2)
            except Exception:
                memory_usage_mb = 0.0
            shape_txt = f"{rows} × {cols}"

        col1, col2 = st.columns(2)
        with col1:
            st.metric("📄 Wiersze", f"{rows:,}")
            st.metric("🔢 Kolumny", cols)
        with col2:
            st.metric("💾 MB", f"{_fmt_float_safe(memory_usage_mb, 1)}")
            st.metric("📐 Kształt", shape_txt)

    def _reset_session(self):
        """Resetuje sesję aplikacji"""
        keys_to_reset = [
            'df', 'data_loaded', 'model_trained', 'analysis_complete',
            'model_results', 'target_column', 'problem_type', 'show_secure_config'
        ]

        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]

        st.success("🔄 Sesja została zresetowana!")
        rerun_once()

    def _render_export_section(self):
        """NAPRAWIONA sekcja eksportu - aktywna automatycznie po treningu"""
        st.markdown("### 💾 Eksport wyników")

        # Sprawdź czy model został wytrenowany
        model_trained = st.session_state.get('model_trained', False)
        model_results = st.session_state.get('model_results', {})

        if not model_trained or not model_results:
            st.info("ℹ️ Eksport będzie dostępny po wytrenowaniu modelu")
            return

        # Model wytrenowany - pokaż opcje eksportu
        st.success("✅ Model wytrenowany - eksport aktywny!")

        # Przygotuj dane do eksportu
        results = model_results
        target_column = st.session_state.get('target_column', 'target')
        problem_type = st.session_state.get('problem_type', 'Unknown')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_trainer = st.session_state.get('ml_trainer_instance')

        st.markdown("#### 📦 Pobierz pliki:")

        # Siatka przycisków eksportu 2x3
        col1, col2 = st.columns(2)

        with col1:
            # 1. JSON Report - zawsze dostępny
            json_data = self._generate_json_report(results, target_column, problem_type)
            # Odporne serializowanie
            json_string = json.dumps(_to_serializable(json_data), indent=2, ensure_ascii=False)

            st.download_button(
                label="📄 Pobierz JSON Report",
                data=json_string,
                file_name=f"ML_Report_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )

            # 3. HTML Report - zawsze dostępny
            html_content = self._generate_html_report(results, target_column, problem_type)

            st.download_button(
                label="🌐 Pobierz HTML Report",
                data=html_content,
                file_name=f"ML_Report_{timestamp}.html",
                mime="text/html",
                use_container_width=True
            )

            # 5. TXT Summary - zawsze dostępny
            txt_content = self._generate_txt_summary(results, target_column, problem_type)

            st.download_button(
                label="📝 Pobierz TXT Summary",
                data=txt_content,
                file_name=f"ML_Summary_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )

        with col2:
            # 2. CSV Feature Importance - jeśli dostępne
            if 'feature_importance' in results and isinstance(results['feature_importance'], pd.DataFrame):
                csv_data = results['feature_importance'].to_csv(index=False)

                st.download_button(
                    label="📊 Pobierz CSV Features",
                    data=csv_data,
                    file_name=f"Feature_Importance_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("📊 CSV Features\n(niedostępne)")

            # 4. Model Pickle - jeśli dostępny
            try:
                if truthy_df_safe(model_trainer) and hasattr(model_trainer, 'trained_models') and target_column in model_trainer.trained_models:
                    model_data = model_trainer.trained_models[target_column]
                    pickled_model = pickle.dumps(model_data)

                    st.download_button(
                        label="🤖 Pobierz Model PKL",
                        data=pickled_model,
                        file_name=f"Trained_Model_{timestamp}.pkl",
                        mime="application/octet-stream",
                        use_container_width=True
                    )
                else:
                    st.info("🤖 Model PKL\n(niedostępny)")
            except Exception as e:
                st.info(f"🤖 Model PKL\n(błąd: {str(e)[:20]}...)")

            # 6. Wszystko w ZIP - premium opcja
            if st.button("📦 Pobierz wszystko (ZIP)", key="btn_export_zip_all", use_container_width=True):
                import zipfile
                import io

                # Utwórz ZIP w pamięci
                zip_buffer = io.BytesIO()

                with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                    # Dodaj JSON
                    zip_file.writestr(f"ML_Report_{timestamp}.json", json_string)

                    # Dodaj HTML
                    zip_file.writestr(f"ML_Report_{timestamp}.html", html_content)

                    # Dodaj TXT
                    zip_file.writestr(f"ML_Summary_{timestamp}.txt", txt_content)

                    # Dodaj CSV jeśli dostępny
                    if 'feature_importance' in results and isinstance(results['feature_importance'], pd.DataFrame):
                        csv_data = results['feature_importance'].to_csv(index=False)
                        zip_file.writestr(f"Feature_Importance_{timestamp}.csv", csv_data)

                    # Dodaj model jeśli dostępny
                    try:
                        if truthy_df_safe(model_trainer) and hasattr(model_trainer, 'trained_models') and target_column in model_trainer.trained_models:
                            model_data = model_trainer.trained_models[target_column]
                            pickled_model = pickle.dumps(model_data)
                            zip_file.writestr(f"Trained_Model_{timestamp}.pkl", pickled_model)
                    except Exception:
                        pass

                zip_buffer.seek(0)

                st.download_button(
                    label="⬇️ Pobierz kompletny pakiet",
                    data=zip_buffer.getvalue(),
                    file_name=f"TMIV_Complete_Export_{timestamp}.zip",
                    mime="application/zip",
                    use_container_width=True
                )

    def _generate_json_report(self, results: Dict, target_column: str, problem_type: str) -> Dict:
        """Generuje pełny JSON report"""
        return {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'app_version': '2.1.0',
                'created_by': 'Marksio AI Solutions - The Most Important Variables'
            },
            'dataset_info': {
                'target_column': target_column,
                'problem_type': problem_type,
                'shape': list(st.session_state.df.shape) if 'df' in st.session_state else None,
                'memory_usage_mb': float(st.session_state.df.memory_usage(deep=True).sum() / (1024**2)) if 'df' in st.session_state else None
            },
            'model_performance': {
                'best_model': results.get('best_model', ''),
                'all_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v
                                for k, v in results.items()
                                if k in ['r2', 'mae', 'mse', 'rmse', 'mape', 'explained_variance', 'max_error', 'mean_residual', 'std_residual', 'accuracy', 'precision', 'recall', 'f1']},
                # ✅ Odporne serializowanie porównania modeli
                'model_comparison': _to_serializable(results.get('model_scores', {}))
            },
            'feature_analysis': {
                'feature_importance': results.get('feature_importance', pd.DataFrame()).to_dict('records') if 'feature_importance' in results else [],
                'top_10_features': results.get('feature_importance', pd.DataFrame()).head(10).to_dict('records') if 'feature_importance' in results else []
            },
            'recommendations': self._generate_export_recommendations(results, problem_type)
        }

    def _generate_html_report(self, results: Dict, target_column: str, problem_type: str) -> str:
        """Generuje HTML report z wizualizacjami"""

        html_template = f"""
        <!DOCTYPE html>
        <html lang="pl">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ML Analysis Report - {target_column}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px; }}
                .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
                .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border-left: 4px solid #667eea; }}
                .metric h3 {{ margin: 0; color: #667eea; font-size: 1.5em; }}
                .metric p {{ margin: 5px 0 0 0; color: #666; }}
                .feature-list {{ list-style: none; padding: 0; }}
                .feature-list li {{ background: #e3f2fd; margin: 5px 0; padding: 10px; border-radius: 5px; border-left: 3px solid #2196f3; }}
                .bar {{ background: #667eea; height: 5px; border-radius: 3px; margin-top: 5px; }}
                .footer {{ text-align: center; margin-top: 40px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>The Most Important Variables</h1>
                <h2>Advanced ML Analysis Report: {target_column}</h2>
                <p>Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Marksio AI Solutions</p>
            </div>

            <div class="section">
                <h2>Podsumowanie modelu</h2>
                <p><strong>Typ problemu:</strong> {problem_type}</p>
                <p><strong>Najlepszy model:</strong> {results.get('best_model', 'N/A')}</p>
                <p><strong>Dataset:</strong> {st.session_state.df.shape if 'df' in st.session_state else 'N/A'}</p>
            </div>

            <div class="section">
                <h2>Metryki wydajności</h2>
                <div class="metrics">
        """

        # Dodaj metryki
        if problem_type == "Regresja":
            metrics = [
                ("R² Score", results.get('r2', 0), "Współczynnik determinacji"),
                ("MAE", results.get('mae', 0), "Średni błąd bezwzględny"),
                ("RMSE", results.get('rmse', 0), "Pierwiastek błędu kwadratowego"),
                ("MAPE", f"{_fmt_float_safe(results.get('mape', 0), 2)}%", "Błąd procentowy")
            ]
        else:
            metrics = [
                ("Accuracy", results.get('accuracy', 0), "Dokładność"),
                ("Precision", results.get('precision', 0), "Precyzja"),
                ("Recall", results.get('recall', 0), "Czułość"),
                ("F1-Score", results.get('f1', 0), "F1 Score")
            ]

        for name, value, desc in metrics:
            if isinstance(value, float):
                value = f"{_fmt_float_safe(value, 4)}"
            html_template += f"""
                    <div class="metric">
                        <h3>{value}</h3>
                        <p>{name}</p>
                        <small>{desc}</small>
                    </div>
            """

        # Dodaj feature importance
        html_template += """
                </div>
            </div>

            <div class="section">
                <h2>Najważniejsze cechy</h2>
                <ul class="feature-list">
        """

        if 'feature_importance' in results and isinstance(results['feature_importance'], pd.DataFrame):
            top_features = results['feature_importance'].head(10)
            for _, row in top_features.iterrows():
                # Obsłuż różne nazwy kolumn z ważnością
                val = (
                    row.get('importance')
                    if 'importance' in row
                    else row.get('importance_mean', None)
                )
                if val is None:
                    # Ostateczny fallback: pierwsza kolumna numeryczna
                    nums = [float(v) for v in row.values if isinstance(v, (int, float, np.number))]
                    val = nums[0] if nums else 0.0
                width_pct = max(0.0, min(100.0, float(val) * 100.0))
                html_template += f"""
                    <li>
                        <strong>{row.get('feature','(feature)')}</strong>: {_fmt_float_safe(val, 4)}
                        <div class="bar" style="width: {width_pct:.1f}%;"></div>
                    </li>
                """

        html_template += """
                </ul>
            </div>

            <div class="footer">
                <p>Made by Marksio AI Solutions | The Most Important Variables v2.1.0</p>
            </div>
        </body>
        </html>
        """

        return html_template

    def _generate_txt_summary(self, results, target_column, problem_type):
        """Generuje podsumowanie w formacie tekstowym"""

        summary = f"""
=== RAPORT ANALIZY MACHINE LEARNING ===
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Platforma: The Most Important Variables v2.1.0
Autor: Marksio AI Solutions

DANE:
- Zmienna docelowa: {target_column}
- Typ problemu: {problem_type}
- Najlepszy model: {results.get('best_model', 'N/A')}

METRYKI WYDAJNOŚCI:
"""

        if problem_type == "Regresja":
            summary += f"""- R² Score: {_fmt_float_safe(results.get('r2', 'N/A'), 4)}
- MAE: {_fmt_float_safe(results.get('mae', 'N/A'), 4)}
- RMSE: {_fmt_float_safe(results.get('rmse', 'N/A'), 4)}
- MAPE: {_fmt_float_safe(results.get('mape', 'N/A'), 2)}%"""
        else:
            summary += f"""- Accuracy: {_fmt_float_safe(results.get('accuracy', 'N/A'), 4)}
- Precision: {_fmt_float_safe(results.get('precision', 'N/A'), 4)}
- Recall: {_fmt_float_safe(results.get('recall', 'N/A'), 4)}
- F1-Score: {_fmt_float_safe(results.get('f1', 'N/A'), 4)}"""

        if 'feature_importance' in results and isinstance(results['feature_importance'], pd.DataFrame):
            summary += "\n\nNAJWAŻNIEJSZE CECHY:\n"
            top_features = results['feature_importance'].head(10)
            for _, row in top_features.iterrows():
                val = row.get('importance', row.get('importance_mean', 0.0))
                summary += f"- {row.get('feature','(feature)')}: {_fmt_float_safe(val, 4)}\n"

        summary += f"\n\nREKOMENDACJE:\n"
        recommendations = self._generate_export_recommendations(results, problem_type)
        for rec in recommendations:
            summary += f"- {rec}\n"

        summary += f"\n\n=== KONIEC RAPORTU ===\nWygenerowano przez: Marksio AI Solutions\nPlatforma: The Most Important Variables v2.1.0"

        return summary

    def _generate_export_recommendations(self, results, problem_type):
        """Generuje rekomendacje dla eksportu"""
        recommendations = []

        if problem_type == "Regresja":
            r2_score = results.get('r2', 0)
            try:
                r2_score = float(r2_score)
            except Exception:
                r2_score = 0.0
            if r2_score > 0.8:
                recommendations.append("Model ma bardzo dobrą wydajność - można go używać do predykcji")
            elif r2_score > 0.6:
                recommendations.append("Model ma umiarkowaną wydajność - warto rozważyć dodatkowe feature engineering")
            else:
                recommendations.append("Model wymaga poprawy - zbadaj outliers i jakość danych")
        else:
            accuracy = results.get('accuracy', 0)
            try:
                accuracy = float(accuracy)
            except Exception:
                accuracy = 0.0
            if accuracy > 0.9:
                recommendations.append("Model ma doskonałą dokładność klasyfikacji")
            elif accuracy > 0.8:
                recommendations.append("Model ma dobrą dokładność - można go używać w produkcji")
            else:
                recommendations.append("Model wymaga optymalizacji - rozważ inne algorytmy lub więcej danych")

        recommendations.append("Sprawdź feature importance aby zrozumieć kluczowe zmienne")
        recommendations.append("Rozważ zbieranie dodatkowych danych dla słabych cech")
        recommendations.append("Użyj modelu do automatyzacji procesów biznesowych")

        return recommendations

    def render_dataset_metrics(self, metrics: dict | None = None):
        """Bezpieczny placeholder: wyświetla podstawowe metryki zbioru danych.
        Oczekuje słownika {'rows': int, 'cols': int, ...}. Nie rzuca wyjątków gdy brak.
        """
        st.subheader("Metryki zbioru")
        if not truthy_df_safe(metrics):
            st.info("Brak metryk do wyświetlenia.")
            return
        cols = st.columns(min(4, len(metrics)))
        for i, (k, v) in enumerate(metrics.items()):
            with cols[i % len(cols)]:
                st.metric(label=str(k), value=str(v))


def render_ai_recommendations(plan: Dict[str, Any], recs: Dict[str, Any]):
    st.markdown("### Rekomendacje AI - przygotowanie danych")
    with st.expander("Pokaż kroki AI Data Prep", expanded=True):
        steps = plan.get("steps", [])
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}. {step.get('name','Step')}** - {step.get('detail','')}")

    st.markdown("### Rekomendacje AI - trening")
    st.write(f"Problem: **{recs.get('problem','?')}**, Folds: **{recs.get('cv_folds',5)}**")
    st.write("Metryki (priorytet pierwszej):", recs.get("metrics", []))
    st.write("Modele (kandydaci):", [m['name'] for m in recs.get("models", [])])


def render_training_controls():
    """
    Przyciski sterujące + hook 'Ustaw rekomendacje AI' (auto dobór rodzin modeli).
    Po kliknięciu ustawiamy st.session_state['final_config']['model_families']
    ORAZ st.session_state['model_families'] (key multiselecta), a na końcu rerun_once().
    """
    import numpy as np
    import pandas as pd
    import streamlit as st

    def _detect_problem_fallback(df: pd.DataFrame, target: str) -> str:
        if truthy_df_safe(target) and target in df.columns:
            y = df[target].dropna()
            if y.dtype.name in ("object", "bool", "category"):
                return "Klasyfikacja"
            if y.nunique() <= 20 and not np.issubdtype(y.dtype, np.floating):
                return "Klasyfikacja"
            return "Regresja"
        return "Regresja"

    def _class_imbalance_ratio(y: pd.Series) -> float:
        vc = y.value_counts(dropna=True)
        if len(vc) < 2 or vc.min() == 0:
            return 1.0
        return float(vc.max()) / float(vc.min())

    def _families_all():
        return [
            "linear", "tree", "random_forest", "gbm",
            "xgboost", "lightgbm", "catboost",
            "svm", "knn", "naive_bayes", "mlp"
        ]

    def _auto_select_model_families(df: pd.DataFrame, target_column: str, problem_type: str):
        rows, cols = df.shape
        X = df.drop(columns=[target_column], errors="ignore")
        num_cols = X.select_dtypes(include=[np.number]).shape[1]
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).shape[1]
        families = []

        if str(problem_type).lower().startswith("klasyf") or "class" in str(problem_type).lower():
            families += ["linear", "tree", "random_forest", "gbm"]
            if rows <= 100_000:
                families += ["xgboost"]
            families += ["lightgbm"]
            if cat_cols > 0:
                families += ["catboost"]
            if rows < 2_000:
                families += ["svm", "knn", "naive_bayes", "mlp"]
            try:
                imb = _class_imbalance_ratio(df[target_column].dropna())
                if imb > 3:
                    # wyrzuć delikatniejsze na niezbalansowanych
                    families = [f for f in families if f not in ("knn", "svm", "naive_bayes")]
                    families = list(dict.fromkeys(families + ["random_forest", "gbm", "lightgbm", "xgboost"]))
            except Exception:
                pass
            if cols > 200:
                families = [f for f in families if f not in ("knn", "svm", "mlp")]
        else:
            families += ["linear", "tree", "random_forest", "gbm", "lightgbm"]
            if rows <= 100_000:
                families += ["xgboost"]
            if cat_cols > 0:
                families += ["catboost"]
            if rows < 3_000:
                families += ["svm", "mlp"]
            if cols > 200:
                families = [f for f in families if f not in ("svm", "mlp")]

        families = list(dict.fromkeys(families))
        known = set(_families_all())
        return [f for f in families if f in known]

    # --- UI: guziki ---
    cols = st.columns([1, 1, 2])
    with cols[0]:
        apply = st.button("Ustaw rekomendacje AI", use_container_width=True, key="btn_apply_ai_plan")
    with cols[2]:
        train = st.button("Trenuj model", type="primary", use_container_width=True, key="btn_train_model")

    # --- HOOK: po kliknięciu ustaw stan i wymuś rerender ---
    if truthy_df_safe(apply):
        st.session_state["apply_ai_plan_clicked"] = True
        st.session_state["ai_plan_applied"] = True

        df = st.session_state.get('uploaded_df') or st.session_state.get('df')
        target = (
            st.session_state.get('target') or
            st.session_state.get('target_column') or
            st.session_state.get('y_col')
        )

        if isinstance(df, pd.DataFrame) and target and target in df.columns:
            # problem type: z sesji / z trenera (jeśli masz) / fallback
            problem_type = st.session_state.get('problem_type')
            if not truthy_df_safe(problem_type):
                app_ref = st.session_state.get('app')
                if truthy_df_safe(app_ref) and getattr(app_ref, 'ml_trainer', None):
                    try:
                        problem_type = app_ref.ml_trainer.detect_problem_type(df, target)
                    except Exception:
                        problem_type = None
            if not truthy_df_safe(problem_type):
                problem_type = _detect_problem_fallback(df, target)

            # auto-dobór rodzin
            try:
                auto_fams = _auto_select_model_families(df, target, problem_type)
            except Exception:
                auto_fams = _families_all()

            # ZAPIS: final_config + stan widgetu (KLUCZOWE!)
            st.session_state.setdefault('final_config', {})
            st.session_state['final_config']['model_families'] = list(auto_fams)

            # Ustawiamy *bezpośrednio* wartość pod key multiselecta:
            st.session_state['model_families'] = list(auto_fams)
            st.session_state["use_auto_families_default"] = True
            st.session_state["model_families_user_override"] = False

            # (opcjonalnie uzupełnij sensowne metryki/strategie)
            st.session_state['final_config'].setdefault(
                'recommended_strategy',
                'balanced'
            )
            if str(problem_type).lower().startswith("klasyf"):
                st.session_state['final_config'].setdefault('recommended_metric', 'f1_weighted')
            else:
                st.session_state['final_config'].setdefault('recommended_metric', 'r2')

            # natychmiastowy rerun, żeby multiselect pokazał nowe wybory
            rerun_once()
        else:
            st.warning("Brak danych lub kolumny celu – nie mogę dobrać rodzin modeli.")

    return apply, train


def render_advanced_overrides(recs: Dict[str, Any]):
    st.markdown("### Zaawansowane (opcjonalne)")
    with st.expander("Pokaż / ukryj zaawansowane", expanded=False):
        enable = st.checkbox("Chcę ręcznie nadpisać rekomendacje", value=False)
        if not truthy_df_safe(enable):
            st.info("AI ustawi wszystko automatycznie. Możesz włączyć nadpisywanie powyżej.")
            return recs, False
        mod = dict(recs)
        c1, c2 = st.columns(2)
        with c1:
            mod["cv_folds"] = st.number_input("CV folds", min_value=2, max_value=20, value=int(recs.get("cv_folds", 5)))
            metrics_all = ["f1_weighted", "roc_auc_ovr", "accuracy", "precision_weighted", "recall_weighted", "rmse", "mae", "r2", "mape", "smape"]
            mod["metrics"] = st.multiselect("Metryki (priorytet pierwszej)", metrics_all, default=recs.get("metrics", []))
        with c2:
            model_names = [m["name"] for m in recs.get("models", [])]
            selected = st.multiselect("Modele do trenowania", model_names, default=model_names)
            mod["models"] = [m for m in recs.get("models", []) if m["name"] in selected]
        st.success("Zastosowano nadpisania (tymczasowe w tej sesji).")
        return mod, True