import streamlit as st


def _fmt_float_safe(x, digits=2):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import pickle
import base64
import time

# ✅ DODANE: bezpieczny menedżer poświadczeń
try:
    from backend.security_manager import credential_manager
except ImportError:
    # Fallback jeśli security_manager nie jest dostępny
    credential_manager = None


class UIComponents:
    """
    Klasa zawierająca wszystkie komponenty interfejsu użytkownika
    """
    
    def __init__(self):
        self.theme_colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#ff9800',
            'error': '#d32f2f',
            'info': '#17a2b8'
        }
    
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
        .css-1d391kg {
            padding-top: 1rem;
        }
        
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
            self._render_dataset_metrics()
    
    def _render_dataset_metrics(self):
        """Renderuje ulepszone metryki datasetu"""
        if hasattr(st.session_state, 'df') and st.session_state.get('data_loaded', False):
            df = st.session_state.df
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rows_formatted = f"{int(len(df)):,}"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 {rows_formatted}</h3>
                    <p>Wierszy danych</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                numeric_cols = int(len(df.select_dtypes(include=[np.number]).columns))
                total_cols = int(len(df.columns))
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔢 {numeric_cols}/{total_cols}</h3>
                    <p>Kolumn numerycznych</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                missing_pct = float((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100)
                data_density = 100 - missing_pct
                st.markdown(f"""
                <div class="metric-card">
                    <h3>✨ {_fmt_float_safe(data_density, 1)}%</h3>
                    <p>Kompletność danych</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                memory_usage = float(df.memory_usage(deep=True).sum() / (1024**2))
                st.markdown(f"""
                <div class="metric-card">
                    <h3>💾 {_fmt_float_safe(memory_usage, 1)}MB</h3>
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
            # ✅ Bezpieczna konfiguracja AI (keyring)
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
            
            if st.button("🔄 Resetuj sesję", help="Wyczyść wszystkie dane i rozpocznij od nowa"):
                self._reset_session()
            
            # NAPRAWIONY EKSPORT
            self._render_export_section()
            
            # Footer sidebar - ZAKTUALIZOWANY BRANDING
            st.markdown("---")
            st.markdown("**🛠️ Wersja:** 2.0.0 Pro")
            st.markdown("**🚀 Made by Marksio AI Solutions**")
            st.markdown("**⭐ Advanced ML Platform**")
            
        return selected_page
    
    def _render_ai_config_simple(self):
        """BEZPIECZNA konfiguracja AI - KOMPLETNA IMPLEMENTACJA"""
        st.markdown("### 🤖 Konfiguracja AI")
        
        # --- NAPRAWIONE helper functions z lepszą obsługą błędów ---
        def _has_key(provider: str) -> bool:
            """Sprawdź czy mamy klucz dla dostawcy"""
            try:
                # Sprawdź różne metody credential_manager
                if credential_manager and hasattr(credential_manager, "get_api_key"):
                    val = credential_manager.get_api_key(provider)
                    return bool(val and val.strip())
                elif credential_manager and hasattr(credential_manager, "get_password"):
                    val = credential_manager.get_password("tmiv_ml_platform", f"{provider}_key_default")
                    return bool(val and val.strip())
                # Fallback do session_state
                return bool(st.session_state.get(f"{provider}_api_key"))
            except Exception:
                return bool(st.session_state.get(f"{provider}_api_key"))

        def _store_key(provider: str, api_key: str) -> bool:
            """Bezpiecznie zapisz klucz API"""
            try:
                # Walidacja klucza
                if provider == "openai" and not api_key.startswith("sk-"):
                    st.error("❌ OpenAI klucz musi zaczynać się od 'sk-'")
                    return False
                elif provider == "anthropic" and not api_key.startswith("sk-ant-"):
                    st.error("❌ Anthropic klucz musi zaczynać się od 'sk-ant-'")
                    return False
                
                # Próba zapisu przez credential_manager
                if credential_manager and hasattr(credential_manager, "store_api_key"):
                    success = credential_manager.store_api_key(provider, api_key)
                    if success:
                        st.session_state[f"{provider}_api_key"] = api_key  # Backup do session
                        return True
                
                # Fallback do keyring
                if credential_manager and hasattr(credential_manager, "set_password"):
                    credential_manager.set_password("tmiv_ml_platform", f"{provider}_key_default", api_key)
                    st.session_state[f"{provider}_api_key"] = api_key
                    return True
                
                # Last resort - tylko session_state (ostrzeżenie)
                st.session_state[f"{provider}_api_key"] = api_key
                st.warning("⚠️ Klucz zapisany tylko w sesji (nie jest trwały)")
                return True
                
            except Exception as e:
                st.error(f"❌ Nie udało się zapisać klucza: {e}")
                # Fallback do session_state
                try:
                    st.session_state[f"{provider}_api_key"] = api_key
                    st.warning("⚠️ Klucz zapisany tylko w sesji ze względu na błąd keyring")
                    return True
                except:
                    return False

        def _remove_key(provider: str) -> bool:
            """Bezpiecznie usuń klucz API"""
            try:
                success = False
                
                # Usuń z credential_manager
                if credential_manager and hasattr(credential_manager, "remove_api_key"):
                    try:
                        credential_manager.remove_api_key(provider)
                        success = True
                    except:
                        pass
                
                # Usuń z keyring
                if credential_manager and hasattr(credential_manager, "delete_password"):
                    try:
                        credential_manager.delete_password("tmiv_ml_platform", f"{provider}_key_default")
                        success = True
                    except:
                        pass
                
                # Usuń z session_state
                if f"{provider}_api_key" in st.session_state:
                    del st.session_state[f"{provider}_api_key"]
                    success = True
                
                return success
                
            except Exception as e:
                st.error(f"❌ Błąd podczas usuwania klucza: {e}")
                return False

        # --- UI: status + przycisk wejścia do panelu ---
        has_openai = _has_key("openai")
        has_anthropic = _has_key("anthropic")

        if has_openai or has_anthropic:
            providers = []
            if has_openai:
                providers.append("OpenAI GPT")
            if has_anthropic:
                providers.append("Anthropic Claude")
            
            provider_text = " + ".join(providers)
            st.success(f"🔒 Zabezpieczone: {provider_text}")
            
            if st.button("🔧 Zarządzaj kluczami", key="btn_keys"):
                st.session_state.show_secure_config = True
                st.rerun()
        else:
            st.info("🎭 Tryb: Symulacja AI")
            if st.button("🚀 Dodaj bezpieczne API", key="btn_add_keys"):
                st.session_state.show_secure_config = True
                st.rerun()

        # --- Panel zarządzania kluczami ---
        if st.session_state.get("show_secure_config", False):
            st.markdown("---")
            with st.container():
                st.markdown("### 🔒 Bezpieczne zarządzanie API")
                
                # Status obecnych kluczy
                col_status1, col_status2 = st.columns(2)
                with col_status1:
                    if has_openai:
                        st.success("✅ OpenAI: Skonfigurowany")
                    else:
                        st.error("❌ OpenAI: Brak klucza")
                with col_status2:
                    if has_anthropic:
                        st.success("✅ Anthropic: Skonfigurowany")
                    else:
                        st.error("❌ Anthropic: Brak klucza")
                
                # Wybór dostawcy
                provider = st.selectbox(
                    "🎯 Wybierz dostawcy:",
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
                        if not api_key:
                            st.warning("⚠️ Wpisz klucz przed zapisem")
                        else:
                            if _store_key(provider, api_key):
                                st.success("✅ Klucz zapisany pomyślnie!")
                                time.sleep(1)  # Krótka pauza
                                st.session_state.show_secure_config = False
                                st.rerun()

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
                            st.rerun()

                with col3:
                    if st.button("❌ Zamknij", key="btn_cancel_key", use_container_width=True):
                        st.session_state.show_secure_config = False
                        st.rerun()

                st.info("🔒 Klucze są szyfrowane i przechowywane w systemowym keyring lub bezpiecznie w sesji")
    
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
        st.rerun()
    
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
            json_string = json.dumps(json_data, indent=2, ensure_ascii=False)
            
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
                if model_trainer and hasattr(model_trainer, 'trained_models') and target_column in model_trainer.trained_models:
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
            if st.button("📦 Pobierz wszystko (ZIP)", use_container_width=True):
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
                        if model_trainer and hasattr(model_trainer, 'trained_models') and target_column in model_trainer.trained_models:
                            model_data = model_trainer.trained_models[target_column]
                            pickled_model = pickle.dumps(model_data)
                            zip_file.writestr(f"Trained_Model_{timestamp}.pkl", pickled_model)
                    except:
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
                'app_version': '2.0.0 Pro',
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
                'model_comparison': results.get('model_scores', {})
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
                .recommendation {{ background: #e8f5e8; border: 1px solid #4caf50; border-radius: 5px; padding: 15px; margin: 10px 0; }}
                .footer {{ text-align: center; margin-top: 40px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 The Most Important Variables</h1>
                <h2>Advanced ML Analysis Report: {target_column}</h2>
                <p>Wygenerowano: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>🚀 Marksio AI Solutions</p>
            </div>
            
            <div class="section">
                <h2>📊 Podsumowanie modelu</h2>
                <p><strong>Typ problemu:</strong> {problem_type}</p>
                <p><strong>Najlepszy model:</strong> {results.get('best_model', 'N/A')}</p>
                <p><strong>Dataset:</strong> {st.session_state.df.shape if 'df' in st.session_state else 'N/A'}</p>
            </div>
            
            <div class="section">
                <h2>📈 Metryki wydajności</h2>
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
                <h2>🎯 Najważniejsze cechy</h2>
                <ul class="feature-list">
        """
        
        if 'feature_importance' in results and isinstance(results['feature_importance'], pd.DataFrame):
            top_features = results['feature_importance'].head(10)
            for _, row in top_features.iterrows():
                html_template += f"""
                    <li>
                        <strong>{row['feature']}</strong>: {_fmt_float_safe(row['importance'], 4)}
                        <div style="background: #667eea; height: 5px; width: {_fmt_float_safe(row['importance']*100, 1)}%; border-radius: 3px; margin-top: 5px;"></div>
                    </li>
                """
        
        html_template += """
                </ul>
            </div>
            
            <div class="footer">
                <p>🚀 Made by Marksio AI Solutions | The Most Important Variables v2.0 Pro</p>
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
Platforma: The Most Important Variables v2.0 Pro
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
            for idx, row in top_features.iterrows():
                summary += f"- {row['feature']}: {_fmt_float_safe(row['importance'], 4)}\n"
        
        summary += f"\n\nREKOMENDACJE:\n"
        recommendations = self._generate_export_recommendations(results, problem_type)
        for rec in recommendations:
            summary += f"- {rec}\n"
        
        summary += f"\n\n=== KONIEC RAPORTU ===\nWygenerowano przez: Marksio AI Solutions\nPlatforma: The Most Important Variables v2.0 Pro"
        
        return summary
    
    def _generate_export_recommendations(self, results, problem_type):
        """Generuje rekomendacje dla eksportu"""
        recommendations = []
        
        if problem_type == "Regresja":
            r2_score = results.get('r2', 0)
            if r2_score > 0.8:
                recommendations.append("✅ Model ma bardzo dobrą wydajność - można go używać do predykcji")
            elif r2_score > 0.6:
                recommendations.append("⚠️ Model ma umiarkowaną wydajność - warto rozważyć dodatkowe feature engineering")
            else:
                recommendations.append("❌ Model wymaga poprawy - zbadaj outliers i jakość danych")
        else:
            accuracy = results.get('accuracy', 0)
            if accuracy > 0.9:
                recommendations.append("✅ Model ma doskonałą dokładność klasyfikacji")
            elif accuracy > 0.8:
                recommendations.append("⚠️ Model ma dobrą dokładność - można go używać w produkcji")
            else:
                recommendations.append("❌ Model wymaga optymalizacji - rozważ inne algorytmy lub więcej danych")
        
        recommendations.append("🚀 Sprawdź feature importance aby zrozumieć kluczowe zmienne")
        recommendations.append("📊 Rozważ zbieranie dodatkowych danych dla słabych cech")
        recommendations.append("⚡ Użyj modelu do automatyzacji procesów biznesowych")
        
        return recommendations

    def render_dataset_metrics(self, metrics: dict | None = None):
        """Bezpieczny placeholder: wyświetla podstawowe metryki zbioru danych.
        Oczekuje słownika {'rows': int, 'cols': int, ...}. Nie rzuca wyjątków gdy brak.
        """
        st.subheader("📏 Metryki zbioru")
        if not metrics:
            st.info("Brak metryk do wyświetlenia.")
            return
        cols = st.columns(min(4, len(metrics)))
        for i, (k, v) in enumerate(metrics.items()):
            with cols[i % len(cols)]:
                st.metric(label=str(k), value=str(v))


import streamlit as st
from typing import Dict, Any, List

def render_ai_recommendations(plan: Dict[str, Any], recs: Dict[str, Any]):
    st.markdown("### 🤖 Rekomendacje AI — przygotowanie danych")
    with st.expander("Pokaż kroki AI Data Prep", expanded=True):
        steps = plan.get("steps", [])
        for i, step in enumerate(steps, 1):
            st.markdown(f"**{i}. {step.get('name','Step')}** — {step.get('detail','')}")

    st.markdown("### 🧠 Rekomendacje AI — trening")
    st.write(f"Problem: **{recs.get('problem','?')}**, Folds: **{recs.get('cv_folds',5)}**")
    st.write("Metryki (priorytet pierwszej):", recs.get("metrics", []))
    st.write("Modele (kandydaci):", [m['name'] for m in recs.get("models", [])])

def render_training_controls():
    # guziki sterujące
    cols = st.columns([1,1,2])
    with cols[0]:
        apply = st.button("⚙️ Ustaw rekomendacje AI", use_container_width=True)
    with cols[2]:
        train = st.button("🚀 Trenuj model", type="primary", use_container_width=True)
    return apply, train


def render_advanced_overrides(recs: Dict[str, Any]):
    st.markdown("### ⚙️ Zaawansowane (opcjonalne)")
    with st.expander("Pokaż / ukryj zaawansowane", expanded=False):
        enable = st.checkbox("Chcę ręcznie nadpisać rekomendacje", value=False)
        if not enable:
            st.info("AI ustawi wszystko automatycznie. Możesz włączyć nadpisywanie powyżej.")
            return recs, False
        mod = dict(recs)
        c1, c2 = st.columns(2)
        with c1:
            mod["cv_folds"] = st.number_input("CV folds", min_value=2, max_value=20, value=int(recs.get("cv_folds",5)))
            metrics_all = ["f1_weighted","roc_auc_ovr","accuracy","precision_weighted","recall_weighted","rmse","mae","r2","mape","smape"]
            mod["metrics"] = st.multiselect("Metryki (priorytet pierwszej)", metrics_all, default=recs.get("metrics", []))
        with c2:
            model_names = [m["name"] for m in recs.get("models",[])]
            selected = st.multiselect("Modele do trenowania", model_names, default=model_names)
            mod["models"] = [m for m in recs.get("models",[]) if m["name"] in selected]
        st.success("Zastosowano nadpisania (tymczasowe w tej sesji).")
        return mod, True
