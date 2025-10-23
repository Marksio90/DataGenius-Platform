"""
Podstawowe komponenty UI Streamlit.

Funkcjonalności:
- Sidebar configuration
- Status indicators
- File uploader
- Parameter selectors
- Progress displays
"""

import logging
from typing import Dict, List, Optional, Tuple

import streamlit as st

from backend.ai_integration import get_ai_integration
from backend.security_manager import get_security_manager
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
security = get_security_manager()


def render_sidebar() -> Dict:
    """
    Renderuje sidebar z konfiguracją.

    Returns:
        Dict: Słownik z ustawieniami użytkownika

    Example:
        >>> # W aplikacji Streamlit
        >>> config = render_sidebar()
    """
    st.sidebar.title("⚙️ Konfiguracja")

    config = {}

    # Sekcja: Klucze API
with st.sidebar.expander("🔑 Klucze API", expanded=False):
    st.markdown("**OpenAI API Key**")
    openai_key = st.text_input(
        "OpenAI Key",
        type="password",
        value=settings.openai_api_key or "",
        key="openai_key_input",
        label_visibility="collapsed"
    )
    
    # ZAPISZ klucz do session_state
    if openai_key and openai_key != settings.openai_api_key:
        st.session_state['openai_api_key'] = openai_key
        # Aktualizuj settings
        settings.openai_api_key = openai_key
        
    if openai_key:
        config['openai_key'] = openai_key
        if security.validate_api_key(openai_key, "openai"):
            st.success("✅ Klucz OpenAI wygląda poprawnie")
        else:
            st.warning("⚠️ Klucz OpenAI może być niepoprawny")

    st.markdown("**Anthropic API Key**")
    anthropic_key = st.text_input(
        "Anthropic Key",
        type="password",
        value=settings.anthropic_api_key or "",
        key="anthropic_key_input",
        label_visibility="collapsed"
    )
    
    # ZAPISZ klucz do session_state
    if anthropic_key and anthropic_key != settings.anthropic_api_key:
        st.session_state['anthropic_api_key'] = anthropic_key
        # Aktualizuj settings
        settings.anthropic_api_key = anthropic_key
        
    if anthropic_key:
        config['anthropic_key'] = anthropic_key
        if security.validate_api_key(anthropic_key, "anthropic"):
            st.success("✅ Klucz Anthropic wygląda poprawnie")
        else:
            st.warning("⚠️ Klucz Anthropic może być niepoprawny")
    
    # Przycisk Apply
    if st.button("💾 Zastosuj Klucze", use_container_width=True):
        # Reinicjalizuj AI integration z nowymi kluczami
        from backend.ai_integration import get_ai_integration
        ai = get_ai_integration()
        ai._init_clients()
        st.success("✅ Klucze zaktualizowane!")
        st.rerun()

    # Status providerów
    with st.sidebar.expander("📊 Status Providerów", expanded=True):
        ai = get_ai_integration()
        status = ai.get_provider_status()

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "OpenAI",
                "✅" if status['openai'] else "❌",
                "Aktywny" if status['openai'] else "Nieaktywny"
            )
        with col2:
            st.metric(
                "Anthropic",
                "✅" if status['anthropic'] else "❌",
                "Aktywny" if status['anthropic'] else "Nieaktywny"
            )

        if not status['any']:
            st.warning("⚠️ Brak aktywnych providerów LLM - działanie w trybie fallback")

    # Sekcja: Narzędzia
    with st.sidebar.expander("🛠️ Narzędzia", expanded=False):
        if st.button("🔄 Reset Sesji", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        if st.button("🗑️ Wyczyść Cache", use_container_width=True):
            from backend.cache_manager import get_cache_manager
            cache_manager = get_cache_manager()
            count = cache_manager.clear_cache()
            st.success(f"Wyczyszczono {count} plików cache")

    # Stopka
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**TMIV v{settings.app_version}**")
    st.sidebar.markdown("🚀 Advanced ML Platform")

    return config


def render_file_uploader() -> Optional[any]:
    """
    Renderuje file uploader.

    Returns:
        Optional[any]: Obiekt pliku lub None

    Example:
        >>> # W aplikacji Streamlit
        >>> uploaded_file = render_file_uploader()
    """
    st.subheader("📁 Wczytaj Dane")

    uploaded_file = st.file_uploader(
        "Wybierz plik z danymi",
        type=['csv', 'xlsx', 'xls', 'parquet', 'json'],
        help="Wspierane formaty: CSV, Excel, Parquet, JSON"
    )

    if uploaded_file:
        st.success(f"✅ Wczytano plik: {uploaded_file.name}")

    return uploaded_file


def render_target_selector(
    columns: List[str],
    auto_detected: Optional[str] = None
) -> str:
    """
    Renderuje selektor kolumny target.

    Args:
        columns: Lista nazw kolumn
        auto_detected: Automatycznie wykryta kolumna

    Returns:
        str: Wybrana kolumna target

    Example:
        >>> # W aplikacji Streamlit
        >>> columns = ['feature1', 'feature2', 'target']
        >>> target = render_target_selector(columns, auto_detected='target')
    """
    st.subheader("🎯 Wybierz Target")

    if auto_detected:
        st.info(f"💡 Automatycznie wykryto: **{auto_detected}**")
        default_index = columns.index(auto_detected) if auto_detected in columns else 0
    else:
        default_index = len(columns) - 1  # Ostatnia kolumna

    target_col = st.selectbox(
        "Kolumna target (zmienna do predykcji):",
        options=columns,
        index=default_index,
        help="Wybierz kolumnę, którą chcesz przewidywać"
    )

    return target_col


def render_strategy_selector() -> Tuple[str, bool, bool]:
    """
    Renderuje selektor strategii treningu.

    Returns:
        Tuple: (strategy, use_tuning, use_ensemble)

    Example:
        >>> # W aplikacji Streamlit
        >>> strategy, tuning, ensemble = render_strategy_selector()
    """
    st.subheader("⚙️ Strategia Treningu")

    col1, col2 = st.columns([2, 1])

    with col1:
        strategy = st.selectbox(
            "Wybierz strategię:",
            options=['fast_small', 'balanced', 'accurate', 'advanced'],
            index=1,  # Default: balanced
            help="""
            - **fast_small**: Szybkie modele dla małych datasetsów
            - **balanced**: Zbalansowany kompromis (zalecane)
            - **accurate**: Więcej modeli, lepsze wyniki
            - **advanced**: Wszystkie modele + zaawansowane opcje
            """
        )

    with col2:
        st.markdown("**Opcje:**")
        use_tuning = st.checkbox(
            "Tuning",
            value=False,
            help="Automatyczny tuning hyperparametrów (wolniejsze)"
        )
        use_ensemble = st.checkbox(
            "Ensemble",
            value=settings.enable_ensemble,
            help="Tworzenie modeli ensemble"
        )

    # Wyświetl opis strategii
    strategy_descriptions = {
        'fast_small': "⚡ Szybkie modele podstawowe - idealne dla małych zbiorów i prototypowania",
        'balanced': "⚖️ Zbalansowany zestaw modeli - dobry kompromis szybkość/jakość",
        'accurate': "🎯 Rozszerzony zestaw modeli - focus na jakość",
        'advanced': "🚀 Wszystkie dostępne modele - maksymalna eksploracja"
    }

    st.info(strategy_descriptions[strategy])

    return strategy, use_tuning, use_ensemble


def render_progress_indicator(
    current: int,
    total: int,
    message: str = "Przetwarzanie..."
) -> None:
    """
    Renderuje wskaźnik postępu.

    Args:
        current: Aktualny krok
        total: Całkowita liczba kroków
        message: Komunikat

    Example:
        >>> # W aplikacji Streamlit
        >>> render_progress_indicator(3, 10, "Trening modeli...")
    """
    progress = current / total if total > 0 else 0
    st.progress(progress)
    st.text(f"{message} ({current}/{total})")


def render_metric_card(
    label: str,
    value: any,
    delta: Optional[any] = None,
    help_text: Optional[str] = None
) -> None:
    """
    Renderuje kartę metryki.

    Args:
        label: Etykieta metryki
        value: Wartość
        delta: Zmiana (opcjonalna)
        help_text: Tekst pomocy

    Example:
        >>> # W aplikacji Streamlit
        >>> render_metric_card("Accuracy", 0.85, delta="+0.05")
    """
    st.metric(
        label=label,
        value=value,
        delta=delta,
        help=help_text
    )


def render_model_status(
    model_name: str,
    status: str,
    metrics: Optional[Dict] = None
) -> None:
    """
    Renderuje status modelu.

    Args:
        model_name: Nazwa modelu
        status: Status ('pending', 'training', 'completed', 'failed')
        metrics: Metryki modelu (opcjonalne)

    Example:
        >>> # W aplikacji Streamlit
        >>> render_model_status('RandomForest', 'completed', {'accuracy': 0.85})
    """
    status_icons = {
        'pending': '⏳',
        'training': '🔄',
        'completed': '✅',
        'failed': '❌'
    }

    icon = status_icons.get(status, '❓')

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f"{icon} **{model_name}**")
        if metrics:
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            st.caption(metric_str)

    with col2:
        if status == 'training':
            st.spinner()


def render_warning_box(warnings: List[str]) -> None:
    """
    Renderuje box z ostrzeżeniami.

    Args:
        warnings: Lista ostrzeżeń

    Example:
        >>> # W aplikacji Streamlit
        >>> render_warning_box(['Brakujące wartości', 'Niezbalansowane klasy'])
    """
    if not warnings:
        return

    with st.expander("⚠️ Ostrzeżenia", expanded=True):
        for warning in warnings:
            st.warning(warning)


def render_info_box(title: str, content: str, expanded: bool = True) -> None:
    """
    Renderuje box informacyjny.

    Args:
        title: Tytuł
        content: Treść
        expanded: Czy rozwinięty

    Example:
        >>> # W aplikacji Streamlit
        >>> render_info_box("Info", "To jest informacja")
    """
    with st.expander(title, expanded=expanded):
        st.info(content)


def render_download_button(
    data: bytes,
    filename: str,
    label: str = "📥 Pobierz",
    mime: str = "application/octet-stream"
) -> None:
    """
    Renderuje przycisk pobierania.

    Args:
        data: Dane do pobrania
        filename: Nazwa pliku
        label: Etykieta przycisku
        mime: Typ MIME

    Example:
        >>> # W aplikacji Streamlit
        >>> render_download_button(b'data', 'file.txt', 'Pobierz plik')
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime,
        use_container_width=True
    )