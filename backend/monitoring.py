"""
Panel monitorowania aplikacji.

Funkcjonalności:
- Health check
- Metryki wydajności
- Status providerów
- Informacje o cache
"""

import logging
import platform
import sys
from typing import Dict

import pandas as pd
import streamlit as st

from backend.cache_manager import get_cache_manager
from backend.telemetry import get_telemetry_collector
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def get_system_info() -> Dict:
    """
    Zwraca informacje o systemie.

    Returns:
        Dict: Informacje systemowe

    Example:
        >>> info = get_system_info()
        >>> 'python_version' in info
        True
    """
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "Unknown",
        "pandas_version": pd.__version__,
        "streamlit_version": st.__version__,
    }


def get_application_health() -> Dict:
    """
    Zwraca stan zdrowia aplikacji.

    Returns:
        Dict: Stan aplikacji

    Example:
        >>> health = get_application_health()
        >>> 'status' in health
        True
    """
    health = {
        "status": "healthy",
        "checks": {},
        "warnings": [],
    }

    # Sprawdź cache
    try:
        cache_manager = get_cache_manager()
        cache_info = cache_manager.get_cache_info()
        health["checks"]["cache"] = "OK"
        health["cache_size_mb"] = cache_info["size_mb"]
    except Exception as e:
        health["checks"]["cache"] = "ERROR"
        health["warnings"].append(f"Cache error: {str(e)}")

    # Sprawdź telemetrię
    try:
        telemetry = get_telemetry_collector()
        stats = telemetry.get_statistics()
        health["checks"]["telemetry"] = "OK"
        health["total_events"] = stats["total_events"]
    except Exception as e:
        health["checks"]["telemetry"] = "ERROR"
        health["warnings"].append(f"Telemetry error: {str(e)}")

    # Sprawdź AI providers
    try:
        from backend.ai_integration import get_ai_integration
        ai = get_ai_integration()
        provider_status = ai.get_provider_status()
        health["checks"]["ai_providers"] = "OK" if provider_status["any"] else "WARNING"
        health["ai_providers"] = provider_status
        if not provider_status["any"]:
            health["warnings"].append("No AI providers available - using fallback mode")
    except Exception as e:
        health["checks"]["ai_providers"] = "ERROR"
        health["warnings"].append(f"AI providers error: {str(e)}")

    # Ogólny status
    if health["warnings"]:
        health["status"] = "degraded"

    if any(v == "ERROR" for v in health["checks"].values()):
        health["status"] = "unhealthy"

    return health


def get_performance_metrics() -> Dict:
    """
    Zwraca metryki wydajności.

    Returns:
        Dict: Metryki wydajności

    Example:
        >>> metrics = get_performance_metrics()
        >>> 'session_duration_seconds' in metrics
        True
    """
    telemetry = get_telemetry_collector()
    stats = telemetry.get_statistics()

    metrics = {
        "session_duration_seconds": stats["session_duration_seconds"],
        "total_operations": stats["total_events"],
        "function_calls": stats["function_calls"],
        "execution_times": stats["execution_times"],
    }

    return metrics


def display_monitoring_panel():
    """
    Wyświetla panel monitorowania w Streamlit.
    """
    try:
        st.subheader("🔧 Panel Monitorowania")
        
        # System info
        with st.expander("ℹ️ Informacje Systemowe", expanded=False):
            try:
                sys_info = get_system_info()
                for key, value in sys_info.items():
                    st.text(f"{key}: {value}")
            except Exception as e:
                st.error(f"Błąd pobierania info systemowych: {e}")

        # Health check
        with st.expander("❤️ Stan Zdrowia Aplikacji", expanded=True):
            try:
                health = get_application_health()

                status_emoji = {
                    "healthy": "✅",
                    "degraded": "⚠️",
                    "unhealthy": "❌",
                }

                st.markdown(f"### Status: {status_emoji.get(health['status'], '❓')} {health['status'].upper()}")

                # Checks
                st.markdown("**Sprawdzenia:**")
                for check, status in health.get("checks", {}).items():
                    status_icon = "✅" if status == "OK" else "⚠️" if status == "WARNING" else "❌"
                    st.text(f"{status_icon} {check}: {status}")

                # Warnings
                if health.get("warnings"):
                    st.markdown("**Ostrzeżenia:**")
                    for warning in health["warnings"]:
                        st.warning(warning)

                # AI Providers
                if "ai_providers" in health:
                    st.markdown("**AI Providers:**")
                    providers = health["ai_providers"]
                    st.text(f"OpenAI: {'✅' if providers.get('openai') else '❌'}")
                    st.text(f"Anthropic: {'✅' if providers.get('anthropic') else '❌'}")
            except Exception as e:
                st.error(f"Błąd health check: {e}")

        # Performance metrics
        with st.expander("📊 Metryki Wydajności", expanded=False):
            try:
                metrics = get_performance_metrics()

                st.metric("Czas sesji", f"{metrics.get('session_duration_seconds', 0):.1f}s")
                st.metric("Liczba operacji", metrics.get('total_operations', 0))

                if metrics.get('execution_times'):
                    st.markdown("**Czasy wykonania:**")
                    exec_df = pd.DataFrame.from_dict(metrics['execution_times'], orient='index')
                    st.dataframe(exec_df)
            except Exception as e:
                st.error(f"Błąd metryk: {e}")

        # Cache info
        with st.expander("💾 Informacje o Cache", expanded=False):
            try:
                cache_manager = get_cache_manager()
                cache_info = cache_manager.get_cache_info()

                st.metric("Rozmiar cache", f"{cache_info.get('size_mb', 0):.2f} MB")
                st.metric("Liczba plików", cache_info.get('n_files', 0))

                if st.button("🗑️ Wyczyść Cache"):
                    count = cache_manager.clear_cache()
                    st.success(f"Usunięto {count} plików cache")
                    st.rerun()
            except Exception as e:
                st.error(f"Błąd cache info: {e}")
                
    except Exception as e:
        st.error(f"Krytyczny błąd panelu monitoringu: {e}")
        import traceback
        st.code(traceback.format_exc())