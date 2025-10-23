"""
Centralna obsługa błędów aplikacji.

Funkcjonalności:
- Definicje niestandardowych wyjątków
- Wrapper funkcji z obsługą błędów
- Formatowanie komunikatów dla UI
- Logowanie błędów z kodami
"""

import functools
import logging
import traceback
from typing import Any, Callable, Optional, TypeVar, Union

import streamlit as st

# Konfiguracja loggera
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type variable dla dekoratora
F = TypeVar('F', bound=Callable[..., Any])


class TMIVException(Exception):
    """Bazowy wyjątek dla aplikacji TMIV."""

    def __init__(self, message: str, code: str = "TMIV-000"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DataLoadException(TMIVException):
    """Wyjątek przy wczytywaniu danych."""

    def __init__(self, message: str):
        super().__init__(message, code="TMIV-DATA-001")


class DataValidationException(TMIVException):
    """Wyjątek przy walidacji danych."""

    def __init__(self, message: str):
        super().__init__(message, code="TMIV-DATA-002")


class EDAException(TMIVException):
    """Wyjątek podczas analizy EDA."""

    def __init__(self, message: str):
        super().__init__(message, code="TMIV-EDA-001")


class MLTrainingException(TMIVException):
    """Wyjątek podczas treningu modelu."""

    def __init__(self, message: str):
        super().__init__(message, code="TMIV-ML-001")


class ExportException(TMIVException):
    """Wyjątek podczas eksportu."""

    def __init__(self, message: str):
        super().__init__(message, code="TMIV-EXPORT-001")


class AIIntegrationException(TMIVException):
    """Wyjątek podczas integracji z AI."""

    def __init__(self, message: str):
        super().__init__(message, code="TMIV-AI-001")


def handle_errors(
    show_in_ui: bool = True,
    default_return: Any = None,
    error_message: str = "Wystąpił błąd podczas operacji"
) -> Callable[[F], F]:
    """
    Dekorator do obsługi błędów z opcjonalnym wyświetlaniem w UI.

    Args:
        show_in_ui: Czy pokazać błąd w Streamlit
        default_return: Wartość zwracana w przypadku błędu
        error_message: Niestandardowy komunikat błędu

    Returns:
        Callable: Zdekorowana funkcja

    Example:
        @handle_errors(show_in_ui=True)
        def risky_function():
            return 1 / 0
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except TMIVException as e:
                logger.error(f"[{e.code}] {e.message}")
                if show_in_ui:
                    st.error(f"❌ **Błąd [{e.code}]**: {e.message}")
                return default_return
            except Exception as e:
                error_details = traceback.format_exc()
                logger.error(f"Nieoczekiwany błąd w {func.__name__}: {error_details}")
                if show_in_ui:
                    st.error(f"❌ **{error_message}**: {str(e)}")
                    with st.expander("🔍 Szczegóły techniczne"):
                        st.code(error_details)
                return default_return

        return wrapper  # type: ignore

    return decorator


def log_error(
    error: Exception,
    context: str = "",
    severity: str = "ERROR"
) -> None:
    """
    Loguje błąd z kontekstem.

    Args:
        error: Wyjątek do zalogowania
        context: Dodatkowy kontekst
        severity: Poziom severity (ERROR, WARNING, INFO)

    Example:
        >>> try:
        ...     1 / 0
        ... except Exception as e:
        ...     log_error(e, "podczas obliczeń")
    """
    log_message = f"{context}: {str(error)}" if context else str(error)

    if severity == "ERROR":
        logger.error(log_message)
    elif severity == "WARNING":
        logger.warning(log_message)
    else:
        logger.info(log_message)


def format_error_for_ui(error: Exception) -> str:
    """
    Formatuje błąd do przyjaznej wiadomości dla użytkownika.

    Args:
        error: Wyjątek do sformatowania

    Returns:
        str: Sformatowana wiadomość

    Example:
        >>> e = ValueError("Invalid input")
        >>> msg = format_error_for_ui(e)
        >>> "Invalid input" in msg
        True
    """
    if isinstance(error, TMIVException):
        return f"[{error.code}] {error.message}"

    error_name = type(error).__name__
    error_msg = str(error)

    # Skróć długie komunikaty
    if len(error_msg) > 200:
        error_msg = error_msg[:200] + "..."

    return f"{error_name}: {error_msg}"


def safe_execute(
    func: Callable,
    *args: Any,
    fallback_value: Any = None,
    error_message: str = "Operacja nie powiodła się",
    **kwargs: Any
) -> Any:
    """
    Bezpiecznie wykonuje funkcję z fallbackiem.

    Args:
        func: Funkcja do wykonania
        *args: Argumenty pozycyjne
        fallback_value: Wartość zwracana przy błędzie
        error_message: Komunikat błędu
        **kwargs: Argumenty nazwane

    Returns:
        Any: Wynik funkcji lub fallback_value

    Example:
        >>> result = safe_execute(lambda: 1/0, fallback_value=0)
        >>> result
        0
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"{error_message}: {str(e)}")
        return fallback_value