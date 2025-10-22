# backend/error_handler.py
"""
Centralized error handling for TMIV – Advanced ML Platform.

Goals
-----
- Convert raw exceptions into SAFE, user-friendly messages (with actionable hints).
- Generate short error IDs for cross-referencing logs and UI.
- Redact secrets from tracebacks (API keys, tokens, passwords).
- Provide utilities for Streamlit UI rendering and simple decorators/context managers.

Public API
---------
- classify_exception(exc) -> ErrorKind
- build_error_report(exc, *, context: dict | None = None) -> ErrorReport
- render_streamlit_error(report: ErrorReport) -> None              # optional, lazy import
- capture_and_report(context: str | None = None) -> decorator      # wraps a function
- capture_errors(context: str | None = None) -> context manager    # wraps a code block
- safe_call(func, *args, **kwargs) -> tuple[result, ErrorReport | None]

Notes
-----
- No hard dependency on Streamlit. We import it only inside `render_streamlit_error`.
- Logging uses the stdlib `logging` package; wire it to your YAML config in `config/logging.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Any, Callable, Optional, Tuple
import logging
import os
import re
import sys
import time
import traceback
import uuid

logger = logging.getLogger("app.error")


# =========================
# Redaction utilities
# =========================

_API_KEY_PATTERNS = [
    r"sk-[A-Za-z0-9]{20,}",                           # OpenAI-like
    r"Bearer\s+[A-Za-z0-9\-._~+/=]{20,}",             # Bearer tokens
    r"AIza[0-9A-Za-z\-_]{35}",                        # Google API
    r"ssh-rsa\s+[A-Za-z0-9+/=]{100,}",                # SSH keys in logs
]
_SECRET_FIELD_NAMES = {
    "password", "passwd", "pwd", "secret", "token", "api_key", "apikey",
    "authorization", "auth", "access_key", "private_key",
}

_REDACTION = "•••REDACTED•••"


def _redact_text(text: str) -> str:
    """Redact API keys/tokens from arbitrary text."""
    if not text:
        return text
    red = text
    for pat in _API_KEY_PATTERNS:
        red = re.sub(pat, _REDACTION, red, flags=re.IGNORECASE)
    # Also redact long hex/base64-ish blobs (heuristic)
    red = re.sub(r"[A-F0-9]{32,}", _REDACTION, red, flags=re.IGNORECASE)
    red = re.sub(r"[A-Za-z0-9+/=]{40,}", _REDACTION, red)
    return red


def _redact_mapping(obj: Any) -> Any:
    """Redact secrets in dict-like structures by key name."""
    try:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if str(k).lower() in _SECRET_FIELD_NAMES:
                    out[k] = _REDACTION
                else:
                    out[k] = _redact_mapping(v)
            return out
        if isinstance(obj, (list, tuple, set)):
            t = type(obj)
            return t(_redact_mapping(v) for v in obj)
        if isinstance(obj, str):
            return _redact_text(obj)
        return obj
    except Exception:
        return "<unserializable>"


# =========================
# Error taxonomy
# =========================

class ErrorKind(str, Enum):
    IO = "IO"
    PARSE = "PARSE"
    ENCODING = "ENCODING"
    VALIDATION = "VALIDATION"
    DATA = "DATA"
    MEMORY = "MEMORY"
    TIMEOUT = "TIMEOUT"
    DEPENDENCY = "DEPENDENCY"
    MODEL = "MODEL"
    PERMISSION = "PERMISSION"
    UNKNOWN = "UNKNOWN"


def classify_exception(exc: BaseException) -> ErrorKind:
    """Coarse mapping from Python/library exceptions to ErrorKind."""
    name = exc.__class__.__name__
    msg = str(exc).lower()

    # Specific libraries (optional import names handled by string check)
    if name in {"ParserError"} or "could not convert string" in msg or "parsing" in msg:
        return ErrorKind.PARSE
    if isinstance(exc, UnicodeDecodeError) or "utf-8" in msg or "encoding" in msg:
        return ErrorKind.ENCODING
    if isinstance(exc, (FileNotFoundError, IsADirectoryError)):
        return ErrorKind.IO
    if isinstance(exc, PermissionError):
        return ErrorKind.PERMISSION
    if isinstance(exc, MemoryError) or "out of memory" in msg or "oom" in msg:
        return ErrorKind.MEMORY
    if isinstance(exc, TimeoutError) or "timed out" in msg or "timeout" in msg:
        return ErrorKind.TIMEOUT
    if isinstance(exc, (ImportError, ModuleNotFoundError)) or "version" in msg and "mismatch" in msg:
        return ErrorKind.DEPENDENCY
    if isinstance(exc, (ValueError, KeyError, IndexError)):
        return ErrorKind.DATA
    if "convergence" in msg or "singular" in msg or "non-invertible" in msg:
        return ErrorKind.MODEL

    return ErrorKind.UNKNOWN


# =========================
# Error report dataclass
# =========================

@dataclass
class ErrorReport:
    error_id: str
    kind: ErrorKind
    user_message: str
    hint: str
    tech_message: str
    traceback: str
    context: dict[str, Any] = field(default_factory=dict)
    ts_unix: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["context"] = _redact_mapping(d.get("context", {}))
        d["tech_message"] = _redact_text(d.get("tech_message", ""))
        d["traceback"] = _redact_text(d.get("traceback", ""))
        return d


# =========================
# Report building
# =========================

def _short_id() -> str:
    return uuid.uuid4().hex[:8].upper()


def _friendly_hint(kind: ErrorKind, exc: BaseException) -> str:
    m = str(exc)
    if kind is ErrorKind.IO:
        return "Sprawdź ścieżkę i uprawnienia do pliku/katalogu."
    if kind is ErrorKind.ENCODING:
        return "Spróbuj wskazać kodowanie (np. UTF-8/Windows-1250) podczas wczytywania."
    if kind is ErrorKind.PARSE:
        return "Zweryfikuj separator (',' vs ';'), nagłówki i nietypowe wartości w pliku."
    if kind is ErrorKind.DATA:
        if "column" in m.lower() and "not found" in m.lower():
            return "Upewnij się, że nazwa kolumny jest poprawna po normalizacji nazw."
        return "Sprawdź typy kolumn i brakujące wartości; rozważ sanityzację danych."
    if kind is ErrorKind.MEMORY:
        return "Zredukuj wielkość danych (próbkowanie), wyłącz część wykresów lub uruchom na maszynie z większą pamięcią."
    if kind is ErrorKind.TIMEOUT:
        return "Wydłuż limit czasu albo uprość zadanie (mniej modeli, mniej foldów)."
    if kind is ErrorKind.DEPENDENCY:
        return "Zainstaluj/upewnij zgodność wersji pakietów (patrz wymagania i `pip/conda`)."
    if kind is ErrorKind.MODEL:
        return "Dostrój hiperparametry, spróbuj innego algorytmu lub skalowania cech."
    if kind is ErrorKind.PERMISSION:
        return "Uruchom aplikację z odpowiednimi uprawnieniami lub zmień lokalizację pliku."
    return "Spróbuj ponownie po drobnych poprawkach; jeśli problem wraca — podaj szczegóły błędu."


def _user_message(kind: ErrorKind, exc: BaseException) -> str:
    base = {
        ErrorKind.IO: "Problem z dostępem do pliku/katalogu.",
        ErrorKind.ENCODING: "Problem z kodowaniem znaków.",
        ErrorKind.PARSE: "Problem z parsowaniem danych.",
        ErrorKind.DATA: "Problem z danymi wejściowymi.",
        ErrorKind.MEMORY: "Brak pamięci dla tej operacji.",
        ErrorKind.TIMEOUT: "Przekroczono limit czasu.",
        ErrorKind.DEPENDENCY: "Problem z zależnościami/wersjami pakietów.",
        ErrorKind.MODEL: "Problem podczas treningu/ewaluacji modelu.",
        ErrorKind.PERMISSION: "Brak uprawnień.",
        ErrorKind.UNKNOWN: "Nieoczekiwany błąd.",
    }[kind]
    detail = str(exc).strip()
    if detail:
        detail = _redact_text(detail)
        return f"{base} Szczegóły: {detail}"
    return base


def build_error_report(exc: BaseException, *, context: dict[str, Any] | None = None) -> ErrorReport:
    """Create a sanitized, user-facing error report with traceback."""
    kind = classify_exception(exc)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    report = ErrorReport(
        error_id=_short_id(),
        kind=kind,
        user_message=_user_message(kind, exc),
        hint=_friendly_hint(kind, exc),
        tech_message=str(exc),
        traceback=tb,
        context=_redact_mapping(context or {}),
    )
    # Log the structured report (sanitized)
    try:
        logger.error("ErrorReport %s :: %s", report.error_id, report.to_dict())
    except Exception:
        pass
    return report


# =========================
# Streamlit integration (optional)
# =========================

def render_streamlit_error(report: ErrorReport) -> None:
    """
    Render an error box with expandable technical details in Streamlit.
    Lazy-imports Streamlit to avoid hard dependency for non-UI contexts.
    """
    try:
        import streamlit as st  # type: ignore

        st.error(f"❗ {report.user_message}")
        st.caption(f"Error ID: `{report.error_id}` • Typ: `{report.kind}`")
        with st.expander("Szczegóły techniczne (kliknij, aby rozwinąć)"):
            st.code(_redact_text(report.traceback), language="text")
            st.json(report.to_dict())
            st.info(report.hint)
    except Exception:
        # If Streamlit not available or rendering fails, log only
        logger.exception("Failed to render Streamlit error for %s", report.error_id)


# =========================
# Decorators / context managers
# =========================

def capture_and_report(context: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Tuple[Any, Optional[ErrorReport]]]]:
    """
    Decorator: run a function and return (result, ErrorReport|None).
    On exception, returns (None, report) and logs the error.
    """

    def wrapper(fn: Callable[..., Any]) -> Callable[..., Tuple[Any, Optional[ErrorReport]]]:
        def inner(*args: Any, **kwargs: Any) -> Tuple[Any, Optional[ErrorReport]]:
            try:
                return fn(*args, **kwargs), None
            except BaseException as exc:  # noqa: BLE001
                ctx = {"decorator_context": context, "function": getattr(fn, "__qualname__", str(fn))}
                rep = build_error_report(exc, context=ctx)
                return None, rep

        inner.__name__ = getattr(fn, "__name__", "captured_func")
        inner.__doc__ = fn.__doc__
        inner.__qualname__ = getattr(fn, "__qualname__", inner.__name__)
        return inner

    return wrapper


class capture_errors:
    """
    Context manager: capture any exception and expose an ErrorReport.

    Usage:
        with capture_errors("training") as cap:
            ... risky code ...
        if cap.error:
            render_streamlit_error(cap.error)
    """

    def __init__(self, context: str | None = None):
        self.context = context
        self.error: Optional[ErrorReport] = None

    def __enter__(self) -> "capture_errors":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            return False  # no exception
        self.error = build_error_report(exc, context={"context": self.context})
        # Suppress exception (already handled upstream); return True to prevent crash
        return True


def safe_call(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, Optional[ErrorReport]]:
    """
    Execute callable and return (result, ErrorReport|None). Does not raise.
    """
    try:
        return func(*args, **kwargs), None
    except BaseException as exc:  # noqa: BLE001
        rep = build_error_report(exc, context={"function": getattr(func, "__qualname__", str(func))})
        return None, rep


__all__ = [
    "ErrorKind",
    "ErrorReport",
    "classify_exception",
    "build_error_report",
    "render_streamlit_error",
    "capture_and_report",
    "capture_errors",
    "safe_call",
]
