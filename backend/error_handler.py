from __future__ import annotations

# ============================================================
#  Utils – pojedynczy rerun (bez zapętlania)
# ============================================================

from backend.safe_utils import truthy_df_safe

def rerun_once(key: str = "_rerun_once") -> None:
    """
    Wykonaj pojedynczy rerun aplikacji (na danym kluczu), bez zapętlania.
    Działa zarówno z st.rerun (>=1.30) jak i experimental_rerun (starsze).
    Poza Streamlitem – cicho nic nie robi.
    """
    try:
        import streamlit as st  # lokalny import (bez twardej zależności przy imporcie modułu)
        if not st.session_state.get(key, False):
            st.session_state[key] = True
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
    except Exception:
        # poza środowiskiem Streamlit / brak session_state – ignorujemy
        pass


# ============================================================
#  backend/error_handler.py
# ============================================================

import functools
import logging
import traceback
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import json

import pandas as pd

# Streamlit jako opcjonalny — UI wywołujemy tylko jeśli dostępny
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore


# ==================== LOGGING (z bezpiecznym fallbackiem) ====================

def _build_logger():
    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger()
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

        class _SimpleLogger:
            def _log(self, level, msg, **kw):
                try:
                    payload = json.dumps(kw, default=str, ensure_ascii=False)
                except Exception:
                    payload = str(kw)
                log_fn = getattr(logging, level, logging.info)
                log_fn(f"{msg} {payload}")

            def info(self, msg, **kw): self._log("info", msg, **kw)
            def warning(self, msg, **kw): self._log("warning", msg, **kw)
            def error(self, msg, **kw): self._log("error", msg, **kw)
            def exception(self, msg, **kw): logging.exception(msg + " " + str(kw))

        return _SimpleLogger()

logger = _build_logger()


# ==================== POMOCNICZE ====================

def _truthy(x: Any) -> bool:
    """Bezpieczne sprawdzenie prawdziwości (z użyciem truthy_df_safe jeśli dostępne)."""
    try:
        return bool(truthy_df_safe(x))
    except Exception:
        return bool(x)


# ==================== BŁĘDY / TYPY ====================

class ErrorType(Enum):
    DATA_VALIDATION = "data_validation"
    MODEL_TRAINING = "model_training"
    API_ERROR = "api_error"
    MEMORY_ERROR = "memory_error"
    SECURITY_ERROR = "security_error"
    SYSTEM_ERROR = "system_error"
    USER_INPUT_ERROR = "user_input_error"


class MLPlatformError(Exception):
    def __init__(
        self,
        error_type: ErrorType,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        user_message: Optional[str] = None,
    ):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.user_message = user_message or message
        self.timestamp = datetime.now().isoformat()
        super().__init__(message)


# ==================== SUGESTIE NAPRAW ====================

class ErrorRecovery:
    @staticmethod
    def suggest_memory_fixes():
        if st is None:
            return
        st.error("🚫 **Błąd pamięci** – zbiór może być zbyt duży")
        with st.expander("💡 Jak naprawić problem z pamięcią"):
            st.markdown(
                "- 📉 Zmniejsz dane (sampling, np. 10k wierszy)\n"
                "- ⚡ Tryb *fast* / mniej algorytmów\n"
                "- 🔧 Wyłącz tuning hiperparametrów\n"
                "- 💾 Zamknij inne aplikacje, zwolnij RAM\n"
                "- 🗂️ Usuń zbędne kolumny"
            )
            if st.button("🔄 Ustaw sampling 10 000 wierszy", key="btn_mem_sampling"):
                st.session_state["force_sampling"] = 10000
                rerun_once()

    @staticmethod
    def suggest_data_fixes(error_details: Dict[str, Any]):
        if st is None:
            return
        st.error("🚫 **Problem z danymi**")
        with st.expander("💡 Jak naprawić dane"):
            if error_details.get("insufficient_samples"):
                st.markdown("- Minimum ~10 próbek na klasę\n- Scal podobne klasy\n- Zbierz więcej danych")
            if error_details.get("missing_target"):
                st.markdown("- Sprawdź kolumnę docelową (nazwa, NaN)\n- Czy wartości są poprawne?")
            if error_details.get("multicollinearity"):
                st.markdown("- Usuń cechy >0.95 korelacji\n- Rozważ PCA / redukcję wymiaru")

    @staticmethod
    def suggest_algorithm_fixes():
        if st is None:
            return
        st.warning("⚠️ **Zaawansowane algorytmy niedostępne**")
        with st.expander("💡 Jak dodać XGBoost/LightGBM/CatBoost"):
            st.code("pip install xgboost lightgbm catboost\n# lub\npip install -r requirements.txt")
            st.info("Po instalacji uruchom aplikację ponownie.")


# ==================== DEKORATORY BŁĘDÓW ====================

class SmartErrorHandler:
    @staticmethod
    def training_error_handler(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except MemoryError as e:
                logger.error("Memory error in training", function=func.__name__, error=str(e))
                ErrorRecovery.suggest_memory_fixes()
                raise MLPlatformError(
                    ErrorType.MEMORY_ERROR,
                    "Insufficient memory for model training",
                    {"function": func.__name__, "suggestion": "reduce_data_size"},
                    "Dataset zbyt duży dla dostępnej pamięci",
                )

            except ImportError as e:
                msg = str(e).lower()
                logger.warning("Missing ML library", library=msg, function=func.__name__)
                if any(lib in msg for lib in ("xgboost", "lightgbm", "catboost")):
                    ErrorRecovery.suggest_algorithm_fixes()
                    # ✅ wstrzykujemy tylko gdy funkcja przyjmuje ten parametr
                    try:
                        sig = inspect.signature(func)
                        if "skip_advanced" in sig.parameters:
                            kwargs["skip_advanced"] = True
                            return func(*args, **kwargs)
                        # brak parametru w sygnaturze – ustaw flaga sesyjna
                        if st is not None and hasattr(st, "session_state"):
                            st.session_state["skip_advanced"] = True
                        return func(*args, **kwargs)
                    except Exception:
                        raise MLPlatformError(
                            ErrorType.SYSTEM_ERROR,
                            f"Required library not found: {e}",
                            {"missing_library": msg},
                        )
                # inna biblioteka brakująca
                raise MLPlatformError(
                    ErrorType.SYSTEM_ERROR,
                    f"Required library not found: {e}",
                    {"missing_library": msg},
                )

            except ValueError as e:
                msg = str(e).lower()
                logger.error("Value error in training", error=str(e), function=func.__name__)
                details: Dict[str, Any] = {}

                # Szersze sygnały o brakach i jakości danych
                if any(k in msg for k in (
                    "too few samples", "not enough samples", "insufficient", "at least 2 classes",
                    "only one class", "need samples", "n_splits", "folds"
                )):
                    details["insufficient_samples"] = True

                if any(k in msg for k in (
                    "target", "y", "unknown label", "contains nan", "contains infinity", "y should",
                    "input y", "bad input", "labels", "empty"
                )):
                    details["missing_target"] = True

                if any(k in msg for k in (
                    "multicollinear", "multicollinearity", "singular matrix",
                    "matrix is not invertible", "ill-conditioned", "collinearity"
                )):
                    details["multicollinearity"] = True

                ErrorRecovery.suggest_data_fixes(details)
                raise MLPlatformError(ErrorType.DATA_VALIDATION, str(e), details, f"Problem z danymi: {str(e)}")

            except Exception as e:
                logger.exception(
                    "Unexpected training error",
                    function=func.__name__, error=str(e),
                    traceback=traceback.format_exc()
                )
                if st is not None:
                    try:
                        st.error(f"🚫 **Nieoczekiwany błąd**: {e}")
                        with st.expander("🔧 Opcje naprawy"):
                            st.markdown("1) Odśwież stronę  \n2) Sprawdź dane  \n3) Zmień parametry  \n4) Zgłoś błąd")
                            if st.button(f"📋 Skopiuj szczegóły błędu", key=f"btn_copy_{func.__name__}"):
                                st.code(json.dumps({
                                    "error": str(e),
                                    "function": func.__name__,
                                    "timestamp": datetime.now().isoformat(),
                                    "traceback": traceback.format_exc()
                                }, indent=2, ensure_ascii=False))
                    except Exception:
                        pass
                raise MLPlatformError(
                    ErrorType.SYSTEM_ERROR,
                    f"Unexpected error in {func.__name__}",
                    {"original_error": str(e), "traceback": traceback.format_exc()},
                )
        return wrapper

    @staticmethod
    def data_processing_handler(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)

            except pd.errors.EmptyDataError:
                logger.error("Empty dataset error", function=func.__name__)
                if st is not None:
                    try:
                        st.error("🚫 **Pusty plik** – brak danych")
                    except Exception:
                        pass
                raise MLPlatformError(
                    ErrorType.DATA_VALIDATION, "Dataset is empty",
                    {"suggestion": "check_file_integrity"}
                )

            except pd.errors.ParserError as e:
                logger.error("CSV parsing error", error=str(e), function=func.__name__)
                if st is not None:
                    try:
                        st.error("🚫 **Błąd parsowania pliku**")
                        with st.expander("💡 Jak naprawić CSV"):
                            st.markdown("- Spróbuj separatora `;`  \n- Zapisz jako UTF-8  \n- Sprawdź nagłówki  \n- Usuń zbędne cudzysłowy")
                    except Exception:
                        pass
                raise MLPlatformError(
                    ErrorType.DATA_VALIDATION, f"File parsing error: {e}",
                    {"suggestion": "fix_csv_format"}
                )

            except UnicodeDecodeError:
                logger.error("Unicode decode error", function=func.__name__)
                if st is not None:
                    try:
                        st.error("🚫 **Problem z kodowaniem pliku** (użyj UTF-8)")
                    except Exception:
                        pass
                raise MLPlatformError(
                    ErrorType.DATA_VALIDATION, "File encoding error - use UTF-8",
                    {"suggestion": "convert_to_utf8"}
                )

            except Exception as e:
                logger.exception("Data processing error", error=str(e), function=func.__name__)
                raise MLPlatformError(
                    ErrorType.DATA_VALIDATION,
                    f"Data processing failed: {e}",
                    {"original_error": str(e)}
                )
        return wrapper

    @staticmethod
    def api_error_handler(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                msg = str(e).lower()
                if st is not None:
                    try:
                        if "401" in msg or "unauthorized" in msg:
                            logger.error("API unauthorized", error=str(e))
                            st.error("🔒 **Błąd autoryzacji API** – sprawdź klucz")
                        elif "429" in msg or "rate limit" in msg:
                            logger.warning("API rate limit exceeded", error=str(e))
                            st.warning("⏳ **Przekroczono limit API** – spróbuj później")
                        elif "timeout" in msg:
                            logger.warning("API timeout", error=str(e))
                            st.warning("⏱️ **Timeout API** – serwer nie odpowiada")
                        else:
                            logger.error("API error", error=str(e), function=func.__name__)
                            st.error(f"🚫 **Błąd API**: {e}")
                        st.info("🎭 Przełączam na tryb symulacji…")
                    except Exception:
                        pass
                raise MLPlatformError(
                    ErrorType.API_ERROR, f"API error: {e}",
                    {"fallback": "simulation_mode"}
                )
        return wrapper

    # ✅ Aliasy dla kompatybilności wstecznej:
    #    Pozwala używać @SmartErrorHandler.data_processing dokładnie jak wcześniej.
    data_processing = data_processing_handler
    training_error = training_error_handler
    api_error = api_error_handler


# ==================== ✅ Context manager dla bezpiecznego wykonywania ====================

@dataclass
class _SafeBox:
    value: Any = None
    error: Optional[Exception] = None

@contextmanager
def safe_execution(operation_name: str, fallback_value=None, suppress_errors: bool = False):
    """
    Użycie:
        with safe_execution("training model", fallback_value=default) as box:
            box.value = train_model()
        result = box.value  # = default gdy poleci wyjątek
    """
    box = _SafeBox(value=fallback_value)
    try:
        yield box
    except Exception as e:
        box.error = e
        logger.error(f"Error in {operation_name}", error=str(e))
        if not _truthy(suppress_errors) and st is not None:
            try:
                st.error(f"❌ Błąd w {operation_name}: {str(e)}")
            except Exception:
                pass
        # brak raise -> zwracamy fallback przez box.value


# ==================== MONITORING ZDROWIA ====================

class HealthMonitor:
    """Monitoring zdrowia aplikacji."""

    def __init__(self):
        self._ensure_initialized()

    def _ensure_initialized(self):
        if st is None or not hasattr(st, "session_state"):
            return
        if 'health_metrics' not in st.session_state:
            st.session_state.health_metrics = {
                'total_trainings': 0,
                'successful_trainings': 0,
                'failed_trainings': 0,
                'avg_training_time': 0.0,
                'errors_by_type': {},
                'last_error': None,
            }

    def _metrics(self) -> dict:
        self._ensure_initialized()
        if st is None or not hasattr(st, "session_state"):
            # fallback poza Streamlit
            return {
                'total_trainings': 0,
                'successful_trainings': 0,
                'failed_trainings': 0,
                'avg_training_time': 0.0,
                'errors_by_type': {},
                'last_error': None,
            }
        return st.session_state.health_metrics

    def record_training_attempt(self, success: bool, duration: float = 0.0, error_type: Optional[str] = None):
        m = self._metrics()
        m['total_trainings'] += 1
        if _truthy(success):
            m['successful_trainings'] += 1
            n = m['successful_trainings']
            old_avg = float(m.get('avg_training_time', 0.0))
            m['avg_training_time'] = (old_avg * (n - 1) + float(duration)) / max(n, 1)
        else:
            m['failed_trainings'] += 1
            if _truthy(error_type):
                m['errors_by_type'][error_type] = m['errors_by_type'].get(error_type, 0) + 1
            m['last_error'] = {'type': error_type, 'timestamp': datetime.now().isoformat()}

        try:
            logger.info(
                "Training attempt recorded",
                success=success, duration=duration,
                total_trainings=m['total_trainings'],
                success_rate=self.get_success_rate()
            )
        except Exception:
            pass

    def get_success_rate(self) -> float:
        m = self._metrics()
        total = int(m['total_trainings'])
        return 1.0 if total == 0 else float(m['successful_trainings']) / max(total, 1)

    def get_health_status(self) -> Dict[str, Any]:
        m = self._metrics()
        success_rate = self.get_success_rate()
        status = 'healthy' if success_rate > 0.8 else ('degraded' if success_rate > 0.5 else 'unhealthy')
        return {
            'status': status,
            'success_rate': success_rate,
            'total_trainings': int(m['total_trainings']),
            'avg_training_time': float(m['avg_training_time']),
            'common_errors': dict(sorted(m['errors_by_type'].items(), key=lambda x: x[1], reverse=True)[:3]),
            'last_error': m['last_error'],
        }

    def render_health_dashboard(self):
        """Rysuje panel zdrowia (jeśli Streamlit dostępny)."""
        if st is None:
            return
        health = self.get_health_status()

        st.markdown("### 🏥 System Health")
        colors = {'healthy': 'green', 'degraded': 'orange', 'unhealthy': 'red'}
        color = colors.get(health['status'], 'gray')
        st.markdown(f"**Status:** :{color}[{health['status'].upper()}]")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Success Rate", f"{health['success_rate']:.1%}")
        with c2:
            st.metric("Total Trainings", health['total_trainings'])
        with c3:
            if health['avg_training_time'] > 0:
                st.metric("Avg Time", f"{health['avg_training_time']:.1f}s")
            else:
                st.metric("Avg Time", "—")

        if health['common_errors']:
            st.markdown("**Top Errors:**")
            for et, cnt in health['common_errors'].items():
                st.text(f"• {et}: {cnt}")


# ==================== SINGLETON ====================

health_monitor = HealthMonitor()
