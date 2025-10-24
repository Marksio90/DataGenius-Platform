import logging
import traceback
import functools
from enum import Enum
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd
from datetime import datetime
import json

# Setup structured logging
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

class ErrorType(Enum):
    """Typy błędów w systemie"""
    DATA_VALIDATION = "data_validation"
    MODEL_TRAINING = "model_training"
    API_ERROR = "api_error"
    MEMORY_ERROR = "memory_error"
    SECURITY_ERROR = "security_error"
    SYSTEM_ERROR = "system_error"
    USER_INPUT_ERROR = "user_input_error"

class MLPlatformError(Exception):
    """Bazowa klasa błędów platformy"""
    def __init__(self, error_type: ErrorType, message: str, details: Dict = None, user_message: str = None):
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.user_message = user_message or message
        self.timestamp = datetime.now().isoformat()
        super().__init__(message)

class ErrorRecovery:
    """System odzyskiwania po błędach"""
    
    @staticmethod
    def suggest_memory_fixes():
        """Sugestie dla problemów pamięciowych"""
        st.error("🚫 **Błąd pamięci** - Dataset może być zbyt duży")
        
        with st.expander("💡 Jak naprawić problem z pamięcią"):
            st.markdown("""
            **Opcje naprawy:**
            1. 📉 **Zmniejsz dane**: Użyj sampling (np. 10,000 wierszy)
            2. ⚡ **Strategia 'fast'**: Wybierz szybsze algorytmy  
            3. 🔧 **Wyłącz tuning**: Hyperparameter tuning zużywa dużo RAM
            4. 💾 **Zamknij inne aplikacje**: Zwolnij pamięć RAM
            5. 🗂️ **Usuń kolumny**: Usuń niepotrzebne cechy przed treningiem
            """)
            
            if st.button("🔄 Spróbuj z sampling 10k wierszy"):
                st.session_state.force_sampling = 10000
                st.rerun()
    
    @staticmethod 
    def suggest_data_fixes(error_details: Dict):
        """Sugestie dla problemów z danymi"""
        st.error("🚫 **Problem z danymi**")
        
        with st.expander("💡 Jak naprawić dane"):
            if "insufficient_samples" in error_details:
                st.markdown("""
                **Za mało próbek:**
                - Potrzebujesz minimum 10 próbek na klasę
                - Spróbuj połączyć podobne klasy
                - Zbierz więcej danych
                """)
            
            elif "missing_target" in error_details:
                st.markdown("""
                **Problem ze zmienną docelową:**
                - Sprawdź czy kolumna docelowa istnieje
                - Czy ma prawidłowe wartości (nie tylko NaN)?
                - Czy nazwa kolumny jest poprawna?
                """)
                
            elif "multicollinearity" in error_details:
                st.markdown("""
                **Multikolinearność:**
                - Usuń skorelowane kolumny (>0.95 korelacja)
                - Użyj PCA do redukcji wymiarowości
                - Wybierz najważniejsze cechy
                """)
    
    @staticmethod
    def suggest_algorithm_fixes():
        """Sugestie dla problemów z algorytmami"""
        st.warning("⚠️ **Niektóre algorytmy niedostępne**")
        
        with st.expander("💡 Jak dodać zaawansowane algorytmy"):
            st.code("""
# Zainstaluj w terminalu:
pip install xgboost lightgbm catboost

# Lub wszystkie naraz:
pip install -r requirements.txt
            """)
            st.info("Po instalacji uruchom aplikację ponownie")

class SmartErrorHandler:
    """Inteligentne zarządzanie błędami z kontekstowymi sugestiami"""
    
    @staticmethod
    def training_error_handler(func):
        """Decorator dla błędów treningu modelu"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                
            except MemoryError as e:
                logger.error("Memory error in training", 
                           function=func.__name__,
                           error=str(e))
                
                ErrorRecovery.suggest_memory_fixes()
                raise MLPlatformError(
                    ErrorType.MEMORY_ERROR,
                    "Insufficient memory for model training",
                    {"function": func.__name__, "suggestion": "reduce_data_size"},
                    "Dataset zbyt duży dla dostępnej pamięci"
                )
            
            except ImportError as e:
                error_msg = str(e).lower()
                logger.warning("Missing ML library", 
                             library=error_msg,
                             function=func.__name__)
                
                if any(lib in error_msg for lib in ['xgboost', 'lightgbm', 'catboost']):
                    ErrorRecovery.suggest_algorithm_fixes()
                    # Kontynuuj z dostępnymi algorytmami
                    kwargs['skip_advanced'] = True
                    return func(*args, **kwargs)
                else:
                    raise MLPlatformError(
                        ErrorType.SYSTEM_ERROR,
                        f"Required library not found: {e}",
                        {"missing_library": error_msg}
                    )
            
            except ValueError as e:
                error_msg = str(e).lower()
                logger.error("Value error in training",
                           error=str(e),
                           function=func.__name__)
                
                error_details = {}
                
                if "sample" in error_msg or "insufficient" in error_msg:
                    error_details["insufficient_samples"] = True
                elif "target" in error_msg or "y" in error_msg:
                    error_details["missing_target"] = True
                elif "multicollinear" in error_msg:
                    error_details["multicollinearity"] = True
                
                ErrorRecovery.suggest_data_fixes(error_details)
                raise MLPlatformError(
                    ErrorType.DATA_VALIDATION,
                    str(e),
                    error_details,
                    f"Problem z danymi: {str(e)}"
                )
            
            except Exception as e:
                # Nieoczekiwany błąd
                logger.exception("Unexpected training error",
                               function=func.__name__,
                               error=str(e),
                               traceback=traceback.format_exc())
                
                st.error(f"🚫 **Nieoczekiwany błąd**: {str(e)}")
                
                with st.expander("🔧 Opcje naprawy"):
                    st.markdown("""
                    **Spróbuj:**
                    1. 🔄 Odśwież stronę i spróbuj ponownie
                    2. 📊 Sprawdź czy dane są poprawne
                    3. ⚙️ Zmień parametry treningu
                    4. 🐛 Jeśli problem się powtarza - zgłoś błąd
                    """)
                    
                    if st.button("📋 Skopiuj szczegóły błędu"):
                        error_report = {
                            "error": str(e),
                            "function": func.__name__,
                            "timestamp": datetime.now().isoformat(),
                            "traceback": traceback.format_exc()
                        }
                        st.code(json.dumps(error_report, indent=2))
                
                raise MLPlatformError(
                    ErrorType.SYSTEM_ERROR,
                    f"Unexpected error in {func.__name__}",
                    {"original_error": str(e), "traceback": traceback.format_exc()}
                )
        
        return wrapper
    
    @staticmethod
    def data_processing_handler(func):
        """Decorator dla błędów przetwarzania danych"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                
            except pd.errors.EmptyDataError:
                logger.error("Empty dataset error", function=func.__name__)
                st.error("🚫 **Pusty plik** - nie zawiera danych")
                st.info("💡 Sprawdź czy plik nie jest uszkodzony")
                
                raise MLPlatformError(
                    ErrorType.DATA_VALIDATION,
                    "Dataset is empty",
                    {"suggestion": "check_file_integrity"}
                )
            
            except pd.errors.ParserError as e:
                logger.error("CSV parsing error", error=str(e), function=func.__name__)
                st.error("🚫 **Błąd parsowania pliku**")
                
                with st.expander("💡 Jak naprawić plik CSV"):
                    st.markdown("""
                    **Najczęstsze problemy:**
                    1. 📝 **Separator**: Spróbuj `;` zamiast `,`
                    2. 🔤 **Encoding**: Zapisz plik jako UTF-8
                    3. 📊 **Nagłówki**: Sprawdź czy pierwsza linia to nazwy kolumn
                    4. ✂️ **Cudzysłowy**: Usuń dodatkowe cudzysłowy z danych
                    """)
                
                raise MLPlatformError(
                    ErrorType.DATA_VALIDATION,
                    f"File parsing error: {str(e)}",
                    {"suggestion": "fix_csv_format"}
                )
            
            except UnicodeDecodeError:
                logger.error("Unicode decode error", function=func.__name__)
                st.error("🚫 **Problem z kodowaniem pliku**")
                
                with st.expander("💡 Jak naprawić kodowanie"):
                    st.markdown("""
                    **Rozwiązania:**
                    1. 🔤 Otwórz plik w Excel/LibreOffice
                    2. 💾 Zapisz jako CSV UTF-8
                    3. 📝 Lub użyj Notatnika i zapisz z kodowaniem UTF-8
                    """)
                
                raise MLPlatformError(
                    ErrorType.DATA_VALIDATION,
                    "File encoding error - use UTF-8",
                    {"suggestion": "convert_to_utf8"}
                )
            
            except Exception as e:
                logger.exception("Data processing error",
                               error=str(e),
                               function=func.__name__)
                
                raise MLPlatformError(
                    ErrorType.DATA_VALIDATION,
                    f"Data processing failed: {str(e)}",
                    {"original_error": str(e)}
                )
        
        return wrapper
    
    @staticmethod
    def api_error_handler(func):
        """Decorator dla błędów API"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
                
            except Exception as e:
                error_msg = str(e).lower()
                
                if "401" in error_msg or "unauthorized" in error_msg:
                    logger.error("API unauthorized", error=str(e))
                    st.error("🔑 **Błąd autoryzacji API**")
                    st.info("💡 Sprawdź czy API key jest prawidłowy")
                    
                elif "429" in error_msg or "rate limit" in error_msg:
                    logger.warning("API rate limit exceeded", error=str(e))
                    st.warning("⏳ **Przekroczono limit API** - spróbuj za chwilę")
                    
                elif "timeout" in error_msg:
                    logger.warning("API timeout", error=str(e))
                    st.warning("⏱️ **Timeout API** - serwer nie odpowiada")
                    st.info("💡 Spróbuj ponownie za moment")
                    
                else:
                    logger.error("API error", error=str(e), function=func.__name__)
                    st.error(f"🚫 **Błąd API**: {str(e)}")
                
                # Fallback do symulacji
                st.info("🎭 Przełączam na tryb symulacji...")
                
                raise MLPlatformError(
                    ErrorType.API_ERROR,
                    f"API error: {str(e)}",
                    {"fallback": "simulation_mode"}
                )
        
        return wrapper

class HealthMonitor:
    """Monitoring zdrowia aplikacji"""
    
    def __init__(self):
        if 'health_metrics' not in st.session_state:
            st.session_state.health_metrics = {
                'total_trainings': 0,
                'successful_trainings': 0,
                'failed_trainings': 0,
                'avg_training_time': 0,
                'errors_by_type': {},
                'last_error': None
            }
    
    def record_training_attempt(self, success: bool, duration: float = 0, error_type: str = None):
        """Zapisz próbę treningu"""
        metrics = st.session_state.health_metrics
        
        metrics['total_trainings'] += 1
        
        if success:
            metrics['successful_trainings'] += 1
            # Update average training time
            old_avg = metrics['avg_training_time']
            n = metrics['successful_trainings']
            metrics['avg_training_time'] = (old_avg * (n-1) + duration) / n
        else:
            metrics['failed_trainings'] += 1
            if error_type:
                metrics['errors_by_type'][error_type] = metrics['errors_by_type'].get(error_type, 0) + 1
            metrics['last_error'] = {
                'type': error_type,
                'timestamp': datetime.now().isoformat()
            }
        
        # Log metrics
        logger.info("Training attempt recorded",
                   success=success,
                   duration=duration,
                   total_trainings=metrics['total_trainings'],
                   success_rate=self.get_success_rate())
    
    def get_success_rate(self) -> float:
        """Oblicz success rate"""
        total = st.session_state.health_metrics['total_trainings']
        if total == 0:
            return 1.0
        successful = st.session_state.health_metrics['successful_trainings']
        return successful / total
    
    def get_health_status(self) -> Dict[str, Any]:
        """Status zdrowia aplikacji"""
        success_rate = self.get_success_rate()
        metrics = st.session_state.health_metrics
        
        return {
            'status': 'healthy' if success_rate > 0.8 else 'degraded' if success_rate > 0.5 else 'unhealthy',
            'success_rate': success_rate,
            'total_trainings': metrics['total_trainings'],
            'avg_training_time': metrics['avg_training_time'],
            'common_errors': dict(sorted(metrics['errors_by_type'].items(), key=lambda x: x[1], reverse=True)[:3]),
            'last_error': metrics['last_error']
        }
    
    def render_health_dashboard(self):
        """Renderuj dashboard zdrowia"""
        if st.sidebar.checkbox("📊 Health Dashboard"):
            health = self.get_health_status()
            
            st.sidebar.markdown("### 🏥 System Health")
            
            # Status indicator
            status_colors = {
                'healthy': 'green',
                'degraded': 'orange', 
                'unhealthy': 'red'
            }
            color = status_colors.get(health['status'], 'gray')
            st.sidebar.markdown(f"**Status:** :{color}[{health['status'].upper()}]")
            
            # Key metrics
            st.sidebar.metric("Success Rate", f"{health['success_rate']:.1%}")
            st.sidebar.metric("Total Trainings", health['total_trainings'])
            
            if health['avg_training_time'] > 0:
                st.sidebar.metric("Avg Time", f"{health['avg_training_time']:.1f}s")
            
            # Common errors
            if health['common_errors']:
                st.sidebar.markdown("**Top Errors:**")
                for error_type, count in health['common_errors'].items():
                    st.sidebar.text(f"• {error_type}: {count}")

# Global health monitor
health_monitor = HealthMonitor()# backend/error_handler.py - FRAGMENT Z POPRAWKAMI

class HealthMonitor:
    """Monitoring zdrowia aplikacji z bezpieczną inicjalizacją"""
    
    def __init__(self):
        self._ensure_health_metrics_initialized()
    
    def _ensure_health_metrics_initialized(self):
        """Bezpieczna inicjalizacja health metrics w session_state"""
        try:
            if hasattr(st, 'session_state'):
                if 'health_metrics' not in st.session_state:
                    st.session_state.health_metrics = {
                        'total_trainings': 0,
                        'successful_trainings': 0,
                        'failed_trainings': 0,
                        'avg_training_time': 0,
                        'errors_by_type': {},
                        'last_error': None
                    }
            else:
                # Fallback gdy session_state nie jest dostępny
                self._fallback_metrics = {
                    'total_trainings': 0,
                    'successful_trainings': 0,
                    'failed_trainings': 0,
                    'avg_training_time': 0,
                    'errors_by_type': {},
                    'last_error': None
                }
        except Exception:
            # Ostateczny fallback
            self._fallback_metrics = {
                'total_trainings': 0,
                'successful_trainings': 0,
                'failed_trainings': 0,
                'avg_training_time': 0,
                'errors_by_type': {},
                'last_error': None
            }
    
    def _get_metrics(self) -> dict:
        """Pobierz metryki z bezpiecznym fallbackiem"""
        try:
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'health_metrics'):
                return st.session_state.health_metrics
            else:
                return getattr(self, '_fallback_metrics', {
                    'total_trainings': 0,
                    'successful_trainings': 0,
                    'failed_trainings': 0,
                    'avg_training_time': 0,
                    'errors_by_type': {},
                    'last_error': None
                })
        except Exception:
            return {
                'total_trainings': 0,
                'successful_trainings': 0,
                'failed_trainings': 0,
                'avg_training_time': 0,
                'errors_by_type': {},
                'last_error': None
            }
    
    def _update_metrics(self, updates: dict):
        """Aktualizuj metryki z bezpiecznym fallbackiem"""
        try:
            if hasattr(st, 'session_state'):
                self._ensure_health_metrics_initialized()
                for key, value in updates.items():
                    st.session_state.health_metrics[key] = value
            else:
                if not hasattr(self, '_fallback_metrics'):
                    self._fallback_metrics = {}
                for key, value in updates.items():
                    self._fallback_metrics[key] = value
        except Exception:
            if not hasattr(self, '_fallback_metrics'):
                self._fallback_metrics = {}
            for key, value in updates.items():
                self._fallback_metrics[key] = value
    
    def record_training_attempt(self, success: bool, duration: float = 0, error_type: str = None):
        """Zapisz próbę treningu z bezpieczną obsługą"""
        try:
            metrics = self._get_metrics()
            
            metrics['total_trainings'] += 1
            
            if success:
                metrics['successful_trainings'] += 1
                # Update average training time
                old_avg = metrics['avg_training_time']
                n = metrics['successful_trainings']
                if n > 0:
                    metrics['avg_training_time'] = (old_avg * (n-1) + duration) / n
            else:
                metrics['failed_trainings'] += 1
                if error_type:
                    if 'errors_by_type' not in metrics:
                        metrics['errors_by_type'] = {}
                    metrics['errors_by_type'][error_type] = metrics['errors_by_type'].get(error_type, 0) + 1
                metrics['last_error'] = {
                    'type': error_type,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Update metrics
            self._update_metrics(metrics)
            
            # Log metrics (bezpiecznie)
            try:
                logger.info("Training attempt recorded",
                           success=success,
                           duration=duration,
                           total_trainings=metrics['total_trainings'],
                           success_rate=self.get_success_rate())
            except:
                pass
                
        except Exception as e:
            # Nie crashuj aplikacji jeśli logging się nie uda
            try:
                logger.warning(f"Failed to record training attempt: {e}")
            except:
                pass
    
    def get_success_rate(self) -> float:
        """Oblicz success rate z bezpieczną obsługą"""
        try:
            metrics = self._get_metrics()
            total = metrics.get('total_trainings', 0)
            if total == 0:
                return 1.0
            successful = metrics.get('successful_trainings', 0)
            return successful / total
        except Exception:
            return 1.0
    
    def get_health_status(self) -> Dict[str, Any]:
        """Status zdrowia aplikacji z bezpieczną obsługą"""
        try:
            success_rate = self.get_success_rate()
            metrics = self._get_metrics()
            
            return {
                'status': 'healthy' if success_rate > 0.8 else 'degraded' if success_rate > 0.5 else 'unhealthy',
                'success_rate': success_rate,
                'total_trainings': metrics.get('total_trainings', 0),
                'avg_training_time': metrics.get('avg_training_time', 0),
                'common_errors': dict(sorted(metrics.get('errors_by_type', {}).items(), key=lambda x: x[1], reverse=True)[:3]),
                'last_error': metrics.get('last_error')
            }
        except Exception:
            return {
                'status': 'unknown',
                'success_rate': 0.0,
                'total_trainings': 0,
                'avg_training_time': 0,
                'common_errors': {},
                'last_error': None
            }
    
    def render_health_dashboard(self):
        """Renderuj dashboard zdrowia z bezpieczną obsługą"""
        try:
            if not hasattr(st, 'sidebar') or not st.sidebar.checkbox("📊 Health Dashboard"):
                return
                
            health = self.get_health_status()
            
            st.sidebar.markdown("### 🏥 System Health")
            
            # Status indicator
            status_colors = {
                'healthy': 'green',
                'degraded': 'orange', 
                'unhealthy': 'red',
                'unknown': 'gray'
            }
            color = status_colors.get(health['status'], 'gray')
            st.sidebar.markdown(f"**Status:** :{color}[{health['status'].upper()}]")
            
            # Key metrics
            st.sidebar.metric("Success Rate", f"{health['success_rate']:.1%}")
            st.sidebar.metric("Total Trainings", health['total_trainings'])
            
            if health['avg_training_time'] > 0:
                st.sidebar.metric("Avg Time", f"{health['avg_training_time']:.1f}s")
            
            # Common errors
            if health['common_errors']:
                st.sidebar.markdown("**Top Errors:**")
                for error_type, count in health['common_errors'].items():
                    st.sidebar.text(f"• {error_type}: {count}")
                    
        except Exception as e:
            try:
                st.sidebar.error(f"Health dashboard error: {e}")
            except:
                pass

# Global health monitor z bezpieczną inicjalizacją
try:
    health_monitor = HealthMonitor()
except Exception:
    # Fallback health monitor
    class FallbackHealthMonitor:
        def record_training_attempt(self, *args, **kwargs): pass
        def get_success_rate(self): return 1.0
        def get_health_status(self): return {'status': 'unknown'}
        def render_health_dashboard(self): pass
    
    health_monitor = FallbackHealthMonitor()