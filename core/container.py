from __future__ import annotations

from backend.safe_utils import truthy_df_safe

from typing import Optional, Any, Dict
import threading
import time

# Streamlit jest opcjonalny – kontener działa też bez niego
try:
    import streamlit as st  # type: ignore
    _HAS_ST = hasattr(st, "session_state")
except Exception:  # pragma: no cover
    st = None  # type: ignore
    _HAS_ST = False


class DependencyContainer:
    """Dependency Injection Container (thread-safe singleton, bezpieczne fallbacki)."""

    _instance: Optional["DependencyContainer"] = None
    _lock = threading.Lock()

    # --- konstrukcja singletona ---
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # zapobiega wielokrotnej inicjalizacji tego samego obiektu
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._init_lock = threading.RLock()

        # Puste pola do wypełnienia w _setup_dependencies
        self._cache_service: Any = None
        self._ml_trainer: Any = None
        self._ai_insights: Any = None
        self._data_repo: Any = None
        self._platform_service: Any = None

        self._setup_dependencies()

    # --- inicjalizacja zależności (leniwe importy + bezpieczne fallbacki) ---
    def _setup_dependencies(self) -> None:
        with self._init_lock:
            # ML trainer
            MLModelTrainer: Any
            try:
                from backend.ml_integration import MLModelTrainer as _MLT  # type: ignore
                MLModelTrainer = _MLT
            except Exception:
                class _MLTStub:  # pragma: no cover
                    def train_model(self, *_, **__):
                        raise ImportError("Brak backend.ml_integration.MLModelTrainer – nie można trenować modeli.")
                    def train_model_async(self, *_, **__):
                        raise ImportError("Brak backend.ml_integration.MLModelTrainer – nie można trenować modeli.")
                MLModelTrainer = _MLTStub

            # AI generator
            AIDescriptionGenerator: Any
            try:
                from backend.ai_integration import AIDescriptionGenerator as _AIG  # type: ignore
                AIDescriptionGenerator = _AIG
            except Exception:
                class _AIGStub:  # pragma: no cover
                    def generate_recommendations(self, *_, **__):
                        return {
                            "key_insights": "Generator AI niedostępny.",
                            "action_items": ["Zainstaluj moduł backend.ai_integration"],
                        }
                AIDescriptionGenerator = _AIGStub

            # Cache – preferuj SmartCache, fallback prosty z TTL
            try:
                from backend.cache_manager import SmartCache  # type: ignore
                CacheImpl = SmartCache
            except Exception:
                class _SimpleCache:
                    def __init__(self):
                        self._d: Dict[str, Any] = {}
                        self._ttl: Dict[str, float] = {}
                        self._lock = threading.Lock()

                    def get(self, k: str) -> Any:
                        with self._lock:
                            exp = self._ttl.get(k)
                            if exp is not None and time.time() > exp:
                                self._d.pop(k, None)
                                self._ttl.pop(k, None)
                                return None
                            return self._d.get(k)

                    def set(self, k: str, v: Any, ttl: int = 3600) -> None:
                        with self._lock:
                            self._d[k] = v
                            self._ttl[k] = time.time() + max(0, int(ttl))

                    # kompatybilność z niektórymi interfejsami
                    def invalidate(self, pattern: str) -> None:
                        frag = pattern.replace("*", "")
                        with self._lock:
                            for key in list(self._d.keys()):
                                if frag in key:
                                    self._d.pop(key, None)
                                    self._ttl.pop(key, None)
                CacheImpl = _SimpleCache

            # Adaptery i warstwa serwisowa (wymagane do pełnej funkcjonalności)
            try:
                from core.adapters import (  # type: ignore
                    MLTrainerAdapter,
                    AIInsightsAdapter,
                    CacheServiceAdapter,
                    DataRepositoryAdapter,
                )
                from core.services import MLPlatformService  # type: ignore
            except Exception as e:
                # Jeżeli te moduły nie istnieją – nie ma sensu iść dalej
                raise ImportError("Brakuje core.adapters lub core.services – nie można zbudować kontenera.") from e

            # Instancje adapterów / usług
            self._cache_service = CacheServiceAdapter(CacheImpl())
            self._ml_trainer = MLTrainerAdapter(MLModelTrainer())
            self._ai_insights = AIInsightsAdapter(AIDescriptionGenerator())
            self._data_repo = DataRepositoryAdapter()

            # Główny serwis orkiestrujący
            self._platform_service = MLPlatformService(
                ml_trainer=self._ml_trainer,
                data_repo=self._data_repo,
                ai_insights=self._ai_insights,
                cache_service=self._cache_service,
            )

    # --- API: properties ---
    @property
    def platform_service(self) -> Any:
        """Główny serwis platformy (orchestrator)."""
        return self._platform_service

    @property
    def ml_trainer(self) -> Any:
        """Kompatybilność wsteczna: zwróć *oryginalny* trainer (bez adaptera)."""
        return getattr(self._ml_trainer, "trainer", self._ml_trainer)

    @property
    def ai_insights(self) -> Any:
        """Kompatybilność wsteczna: zwróć *oryginalny* generator (bez adaptera)."""
        return getattr(self._ai_insights, "generator", self._ai_insights)

    @property
    def cache_service(self) -> Any:
        """Serwis cache (opakowany adapterem)."""
        return self._cache_service

    @property
    def data_repo(self) -> Any:
        """Repozytorium modeli (adapter)."""
        return self._data_repo

    # --- żywotność zasobów ---
    def cleanup(self) -> None:
        """Zwolnij zasoby (np. wątki asynchronicznego trenera)."""
        try:
            tr = getattr(self._ml_trainer, "trainer", None)
            if tr is not None and hasattr(tr, "async_trainer") and hasattr(tr.async_trainer, "cleanup"):
                tr.async_trainer.cleanup()
        except Exception:
            pass

    # --- reset kontenera (np. po zmianie konfiguracji) ---
    @classmethod
    def _reset_singleton(cls) -> "DependencyContainer":
        with cls._lock:
            if cls._instance is not None:
                try:
                    cls._instance.cleanup()
                except Exception:
                    pass
            cls._instance = None
            return cls()


# ------------------------------------------------------------
# Global helpers
# ------------------------------------------------------------
def get_container(reset: bool = False) -> DependencyContainer:
    """
    Pobierz globalny kontener zależności.
    - W Streamlit: trzymamy go w `st.session_state` (stabilny między rerunami).
    - Poza Streamlit: klasyczny singleton w obrębie procesu.
    """
    key = "_di_container"

    if _HAS_ST:
        try:
            if reset or key not in st.session_state:
                st.session_state[key] = DependencyContainer._reset_singleton() if reset else DependencyContainer()
            return st.session_state[key]
        except Exception:
            # fallback do procesu, jeśli session_state był niedostępny
            pass

    if reset:
        return DependencyContainer._reset_singleton()
    return DependencyContainer()


def reset_container() -> DependencyContainer:
    """Wymuś przeładowanie zależności (również czyści zasoby)."""
    return get_container(reset=True)
