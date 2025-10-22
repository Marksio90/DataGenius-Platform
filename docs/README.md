# core/container.py
"""
DI Container – rejestr i lazy-singletony usług TMIV.

Użycie:
    from core.container import get_container
    c = get_container()
    ml = c.resolve("ml")
    data = c.resolve("data")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# Usługi (łagodne importy – nie crashujemy edytora)
from core.services.cache_service import CacheService
from core.services.data_service import DataService
from core.services.eda_service import EDAService
from core.services.ml_service import MLService
from core.services.explain_service import ExplainService
from core.services.export_service import ExportService
from core.services.insights_service import InsightsService
from core.services.telemetry_service import TelemetryService

try:  # settings
    from config.settings import get_settings as _get_settings
except Exception:
    def _get_settings():
        class _S:  # minimal fallback
            app_name = "TMIV Advanced ML Platform"
            env = "development"
            database_url = None
        return _S()

# DB opcjonalnie
try:
    from core.services.db_service import DBService
except Exception:
    DBService = None  # type: ignore


@dataclass
class _Container:
    _singletons: Dict[str, Any] = field(default_factory=dict)
    _factories: Dict[str, Callable[[], Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Rejestr fabryk (lazy)
        self._factories.update({
            "settings": lambda: _get_settings(),
            "cache": lambda: CacheService(),
            "telemetry": lambda: TelemetryService(),
            "data": lambda: DataService(),
            "eda": lambda: EDAService(),
            "ml": lambda: MLService(),
            "explain": lambda: ExplainService(),
            "export": lambda: ExportService(),
            "insights": lambda: InsightsService(),
            "db": (lambda: DBService()) if DBService is not None else (lambda: None),
        })

    # ---- API ----
    def resolve(self, key: str) -> Any:
        if key in self._singletons:
            return self._singletons[key]
        if key not in self._factories:
            raise KeyError(f"Service '{key}' is not registered.")
        obj = self._factories[key]()
        self._singletons[key] = obj
        return obj

    def resolve_or_none(self, key: str) -> Any | None:
        try:
            return self.resolve(key)
        except Exception:
            return None

    def snapshot(self) -> dict:
        return {
            "registered": sorted(self._factories.keys()),
            "instantiated": sorted(self._singletons.keys()),
        }


# singleton na proces / sesję (Streamlit trzyma to zwykle w session_state)
_CONTAINER: _Container | None = None

def get_container() -> _Container:
    global _CONTAINER
    if _CONTAINER is None:
        _CONTAINER = _Container()
    return _CONTAINER
