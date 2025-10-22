# core/container.py
"""
Lightweight DI container for TMIV – Advanced ML Platform.

Why this exists
---------------
- Centralized place to construct and share services across the app
  (settings, telemetry, security, cache, data/ml/eda services, ...).
- Works both in plain Python and inside Streamlit sessions.
- Lazy, import-on-first-use factories (no heavy imports at module load).
- Safe fallbacks when optional modules are not installed yet.

Usage
-----
    from core.container import get_container

    c = get_container()
    settings = c.resolve("settings")         # config.settings.Settings (singleton)
    telemetry = c.resolve("telemetry")       # backend.telemetry.Telemetry (singleton)
    security = c.resolve_or_none("security") # backend.security_manager.SecurityManager
    ml = c.resolve_or_none("ml")             # core.services.ml_service (if present)

You can also register custom instances/factories:

    c.register_instance("my_service", obj)
    c.register_factory("db", lambda: DBService(...))

Notes
-----
- Inside Streamlit, the container is stored under `st.session_state["_tmiv_container"]`
  to avoid cross-session state leaks.
- `resolve(...)` is thread-safe (per-process lock).
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# ---------------------------------
# DI container implementation
# ---------------------------------


Factory = Callable[[], Any]


@dataclass
class _Registry:
    instances: Dict[str, Any] = field(default_factory=dict)
    factories: Dict[str, Factory] = field(default_factory=dict)
    locks: Dict[str, threading.Lock] = field(default_factory=dict)

    def lock_for(self, name: str) -> threading.Lock:
        if name not in self.locks:
            self.locks[name] = threading.Lock()
        return self.locks[name]


class DIContainer:
    """
    Minimalistic DI container with:
    - lazy factories,
    - singletons per key,
    - optional dependencies tolerated.
    """

    def __init__(self) -> None:
        self._r = _Registry()
        self._bootstrap_defaults()

    # ---- Registration API ----
    def register_factory(self, name: str, factory: Factory) -> None:
        self._r.factories[name] = factory

    def register_instance(self, name: str, instance: Any) -> None:
        self._r.instances[name] = instance

    # ---- Resolution API ----
    def resolve(self, name: str) -> Any:
        """
        Return an instance for a registered `name`, constructing it once via factory if needed.
        Raises KeyError if no factory/instance is known.
        """
        # fast path
        if name in self._r.instances:
            return self._r.instances[name]

        # factory path
        if name not in self._r.factories:
            raise KeyError(f"Service '{name}' is not registered.")

        lock = self._r.lock_for(name)
        with lock:
            # double-checked
            if name in self._r.instances:
                return self._r.instances[name]
            inst = self._r.factories[name]()
            self._r.instances[name] = inst
            return inst

    def resolve_or_none(self, name: str) -> Any | None:
        """
        Same as resolve(), but returns None if missing or factory fails to import.
        """
        try:
            return self.resolve(name)
        except Exception:
            return None

    def has(self, name: str) -> bool:
        return name in self._r.instances or name in self._r.factories

    def snapshot(self) -> dict:
        """
        Safe snapshot for UI/debug (no secrets).
        """
        return {
            "registered": sorted(set(self._r.factories.keys()) | set(self._r.instances.keys())),
            "initialized": sorted(self._r.instances.keys()),
        }

    def reset(self) -> None:
        """
        Clear all created instances (factories remain).
        """
        self._r.instances.clear()

    # ---- Defaults ----
    def _bootstrap_defaults(self) -> None:
        """
        Register the default core services. All imports are lazy inside factories.
        """
        # Settings
        def _settings_factory():
            from config.settings import get_settings

            return get_settings()

        # Telemetry
        def _telemetry_factory():
            from backend.telemetry import get_telemetry

            t = get_telemetry()
            # The app should have called init() once; we don't force-init here.
            return t

        # Security
        def _security_factory():
            from backend.security_manager import get_security

            return get_security()

        # Cache facade (wraps backend.cache_manager, but exposes a small, stable API)
        def _cache_factory():
            try:
                from backend import cache_manager as cm  # type: ignore

                class CacheFacade:
                    cached_path = staticmethod(cm.cached_path)
                    cache_result = staticmethod(cm.cache_result)
                    df_fingerprint = staticmethod(cm.df_fingerprint)

                return CacheFacade()
            except Exception:
                # Minimal fallback if cache_manager is absent
                import os
                from pathlib import Path
                import hashlib
                import json

                class _FallbackCache:
                    @staticmethod
                    def cached_path(subdir: str, name: str) -> "Path":
                        p = Path("cache").joinpath("artifacts", subdir)
                        p.mkdir(parents=True, exist_ok=True)
                        return p.joinpath(name).resolve()

                    @staticmethod
                    def cache_result(namespace: str = "default", ttl: int | None = None):
                        def _decor(fn):
                            return fn  # no-op
                        return _decor

                    @staticmethod
                    def df_fingerprint(df) -> str:
                        try:
                            sample = df.head(1000)
                            payload = {
                                "cols": list(map(str, sample.columns)),
                                "dtypes": [str(t) for t in sample.dtypes],
                                "shape": list(df.shape),
                                "preview": sample.astype(str).to_dict(orient="list"),
                            }
                            h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
                            return h[:12].upper()
                        except Exception:
                            return "UNKNOWN"

                return _FallbackCache()

        # Optional domain services (import on demand; tolerate absence)
        def _optional_service(import_path: str, attr: str | None = None) -> Factory:
            def _factory():
                try:
                    mod = importlib.import_module(import_path)
                    obj = getattr(mod, attr) if attr else mod
                    # If it's a class with zero-arg ctor, instantiate; else return module/object.
                    try:
                        return obj()  # type: ignore[call-arg]
                    except TypeError:
                        return obj
                except Exception as e:
                    raise RuntimeError(f"Optional service '{import_path}' is not available: {e}") from e

            return _factory

        # Register core, always-present factories
        self.register_factory("settings", _settings_factory)
        self.register_factory("telemetry", _telemetry_factory)
        self.register_factory("security", _security_factory)
        self.register_factory("cache", _cache_factory)

        # Register optional service factories (may not exist yet in the repo)
        self.register_factory("data", _optional_service("core.services.data_service", "DataService"))
        self.register_factory("eda", _optional_service("core.services.eda_service", "EDAService"))
        self.register_factory("ml", _optional_service("core.services.ml_service", "MLService"))
        self.register_factory("explain", _optional_service("core.services.explain_service", "ExplainService"))
        self.register_factory("export", _optional_service("core.services.export_service", "ExportService"))
        self.register_factory("insights", _optional_service("core.services.insights_service", "InsightsService"))
        self.register_factory("db", _optional_service("core.services.db_service", "DBService"))


# ---------------------------------
# Streamlit-aware singleton access
# ---------------------------------

_singleton: Optional[DIContainer] = None
_singleton_lock = threading.Lock()


def _get_streamlit_container() -> Optional[DIContainer]:
    """
    If running under Streamlit, keep the container in session_state to avoid cross-user leaks.
    """
    try:
        import streamlit as st  # type: ignore

        key = "_tmiv_container"
        if key not in st.session_state:
            st.session_state[key] = DIContainer()
        return st.session_state[key]
    except Exception:
        return None


def get_container() -> DIContainer:
    """
    Return the process/session singleton container.
    """
    # Prefer Streamlit session container if available
    c = _get_streamlit_container()
    if c is not None:
        return c

    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = DIContainer()
        return _singleton


def reset_container() -> None:
    """
    Clear the singleton (useful for tests). Streamlit session container is reset per session.
    """
    c = _get_streamlit_container()
    if c is not None:
        # Replace with a new one
        try:
            import streamlit as st  # type: ignore

            st.session_state["_tmiv_container"] = DIContainer()
            return
        except Exception:
            pass

    global _singleton
    with _singleton_lock:
        _singleton = DIContainer()


__all__ = [
    "DIContainer",
    "get_container",
    "reset_container",
]
