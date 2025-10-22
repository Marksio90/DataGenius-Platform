# core/services/cache_service.py
"""
CacheService – cienka fasada nad backend.cache_manager dla TMIV – Advanced ML Platform.

Co dostajesz:
- `cached_path(subdir, name)`  -> Path do pliku w cache/artifacts/<subdir>/<name> (tworzy katalogi).
- `cache_result(namespace, ttl)` -> dekorator cache'ujący wynik funkcji.
- `df_fingerprint(df)` -> stabilny odcisk danych (hash struktury + próbki zawartości).

Uwagi:
- Jeśli `backend.cache_manager` jest dostępny, korzystamy z niego w 100%.
- W przeciwnym razie stosujemy bezpieczne fallbacki:
  * `cached_path` tworzy ścieżkę pod ./cache/artifacts/...
  * `cache_result` staje się lekkim cache'em w pamięci (z TTL).
  * `df_fingerprint` liczy skrót na podstawie kolumn/dtypów i pierwszych 1000 wierszy.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar

T = TypeVar("T")


class CacheService:
    def __init__(self) -> None:
        # Spróbuj wciągnąć w pełni-feature'owy backend
        self._cm = None
        try:  # pragma: no cover
            from backend import cache_manager as cm  # type: ignore

            self._cm = cm
        except Exception:
            self._cm = None

        # Prosty fallback dla cache_result, gdy backend niedostępny
        self._mem_cache: dict[str, Tuple[float, Any]] = {}  # key -> (expires_at, value)

    # --------------------------
    # Public API
    # --------------------------

    def cached_path(self, subdir: str, name: str) -> Path:
        """
        Zwraca absolutną ścieżkę pliku w `cache/artifacts/<subdir>/<name>`.
        Tworzy brakujące katalogi.
        """
        if self._cm is not None and hasattr(self._cm, "cached_path"):
            return self._cm.cached_path(subdir, name)  # type: ignore[attr-defined]

        base = Path("cache").joinpath("artifacts", subdir)
        base.mkdir(parents=True, exist_ok=True)
        return base.joinpath(name).resolve()

    def cache_result(self, namespace: str = "default", ttl: Optional[int] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """
        Dekorator cache’ujący wynik funkcji. Klucz budowany z:
        - namespacu,
        - nazwy funkcji,
        - JSON-owalnych argumentów (args/kwargs).

        Jeśli dostępny jest backendowy `cache_manager.cache_result`, dekorator zostaje delegowany.
        """
        if self._cm is not None and hasattr(self._cm, "cache_result"):
            return self._cm.cache_result(namespace=namespace, ttl=ttl)  # type: ignore[attr-defined]

        def _decor(fn: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(fn)
            def _wrap(*args: Any, **kwargs: Any) -> T:
                key = self._make_key(namespace, fn, args, kwargs)
                now = time.monotonic()
                ent = self._mem_cache.get(key)
                if ent is not None:
                    expires_at, value = ent
                    if ttl is None or now < expires_at:
                        return value  # type: ignore[return-value]
                # compute
                value = fn(*args, **kwargs)
                expires_at = now + float(ttl) if ttl else now + 365 * 24 * 3600.0  # długa żywotność bez TTL
                self._mem_cache[key] = (expires_at, value)
                return value

            return _wrap

        return _decor

    def df_fingerprint(self, df) -> str:
        """
        Stabilny skrót danych (12 hex), deleguje do backendu jeśli dostępny.
        """
        if self._cm is not None and hasattr(self._cm, "df_fingerprint"):
            try:
                return str(self._cm.df_fingerprint(df))  # type: ignore[attr-defined]
            except Exception:
                pass
        # Fallback
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

    # --------------------------
    # Helpers
    # --------------------------

    @staticmethod
    def _jsonable(obj: Any) -> Any:
        try:
            json.dumps(obj)
            return obj
        except Exception:
            if isinstance(obj, dict):
                return {str(k): CacheService._jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple, set)):
                return [CacheService._jsonable(v) for v in obj]
            try:
                return str(obj)
            except Exception:
                return "<unrepr>"

    def _make_key(self, namespace: str, fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
        try:
            payload = {"ns": namespace, "fn": f"{fn.__module__}.{fn.__qualname__}", "args": self._jsonable(args), "kwargs": self._jsonable(kwargs)}
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            return hashlib.sha256(blob).hexdigest()
        except Exception:
            return f"{namespace}:{fn.__name__}:{hash(args) ^ hash(tuple(kwargs.items()))}"


__all__ = ["CacheService"]
