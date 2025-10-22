# backend/cache_manager.py
"""
Cache Manager for TMIV – disk-backed, fast, and safe.

Features
--------
- Unified API over `diskcache` (if available) with in-memory fallback.
- Helpers for hashing DataFrames/ndarrays and building robust cache keys.
- Simple decorator `@cache_result(...)` for function-level memoization.
- Namespaced cache with TTL support.
- Utility to obtain an artifacts path inside the cache subtree.

Environment
-----------
TMIV_CACHE_DIR   : base directory for cache (default: ./cache)

Usage
-----
    from backend.cache_manager import get_cache, cache_result, df_fingerprint

    cache = get_cache()
    cache.set("hello", {"x": 1}, ttl=60)
    cache.get("hello")

    @cache_result(namespace="eda", ttl=600)
    def heavy_stats(df: pd.DataFrame) -> dict: ...

Notes
-----
- Keys must be JSON-serializable; for DataFrames/arrays we store fingerprints.
- TTL is best-effort (enforced by backend if diskcache available).
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import threading
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd


# =========================
# Utilities
# =========================


def _json_dumps_safe(obj: Any) -> str:
    """Stable JSON for key building; pickle fallback only for tiny objects."""
    try:
        return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        # As last resort, hash a pickle of the object to avoid huge keys
        try:
            b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return f"__pickle__:{sha256(b).hexdigest()}"
        except Exception:
            return f"__repr__:{repr(obj)[:500]}"


def df_fingerprint(df: pd.DataFrame, max_cells: int = 5000) -> str:
    """
    Cheap but stable-ish fingerprint of a DataFrame (schema + sample of values).
    """
    h = sha256()
    cols = tuple(map(str, df.columns))
    h.update(repr(cols).encode("utf-8"))
    # include dtypes
    h.update(repr(tuple(map(str, df.dtypes))).encode("utf-8"))

    if len(df) and len(cols):
        n = min(max_cells // max(1, len(cols)), len(df))
        sample = df.head(n).reset_index(drop=True)
        # use pandas hashing; stable across runs for same values
        hv = pd.util.hash_pandas_object(sample, index=False).values.tobytes()
        h.update(hv)
    return h.hexdigest()


def ndarray_fingerprint(arr: np.ndarray, max_elems: int = 5000) -> str:
    h = sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    if arr.size:
        flat = arr.ravel()[:max_elems]
        h.update(flat.tobytes())
    return h.hexdigest()


def auto_arg_fingerprint(x: Any) -> str:
    """Build a short fingerprint for common heavy types used in keys."""
    try:
        import polars as pl  # optional
    except Exception:
        pl = None  # type: ignore

    if isinstance(x, pd.DataFrame):
        return f"DF:{df_fingerprint(x)}"
    if isinstance(x, pd.Series):
        return f"SER:{sha256(pd.util.hash_pandas_object(x, index=True).values.tobytes()).hexdigest()}"
    if isinstance(x, np.ndarray):
        return f"ND:{ndarray_fingerprint(x)}"
    if pl is not None and isinstance(getattr(pl, "DataFrame", object), type) and isinstance(x, pl.DataFrame):  # type: ignore
        # light-weight pl df hash: schema + head rows
        return f"PL:{sha256(str((list(x.columns), x.head(50).to_dict())).encode()).hexdigest()}"
    # Small primitives and tuples/lists/dicts
    return _json_dumps_safe(x)


def build_key(func: Callable[..., Any], args: tuple, kwargs: dict[str, Any], namespace: str | None) -> str:
    """Construct a robust namespaced key from function and arguments."""
    parts: list[str] = []
    parts.append(namespace or func.__module__)
    parts.append(func.__qualname__)

    # fingerprint args
    a = [auto_arg_fingerprint(v) for v in args]
    k = {kk: auto_arg_fingerprint(vv) for kk, vv in sorted(kwargs.items())}
    parts.append(_json_dumps_safe(a))
    parts.append(_json_dumps_safe(k))
    raw = "||".join(parts)
    return sha256(raw.encode("utf-8")).hexdigest()


# =========================
# Cache backends
# =========================


class _FallbackDictCache:
    """In-memory fallback cache with naive TTL; process-local only."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self._data: dict[str, tuple[float | None, Any]] = {}
        self._lock = threading.Lock()

    # API compatibility with diskcache.Cache
    def set(self, key: str, value: Any, expire: float | None = None) -> None:
        with self._lock:
            expiry = time.time() + float(expire) if expire else None
            self._data[key] = (expiry, value)

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if key not in self._data:
                return default
            expiry, value = self._data[key]
            if expiry is not None and time.time() > expiry:
                del self._data[key]
                return default
            return value

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


@dataclass
class CacheManager:
    """Thin wrapper over diskcache/in-memory backends with convenience helpers."""

    base_dir: Path
    _backend: Any

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        expire = float(ttl) if ttl else None
        # diskcache uses 'expire' seconds; fallback imitates
        self._backend.set(key, value, expire=expire)

    def get(self, key: str, default: Any = None) -> Any:
        return self._backend.get(key, default)

    def delete(self, key: str) -> None:
        self._backend.delete(key)

    def clear(self) -> None:
        self._backend.clear()
        # also clean artifacts dir
        art = self.artifacts_dir()
        if art.exists():
            shutil.rmtree(art, ignore_errors=True)

    # ---------- helpers ----------

    def artifacts_dir(self) -> Path:
        p = self.base_dir / "artifacts"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def path_in_artifacts(self, *parts: str) -> Path:
        p = self.artifacts_dir().joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


_singleton_lock = threading.Lock()
_singleton_cache: CacheManager | None = None


def get_cache() -> CacheManager:
    """Get (or create) process-global CacheManager instance."""
    global _singleton_cache
    with _singleton_lock:
        if _singleton_cache is not None:
            return _singleton_cache

        base = Path(os.getenv("TMIV_CACHE_DIR", "cache")).resolve()
        base.mkdir(parents=True, exist_ok=True)
        backend: Any
        try:
            from diskcache import Cache  # type: ignore

            backend = Cache(str(base))
        except Exception:
            backend = _FallbackDictCache(base)

        _singleton_cache = CacheManager(base_dir=base, _backend=backend)
        return _singleton_cache


# =========================
# Decorators
# =========================


def cache_result(namespace: str | None = None, ttl: float | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to cache function results using argument fingerprints (safe for DataFrames/ndarrays).

    Example
    -------
        @cache_result(namespace="eda", ttl=600)
        def correlations(df: pd.DataFrame) -> pd.DataFrame: ...
    """

    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        def inner(*args: Any, **kwargs: Any) -> Any:
            cache = get_cache()
            key = build_key(func, args, kwargs, namespace)
            hit = cache.get(key, default=None)
            if hit is not None:
                return hit
            res = func(*args, **kwargs)
            try:
                cache.set(key, res, ttl)
            except Exception:
                # As a last resort, try to store a light-weight placeholder (e.g., hash only)
                try:
                    cache.set(key, _json_dumps_safe(res), ttl)
                except Exception:
                    pass
            return res

        inner.__name__ = getattr(func, "__name__", "cached_func")
        inner.__doc__ = func.__doc__
        inner.__qualname__ = getattr(func, "__qualname__", inner.__name__)
        return inner

    return wrap


# =========================
# High-level helpers
# =========================


def cached_path(*subparts: str) -> Path:
    """
    Create (and return) a path under the cache's artifacts directory.
    Use this for temporary exports, plots, small models, etc.
    """
    cache = get_cache()
    return cache.path_in_artifacts(*subparts)


def clear_cache() -> None:
    """Remove all cached entries and artifacts (careful!)."""
    get_cache().clear()


# =========================
# Self-test
# =========================

if __name__ == "__main__":
    import pandas as pd

    cache = get_cache()
    print("Cache dir:", cache.base_dir)

    @cache_result(namespace="demo", ttl=2)
    def heavy_sum(df: pd.DataFrame, x: int) -> int:
        time.sleep(0.2)
        return x + len(df)

    df = pd.DataFrame({"a": range(10)})
    print("First:", heavy_sum(df, 3))
    print("Second (hit):", heavy_sum(df, 3))
    time.sleep(2.1)
    print("Third (expired):", heavy_sum(df, 3))
