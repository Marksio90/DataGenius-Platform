"""
cache_manager.py
Docstring (PL): Warstwa cache na bazie diskcache (jeśli dostępny) lub prosty dict w pamięci.
"""
from __future__ import annotations
from typing import Any
import os

try:
    import diskcache as dc
    HAS_DC = True
except Exception:
    HAS_DC = False

_CACHE = None

def get_cache(path: str = "artifacts/cache") -> Any:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    os.makedirs(path, exist_ok=True)
    if HAS_DC:
        _CACHE = dc.Cache(path)
    else:
        class _Dummy(dict):
            def close(self): ...
        _CACHE = _Dummy()
    return _CACHE