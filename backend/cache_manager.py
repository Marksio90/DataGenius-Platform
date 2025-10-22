from __future__ import annotations

from backend.safe_utils import truthy_df_safe

"""
backend/cache_manager.py — WERSJA STABILNA (3-poziomowy cache, bez logów UI)

L1: st.session_state (ultra-szybki, per-session)
L2: pamięć procesu (TTL)
L3: dysk (pickle; TTL po mtime, zapisy atomowe)

Uwaga dot. reloadów Streamlit:
- Jeśli zapisy w L3 trafiają do katalogu aplikacji, watcher plików może wywoływać rerun.
- Ustaw TMIV_CACHE_DIR na ścieżkę poza repo ALBO TMIV_CACHE_TO_TMP=1, aby trafić do katalogu tymczasowego OS.
"""

from typing import Any, Optional, Tuple, Iterable, Dict
from pathlib import Path
from functools import wraps
import hashlib
import pickle
import threading
import json
import time
import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st


class SmartCache:
    """Multi-level cache:
       L1: st.session_state (ultra-fast, per-session)
       L2: memory (process-level; TTL)
       L3: disk (pickle; TTL wg mtime, zapisy atomowe)
    Bez komunikatów UI. Thread-safe dla L2/L3.
    """

    def __init__(
        self,
        cache_dir: str | Path = "cache",
        *,
        enable_disk: bool = True,
        max_disk_bytes: Optional[int] = None,
    ):
        """
        :param cache_dir: katalog dla L3 (pickle). Można nadpisać envem TMIV_CACHE_DIR.
                          Jeśli TMIV_CACHE_TO_TMP=1 i TMIV_CACHE_DIR nie jest ustawiony,
                          cache trafi do katalogu tymczasowego OS.
        :param enable_disk: włącza/wyłącza L3 (dysk).
        :param max_disk_bytes: jeśli ustawione, nie zapisze obiektu na L3, gdy pickle > limit.
        """
        # L1 (session_state) – inicjalizacja bezpieczna
        self._ensure_session_cache_initialized()

        # L2 (memory) – dane i data wygaśnięcia
        self._mem_lock = threading.RLock()
        self.memory_cache: Dict[str, Any] = {}
        self.memory_expire_at: Dict[str, float] = {}

        # L3 (disk)
        self.enable_disk = bool(enable_disk)
        self._disk_lock = threading.RLock()
        self.max_disk_bytes = max_disk_bytes

        cache_dir_env = os.getenv("TMIV_CACHE_DIR")
        if cache_dir_env:
            base_dir = Path(cache_dir_env)
        else:
            to_tmp = os.getenv("TMIV_CACHE_TO_TMP") == "1"
            base_dir = Path(tempfile.gettempdir()) / "tmiv_cache" if to_tmp else Path(cache_dir)

        self.cache_dir = Path(base_dir)
        if self.enable_disk:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                # Jeżeli nie możemy utworzyć katalogu na dysku – wyłącz L3
                self.enable_disk = False

    # ---------------------------------------------------------------------
    # L1 helpers
    # ---------------------------------------------------------------------
    def _ensure_session_cache_initialized(self):
        """Bezpieczna inicjalizacja cache w session_state."""
        try:
            if hasattr(st, "session_state"):
                st.session_state.setdefault("smart_cache_l1", {})
            else:
                self._fallback_l1_cache = {}
        except Exception:
            self._fallback_l1_cache = {}

    def _get_l1_cache(self) -> dict:
        try:
            if hasattr(st, "session_state") and "smart_cache_l1" in st.session_state:
                return st.session_state.smart_cache_l1  # type: ignore[attr-defined]
            return getattr(self, "_fallback_l1_cache", {})
        except Exception:
            return getattr(self, "_fallback_l1_cache", {})

    def _set_l1_cache(self, key: str, value: Any):
        try:
            if hasattr(st, "session_state"):
                st.session_state.setdefault("smart_cache_l1", {})
                st.session_state.smart_cache_l1[key] = value  # type: ignore[attr-defined]
            else:
                setattr(self, "_fallback_l1_cache", getattr(self, "_fallback_l1_cache", {}))
                self._fallback_l1_cache[key] = value
        except Exception:
            setattr(self, "_fallback_l1_cache", getattr(self, "_fallback_l1_cache", {}))
            self._fallback_l1_cache[key] = value

    # ---------------------------------------------------------------------
    # Keying (stabilny fingerprint argumentów)
    # ---------------------------------------------------------------------
    def _fp_df(self, df: pd.DataFrame) -> str:
        """Stabilny fingerprint DataFrame — szybki i odporny na drobne różnice."""
        # kolumny + dtypes + 50 wierszy (split JSON) – lekko i stabilnie
        try:
            cols = list(map(str, df.columns))
            dtypes = df.dtypes.astype(str).to_dict()
            sample = df.head(50).to_json(orient="split", index=False)
            payload = json.dumps({"c": cols, "t": dtypes, "s": sample}, sort_keys=True, ensure_ascii=False)
            return hashlib.md5(payload.encode("utf-8")).hexdigest()
        except Exception:
            return f"df:{hash(df.shape)}:{hash(tuple(map(str, getattr(df, 'columns', ()))))}"

    def _fp_np(self, a: np.ndarray) -> str:
        try:
            shape = a.shape
            dtype = str(a.dtype)
            b = memoryview(a).tobytes()
            if len(b) > 100_000:
                b = b[:100_000]
            h = hashlib.md5(b).hexdigest()
            return f"np:{shape}:{dtype}:{h}"
        except Exception:
            try:
                return f"np:{getattr(a, 'shape', None)}:{str(getattr(a, 'dtype', None))}:{hash(getattr(a, 'data', None))}"
            except Exception:
                return f"np:unknown:{hash(repr(a))}"

    def _arg_fingerprint(self, x: Any) -> str:
        """Deterministyczny skrót dla typowych obiektów, żeby stable-key działał."""
        try:
            if isinstance(x, pd.DataFrame):
                return f"df:{self._fp_df(x)}"
            if isinstance(x, pd.Series):
                return f"ser:{self._fp_df(x.to_frame())}"
            if isinstance(x, pd.Index):
                return f"idx:{hash(tuple(map(str, x.tolist())))}"
            if isinstance(x, np.ndarray):
                return self._fp_np(x)
            # NumPy scalary / dtypes
            if isinstance(x, (np.generic,)):
                return f"npval:{str(x.dtype)}:{repr(x.item())}"
            # datetime / date
            if hasattr(x, "isoformat"):
                try:
                    return f"dt:{x.isoformat()}"
                except Exception:
                    pass
            if isinstance(x, (list, tuple)):
                return f"seq:[{','.join(self._arg_fingerprint(i) for i in x)}]"
            if isinstance(x, dict):
                items = ",".join(
                    f"{k}:{self._arg_fingerprint(v)}" for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))
                )
                return f"dict:{{{items}}}"
            # prymitywy / inne
            return f"val:{repr(x)}"
        except Exception:
            return f"val:{repr(x)}"

    def get_cache_key(self, *args, **kwargs) -> str:
        """Generuj krótki, stabilny klucz cache oparty o fingerprint argumentów."""
        try:
            args_fp = "|".join(self._arg_fingerprint(a) for a in args)
            kw_fp = "|".join(f"{k}={self._arg_fingerprint(v)}" for k, v in sorted(kwargs.items(), key=lambda kv: str(kv[0])))
            key_string = f"{args_fp}||{kw_fp}"
            return hashlib.sha256(key_string.encode("utf-8")).hexdigest()[:24]
        except Exception:
            fallback = repr((args, sorted(kwargs.items(), key=lambda kv: str(kv[0]))))
            return hashlib.sha256(fallback.encode("utf-8")).hexdigest()[:24]

    # ---------------------------------------------------------------------
    # Core API
    # ---------------------------------------------------------------------
    def get(self, cache_key: str, ttl: int = 3600) -> Optional[Any]:
        """Pobierz z cache (L1 -> L2 -> L3). TTL w sekundach."""
        now = time.time()

        # L1
        l1 = self._get_l1_cache()
        if cache_key in l1:
            return l1[cache_key]

        # L2
        with self._mem_lock:
            exp = self.memory_expire_at.get(cache_key)
            if exp is not None and now < exp and cache_key in self.memory_cache:
                val = self.memory_cache[cache_key]
                # promuj do L1
                self._set_l1_cache(cache_key, val)
                return val
            # wygasł?
            if exp is not None and now >= exp:
                self.memory_cache.pop(cache_key, None)
                self.memory_expire_at.pop(cache_key, None)

        # L3
        if self.enable_disk:
            pkl = self.cache_dir / f"{cache_key}.pkl"
            if pkl.exists():
                try:
                    if now - pkl.stat().st_mtime < ttl:
                        with pkl.open("rb") as f:
                            val = pickle.load(f)
                        # promuj do L1 + L2
                        self._set_l1_cache(cache_key, val)
                        with self._mem_lock:
                            self.memory_cache[cache_key] = val
                            self.memory_expire_at[cache_key] = now + ttl
                        return val
                    else:
                        # plik wygasł
                        pkl.unlink(missing_ok=True)
                except Exception:
                    # plik uszkodzony / częściowy zapis – usuń
                    try:
                        pkl.unlink(missing_ok=True)
                    except Exception:
                        pass

        return None

    def set(self, cache_key: str, value: Any, ttl: int = 3600):
        """Zapisz na wszystkich poziomach."""
        self._set_all_levels(cache_key, value, ttl)

    def _set_all_levels(self, cache_key: str, value: Any, ttl: int):
        now = time.time()

        # L1
        self._set_l1_cache(cache_key, value)

        # L2
        with self._mem_lock:
            self.memory_cache[cache_key] = value
            self.memory_expire_at[cache_key] = now + ttl

        # L3 (atomowy zapis, opcjonalny limit rozmiaru)
        if not self.enable_disk:
            return
        try:
            pkl = self.cache_dir / f"{cache_key}.pkl"
            tmp = pkl.with_suffix(".pkl.tmp")

            # serializacja do tmp (aby sprawdzić rozmiar i uniknąć częściowych plików)
            with tmp.open("wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)

            if self.max_disk_bytes is not None and tmp.stat().st_size > self.max_disk_bytes:
                # za duży plik – nie cache’ujemy na dysku
                tmp.unlink(missing_ok=True)
                return

            # rename atomowy
            with self._disk_lock:
                tmp.replace(pkl)
        except Exception:
            # cicho ignoruj błędy dysku
            try:
                tmp.unlink(missing_ok=True)  # type: ignore[name-defined]
            except Exception:
                pass

    # ---------------------------------------------------------------------
    # Decorator (bez komunikatów UI)
    # ---------------------------------------------------------------------
    def cache_decorator(
        self,
        ttl: int = 3600,
        show_spinner: bool = True,
        spinner_text: Optional[str] = None,
    ):
        """Decorator dla funkcji. Zero logów do UI. Spinner opcjonalny."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_key = f"{func.__name__}_{self.get_cache_key(*args, **kwargs)}"

                try:
                    cached = self.get(func_key, ttl)
                except Exception:
                    cached = None
                if cached is not None:
                    return cached

                # Cache miss – wykonaj i zapisz
                if truthy_df_safe(show_spinner) and hasattr(st, "spinner"):
                    text = spinner_text or f"🔄 Obliczam {func.__name__}..."
                    with st.spinner(text):
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                try:
                    self.set(func_key, result, ttl)
                except Exception:
                    pass
                return result

            return wrapper

        return decorator

    # ---------------------------------------------------------------------
    # Maintenance helpers
    # ---------------------------------------------------------------------
    def clear(self):
        """Wyczyść wszystkie poziomy cache."""
        # L1
        try:
            if hasattr(st, "session_state") and "smart_cache_l1" in st.session_state:
                st.session_state.smart_cache_l1.clear()  # type: ignore[attr-defined]
        except Exception:
            pass

        # L2
        with self._mem_lock:
            self.memory_cache.clear()
            self.memory_expire_at.clear()

        # L3
        if self.enable_disk:
            try:
                for p in self.cache_dir.glob("*.pkl"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
            except Exception:
                pass

    def clear_prefix(self, prefix: str):
        """Usuń wpisy (L1/L2/L3), których klucz zaczyna się od `prefix`."""
        # L1
        try:
            if hasattr(st, "session_state") and "smart_cache_l1" in st.session_state:
                keys = [k for k in st.session_state.smart_cache_l1.keys() if str(k).startswith(prefix)]  # type: ignore[attr-defined]
                for k in keys:
                    st.session_state.smart_cache_l1.pop(k, None)  # type: ignore[attr-defined]
        except Exception:
            pass

        # L2
        with self._mem_lock:
            keys = [k for k in list(self.memory_cache.keys()) if str(k).startswith(prefix)]
            for k in keys:
                self.memory_cache.pop(k, None)
                self.memory_expire_at.pop(k, None)

        # L3
        if self.enable_disk:
            try:
                for p in self.cache_dir.glob(f"{prefix}*.pkl"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
            except Exception:
                pass

    def cleanup_expired_files(self, ttl: int = 3600, max_to_delete: int = 200):
        """Opcjonalny housekeeping dla dysku (usuń przeterminowane pliki)."""
        if not self.enable_disk:
            return
        now = time.time()
        deleted = 0
        try:
            for p in self.cache_dir.glob("*.pkl"):
                if now - p.stat().st_mtime >= ttl:
                    try:
                        p.unlink()
                        deleted += 1
                        if deleted >= max_to_delete:
                            break
                    except Exception:
                        pass
        except Exception:
            pass


# Global cache instance (bezpieczna inicjalizacja)
try:
    smart_cache = SmartCache()
except Exception:
    # Fallback – prosty cache w pamięci (bez dysku i TTL)
    class FallbackCache:
        def __init__(self):
            self.cache: Dict[str, Any] = {}

        def get(self, key, ttl: int = 3600):
            return self.cache.get(key)

        def set(self, key, value, ttl: int = 3600):
            self.cache[key] = value

        def cache_decorator(self, ttl: int = 3600, show_spinner: bool = True, spinner_text: Optional[str] = None):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    k = f"{func.__name__}:{hashlib.sha1(repr((args, kwargs)).encode()).hexdigest()[:16]}"
                    if k in self.cache:
                        return self.cache[k]
                    if truthy_df_safe(show_spinner) and hasattr(st, "spinner"):
                        with st.spinner(spinner_text or f"🔄 Obliczam {func.__name__}..."):
                            res = func(*args, **kwargs)
                    else:
                        res = func(*args, **kwargs)
                    self.cache[k] = res
                    return res
                return wrapper
            return decorator

        def clear(self):
            self.cache.clear()

        def clear_prefix(self, prefix: str):
            for k in [k for k in list(self.cache.keys()) if str(k).startswith(prefix)]:
                self.cache.pop(k, None)

        def cleanup_expired_files(self, ttl: int = 3600, max_to_delete: int = 200):
            return

    smart_cache = FallbackCache()
