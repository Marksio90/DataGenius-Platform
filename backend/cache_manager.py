# backend/cache_manager.py - POPRAWIONA WERSJA
import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Optional
import streamlit as st
import pandas as pd
import json
from functools import wraps

class SmartCache:
    """Multi-level cache system"""
    
    def __init__(self, cache_dir: str = "cache"):
        # POPRAWKA: Bezpieczna inicjalizacja L1 cache
        self._ensure_session_cache_initialized()
        
        # L2: Memory cache
        self.memory_cache = {}
        self.memory_ttl = {}
        
        # L3: Disk cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _ensure_session_cache_initialized(self):
        """Bezpieczna inicjalizacja cache w session_state"""
        try:
            # Sprawdź czy st.session_state jest dostępny
            if hasattr(st, 'session_state'):
                if 'smart_cache_l1' not in st.session_state:
                    st.session_state.smart_cache_l1 = {}
            else:
                # Fallback gdy session_state nie jest dostępny
                self._fallback_l1_cache = {}
        except Exception as e:
            # Fallback na wypadek problemów z session_state
            self._fallback_l1_cache = {}
    
    def _get_l1_cache(self) -> dict:
        """Pobierz L1 cache z bezpiecznym fallbackiem"""
        try:
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'smart_cache_l1'):
                return st.session_state.smart_cache_l1
            else:
                return getattr(self, '_fallback_l1_cache', {})
        except Exception:
            return getattr(self, '_fallback_l1_cache', {})
    
    def _set_l1_cache(self, key: str, value: Any):
        """Ustaw wartość w L1 cache z bezpiecznym fallbackiem"""
        try:
            if hasattr(st, 'session_state'):
                if 'smart_cache_l1' not in st.session_state:
                    st.session_state.smart_cache_l1 = {}
                st.session_state.smart_cache_l1[key] = value
            else:
                if not hasattr(self, '_fallback_l1_cache'):
                    self._fallback_l1_cache = {}
                self._fallback_l1_cache[key] = value
        except Exception:
            if not hasattr(self, '_fallback_l1_cache'):
                self._fallback_l1_cache = {}
            self._fallback_l1_cache[key] = value

    def get_cache_key(self, *args, **kwargs) -> str:
        """Generuj unikalny klucz cache"""
        # Konwertuj pandas DataFrame na hash
        processed_args = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                # Hash DataFrame content
                processed_args.append(hashlib.md5(arg.to_string().encode()).hexdigest())
            else:
                processed_args.append(str(arg))
        
        key_string = f"{processed_args}_{sorted(kwargs.items())}"
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def get(self, cache_key: str, ttl: int = 3600) -> Optional[Any]:
        """Pobierz z cache (L1 -> L2 -> L3)"""
        current_time = time.time()
        
        # L1: Streamlit session (z bezpiecznym dostępem)
        l1_cache = self._get_l1_cache()
        if cache_key in l1_cache:
            return l1_cache[cache_key]
        
        # L2: Memory
        if cache_key in self.memory_cache:
            if current_time - self.memory_ttl.get(cache_key, 0) < ttl:
                # Propaguj do L1
                self._set_l1_cache(cache_key, self.memory_cache[cache_key])
                return self.memory_cache[cache_key]
            else:
                # Wygasł
                del self.memory_cache[cache_key]
                del self.memory_ttl[cache_key]
        
        # L3: Disk
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                # Sprawdź czas modyfikacji
                file_time = cache_file.stat().st_mtime
                if current_time - file_time < ttl:
                    with open(cache_file, 'rb') as f:
                        value = pickle.load(f)
                    
                    # Propaguj do wyższych poziomów
                    self._set_all_levels(cache_key, value)
                    return value
                else:
                    # Plik wygasł
                    cache_file.unlink()
            except Exception:
                pass
        
        return None
    
    def set(self, cache_key: str, value: Any, ttl: int = 3600):
        """Zapisz we wszystkich poziomach"""
        self._set_all_levels(cache_key, value)
        
    def _set_all_levels(self, cache_key: str, value: Any):
        """Zapisz we wszystkich poziomach cache"""
        # L1: Session (bezpiecznie)
        self._set_l1_cache(cache_key, value)
        
        # L2: Memory
        self.memory_cache[cache_key] = value
        self.memory_ttl[cache_key] = time.time()
        
        # L3: Disk
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(value, f)
        except Exception as e:
            # Nie wywalaj całej aplikacji jeśli zapis na dysk nie działa
            try:
                if hasattr(st, 'warning'):
                    st.warning(f"Nie udało się zapisać cache na dysku: {e}")
            except:
                pass
    
    def cache_decorator(self, ttl: int = 3600):
        """Decorator dla funkcji"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generuj klucz z nazwą funkcji
                func_key = f"{func.__name__}_{self.get_cache_key(*args, **kwargs)}"
                
                # Sprawdź cache
                cached_result = self.get(func_key, ttl)
                if cached_result is not None:
                    try:
                        if hasattr(st, 'info'):
                            st.info(f"🚀 Używam cache dla {func.__name__}")
                    except:
                        pass
                    return cached_result
                
                # Cache miss - wykonaj funkcję
                try:
                    if hasattr(st, 'spinner'):
                        with st.spinner(f"🔄 Obliczam {func.__name__}..."):
                            result = func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                except:
                    result = func(*args, **kwargs)
                
                # Zapisz w cache
                self.set(func_key, result, ttl)
                
                try:
                    if hasattr(st, 'success'):
                        st.success(f"💾 Zapisano w cache: {func.__name__}")
                except:
                    pass
                
                return result
            return wrapper
        return decorator

# Global cache instance - BEZPIECZNA INICJALIZACJA
try:
    smart_cache = SmartCache()
except Exception as e:
    # Fallback - prosty cache w pamięci
    class FallbackCache:
        def __init__(self):
            self.cache = {}
        
        def get(self, key, ttl=3600):
            return self.cache.get(key)
        
        def set(self, key, value, ttl=3600):
            self.cache[key] = value
        
        def cache_decorator(self, ttl=3600):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return wrapper
            return decorator
    
    smart_cache = FallbackCache()