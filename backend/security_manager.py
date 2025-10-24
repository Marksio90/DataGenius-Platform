from __future__ import annotations
"""
backend/security_manager.py
Bezpieczne zarządzanie kluczami API z fallbackami i error handling
PLUS DataValidator i RateLimiter.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, Callable

import streamlit as st
import pandas as pd

# Opcjonalny keyring – łagodny fallback
try:
    import keyring  # noqa: F401
    KEYRING_AVAILABLE = True
except Exception:
    keyring = None  # type: ignore
    KEYRING_AVAILABLE = False

logger = logging.getLogger(__name__)


class CredentialManager:
    """Prosty menedżer kluczy API z bezpiecznymi fallbackami."""
    def get_api_key(self, provider: str) -> Optional[str]:
        key_name = f"{provider.lower()}_api_key"
        try:
            key = st.session_state.get(key_name)
            if key and str(key).strip():
                return str(key).strip()
        except Exception:
            pass
        env_key = f"{provider.upper()}_API_KEY"
        env_val = os.getenv(env_key)
        if env_val and str(env_val).strip():
            return str(env_val).strip()
        try:
            if KEYRING_AVAILABLE:
                service = "TMIV"
                kr = keyring.get_password(service, provider)  # type: ignore
                if kr and kr.strip():
                    return kr.strip()
        except Exception:
            pass
        return None


class DataValidator:
    """Prosta walidacja ramki danych używana przed przetwarzaniem."""
    def validate_dataframe(self, df) -> None:
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Brak danych lub niepoprawny obiekt danych.")
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError("Dane są puste.")


class RateLimiter:
    """Lekki limiter oparty o st.session_state (per sesja)."""
    DEFAULTS: Dict[str, Dict[str, int]] = {
        "file_uploads": {"max": 1000, "window": 60},
        "training": {"max": 100, "window": 300},
        "ai_calls": {"max": 500, "window": 60},
        "eda": {"max": 200, "window": 60},
        "export": {"max": 200, "window": 60},
    }

    def _ensure(self) -> None:
        ss = st.session_state
        ss.setdefault("rate_limiter_cache", {})
        ss.setdefault("rate_limit_config", self.DEFAULTS.copy())
        ss.setdefault("disable_rate_limit", True)  # OFF by default

    def get_config(self, channel: str) -> Dict[str, int]:
        self._ensure()
        cfg = st.session_state["rate_limit_config"]
        return cfg.get(channel, {"max": 10**9, "window": 1})

    def _prune_and_count(self, channel: str, window: int) -> int:
        ss = st.session_state
        cache = ss["rate_limiter_cache"]
        now = time.time()
        arr = [t for t in cache.get(channel, []) if now - t <= window]
        cache[channel] = arr
        ss["rate_limiter_cache"] = cache
        return len(arr)

    def get_remaining(self, channel: str) -> int:
        cfg = self.get_config(channel)
        used = self._prune_and_count(channel, cfg["window"])
        return max(0, cfg["max"] - used)

    def allow(self, channel: str) -> bool:
        self._ensure()
        if st.session_state.get("disable_rate_limit", False):
            return True
        cfg = self.get_config(channel)
        used = self._prune_and_count(channel, cfg["window"])
        if used < cfg["max"]:
            st.session_state["rate_limiter_cache"].setdefault(channel, []).append(time.time())
            return True
        return False

    def guard(self, channel: str) -> None:
        if not self.allow(channel):
            cfg = self.get_config(channel)
            rem = self.get_remaining(channel)
            raise RuntimeError(f"⚠️ Osiągnięto limit operacji. Pozostało: {rem} w tym oknie czasowym ({cfg['window']}s).")  # noqa: E501

    def limit(self, channel: str) -> Callable:
        def deco(fn: Callable):
            def wrap(*a, **k):
                self.guard(channel)
                return fn(*a, **k)
            return wrap
        return deco

    # Zgodność wsteczna
    def is_allowed(self, channel: str, *args, **kwargs) -> bool:
        return self.allow(channel)

    def allowed(self, channel: str, *args, **kwargs) -> bool:
        return self.allow(channel)

    def get_remaining_requests(self, channel: str, *args, **kwargs) -> int:
        return self.get_remaining(channel)


credential_manager = CredentialManager()
data_validator = DataValidator()
rate_limiter = RateLimiter()
