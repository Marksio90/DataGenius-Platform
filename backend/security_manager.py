from __future__ import annotations

"""
backend/security_manager.py
Bezpieczne zarządzanie kluczami API z fallbackami i error handling
PLUS DataValidator i RateLimiter (na st.session_state).
"""

from backend.safe_utils import truthy_df_safe

import os
import logging
from time import monotonic
from functools import wraps
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable, Tuple

# ── Pandas (walidator danych) ─────────────────────────────────────────────────
import pandas as pd

# ── Streamlit (łagodny fallback, by działało poza Streamlitem) ────────────────
try:
    import streamlit as st  # type: ignore
    if not hasattr(st, "session_state"):
        class _SS(dict):
            ...
        st.session_state = _SS()  # type: ignore
except Exception:  # pragma: no cover
    class _DummySt:
        session_state: Dict[str, Any] = {}
        class _Secrets(dict):
            ...
        secrets = _Secrets()
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def info(self, *a, **k): pass
        def success(self, *a, **k): pass
    st = _DummySt()  # type: ignore

# ── Opcjonalny keyring – łagodny fallback ─────────────────────────────────────
try:
    import keyring  # type: ignore  # noqa: F401
    KEYRING_AVAILABLE = True
except Exception:  # pragma: no cover
    keyring = None  # type: ignore
    KEYRING_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


# ==============================================================================
# Helpers
# ==============================================================================
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _env_int(name: str, default: int) -> int:
    try:
        raw = os.getenv(name, "")
        raw = raw.strip() if isinstance(raw, str) else ""
        return int(raw or default)
    except Exception:
        return default


def _mask(s: Optional[str], keep: int = 4) -> str:
    """Maskuj klucz dla logów/UI."""
    if not s:
        return ""
    s = str(s)
    return ("*" * max(0, len(s) - keep)) + s[-keep:]


def _provider_key_candidates(provider: str) -> Tuple[str, ...]:
    p = str(provider or "").strip().lower()
    return (p, f"{p}_api_key", f"{p}_token", f"{p}_key")


def _provider_env_candidates(provider: str) -> Tuple[str, ...]:
    P = str(provider or "").strip().upper()
    return (f"{P}_API_KEY", f"{P}_TOKEN", f"{P}_KEY")


# ==============================================================================
# Credential Manager
# ==============================================================================
class CredentialManager:
    """Prosty menedżer kluczy API z bezpiecznymi fallbackami."""

    SERVICE_NAME = "TMIV"

    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Priorytet:
        1) st.session_state["<provider>_api_key"] / ["<provider>_token"] oraz st.secrets.get(...)
        2) ENV: <PROVIDER>_API_KEY || <PROVIDER>_TOKEN || <PROVIDER>_KEY
        3) keyring (jeśli dostępny), service="TMIV", username=<provider>
        """
        prov = str(provider or "").strip().lower()
        if not prov:
            return None

        # 1) Session state
        try:
            for k in (f"{prov}_api_key", f"{prov}_token"):
                val = st.session_state.get(k)  # type: ignore[attr-defined]
                if truthy_df_safe(val) and str(val).strip():
                    return str(val).strip()
        except Exception:
            pass

        # 1b) Streamlit secrets – użyj .get() i typowego wachlarza kluczy
        try:
            if hasattr(st, "secrets"):
                for k in _provider_key_candidates(prov):
                    try:
                        v = st.secrets.get(k)  # type: ignore[attr-defined]
                    except Exception:
                        v = None
                    if truthy_df_safe(v) and str(v).strip():
                        return str(v).strip()
        except Exception:
            pass

        # 2) Env – przetestuj popularne warianty
        for env_key in _provider_env_candidates(prov):
            env_val = os.getenv(env_key)
            if truthy_df_safe(env_val) and str(env_val).strip():
                return str(env_val).strip()

        # 3) Keyring
        try:
            if KEYRING_AVAILABLE:
                kr = keyring.get_password(self.SERVICE_NAME, prov)  # type: ignore
                if truthy_df_safe(kr) and kr.strip():
                    return kr.strip()
        except Exception:
            pass

        return None

    def set_api_key(self, provider: str, value: str, *, persist_keyring: bool = False) -> None:
        """Ustaw w session_state i opcjonalnie w systemowym keyring."""
        prov = str(provider or "").strip().lower()
        key_name = f"{prov}_api_key"
        try:
            st.session_state[key_name] = str(value or "").strip()
        except Exception:
            pass
        if truthy_df_safe(persist_keyring) and KEYRING_AVAILABLE:
            try:
                keyring.set_password(self.SERVICE_NAME, prov, str(value or "").strip())  # type: ignore
            except Exception:
                pass
        logger.info("API key set for %s: %s", prov, _mask(value))

    def clear_api_key(self, provider: str, *, remove_keyring: bool = False) -> None:
        """Wyczyść klucz z session_state i opcjonalnie z keyring."""
        prov = str(provider or "").strip().lower()
        key_name = f"{prov}_api_key"
        try:
            if key_name in st.session_state:
                del st.session_state[key_name]
        except Exception:
            pass
        if truthy_df_safe(remove_keyring) and KEYRING_AVAILABLE:
            try:
                if hasattr(keyring, "delete_password"):  # type: ignore
                    keyring.delete_password(self.SERVICE_NAME, prov)  # type: ignore
                else:
                    keyring.set_password(self.SERVICE_NAME, prov, "")  # type: ignore
            except Exception:
                pass
        logger.info("API key cleared for %s", prov)

    def get_masked_api_key(self, provider: str) -> str:
        """Zwraca zamaskowaną reprezentację klucza (do UI/logów)."""
        key = self.get_api_key(provider)
        return _mask(key)

    # ── Dekorator opcjonalny: wymaga klucza przed wywołaniem funkcji ───────────
    def require_api_key(self, provider: str) -> Callable:
        """
        @credential_manager.require_api_key("openai")
        def call_openai(...): ...
        """
        def deco(fn: Callable):
            @wraps(fn)
            def wrap(*a, **k):
                key = self.get_api_key(provider)
                if not truthy_df_safe(key):
                    msg = f"🔒 Brak klucza dla '{provider}'. Skonfiguruj w sekcji bezpieczeństwa."
                    try:
                        st.error(msg)
                    except Exception:
                        pass
                    raise RuntimeError(msg)
                return fn(*a, **k)
            return wrap
        return deco


# ==============================================================================
# Data Validator
# ==============================================================================
class DataValidator:
    """
    Prosta walidacja ramki danych używana przed przetwarzaniem.
    ZWRACA tę samą ramkę (dla zgodności z resztą kodu).
    """

    def validate_dataframe(self, df: Any) -> pd.DataFrame:
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Brak danych lub niepoprawny obiekt danych (oczekiwano pandas.DataFrame).")
        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError("Dane są puste.")
        return df


# ==============================================================================
# Rate Limiter
# ==============================================================================
class RateLimiter:
    """Lekki limiter oparty o st.session_state (per sesja), z bezpiecznymi aliasami."""

    DEFAULTS: Dict[str, Dict[str, int]] = {
        "file_uploads": {"max": 1000, "window": 60},
        "training": {"max": 100, "window": 300},
        "ai_calls": {"max": 500, "window": 60},
        "eda": {"max": 200, "window": 60},
        "export": {"max": 200, "window": 60},
    }

    # ── Internal helpers ───────────────────────────────────────────────────────
    def _ensure(self) -> None:
        ss = st.session_state  # type: ignore[attr-defined]
        ss.setdefault("rate_limiter_cache", {})  # channel -> list[float monotonic seconds]
        cfg = ss.setdefault("rate_limit_config", self.DEFAULTS.copy())

        # Wczytaj opcjonalny config z ENV (np. TMIV_RATE_LIMIT_TRAINING_MAX=50)
        for ch in list(self.DEFAULTS.keys()):
            kmax = f"TMIV_RATE_LIMIT_{ch.upper()}_MAX"
            kwin = f"TMIV_RATE_LIMIT_{ch.upper()}_WINDOW"
            mx = _env_int(kmax, cfg[ch]["max"])
            wn = _env_int(kwin, cfg[ch]["window"])
            cfg[ch] = {"max": int(mx), "window": int(wn)}
        ss["rate_limit_config"] = cfg  # type: ignore[attr-defined]

        # Domyślnie limiter WYŁĄCZONY (włącz w prod przez disable_rate_limit=False lub env)
        ss.setdefault("disable_rate_limit", _env_bool("TMIV_RATE_LIMIT_DISABLE", True))

    def get_config(self, channel: str) -> Dict[str, int]:
        self._ensure()
        cfg = st.session_state["rate_limit_config"]  # type: ignore[attr-defined]
        ch = str(channel or "").strip().lower()
        if ch in cfg:
            return cfg[ch]
        # dla nieznanych kanałów — łagodny default
        return {"max": 10**9, "window": 1}

    def _prune_and_count(self, channel: str, window: int) -> int:
        cache = st.session_state["rate_limiter_cache"]  # type: ignore[attr-defined]
        now = monotonic()
        arr = [t for t in cache.get(channel, []) if now - t <= window]
        cache[channel] = arr
        st.session_state["rate_limiter_cache"] = cache  # type: ignore[attr-defined]
        return len(arr)

    # ── Public API ────────────────────────────────────────────────────────────
    def get_remaining(self, channel: str) -> int:
        cfg = self.get_config(channel)
        used = self._prune_and_count(channel, cfg["window"])
        return max(0, int(cfg["max"]) - int(used))

    def allow(self, channel: str) -> bool:
        self._ensure()
        if st.session_state.get("disable_rate_limit", False):  # type: ignore[attr-defined]
            return True
        cfg = self.get_config(channel)
        used = self._prune_and_count(channel, cfg["window"])
        if used < cfg["max"]:
            st.session_state["rate_limiter_cache"].setdefault(channel, []).append(monotonic())  # type: ignore[attr-defined]
            return True
        return False

    def guard(self, channel: str) -> None:
        """Rzuć błąd gdy przekroczono limit."""
        if not self.allow(channel):
            cfg = self.get_config(channel)
            rem = self.get_remaining(channel)
            raise RuntimeError(
                f"⚠️ Osiągnięto limit operacji. Pozostało: {rem} w tym oknie czasowym ({cfg['window']}s)."
            )

    def limit(self, channel: str) -> Callable:
        """Dekorator ograniczający wywołania funkcji."""
        def deco(fn: Callable):
            @wraps(fn)
            def wrap(*a, **k):
                self.guard(channel)
                return fn(*a, **k)
            return wrap
        return deco

    @contextmanager
    def context(self, channel: str):
        """
        Użycie:
            with rate_limiter.context("training"):
                do_training()
        """
        self.guard(channel)
        try:
            yield
        finally:
            # nic – pojedyncze wejście liczymy już w guard/allow
            pass

    def set_config(self, channel: str, *, max_calls: Optional[int] = None, window: Optional[int] = None) -> None:
        """Ustaw konfigurację dla kanału w locie."""
        self._ensure()
        ch = str(channel or "").strip().lower()
        cfg = st.session_state["rate_limit_config"]  # type: ignore[attr-defined]
        base = cfg.get(ch, {"max": 10**9, "window": 1})
        if max_calls is not None:
            base["max"] = int(max_calls)
        if window is not None:
            base["window"] = int(window)
        cfg[ch] = base
        st.session_state["rate_limit_config"] = cfg  # type: ignore[attr-defined]

    # ── Aliasy dla zgodności wstecznej ────────────────────────────────────────
    def enforce(self, channel: str) -> bool:
        """
        Alias kompatybilny z wcześniejszym API: podnosi błąd przy przekroczeniu limitu,
        a w przeciwnym razie zwraca True.
        """
        self.guard(channel)
        return True

    def is_allowed(self, channel: str, *args, **kwargs) -> bool:
        return self.allow(channel)

    def allowed(self, channel: str, *args, **kwargs) -> bool:
        return self.allow(channel)

    def get_remaining_requests(self, channel: str, *args, **kwargs) -> int:
        return self.get_remaining(channel)


# ==============================================================================
# Singletons
# ==============================================================================
credential_manager = CredentialManager()
data_validator = DataValidator()
rate_limiter = RateLimiter()

__all__ = [
    "credential_manager",
    "data_validator",
    "rate_limiter",
    "CredentialManager",
    "DataValidator",
    "RateLimiter",
]
