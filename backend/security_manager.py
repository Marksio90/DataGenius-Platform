# backend/security_manager.py
"""
Security & Secrets manager for TMIV – Advanced ML Platform.

Goals
-----
- Single place to **get/set/validate** API keys for external providers (OpenAI, Anthropic, HF, etc.).
- Safe-by-default:
  * Redaction helpers for UI/logs,
  * Optional in-memory encryption (Fernet) if `cryptography` is available,
  * ZERO hard dependency on Streamlit/keyring (lazy/optional).
- Multiple sources (precedence):
  1) Explicitly set in this process (via `set_key`),
  2) `.streamlit/secrets.toml` (if running under Streamlit),
  3) Environment variables,
  4) OS keyring (if available).
- Utilities to build auth headers and gently verify key formats.

Public API (selected)
---------------------
- get_security() -> SecurityManager                   # singleton
- SecurityManager.get_key(provider) -> str | None
- SecurityManager.set_key(provider, key) -> bool
- SecurityManager.validate_key(provider, key) -> (bool, str | None)
- SecurityManager.redact(secret) -> str               # "sk-...abcd"
- SecurityManager.secret_hash(secret) -> str          # "sha256:abcd..."
- SecurityManager.build_auth_headers(provider) -> dict[str, str]
- SecurityManager.merge_into_env(provider) -> None
- SecurityManager.clear(provider | None) -> None
- list_known_providers() -> list[str]

Supported providers & envs
--------------------------
openai       : OPENAI_API_KEY
anthropic    : ANTHROPIC_API_KEY
huggingface  : HUGGINGFACE_API_KEY | HF_TOKEN
openrouter   : OPENROUTER_API_KEY
cohere       : COHERE_API_KEY
azure_openai : AZURE_OPENAI_API_KEY (+ AZURE_OPENAI_ENDPOINT for client code)
"""

from __future__ import annotations

import base64
import os
import re
import threading
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Optional

# ---------- Optional dependencies (lazy) ----------
try:  # Optional encryption
    from cryptography.fernet import Fernet  # type: ignore
except Exception:  # pragma: no cover
    Fernet = None  # type: ignore

try:  # Optional keyring
    import keyring  # type: ignore
except Exception:  # pragma: no cover
    keyring = None  # type: ignore


# ---------- Provider specs ----------
@dataclass(frozen=True)
class ProviderSpec:
    env_vars: tuple[str, ...]
    pattern: re.Pattern[str] | None
    auth_type: str = "bearer"  # or "token" (for HF)
    header_name: str = "Authorization"


def _re(pat: str) -> re.Pattern[str]:
    return re.compile(pat)


PROVIDERS: dict[str, ProviderSpec] = {
    # Keys evolve; patterns are permissive but still helpful.
    "openai": ProviderSpec(
        env_vars=("OPENAI_API_KEY",),
        pattern=_re(r"^sk-[A-Za-z0-9]{20,}$"),
        auth_type="bearer",
    ),
    "anthropic": ProviderSpec(
        env_vars=("ANTHROPIC_API_KEY",),
        pattern=_re(r"^sk-(?:ant|live|prod|test)?-[A-Za-z0-9]{20,}$"),
        auth_type="bearer",
    ),
    "huggingface": ProviderSpec(
        env_vars=("HUGGINGFACE_API_KEY", "HF_TOKEN"),
        pattern=_re(r"^hf_[A-Za-z0-9]{30,}$"),
        auth_type="token",  # HF expects "Authorization: Bearer <token>" (works), or "hf_xxx" in headers for some libs.
    ),
    "openrouter": ProviderSpec(
        env_vars=("OPENROUTER_API_KEY",),
        pattern=_re(r"^sk-or-[A-Za-z0-9]{20,}$"),
        auth_type="bearer",
    ),
    "cohere": ProviderSpec(
        env_vars=("COHERE_API_KEY",),
        pattern=_re(r"^[A-Za-z0-9]{32,64}$"),
        auth_type="bearer",
    ),
    "azure_openai": ProviderSpec(
        env_vars=("AZURE_OPENAI_API_KEY",),
        pattern=_re(r"^[A-Za-z0-9+/=]{20,}$"),  # Azure keys vary; allow base64-ish
        auth_type="bearer",
    ),
}


def list_known_providers() -> list[str]:
    return sorted(PROVIDERS.keys())


# ---------- Redaction & hashing ----------
def _mask_middle(s: str, keep_left: int = 4, keep_right: int = 4) -> str:
    if not s:
        return ""
    s = s.strip()
    if len(s) <= keep_left + keep_right + 3:
        return "•" * len(s)
    return f"{s[:keep_left]}•••{s[-keep_right:]}"


def _sha256_hex(s: str) -> str:
    return sha256(s.encode("utf-8")).hexdigest()[:12]


# ---------- Streamlit secrets (lazy) ----------
def _st_secret(keys: tuple[str, ...]) -> Optional[str]:
    try:
        import streamlit as st  # type: ignore

        for k in keys:
            try:
                v = st.secrets[k]  # type: ignore[index]
                if isinstance(v, str) and v.strip():
                    return v.strip()
            except Exception:
                continue
    except Exception:
        return None
    return None


# ---------- In-memory encrypted store ----------
class _Vault:
    """
    Tiny process-local vault. If `cryptography` is present, values are Fernet-encrypted in RAM.
    This is NOT a substitute for a proper KMS; it's a best-effort safety net to avoid accidental dumps.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._mem: dict[str, bytes] = {}
        self._fernet = None
        if Fernet is not None:
            # Prefer a stable process key from env; else random per run
            key_env = os.getenv("TMIV_FERNET_KEY")
            if key_env:
                try:
                    self._fernet = Fernet(key_env.encode("utf-8"))
                except Exception:
                    pass
            if self._fernet is None:
                # generate ephemeral key for this process
                self._fernet = Fernet(base64.urlsafe_b64encode(os.urandom(32)))

    def set(self, name: str, value: str) -> None:
        v = (value or "").encode("utf-8")
        with self._lock:
            if self._fernet is not None:
                try:
                    v = self._fernet.encrypt(v)
                except Exception:
                    pass
            self._mem[name] = v

    def get(self, name: str) -> Optional[str]:
        with self._lock:
            v = self._mem.get(name)
            if v is None:
                return None
            out = v
            if self._fernet is not None:
                try:
                    out = self._fernet.decrypt(v)
                except Exception:
                    # if decrypt fails, assume plaintext
                    out = v
            try:
                return out.decode("utf-8")
            except Exception:
                return None

    def delete(self, name: str) -> None:
        with self._lock:
            self._mem.pop(name, None)

    def clear(self) -> None:
        with self._lock:
            self._mem.clear()


# ---------- SecurityManager ----------
class SecurityManager:
    def __init__(self) -> None:
        self._vault = _Vault()

    # ---- core helpers ----
    def validate_key(self, provider: str, key: str) -> tuple[bool, Optional[str]]:
        spec = PROVIDERS.get(provider)
        if not spec:
            return False, f"Unknown provider '{provider}'. Known: {', '.join(list_known_providers())}"
        k = (key or "").strip()
        if not k:
            return False, "Empty key."
        if spec.pattern is not None and not spec.pattern.match(k):
            return False, "Key does not match expected format (still may work, but please verify)."
        return True, None

    def set_key(self, provider: str, key: str) -> bool:
        ok, _ = self.validate_key(provider, key)
        # Even if validation fails, we store (user may have non-standard key). We'll return 'ok' for UI hints.
        self._vault.set(provider, key.strip())
        return ok

    def get_key(self, provider: str) -> Optional[str]:
        # 1) Explicitly set this run
        k = self._vault.get(provider)
        if k:
            return k

        # 2) Streamlit secrets
        spec = PROVIDERS.get(provider)
        if spec:
            k = _st_secret(spec.env_vars)
            if k:
                return k

        # 3) Environment
        if spec:
            for env in spec.env_vars:
                k = os.getenv(env)
                if k and k.strip():
                    return k.strip()

        # 4) OS keyring (service name "tmiv", username=provider)
        if keyring is not None:
            try:
                k = keyring.get_password("tmiv", provider)  # type: ignore[arg-type]
                if k and k.strip():
                    return k.strip()
            except Exception:
                pass

        return None

    def merge_into_env(self, provider: str) -> None:
        """Populate os.environ with the resolved key under the provider's env var(s)."""
        key = self.get_key(provider)
        if not key:
            return
        spec = PROVIDERS.get(provider)
        if not spec:
            return
        for env in spec.env_vars:
            os.environ.setdefault(env, key)

    def save_to_keyring(self, provider: str, key: str) -> bool:
        """Persist key to OS keyring (optional)."""
        if keyring is None:
            return False
        try:
            keyring.set_password("tmiv", provider, key)  # type: ignore[arg-type]
            return True
        except Exception:
            return False

    def delete_from_keyring(self, provider: str) -> bool:
        if keyring is None:
            return False
        try:
            keyring.delete_password("tmiv", provider)  # type: ignore[arg-type]
            return True
        except Exception:
            return False

    def clear(self, provider: str | None = None) -> None:
        if provider:
            self._vault.delete(provider)
            return
        self._vault.clear()

    # ---- display/log helpers ----
    @staticmethod
    def redact(secret: str | None) -> str:
        if not secret:
            return ""
        s = secret.strip()
        # keep any well-known prefix like "sk-" / "hf_"
        prefix = ""
        if s.startswith(("sk-", "hf_", "csk_", "key_", "token_", "az_")):
            prefix = s.split("-", 1)[0] + "-" if "-" in s[:5] else s.split("_", 1)[0] + "_"
        return prefix + _mask_middle(s[len(prefix):], keep_left=3, keep_right=4)

    @staticmethod
    def secret_hash(secret: str | None) -> str:
        if not secret:
            return ""
        return f"sha256:{_sha256_hex(secret)}"

    # ---- headers / auth ----
    def build_auth_headers(self, provider: str) -> dict[str, str]:
        """
        Build minimal Authorization headers for a provider.
        Note: Some SDKs ignore headers and expect env variables; `merge_into_env` may be needed.
        """
        key = self.get_key(provider)
        if not key:
            return {}

        spec = PROVIDERS.get(provider)
        if not spec:
            return {}

        if spec.auth_type == "bearer":
            return {spec.header_name: f"Bearer {key}"}
        if spec.auth_type == "token":
            # For HF, both "Bearer <token>" and "hf_xxx" as raw token can work depending on client.
            return {spec.header_name: f"Bearer {key}"}
        return {}

    # ---- convenience ----
    def snapshot(self) -> Dict[str, dict]:
        """Return a redacted snapshot for UI/debug (never raw keys)."""
        out: Dict[str, dict] = {}
        for p, spec in PROVIDERS.items():
            k = self.get_key(p)
            ok, reason = (False, "missing")
            if k:
                ok, reason = self.validate_key(p, k)
            out[p] = {
                "present": bool(k),
                "valid": bool(ok) if k else False,
                "reason": reason if (k and not ok) else None,
                "redacted": self.redact(k) if k else "",
                "hash": self.secret_hash(k) if k else "",
                "envs": list(spec.env_vars),
            }
        return out


# ---------- Singleton ----------
_singleton: Optional[SecurityManager] = None
_singleton_lock = threading.Lock()


def get_security() -> SecurityManager:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = SecurityManager()
        return _singleton


__all__ = [
    "get_security",
    "SecurityManager",
    "list_known_providers",
]
