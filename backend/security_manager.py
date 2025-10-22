from __future__ import annotations
import os
from typing import Optional
import streamlit as st

try:
    import keyring  # type: ignore
    KEYRING_AVAILABLE = True
except Exception:
    keyring = None  # type: ignore
    KEYRING_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    _FERNET_OK = True
except Exception:
    Fernet = None  # type: ignore
    _FERNET_OK = False

def _truthy(x) -> bool:
    try:
        return bool(x) and str(x).lower() not in {"none","nan","null",""}
    except Exception:
        return False

def _provider_env_candidates(prov: str):
    base = prov.upper()
    cands = [f"{base}_API_KEY", f"{base}_KEY", f"{base}_TOKEN"]
    if base == "OPENAI":
        cands.append("OPENAI_API_KEY")
    return cands

def _get_fernet():
    if not _FERNET_OK:
        return None
    key = st.session_state.get("_fernet_key")
    if not key:
        key = Fernet.generate_key().decode("utf-8")
        st.session_state["_fernet_key"] = key
    return Fernet(key.encode("utf-8"))

def _enc_store(provider: str, value: str) -> None:
    try:
        f = _get_fernet()
        enc = value
        if f:
            enc = f.encrypt(value.encode("utf-8")).decode("utf-8")
        bag = st.session_state.get("_enc_keys", {})
        bag[provider] = enc
        st.session_state["_enc_keys"] = bag
    except Exception:
        bag = st.session_state.get("_enc_keys", {})
        bag[provider] = value
        st.session_state["_enc_keys"] = bag

def _enc_load(provider: str) -> Optional[str]:
    try:
        bag = st.session_state.get("_enc_keys", {})
        enc = bag.get(provider)
        if not enc:
            return None
        f = _get_fernet()
        if f:
            return f.decrypt(enc.encode("utf-8")).decode("utf-8")
        return enc
    except Exception:
        return None

class CredentialManager:
    """keyring ➜ szyfrowana sesja (Fernet) ➜ ENV (bez st.secrets)."""
    SERVICE_NAME = "TMIV"

    def get_api_key(self, provider: str) -> Optional[str]:
        prov = str(provider or "").strip().lower()
        if not prov:
            return None
        # 1) keyring
        try:
            if KEYRING_AVAILABLE:
                kr = keyring.get_password(self.SERVICE_NAME, prov)  # type: ignore
                if _truthy(kr):
                    return kr.strip()
        except Exception:
            pass
        # 2) zaszyfrowana sesja
        val = _enc_load(prov)
        if _truthy(val):
            return str(val).strip()
        # 3) ENV
        for env_key in _provider_env_candidates(prov):
            env_val = os.getenv(env_key)
            if _truthy(env_val):
                return str(env_val).strip()
        return None

    def set_api_key(self, provider: str, value: str, *, persist_keyring: bool = True) -> None:
        prov = str(provider or "").strip().lower()
        val = str(value or "").strip()
        if not prov or not val:
            return
        _enc_store(prov, val)
        if persist_keyring and KEYRING_AVAILABLE:
            try:
                keyring.set_password(self.SERVICE_NAME, prov, val)  # type: ignore
            except Exception:
                pass

    def clear_api_key(self, provider: str, *, remove_keyring: bool = False) -> None:
        prov = str(provider or "").strip().lower()
        try:
            bag = st.session_state.get("_enc_keys", {})
            if prov in bag:
                del bag[prov]
                st.session_state["_enc_keys"] = bag
        except Exception:
            pass
        if remove_keyring and KEYRING_AVAILABLE:
            try:
                if hasattr(keyring, "delete_password"):  # type: ignore
                    keyring.delete_password(self.SERVICE_NAME, prov)  # type: ignore
            except Exception:
                pass

# Legacy exports for backward compatibility
try:
    credential_manager  # type: ignore[name-defined]
except Exception:
    credential_manager = CredentialManager()

def get_api_key(provider: str):
    try:
        return credential_manager.get_api_key(provider)
    except Exception:
        return None
