"""
security_manager.py
Docstring (PL): Zarządza kluczami i wrażliwymi danymi w sesji. Nie loguje wartości.
Wspiera opcjonalne szyfrowanie Fernet (jeśli zainstalowane).
"""
from __future__ import annotations
from typing import Optional
import os

try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except Exception:
    HAS_FERNET = False

class SecretBox:
    def __init__(self, key: Optional[bytes] = None):
        self._key = key or (Fernet.generate_key() if HAS_FERNET else None)
        self._fernet = Fernet(self._key) if (HAS_FERNET and self._key) else None

    def seal(self, value: str) -> bytes:
        if not value:
            return b""
        if self._fernet:
            return self._fernet.encrypt(value.encode("utf-8"))
        return value.encode("utf-8")

    def open(self, blob: bytes) -> str:
        if not blob:
            return ""
        if self._fernet:
            return self._fernet.decrypt(blob).decode("utf-8")
        return blob.decode("utf-8")

def redact(s: Optional[str]) -> str:
    """
    Docstring (PL): Zwraca zredagowaną wersję sekretu (np. „sk-***1234”). Bezpieczne do logów.
    """
    if not s:
        return ""
    if len(s) <= 8:
        return "***"
    return f"{s[:2]}***{s[-4:]}"