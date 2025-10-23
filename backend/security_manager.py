"""
Zarządzanie bezpieczeństwem aplikacji.

Funkcjonalności:
- Bezpieczna obsługa kluczy API
- Szyfrowanie danych w sesji
- Sanityzacja promptów
- Walidacja inputów
"""

import logging
import re
from typing import Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SecurityManager:
    """
    Manager bezpieczeństwa aplikacji.
    
    Obsługuje:
    - Szyfrowanie kluczy w sesji
    - Sanityzację promptów dla LLM
    - Walidację danych wejściowych
    """

    def __init__(self):
        """Inicjalizacja security managera."""
        self.encryption_available = False
        self._init_encryption()

    def _init_encryption(self):
        """Inicjalizuje moduł szyfrowania."""
        try:
            from cryptography.fernet import Fernet
            # W produkcji klucz powinien być przechowywany bezpiecznie
            self.cipher_suite = None  # Placeholder
            self.encryption_available = False
            logger.info("Moduł szyfrowania dostępny (nieaktywny w demo)")
        except ImportError:
            logger.warning("Cryptography nie jest dostępna")

    def sanitize_prompt(self, prompt: str, max_length: int = 10000) -> str:
        """
        Sanityzuje prompt przed wysłaniem do LLM.

        Args:
            prompt: Prompt do sanityzacji
            max_length: Maksymalna długość promptu

        Returns:
            str: Sanityzowany prompt

        Example:
            >>> sm = SecurityManager()
            >>> sanitized = sm.sanitize_prompt("Test prompt")
            >>> len(sanitized) > 0
            True
        """
        if not prompt:
            return ""

        # Ogranicz długość
        if len(prompt) > max_length:
            logger.warning(f"Prompt przekracza {max_length} znaków - obcinanie")
            prompt = prompt[:max_length]

        # Usuń potencjalnie niebezpieczne sekwencje
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
        ]

        for pattern in dangerous_patterns:
            prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE | re.DOTALL)

        return prompt.strip()

    def validate_api_key(self, key: str, provider: str = "openai") -> bool:
        """
        Waliduje format klucza API.

        Args:
            key: Klucz API do walidacji
            provider: Provider (openai, anthropic)

        Returns:
            bool: Czy klucz jest poprawny

        Example:
            >>> sm = SecurityManager()
            >>> sm.validate_api_key("sk-test123", "openai")
            False
        """
        if not key or len(key) < 10:
            return False

        if provider == "openai":
            return key.startswith("sk-") and len(key) > 20
        elif provider == "anthropic":
            return key.startswith("sk-ant-") and len(key) > 20

        return False

    def mask_api_key(self, key: str, visible_chars: int = 4) -> str:
        """
        Maskuje klucz API do wyświetlenia.

        Args:
            key: Klucz do zamaskowania
            visible_chars: Liczba widocznych znaków na końcu

        Returns:
            str: Zamaskowany klucz

        Example:
            >>> sm = SecurityManager()
            >>> masked = sm.mask_api_key("sk-test123456789")
            >>> "***" in masked
            True
        """
        if not key or len(key) < visible_chars:
            return "***"

        return f"***{key[-visible_chars:]}"

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanityzuje nazwę pliku.

        Args:
            filename: Nazwa pliku

        Returns:
            str: Bezpieczna nazwa pliku

        Example:
            >>> sm = SecurityManager()
            >>> safe = sm.sanitize_filename("my/bad\\file.txt")
            >>> "/" not in safe
            True
        """
        # Usuń niebezpieczne znaki
        safe_name = re.sub(r'[^\w\s\-\.]', '_', filename)
        safe_name = re.sub(r'[\s_]+', '_', safe_name)
        return safe_name[:255]  # Limit długości

    def check_data_privacy(self, text: str) -> dict:
        """
        Sprawdza czy tekst zawiera potencjalnie wrażliwe dane.

        Args:
            text: Tekst do sprawdzenia

        Returns:
            dict: Raport z wykrytymi wzorcami

        Example:
            >>> sm = SecurityManager()
            >>> report = sm.check_data_privacy("My email is test@example.com")
            >>> report['has_sensitive_data']
            True
        """
        patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        }

        findings = {}
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                findings[pattern_name] = len(matches)

        return {
            "has_sensitive_data": len(findings) > 0,
            "findings": findings,
        }


# Singleton instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """
    Zwraca singleton instancji SecurityManager.

    Returns:
        SecurityManager: Instancja security managera
    """
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager