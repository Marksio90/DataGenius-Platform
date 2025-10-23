"""
Centralna konfiguracja aplikacji wykorzystująca Pydantic Settings.

Funkcjonalności:
- Zarządzanie kluczami API
- Feature flags
- Ustawienia środowiskowe
- Walidacja konfiguracji

Przykład użycia:
    from config.settings import get_settings
    settings = get_settings()
    api_key = settings.openai_api_key
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Główna klasa konfiguracji aplikacji."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Application
    app_name: str = Field(
        default="The Most Important Variables - Advanced ML Platform v2.0 Pro"
    )
    app_version: str = Field(default="2.0.0")
    log_level: str = Field(default="INFO")
    enable_telemetry: bool = Field(default=True)
    enable_cache: bool = Field(default=True)
    max_upload_size_mb: int = Field(default=200)

    # Feature Flags
    enable_pycaret: bool = Field(default=False)
    enable_mlflow: bool = Field(default=False)
    enable_profiling: bool = Field(default=True)
    enable_ensemble: bool = Field(default=True)
    enable_auto_tuning: bool = Field(default=True)

    # Database
    use_database: bool = Field(default=False)
    database_url: str = Field(default="sqlite:///tmiv.db")

    # ML Settings
    default_cv_folds: int = Field(default=5)
    default_test_size: float = Field(default=0.2)
    random_state: int = Field(default=42)
    n_jobs: int = Field(default=-1)

    # Timeouts and limits
    llm_timeout_seconds: int = Field(default=30)
    max_llm_retries: int = Field(default=3)
    max_llm_tokens: int = Field(default=2000)

    # Paths
    @property
    def root_dir(self) -> Path:
        """Katalog główny projektu."""
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        """Katalog z danymi."""
        return self.root_dir / "data"

    @property
    def artifacts_dir(self) -> Path:
        """Katalog z artefaktami."""
        path = self.root_dir / "artifacts"
        path.mkdir(exist_ok=True)
        return path

    @property
    def outputs_dir(self) -> Path:
        """Katalog z outputami."""
        path = self.root_dir / "outputs"
        path.mkdir(exist_ok=True)
        return path

    def has_openai_key(self) -> bool:
        """Sprawdza czy klucz OpenAI jest dostępny."""
        return bool(self.openai_api_key and len(self.openai_api_key) > 10)

    def has_anthropic_key(self) -> bool:
        """Sprawdza czy klucz Anthropic jest dostępny."""
        return bool(self.anthropic_api_key and len(self.anthropic_api_key) > 10)

    def has_any_llm_key(self) -> bool:
        """Sprawdza czy jakikolwiek klucz LLM jest dostępny."""
        return self.has_openai_key() or self.has_anthropic_key()


@lru_cache()
def get_settings() -> Settings:
    """
    Zwraca singleton instancji Settings z cache.

    Returns:
        Settings: Skonfigurowana instancja ustawień
    """
    return Settings()