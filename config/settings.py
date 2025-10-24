"""
TMIV Settings (Pydantic v2) — konfiguracja i feature flags.
Docstring (PL): Ten moduł definiuje centralne ustawienia aplikacji wraz z bezpiecznym
wczytywaniem z .env oraz Streamlit secrets. Zapewnia spójność i możliwość
sterowania funkcjami „gated” bez modyfikacji kodu.
"""

from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional
import os

class TMIVSettings(BaseSettings):
    # Feature flags (domyślnie konserwatywnie OFF)
    use_polars: bool = Field(default=False, alias="USE_POLARS")
    use_shap: bool = Field(default=False, alias="USE_SHAP")
    use_optuna: bool = Field(default=False, alias="USE_OPTUNA")
    use_onnx: bool = Field(default=False, alias="USE_ONNX")
    use_opentelemetry: bool = Field(default=False, alias="USE_OPENTELEMETRY")
    use_gx: bool = Field(default=False, alias="USE_GX")
    use_sbom: bool = Field(default=False, alias="USE_SBOM")
    rag_explain: bool = Field(default=False, alias="RAG_EXPLAIN")

    # Budżety / limity
    max_train_time_sec: int = Field(default=180, alias="MAX_TRAIN_TIME_SEC")
    max_hpo_trials: int = Field(default=25, alias="MAX_HPO_TRIALS")
    max_models_parallel: int = Field(default=2, alias="MAX_MODELS_PARALLEL")

    # Bezpieczeństwo / klucze (trzymane poza repo – .env/secrets)
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # Reproducibility
    global_seed: int = Field(default=42, alias="GLOBAL_SEED")

    model_config = SettingsConfigDict(
        env_file=("config/.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @property
    def strategy_default(self) -> str:
        """
        Docstring (PL): Domyślna strategia mocy treningu w UI.
        Możliwe wartości: 'fast_small' | 'balanced' | 'accurate' | 'advanced'
        """
        return "balanced"

    def provider_status(self) -> dict:
        """
        Docstring (PL): Zwraca status dostępności usług zewnętrznych (LLM), bez logowania kluczy.
        """
        return {
            "OpenAI": bool(self.openai_api_key),
            "Anthropic": bool(self.anthropic_api_key),
        }


def load_settings() -> TMIVSettings:
    """
    Docstring (PL): Ładuje ustawienia z .env oraz zmiennych środowiskowych.
    """
    # Uwaga: Streamlit secrets (st.secrets) wczytujemy po stronie app.py, jeśli dostępne.
    return TMIVSettings()


# Umożliwiamy prosty import: from config.settings import settings
settings = load_settings()