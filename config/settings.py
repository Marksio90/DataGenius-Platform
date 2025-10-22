try:
    # Pydantic v2
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
    from pydantic import Field  # v2
    _PYD2 = True
except Exception:
    # Pydantic v1
    from pydantic import BaseSettings, Field  # type: ignore
    SettingsConfigDict = None  # type: ignore
    _PYD2 = False

from typing import Optional, List
import os
from pathlib import Path
from urllib.parse import quote_plus


class Settings(BaseSettings):
    """
    Konfiguracja aplikacji z wykorzystaniem Pydantic Settings.
    - Wspiera Pydantic v1 i v2.
    - Automatycznie ładuje zmienne z pliku .env (config/.env).
    - Normalizuje ścieżki i weryfikuje podstawowe parametry.
    """

    # Podstawowe ustawienia aplikacji
    app_name: str = Field("The Most Important Variables", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    debug: bool = Field(False, env="DEBUG")

    # Ustawienia Machine Learning
    ml_random_state: int = Field(42, env="ML_RANDOM_STATE")
    ml_test_size: float = Field(0.2, env="ML_TEST_SIZE")
    ml_cv_folds: int = Field(5, env="ML_CV_FOLDS")
    ml_timeout: int = Field(300, env="ML_TIMEOUT")  # 5 minut

    # Ustawienia bazy danych
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    db_host: str = Field("localhost", env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_name: str = Field("hackathon_db", env="DB_NAME")
    db_user: str = Field("postgres", env="DB_USER")
    db_password: Optional[str] = Field(None, env="DB_PASSWORD")

    # Ścieżki do plików
    data_dir: Path = Field(default=Path("data"), env="DATA_DIR")
    models_dir: Path = Field(default=Path("models"), env="MODELS_DIR")
    logs_dir: Path = Field(default=Path("logs"), env="LOGS_DIR")

    # Ustawienia LLM / AI
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    use_local_llm: bool = Field(True, env="USE_LOCAL_LLM")
    llm_model: str = Field("gpt-3.5-turbo", env="LLM_MODEL")
    llm_temperature: float = Field(0.7, env="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(1000, env="LLM_MAX_TOKENS")

    # Ustawienia UI
    max_file_size: int = Field(200, env="MAX_FILE_SIZE")  # MB
    supported_file_types: List[str] = Field(
        default_factory=lambda: ["csv", "json"], env="SUPPORTED_FILE_TYPES"
    )
    theme: str = Field("light", env="THEME")

    # Ustawienia wydajności
    max_rows_display: int = Field(10000, env="MAX_ROWS_DISPLAY")
    max_columns_display: int = Field(50, env="MAX_COLUMNS_DISPLAY")
    cache_ttl: int = Field(3600, env="CACHE_TTL")  # 1 godzina

    # Ustawienia logowania
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("app.log", env="LOG_FILE")  # względna nazwa -> logs_dir / log_file

    # Ustawienia deploymentu
    host: str = Field("localhost", env="HOST")
    port: int = Field(8501, env="PORT")
    workers: int = Field(1, env="WORKERS")

    # Konfiguracja Pydantic Settings (v2) lub Config (v1)
    if _PYD2:
        model_config = SettingsConfigDict(
            env_file="config/.env",
            env_file_encoding="utf-8",
            case_sensitive=False,
            # env_prefix="TMIV_",  # odkomentuj jeśli chcesz wspólny prefiks
            extra="ignore",
        )
    else:
        class Config:
            env_file = "config/.env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            # env_prefix = "TMIV_"
            extra = "ignore"

    # -----------------------
    # Post-init (v1/v2-safe)
    # -----------------------
    if _PYD2:
        # pydantic v2: preferowany hook
        def model_post_init(self, __context) -> None:  # type: ignore[override]
            self._after_init()
    else:
        # pydantic v1: klasyczny __init__
        def __init__(self, **kwargs):  # type: ignore[override]
            super().__init__(**kwargs)
            self._after_init()

    def _after_init(self) -> None:
        """Wspólna część inicjalizacji po zbudowaniu modelu."""
        # Normalizacja ścieżek (expanduser + absolutyzacja)
        self.data_dir = Path(os.path.expanduser(str(self.data_dir))).resolve()
        self.models_dir = Path(os.path.expanduser(str(self.models_dir))).resolve()
        self.logs_dir = Path(os.path.expanduser(str(self.logs_dir))).resolve()

        # Uporządkuj listę rozszerzeń (bez kropek, lowercase, deduplikacja)
        self.supported_file_types = sorted(
            {str(ext).lower().lstrip(".") for ext in (self.supported_file_types or [])}
        )

        # Tworzenie wymaganych katalogów
        self.create_directories()

        # Walidacja ustawień
        self.validate_settings()

    # -----------------------
    # Metody pomocnicze
    # -----------------------
    def create_directories(self) -> None:
        """Tworzy wymagane katalogi jeśli nie istnieją."""
        for directory in (self.data_dir, self.models_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def validate_settings(self) -> None:
        """Waliduje ustawienia konfiguracji."""
        if not (0.0 < float(self.ml_test_size) < 1.0):
            raise ValueError("ml_test_size musi być między 0 a 1")

        if int(self.ml_cv_folds) < 2:
            raise ValueError("ml_cv_folds musi być >= 2")

        if int(self.max_file_size) <= 0:
            raise ValueError("max_file_size musi być > 0")

        if int(self.workers) < 1:
            raise ValueError("workers musi być >= 1")

        if int(self.llm_max_tokens) <= 0:
            raise ValueError("llm_max_tokens musi być > 0")

        if not self.supported_file_types:
            raise ValueError("supported_file_types nie może być puste")

        if not (0.0 <= float(self.llm_temperature) <= 2.0):
            raise ValueError("llm_temperature powinno być w zakresie 0.0–2.0")

    # -----------------------
    # Wygodne gettery/properties
    # -----------------------
    def get_database_url(self) -> str:
        """
        Zwraca URL bazy danych; wspiera bezpośredni DATABASE_URL lub składanie z pól (Postgres/sqlite).
        - Jeśli db_host zaczyna się od 'sqlite', zbuduje sqlite:///path.db
        - Hasło jest bezpiecznie kodowane (URL-encoding)
        """
        if self.database_url:
            return str(self.database_url)

        # sqlite — gdy db_host zaczyna się od 'sqlite'
        if str(self.db_host).lower().startswith("sqlite"):
            # np. db_host="sqlite", db_name="file.db" -> sqlite:///file.db
            db_path = Path(str(self.db_name)).expanduser().resolve()
            return f"sqlite:///{db_path}"

        user_part = quote_plus(self.db_user) if self.db_user else ""
        password_part = f":{quote_plus(self.db_password)}" if self.db_password else ""
        host_part = self.db_host
        return f"postgresql://{user_part}{password_part}@{host_part}:{self.db_port}/{self.db_name}"

    @property
    def log_path(self) -> Path:
        """Pełna ścieżka do pliku logów (logs_dir / log_file, jeśli log_file nie jest już absolutne)."""
        p = Path(str(self.log_file))
        return p if p.is_absolute() else (self.logs_dir / p)

    def is_llm_available(self) -> bool:
        """Sprawdza czy LLM jest dostępny."""
        return bool(self.openai_api_key or self.anthropic_api_key or self.use_local_llm)

    def get_supported_file_extensions(self) -> List[str]:
        """Zwraca listę obsługiwanych rozszerzeń z kropką (np. ['.csv','.json'])."""
        return [f".{ext}" for ext in self.supported_file_types]


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Zwraca globalną instancję ustawień (singleton pattern).
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Przeładowuje ustawienia (przydatne w testach).
    """
    global _settings
    _settings = None
    return get_settings()


# Przykłady użycia:
if __name__ == "__main__":
    settings = get_settings()
    print(f"App: {settings.app_name} v{settings.app_version}")
    print(f"Data dir: {settings.data_dir}")
    print(f"Logs at:  {settings.log_path}")
    print(f"DB URL:   {settings.get_database_url()}")
    print(f"Types:    {settings.get_supported_file_extensions()}")

# Training speed profile
FAST_QUALITY_MODE = True  # szybkie i mocne modele, minimalna utrata jakości
