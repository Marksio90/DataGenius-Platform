try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """
    Konfiguracja aplikacji z wykorzystaniem Pydantic Settings
    Automatycznie ładuje zmienne z pliku .env
    """
    
    # Podstawowe ustawienia aplikacji
    app_name: str = "The Most Important Variables"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Ustawienia Machine Learning
    ml_random_state: int = 42
    ml_test_size: float = 0.2
    ml_cv_folds: int = 5
    ml_timeout: int = 300  # 5 minut timeout dla treningu
    
    # Ustawienia bazy danych
    database_url: Optional[str] = None
    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "hackathon_db"
    db_user: str = "postgres"
    db_password: Optional[str] = None
    
    # Ścieżki do plików
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    logs_dir: Path = Path("logs")
    
    # Ustawienia LLM / AI
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    use_local_llm: bool = True
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000
    
    # Ustawienia UI
    max_file_size: int = 200  # MB
    supported_file_types: list = ["csv", "json"]
    theme: str = "light"
    
    # Ustawienia wydajności
    max_rows_display: int = 10000
    max_columns_display: int = 50
    cache_ttl: int = 3600  # 1 godzina
    
    # Ustawienia logowania
    log_level: str = "INFO"
    log_file: str = "app.log"
    
    # Ustawienia deploymentu
    host: str = "localhost"
    port: int = 8501
    workers: int = 1
    
    class Config:
        env_file = "config/.env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Tworzenie wymaganych katalogów
        self.create_directories()
        
        # Walidacja ustawień
        self.validate_settings()
    
    def create_directories(self):
        """Tworzy wymagane katalogi jeśli nie istnieją"""
        directories = [self.data_dir, self.models_dir, self.logs_dir]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_settings(self):
        """Waliduje ustawienia konfiguracji"""
        if self.ml_test_size <= 0 or self.ml_test_size >= 1:
            raise ValueError("ml_test_size musi być między 0 a 1")
        
        if self.ml_cv_folds < 2:
            raise ValueError("ml_cv_folds musi być >= 2")
        
        if self.max_file_size <= 0:
            raise ValueError("max_file_size musi być > 0")
    
    def get_database_url(self) -> str:
        """Zwraca URL bazy danych"""
        if self.database_url:
            return self.database_url
        
        password_part = f":{self.db_password}" if self.db_password else ""
        return f"postgresql://{self.db_user}{password_part}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    def is_llm_available(self) -> bool:
        """Sprawdza czy LLM jest dostępny"""
        return bool(self.openai_api_key or self.anthropic_api_key or self.use_local_llm)
    
    def get_supported_file_extensions(self) -> list:
        """Zwraca listę obsługiwanych rozszerzeń plików"""
        return [f".{ext}" for ext in self.supported_file_types]

# Singleton instance
_settings: Optional[Settings] = None

def get_settings() -> Settings:
    """
    Zwraca globalną instancję ustawień (singleton pattern)
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings

def reload_settings():
    """
    Przeładowuje ustawienia (przydatne w testach)
    """
    global _settings
    _settings = None
    return get_settings()

# Przykłady użycia:
if __name__ == "__main__":
    settings = get_settings()
    print(f"App: {settings.app_name} v{settings.app_version}")