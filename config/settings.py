"""Pydantic-style settings placeholder (skeleton)."""
from __future__ import annotations

try:
    from pydantic_settings import BaseSettings
except Exception:  # optional at skeleton stage
    class BaseSettings:  # type: ignore
        pass

class Settings(BaseSettings):
    app_name: str = "TMIV Advanced ML Platform"
    env: str = "development"

def get_settings() -> Settings:
    return Settings()
