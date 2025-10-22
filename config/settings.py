# config/settings.py
"""
Pydantic Settings for TMIV – Advanced ML Platform (v2-friendly).

Goals
-----
- Single source of truth for runtime configuration.
- Reads from:
  * `.env` (if present),
  * environment variables (UPPER_SNAKE_CASE by default).
- Light validation & convenient path handling.
- No hard dependency on Streamlit or other app layers.

Notes
-----
- Compatible with `pydantic-settings` v2. If it's missing at skeleton stage,
  the fallback `BaseSettings` keeps imports working (fields won't auto-load from env).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# ---- Pydantic / Settings (v2 preferred, graceful fallback) ----
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # v2
except Exception:  # optional at skeleton stage
    SettingsConfigDict = dict  # type: ignore

    class BaseSettings:  # type: ignore
        def __init__(self, **kwargs):  # very tiny shim to avoid crashes during skeleton phase
            for k, v in kwargs.items():
                setattr(self, k, v)

try:
    # pydantic v2
    from pydantic import field_validator, computed_field  # type: ignore
except Exception:  # pragma: no cover
    # pydantic v1 compatibility shims
    from pydantic import validator as field_validator  # type: ignore

    def computed_field(*args, **kwargs):  # type: ignore
        def _wrap(fn):
            return property(fn)
        return _wrap


class Settings(BaseSettings):
    # --- App ---
    app_name: str = "TMIV Advanced ML Platform"
    env: str = "development"                      # development | staging | production
    log_level: str = "INFO"

    # --- Paths (strings or Paths; will be normalized to absolute Paths) ---
    data_dir: Optional[Path] = None               # default: ./data
    cache_dir: Optional[Path] = None              # default: ./cache
    exports_dir: Optional[Path] = None            # default: ./exports
    logging_config: Path = Path("config/logging.yaml")

    # --- Security / crypto (optional) ---
    tmiv_fernet_key: Optional[str] = None         # base64 urlsafe 32B (optional)

    # --- Providers (optional) ---
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    hf_token: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None

    # --- Database (optional) ---
    database_url: Optional[str] = None            # e.g. sqlite:///tmiv.db, duckdb:///tmiv.duckdb, postgres://...

    # --- Telemetry (optional) ---
    tmiv_telemetry: str = "on"                    # on | off
    tmiv_telemetry_exporter: str = "auto"         # auto | otlp | console | none
    tmiv_otlp_endpoint: Optional[str] = None      # or use OTEL_EXPORTER_OTLP_ENDPOINT

    # --- Training defaults ---
    tmiv_n_jobs: int = -1                         # -1 = all cores
    tmiv_random_state: int = 42

    # pydantic-settings v2 config
    model_config = SettingsConfigDict(
        env_file=(".env",),               # load from .env if present
        env_file_encoding="utf-8",
        case_sensitive=False,             # ENV keys are matched case-insensitively
        extra="ignore",                   # ignore unknown keys
    )

    # ---- Validators & computed fields ----

    @field_validator("env")
    @classmethod
    def _norm_env(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in {"development", "staging", "production"}:
            return "development"
        return v

    @field_validator("log_level")
    @classmethod
    def _norm_log_level(cls, v: str) -> str:
        return (v or "INFO").strip().upper()

    @field_validator("data_dir", "cache_dir", "exports_dir")
    @classmethod
    def _normalize_paths(cls, v: Optional[Path]) -> Path:
        """
        Expand ~, default to project-relative dirs when None.
        """
        if v in (None, "", Path("")):
            # sensible defaults relative to CWD/project root
            # (we don't try to detect repo root here to keep it simple)
            name = "data"
            if cls._normalize_paths.__name__ == "_normalize_paths":  # silence linter
                pass
            # resolve by attribute name (inspect stack would be overkill; choose by identity)
            # we infer by default below
        # Since we don't know which field triggered the validator here (v2 provides info, but we keep it generic),
        # we return path and let `ensure_dirs()` create subfolders.
        p = Path(v).expanduser().resolve() if v else None
        return p or Path.cwd().joinpath("data").resolve()

    @computed_field  # type: ignore[misc]
    @property
    def artifacts_dir(self) -> Path:
        """
        A canonical artifacts subdir under cache_dir: <cache_dir>/artifacts
        """
        base = self.cache_dir or Path.cwd().joinpath("cache").resolve()
        return Path(base).joinpath("artifacts").resolve()

    def ensure_dirs(self) -> None:
        """
        Create required directories if they don't exist.
        """
        # Defaults if not provided
        if not self.data_dir:
            self.data_dir = Path.cwd().joinpath("data").resolve()
        if not self.cache_dir:
            self.cache_dir = Path.cwd().joinpath("cache").resolve()
        if not self.exports_dir:
            self.exports_dir = Path.cwd().joinpath("exports").resolve()

        for p in (self.data_dir, self.cache_dir, self.exports_dir, self.artifacts_dir, Path("logs")):
            try:
                Path(p).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

    # ---- Convenience helpers ----

    def is_telemetry_enabled(self) -> bool:
        return str(self.tmiv_telemetry).strip().lower() not in {"0", "false", "off"}

    @staticmethod
    def _redact(secret: Optional[str]) -> str:
        if not secret:
            return ""
        s = secret.strip()
        prefix = ""
        if s.startswith(("sk-", "hf_", "key_", "token_", "az_")):
            if "-" in s[:5]:
                prefix = s.split("-", 1)[0] + "-"
            elif "_" in s[:5]:
                prefix = s.split("_", 1)[0] + "_"
        body = s[len(prefix) :]
        if len(body) <= 7:
            return prefix + "•••"
        return prefix + body[:3] + "•••" + body[-4:]

    def redacted_snapshot(self) -> dict[str, Any]:
        """
        Lightweight, safe-to-print snapshot (no secrets).
        """
        return {
            "app_name": self.app_name,
            "env": self.env,
            "log_level": self.log_level,
            "paths": {
                "data_dir": str(self.data_dir) if self.data_dir else None,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
                "exports_dir": str(self.exports_dir) if self.exports_dir else None,
                "artifacts_dir": str(self.artifacts_dir),
            },
            "providers": {
                "openai_api_key": self._redact(self.openai_api_key),
                "anthropic_api_key": self._redact(self.anthropic_api_key),
                "huggingface_api_key": self._redact(self.huggingface_api_key or self.hf_token),
                "openrouter_api_key": self._redact(self.openrouter_api_key),
                "cohere_api_key": self._redact(self.cohere_api_key),
                "azure_openai_api_key": self._redact(self.azure_openai_api_key),
                "azure_openai_endpoint": bool(self.azure_openai_endpoint),
            },
            "database_url": bool(self.database_url),
            "telemetry": {
                "enabled": self.is_telemetry_enabled(),
                "exporter": self.tmiv_telemetry_exporter,
                "otlp_endpoint": bool(self.tmiv_otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")),
            },
            "training": {
                "n_jobs": self.tmiv_n_jobs,
                "random_state": self.tmiv_random_state,
            },
        }


# --- Singleton accessor (simple cache) ---

_settings_singleton: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Return a (cached) Settings instance.
    Call with reload=True to re-read environment/.env at runtime.
    """
    global _settings_singleton
    if _settings_singleton is None or reload:
        _settings_singleton = Settings()
        # Normalize & ensure dirs on load
        _settings_singleton.ensure_dirs()
    return _settings_singleton
