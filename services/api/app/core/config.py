"""PHOTO → 3D ENGINE - Configuration

Validated settings loaded from environment variables or .env file.
Enforces loopback binding by default for local privacy.
"""

from pathlib import Path
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Runtime Environment
    app_env: str = Field(default="development", alias="APP_ENV")
    service_name: str = "photo-to-3d-api"
    service_version: str = "0.1.0"
    api_version: str = "v1"
    schema_version: str = "1.0.0"

    # Network Binding (Enforce loopback by default)
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_log_level: str = Field(default="info", alias="API_LOG_LEVEL")

    # Local Storage
    storage_root: Path = Field(default=Path("./data/storage"), alias="STORAGE_ROOT")
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")

    # Allowed CORS Origins (can be comma-separated string or list)
    cors_origins_raw: Union[str, List[str]] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        alias="CORS_ORIGINS"
    )

    @property
    def cors_origins(self) -> List[str]:
        if isinstance(self.cors_origins_raw, list):
            return [str(item).strip() for item in self.cors_origins_raw if str(item).strip()]
        if isinstance(self.cors_origins_raw, str):
            return [item.strip() for item in self.cors_origins_raw.split(",") if item.strip()]
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    @field_validator("api_host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        return v.strip()

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


settings = Settings()
