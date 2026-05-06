import json
import os
from typing import Literal

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "mYrA API"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @classmethod
    def from_environment(cls) -> "Settings":
        values: dict[str, object] = {}
        if app_name := os.getenv("MYRA_APP_NAME"):
            values["app_name"] = app_name
        if log_level := os.getenv("MYRA_LOG_LEVEL"):
            values["log_level"] = log_level
        if cors_origins := os.getenv("MYRA_CORS_ORIGINS"):
            values["cors_origins"] = json.loads(cors_origins)
        return cls.model_validate(values)
