import json
import logging

from app.config import Settings
from app.logging import JsonFormatter


def test_settings_read_environment(monkeypatch) -> None:
    monkeypatch.setenv("MYRA_APP_NAME", "Test API")
    monkeypatch.setenv("MYRA_CORS_ORIGINS", '["http://localhost:3100"]')

    settings = Settings.from_environment()

    assert settings.app_name == "Test API"
    assert settings.cors_origins == ["http://localhost:3100"]


def test_log_formatter_emits_json() -> None:
    record = logging.LogRecord("myra.api", logging.INFO, __file__, 1, "ready", (), None)

    formatted = json.loads(JsonFormatter().format(record))

    assert formatted["level"] == "INFO"
    assert formatted["logger"] == "myra.api"
    assert formatted["message"] == "ready"
    assert "timestamp" in formatted
