import json
import logging

from app.config import Settings
from app.logging import JsonFormatter


def test_settings_read_environment(monkeypatch) -> None:
    monkeypatch.setenv("MYRA_APP_NAME", "Test API")
    monkeypatch.setenv("MYRA_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("MYRA_CORS_ORIGINS", '["http://localhost:3100"]')
    monkeypatch.setenv("MYRA_CORS_METHODS", '["GET", "POST"]')
    monkeypatch.setenv("MYRA_CORS_HEADERS", '["Content-Type"]')
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/myra")
    monkeypatch.setenv("SUPABASE_PROJECT_REF", "proj123")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gcp-test")
    monkeypatch.setenv("GCS_BUCKET_NAME", "my-bucket")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://custom.deepseek.com")
    monkeypatch.setenv("MYRA_MAX_UPLOAD_SIZE_BYTES", "10485760")
    monkeypatch.setenv("MYRA_MAX_PDF_PAGES", "50")

    settings = Settings.from_environment()

    assert settings.app_name == "Test API"
    assert settings.log_level == "DEBUG"
    assert settings.cors_origins == ["http://localhost:3100"]
    assert settings.cors_methods == ["GET", "POST"]
    assert settings.cors_headers == ["Content-Type"]
    assert settings.database_url == "postgresql://user:pass@localhost:5432/myra"
    assert settings.supabase_project_ref == "proj123"
    assert settings.google_cloud_project == "gcp-test"
    assert settings.gcs_bucket_name == "my-bucket"
    assert settings.deepseek_api_key == "sk-test"
    assert settings.deepseek_base_url == "https://custom.deepseek.com"
    assert settings.max_upload_size_bytes == 10485760
    assert settings.max_pdf_pages == 50


def test_log_formatter_emits_json() -> None:
    record = logging.LogRecord("myra.api", logging.INFO, __file__, 1, "ready", (), None)

    formatted = json.loads(JsonFormatter().format(record))

    assert formatted["level"] == "INFO"
    assert formatted["logger"] == "myra.api"
    assert formatted["message"] == "ready"
    assert "timestamp" in formatted
