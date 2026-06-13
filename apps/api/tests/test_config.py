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
    monkeypatch.setenv("MYRA_GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://localhost:7687")
    monkeypatch.setenv("NEO4J_USER", "neo4j")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret-password")
    monkeypatch.setenv("NEO4J_DATABASE", "custom_graph")
    monkeypatch.setenv("NEO4J_TIMEOUT_SECONDS", "15.5")
    monkeypatch.setenv("MYRA_GRAPH_BATCH_LIMIT", "100")

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
    assert settings.graphrag_enabled is True
    assert settings.neo4j_uri == "bolt://localhost:7687"
    assert settings.neo4j_user == "neo4j"
    assert settings.neo4j_password == "secret-password"
    assert settings.neo4j_database == "custom_graph"
    assert settings.neo4j_timeout_seconds == 15.5
    assert settings.graph_batch_limit == 100


def test_settings_defaults_when_env_vars_absent(monkeypatch) -> None:
    for var in [
        "MYRA_GRAPHRAG_ENABLED",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_PASSWORD",
        "NEO4J_DATABASE",
        "NEO4J_TIMEOUT_SECONDS",
        "MYRA_GRAPH_BATCH_LIMIT",
    ]:
        monkeypatch.delenv(var, raising=False)

    settings = Settings.from_environment()

    assert settings.graphrag_enabled is False
    assert settings.neo4j_uri is None
    assert settings.neo4j_user is None
    assert settings.neo4j_password is None
    assert settings.neo4j_database == "neo4j"
    assert settings.neo4j_timeout_seconds == 10.0
    assert settings.graph_batch_limit == 50


def test_graphrag_enabled_boolean_parsing(monkeypatch) -> None:
    for val in ("true", "True", "1", "yes", "YES"):
        monkeypatch.setenv("MYRA_GRAPHRAG_ENABLED", val)
        assert Settings.from_environment().graphrag_enabled is True

    for val in ("false", "False", "0", "no", "anything_else"):
        monkeypatch.setenv("MYRA_GRAPHRAG_ENABLED", val)
        assert Settings.from_environment().graphrag_enabled is False


def test_log_formatter_emits_json() -> None:
    record = logging.LogRecord("myra.api", logging.INFO, __file__, 1, "ready", (), None)

    formatted = json.loads(JsonFormatter().format(record))

    assert formatted["level"] == "INFO"
    assert formatted["logger"] == "myra.api"
    assert formatted["message"] == "ready"
    assert "timestamp" in formatted
