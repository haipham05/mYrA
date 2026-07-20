import json
import logging

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.db.session import (
    get_budget_session_factory,
    resolve_budget_database_url,
    resolve_database_url,
)
from app.logging import JsonFormatter
from app.storage.factory import get_storage, set_storage
from app.storage.gcs import GCSStorage
from app.storage.local import LocalStorage


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


def test_explicit_local_profile_uses_local_database_and_storage(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MYRA_RUNTIME_PROFILE", "local")
    monkeypatch.setenv(
        "DATABASE_URL", "postgresql+psycopg2://local:secret@myra-local-postgres:5432/myra"
    )
    monkeypatch.setenv("GCS_BUCKET_NAME", "owner-cloud-bucket")
    monkeypatch.setenv("MYRA_LOCAL_STORAGE_ROOT", str(tmp_path / "papers"))

    settings = Settings.from_environment()
    set_storage(None)
    try:
        storage = get_storage(settings)
        assert isinstance(storage, LocalStorage)
        assert storage.base_dir == tmp_path / "papers"
        assert resolve_database_url(settings) == settings.database_url
    finally:
        set_storage(None)


def test_explicit_profiles_fail_closed_instead_of_falling_back(monkeypatch) -> None:
    with pytest.raises(ValidationError, match="cloud-data profile requires DATABASE_URL"):
        Settings(runtime_profile="cloud-data", gcs_bucket_name="bucket")

    local_settings = Settings(
        runtime_profile="local",
        database_url="postgresql+psycopg2://local:secret@db.example.com:5432/myra",
    )
    with pytest.raises(ValueError, match="must target the local PostgreSQL service"):
        resolve_database_url(local_settings)


def test_cloud_data_profile_selects_gcs_explicitly(monkeypatch) -> None:
    settings = Settings(
        runtime_profile="cloud-data",
        database_url="postgresql+psycopg2://user:secret@db.example.com:5432/myra",
        budget_database_url="postgresql+psycopg2://user:secret@myra-local-postgres:5432/myra",
        gcs_bucket_name="research-bucket",
    )
    set_storage(None)
    try:
        assert isinstance(get_storage(settings), GCSStorage)
        assert resolve_database_url(settings) == settings.database_url
        assert resolve_budget_database_url(settings) == settings.budget_database_url
    finally:
        set_storage(None)


def test_cloud_data_profile_rejects_sqlite_database_url() -> None:
    settings = Settings(
        runtime_profile="cloud-data",
        database_url="sqlite:///./local.db",
        budget_database_url="postgresql+psycopg2://user:secret@myra-local-postgres:5432/myra",
        gcs_bucket_name="research-bucket",
    )

    with pytest.raises(ValueError, match="must target PostgreSQL"):
        resolve_database_url(settings)


def test_cloud_data_profile_requires_a_local_budget_database() -> None:
    with pytest.raises(ValidationError, match="requires MYRA_BUDGET_DATABASE_URL"):
        Settings(
            runtime_profile="cloud-data",
            database_url="postgresql+psycopg2://user:secret@db.example.com:5432/myra",
            gcs_bucket_name="research-bucket",
        )


def test_cloud_data_budget_rejects_a_remote_database() -> None:
    settings = Settings(
        runtime_profile="cloud-data",
        database_url="postgresql+psycopg2://user:secret@db.example.com:5432/myra",
        budget_database_url="postgresql+psycopg2://user:secret@db.example.com:5432/myra",
        gcs_bucket_name="research-bucket",
    )
    with pytest.raises(ValueError, match="must target local PostgreSQL"):
        resolve_budget_database_url(settings)


def test_cloud_data_budget_session_uses_local_ledger_database() -> None:
    settings = Settings(
        runtime_profile="cloud-data",
        database_url="postgresql+psycopg2://user:secret@cloud-db.example:5432/myra",
        budget_database_url="postgresql+psycopg2://user:secret@myra-local-postgres:5432/myra",
        gcs_bucket_name="research-bucket",
    )

    factory = get_budget_session_factory(settings)
    assert factory.kw["bind"].url.host == "myra-local-postgres"


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


def test_log_formatter_keeps_safe_metrics_and_context_only() -> None:
    from app.observability import OperationContext, use_operation_context

    record = logging.LogRecord("myra.api", logging.INFO, __file__, 1, "ready", (), None)
    record.latency_ms = 12.5
    record.evidence_count = 3
    record.prompt = "private research question"
    record.provider_usage = {"prompt_tokens": 24, "completion_tokens": 7}
    record.requested_model = "deepseek-chat"
    record.reported_model = "deepseek-chat-2026-01"
    record.response_id = "resp-test-123"

    context = OperationContext.validated(correlation_id="request-123")
    with use_operation_context(context):
        formatted = json.loads(JsonFormatter().format(record))

    assert formatted["correlation_id"] == "request-123"
    assert formatted["latency_ms"] == 12.5
    assert formatted["evidence_count"] == 3
    assert formatted["provider_usage"] == {"prompt_tokens": 24, "completion_tokens": 7}
    assert formatted["requested_model"] == "deepseek-chat"
    assert formatted["reported_model"] == "deepseek-chat-2026-01"
    assert formatted["response_id"] == "resp-test-123"
    assert "prompt" not in formatted


def test_log_formatter_keeps_safe_translation_validation_summary_only() -> None:
    record = logging.LogRecord(
        "myra.translation.processor",
        logging.INFO,
        __file__,
        1,
        "translation_segment_validation",
        (),
        None,
    )
    record.stage = "validation"
    record.total_units = 31
    record.completed_units = 29
    record.skipped_units = 2
    record.failure_count = 0
    record.failure_reasons = {}
    record.failure_units = []
    record.translation_id = "translation-test-id"
    record.source_quote = "private research text"

    formatted = json.loads(JsonFormatter().format(record))

    assert formatted["total_units"] == 31
    assert formatted["completed_units"] == 29
    assert formatted["failure_count"] == 0
    assert "failure_units" in formatted
    assert formatted["translation_id"] == "translation-test-id"
    assert "source_quote" not in formatted


def test_log_formatter_redacts_sensitive_text() -> None:
    record = logging.LogRecord(
        "myra.api",
        logging.INFO,
        __file__,
        1,
        "provider failed for owner@example.com",
        (),
        None,
    )
    formatted = json.loads(JsonFormatter().format(record))
    assert "owner@example.com" not in formatted["message"]
    assert "[EMAIL REDACTED]" in formatted["message"]
