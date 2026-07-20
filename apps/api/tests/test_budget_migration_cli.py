import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.cli import migrate_budget
from app.config import Settings
from app.db import compatibility
from app.db import session as database_session
from app.db.compatibility import IncompatibleSchemaError, check_schema_compatibility


def test_migrate_budget_database_upgrades_only_resolved_url(tmp_path, monkeypatch):
    database_url = f"sqlite:///{tmp_path / 'local-budget.sqlite'}"
    monkeypatch.setenv("MYRA_RUNTIME_PROFILE", "auto")
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setattr(
        migrate_budget,
        "resolve_budget_database_url",
        lambda _settings: database_url,
    )

    migrate_budget.migrate_budget_database(Settings(runtime_profile="auto"))

    engine = create_engine(database_url)
    try:
        check_schema_compatibility(engine)
        settings = Settings(
            runtime_profile="cloud-data",
            database_url="postgresql+psycopg2://owner:example@cloud-db.example:5432/myra",
            budget_database_url=(
                "postgresql+psycopg2://myra:example@myra-local-postgres:5432/myra"
            ),
            gcs_bucket_name="research-bucket",
        )
        monkeypatch.setattr(
            database_session,
            "get_budget_session_factory",
            lambda _settings: sessionmaker(bind=engine),
        )
        compatibility.check_budget_schema_compatibility(settings)
    finally:
        engine.dispose()


def test_cloud_profile_readiness_checks_separate_budget_schema(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'unmigrated-budget.sqlite'}")
    factory = sessionmaker(bind=engine)
    settings = Settings(
        runtime_profile="cloud-data",
        database_url="postgresql+psycopg2://owner:example@cloud-db.example:5432/myra",
        budget_database_url=("postgresql+psycopg2://myra:example@myra-local-postgres:5432/myra"),
        gcs_bucket_name="research-bucket",
    )
    monkeypatch.setattr(database_session, "get_budget_session_factory", lambda _settings: factory)

    with pytest.raises(IncompatibleSchemaError, match="no applied migrations"):
        compatibility.check_budget_schema_compatibility(settings)

    engine.dispose()
