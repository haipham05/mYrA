import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from app.db.compatibility import (
    IncompatibleSchemaError,
    check_schema_compatibility,
    get_schema_revisions,
)
from app.main import app


def test_schema_compatibility_unmigrated_fails():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        with pytest.raises(IncompatibleSchemaError, match="no applied migrations"):
            check_schema_compatibility(engine)


def test_schema_compatibility_outdated_revision_fails():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        engine = create_engine(f"sqlite:///{tmp.name}")
        with engine.connect() as conn:
            conn.execute(
                text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL PRIMARY KEY)")
            )
            conn.execute(text("INSERT INTO alembic_version VALUES ('outdated_old_rev_123')"))
            conn.commit()

        with pytest.raises(IncompatibleSchemaError, match="is incompatible"):
            check_schema_compatibility(engine)


def test_schema_compatibility_with_head_passes():
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        ini_path = Path.cwd() / "alembic.ini"
        if not ini_path.exists():
            ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
        cfg = Config(str(ini_path))
        cfg.set_main_option("sqlalchemy.url", f"sqlite:///{tmp.name}")
        command.upgrade(cfg, "head")

        engine = create_engine(f"sqlite:///{tmp.name}")
        # Should not raise
        check_schema_compatibility(engine)

        current_rev, heads = get_schema_revisions(engine)
        assert current_rev is not None
        assert current_rev in heads


def test_health_readiness_probe_success():
    client = TestClient(app)
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["database"] == "connected"
    assert data["storage"] == "available"


def test_health_readiness_probe_database_failure():
    client = TestClient(app)
    with patch("app.db.session.engine.connect", side_effect=RuntimeError("DB down")):
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["status"] == "unready"


def test_health_readiness_probe_storage_failure():
    client = TestClient(app)
    with patch("app.storage.local.LocalStorage.exists", side_effect=RuntimeError("Storage down")):
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["status"] == "unready"


def test_find_alembic_ini_not_found():
    from app.db.compatibility import find_alembic_ini

    with pytest.raises(FileNotFoundError, match="not found at"):
        find_alembic_ini("/nonexistent/path/alembic.ini")
