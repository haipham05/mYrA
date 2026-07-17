"""Safety checks for idempotent local runtime credential setup."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "init_local_runtime.py"
SPEC = importlib.util.spec_from_file_location("local_runtime_bootstrap", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


def test_credentials_are_private_and_idempotent(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"

    database_file, application_file = BOOTSTRAP.ensure_local_credentials(runtime_dir)
    files_dir = runtime_dir / "files"
    first_password = BOOTSTRAP._read_password(database_file)

    assert first_password
    assert database_file.stat().st_mode & 0o777 == 0o600
    assert application_file.stat().st_mode & 0o777 == 0o600
    assert runtime_dir.stat().st_mode & 0o777 == 0o700
    assert files_dir.is_dir()
    assert files_dir.stat().st_mode & 0o777 == 0o700
    assert "MYRA_LOCAL_STORAGE_ROOT=/app/data/storage" in application_file.read_text()

    BOOTSTRAP.ensure_local_credentials(runtime_dir)

    assert BOOTSTRAP._read_password(database_file) == first_password
    assert f"myra:{first_password}@myra-local-postgres" in application_file.read_text()


def test_application_profile_is_repaired_without_rotating_database_password(
    tmp_path: Path,
) -> None:
    runtime_dir = tmp_path / "runtime"
    database_file, application_file = BOOTSTRAP.ensure_local_credentials(runtime_dir)
    password = BOOTSTRAP._read_password(database_file)
    application_file.write_text("MYRA_RUNTIME_PROFILE=local\n", encoding="utf-8")

    BOOTSTRAP.ensure_local_credentials(runtime_dir)

    assert BOOTSTRAP._read_password(database_file) == password
    assert f"myra:{password}@myra-local-postgres" in application_file.read_text()


def test_invalid_existing_database_secret_is_not_silently_rotated(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    database_file = runtime_dir / "local-postgres.env"
    database_file.write_text("POSTGRES_DB=myra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contain no password"):
        BOOTSTRAP.ensure_local_credentials(runtime_dir)

    assert database_file.read_text(encoding="utf-8") == "POSTGRES_DB=myra\n"
