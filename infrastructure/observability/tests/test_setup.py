from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from infrastructure.observability.setup import setup


def test_setup_creates_private_files_and_preserves_credentials_on_rerun(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "private-state"

    setup(state_dir)
    credential_path = state_dir / "credentials.json"
    retention_path = state_dir / "retention" / "retention-state.json"
    first_credentials = credential_path.read_bytes()
    state = json.loads(first_credentials)
    personal_key = state["personal_project"]["secret_key"]
    first_app_env = (state_dir / "app.env").read_text(encoding="utf-8")
    first_retention_state = retention_path.read_bytes()

    setup(state_dir)

    assert credential_path.read_bytes() == first_credentials
    assert retention_path.read_bytes() == first_retention_state
    assert json.loads(first_retention_state) == {"format_version": 1, "projects": {}}
    assert (state_dir / "app.env").read_text(encoding="utf-8") == first_app_env
    assert personal_key in first_app_env
    assert stat.S_IMODE(state_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(retention_path.parent.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600 for path in state_dir.iterdir() if path.is_file()
    )


def test_setup_does_not_create_synthetic_project_credentials(tmp_path: Path) -> None:
    state_dir = tmp_path / "private-state"
    setup(state_dir)
    assert not (state_dir / "synthetic-project.env").exists()


def test_setup_migrates_legacy_retention_state_into_separate_mount_directory(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "private-state"
    state_dir.mkdir()
    legacy = state_dir / "retention-state.json"
    legacy_state = {
        "format_version": 1,
        "projects": {"personal": {"last_success_at": "2026-10-01T12:00:00Z"}},
    }
    legacy.write_text(json.dumps(legacy_state), encoding="utf-8")

    setup(state_dir)

    migrated = state_dir / "retention" / "retention-state.json"
    assert json.loads(migrated.read_text(encoding="utf-8")) == legacy_state
    assert legacy.read_text(encoding="utf-8") == json.dumps(legacy_state)


def test_setup_refuses_symlink_state_or_output(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked_state = tmp_path / "linked-state"
    linked_state.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink observability directory"):
        setup(linked_state)

    output_state = tmp_path / "output-state"
    output_state.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("keep", encoding="utf-8")
    (output_state / "credentials.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink state file"):
        setup(output_state)
    assert outside.read_text(encoding="utf-8") == "keep"


def test_rendered_runtime_configuration_uses_separate_redis_and_disables_telemetry(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "private-state"
    setup(state_dir)
    web_env = (state_dir / "langfuse-web.env").read_text(encoding="utf-8")
    app_env = (state_dir / "app.env").read_text(encoding="utf-8")
    cache_env = (state_dir / "myra-cache.env").read_text(encoding="utf-8")
    langfuse_redis_env = (state_dir / "langfuse-redis.env").read_text(encoding="utf-8")

    assert "TELEMETRY_ENABLED=false" in web_env
    assert "REDIS_HOST=langfuse-redis" in web_env
    assert "myra-cache" in app_env
    assert "MYRA_CACHE_PASSWORD=" in cache_env
    assert "LANGFUSE_REDIS_PASSWORD=" in langfuse_redis_env
    assert "LANGFUSE_REDIS_PASSWORD=" not in cache_env
    assert "MYRA_CACHE_PASSWORD=" not in langfuse_redis_env


def test_setup_leaves_existing_synthetic_credentials_untouched(tmp_path: Path) -> None:
    state_dir = tmp_path / "private-state"
    setup(state_dir)
    synthetic_env = state_dir / "synthetic-project.env"
    synthetic_env.write_text("owner-managed", encoding="utf-8")
    setup(state_dir)
    assert synthetic_env.read_text(encoding="utf-8") == "owner-managed"


def test_cli_does_not_print_generated_secrets(tmp_path: Path) -> None:
    state_dir = tmp_path / "private-state"
    script = Path(__file__).resolve().parents[1] / "setup.py"

    result = subprocess.run(
        [sys.executable, str(script), "--state-dir", str(state_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    secrets_in_state = json.loads((state_dir / "credentials.json").read_text())

    assert "ready" in result.stdout
    assert secrets_in_state["web"]["password"] not in result.stdout
    assert secrets_in_state["personal_project"]["secret_key"] not in result.stdout
    assert not result.stderr
