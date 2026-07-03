"""Create private, repeatable local credentials for the observability overlay."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE_DIR = REPOSITORY_ROOT / ".local" / "observability"
STATE_FILE = "credentials.json"


def _random_secret(byte_count: int = 32) -> str:
    return secrets.token_urlsafe(byte_count)


def _new_state() -> dict[str, Any]:
    personal_public = f"pk-lf-{secrets.token_hex(20)}"
    personal_secret = f"sk-lf-{secrets.token_hex(32)}"
    return {
        "format_version": 1,
        "web": {
            "email": "local-admin@myra.invalid",
            "name": "mYrA Local",
            "password": _random_secret(32),
            "nextauth_secret": _random_secret(32),
        },
        "storage": {
            "postgres_password": _random_secret(32),
            "clickhouse_password": _random_secret(32),
            "minio_user": f"myra-{secrets.token_hex(6)}",
            "minio_password": _random_secret(32),
            "salt": _random_secret(32),
            "encryption_key": secrets.token_hex(32),
            "langfuse_redis_password": _random_secret(32),
            "myra_cache_password": _random_secret(32),
        },
        "personal_project": {
            "org_id": "myra-local",
            "org_name": "mYrA Local",
            "project_id": "myra-personal",
            "project_name": "mYrA Personal",
            "public_key": personal_public,
            "secret_key": personal_secret,
        },
    }


def _load_or_create_state(state_dir: Path) -> dict[str, Any]:
    state_path = state_dir / STATE_FILE
    if state_path.is_symlink():
        raise ValueError(f"Refusing symlink state file: {state_path}")
    if state_path.exists():
        if not state_path.is_file():
            raise ValueError(f"Credential state is not a regular file: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("format_version") != 1:
            raise ValueError("Unsupported observability credential state version")
        _require_state_shape(state)
        os.chmod(state_path, 0o600)
        return state

    state = _new_state()
    _write_private_file(state_path, json.dumps(state, indent=2) + "\n")
    return state


def _require_state_shape(state: dict[str, Any]) -> None:
    required = {
        "web": {"email", "name", "password", "nextauth_secret"},
        "storage": {
            "postgres_password",
            "clickhouse_password",
            "minio_user",
            "minio_password",
            "salt",
            "encryption_key",
            "langfuse_redis_password",
            "myra_cache_password",
        },
        "personal_project": {
            "org_id",
            "org_name",
            "project_id",
            "project_name",
            "public_key",
            "secret_key",
        },
    }
    for section, keys in required.items():
        value = state.get(section)
        if not isinstance(value, dict) or not keys.issubset(value):
            raise ValueError(
                f"Invalid observability credential state section: {section}"
            )
        if any(not isinstance(value[key], str) or not value[key] for key in keys):
            raise ValueError(
                f"Empty or invalid value in credential state section: {section}"
            )


def _write_private_file(path: Path, contents: str) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink output file: {path}")
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _env_file(values: dict[str, str]) -> str:
    return "".join(f"{key}={value}\n" for key, value in values.items())


def _render_env_files(state: dict[str, Any]) -> dict[str, str]:
    web = state["web"]
    storage = state["storage"]
    project = state["personal_project"]
    minio_user = storage["minio_user"]
    minio_password = storage["minio_password"]
    redis_password = storage["langfuse_redis_password"]
    common = {
        "NEXTAUTH_URL": "http://127.0.0.1:3001",
        "DATABASE_URL": (
            "postgresql://postgres:"
            f"{storage['postgres_password']}@langfuse-postgres:5432/langfuse"
        ),
        "SALT": storage["salt"],
        "ENCRYPTION_KEY": storage["encryption_key"],
        "TELEMETRY_ENABLED": "false",
        "LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES": "false",
        "CLICKHOUSE_MIGRATION_URL": "clickhouse://langfuse-clickhouse:9000",
        "CLICKHOUSE_URL": "http://langfuse-clickhouse:8123",
        "CLICKHOUSE_USER": "clickhouse",
        "CLICKHOUSE_PASSWORD": storage["clickhouse_password"],
        "CLICKHOUSE_CLUSTER_ENABLED": "false",
        "LANGFUSE_S3_EVENT_UPLOAD_BUCKET": "langfuse",
        "LANGFUSE_S3_EVENT_UPLOAD_REGION": "us-east-1",
        "LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID": minio_user,
        "LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY": minio_password,
        "LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT": "http://langfuse-minio:9000",
        "LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE": "true",
        "LANGFUSE_S3_MEDIA_UPLOAD_BUCKET": "langfuse",
        "LANGFUSE_S3_MEDIA_UPLOAD_REGION": "us-east-1",
        "LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID": minio_user,
        "LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY": minio_password,
        "LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT": "http://langfuse-minio:9000",
        "LANGFUSE_S3_MEDIA_UPLOAD_INTERNAL_ENDPOINT": "http://langfuse-minio:9000",
        "LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE": "true",
        "LANGFUSE_S3_BATCH_EXPORT_ENABLED": "false",
        "REDIS_HOST": "langfuse-redis",
        "REDIS_PORT": "6379",
        "REDIS_AUTH": redis_password,
        "REDIS_TLS_ENABLED": "false",
        "LANGFUSE_BULLMQ_SKIP_REDIS_VERSION_CHECK": "false",
    }
    web_values = {
        **common,
        "NEXTAUTH_SECRET": web["nextauth_secret"],
        "LANGFUSE_INIT_ORG_ID": project["org_id"],
        "LANGFUSE_INIT_ORG_NAME": project["org_name"],
        "LANGFUSE_INIT_PROJECT_ID": project["project_id"],
        "LANGFUSE_INIT_PROJECT_NAME": project["project_name"],
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY": project["public_key"],
        "LANGFUSE_INIT_PROJECT_SECRET_KEY": project["secret_key"],
        "LANGFUSE_INIT_USER_EMAIL": web["email"],
        "LANGFUSE_INIT_USER_NAME": web["name"],
        "LANGFUSE_INIT_USER_PASSWORD": web["password"],
    }
    worker_values = {**common, "NEXTAUTH_URL": "http://langfuse-web:3000"}
    app_values = {
        "LANGFUSE_PUBLIC_KEY": project["public_key"],
        "LANGFUSE_SECRET_KEY": project["secret_key"],
        "MYRA_REDIS_URL": f"redis://:{storage['myra_cache_password']}@myra-cache:6379/0",
    }
    return {
        "langfuse-web.env": _env_file(web_values),
        "langfuse-worker.env": _env_file(worker_values),
        "postgres.env": _env_file(
            {
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": storage["postgres_password"],
                "POSTGRES_DB": "langfuse",
                "TZ": "UTC",
                "PGTZ": "UTC",
            }
        ),
        "clickhouse.env": _env_file(
            {
                "CLICKHOUSE_DB": "default",
                "CLICKHOUSE_USER": "clickhouse",
                "CLICKHOUSE_PASSWORD": storage["clickhouse_password"],
            }
        ),
        "minio.env": _env_file(
            {"MINIO_ROOT_USER": minio_user, "MINIO_ROOT_PASSWORD": minio_password}
        ),
        "langfuse-redis.env": _env_file({"LANGFUSE_REDIS_PASSWORD": redis_password}),
        "myra-cache.env": _env_file(
            {"MYRA_CACHE_PASSWORD": storage["myra_cache_password"]}
        ),
        "app.env": _env_file(app_values),
    }


def setup(state_dir: Path = DEFAULT_STATE_DIR) -> None:
    state_dir = state_dir.expanduser().absolute()
    if state_dir.parent.is_symlink():
        raise ValueError(f"Refusing symlink parent directory: {state_dir.parent}")
    if state_dir.is_symlink():
        raise ValueError(f"Refusing symlink observability directory: {state_dir}")
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(state_dir, 0o700)

    state = _load_or_create_state(state_dir)
    for name, contents in _render_env_files(state).items():
        _write_private_file(state_dir / name, contents)

    synthetic_path = state_dir / "synthetic-project.env"
    if synthetic_path.is_symlink():
        raise ValueError(f"Refusing symlink output file: {synthetic_path}")
    if not synthetic_path.exists():
        _write_private_file(
            synthetic_path,
            "# After creating a synthetic-test project in the local Langfuse UI,\n"
            "# put that project's API keys here. Do not use the personal project keys.\n"
            "LANGFUSE_PUBLIC_KEY=\n"
            "LANGFUSE_SECRET_KEY=\n",
        )

    retention_directory = state_dir / "retention"
    if retention_directory.is_symlink():
        raise ValueError(f"Refusing symlink output directory: {retention_directory}")
    retention_directory.mkdir(mode=0o700, exist_ok=True)
    os.chmod(retention_directory, 0o700)
    retention_path = retention_directory / "retention-state.json"
    if retention_path.is_symlink():
        raise ValueError(f"Refusing symlink output file: {retention_path}")
    if not retention_path.exists():
        legacy_retention_path = state_dir / "retention-state.json"
        if legacy_retention_path.is_symlink():
            raise ValueError(f"Refusing symlink output file: {legacy_retention_path}")
        initial_state: dict[str, Any] = {"format_version": 1, "projects": {}}
        if legacy_retention_path.is_file():
            try:
                legacy_state = json.loads(
                    legacy_retention_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError):
                raise ValueError("Existing retention state is invalid") from None
            if (
                not isinstance(legacy_state, dict)
                or legacy_state.get("format_version") != 1
                or not isinstance(legacy_state.get("projects"), dict)
            ):
                raise ValueError("Existing retention state is invalid")
            initial_state = legacy_state
        _write_private_file(
            retention_path,
            json.dumps(initial_state, indent=2) + "\n",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or preserve private credentials for local Langfuse and Redis."
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()
    setup(arguments.state_dir)
    print("Observability setup files are ready; credentials were not displayed.")
    print(f"Private state directory: {arguments.state_dir.expanduser().absolute()}")


if __name__ == "__main__":
    main()
