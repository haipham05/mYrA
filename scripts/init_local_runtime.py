"""Create the ignored, idempotent credentials for the local PostgreSQL profile."""

from __future__ import annotations

import os
import secrets
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_DIR = ROOT / ".local" / "runtime"


def _read_password(path: Path) -> str | None:
    if not path.exists():
        return None

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("POSTGRES_PASSWORD="):
            password = line.partition("=")[2]
            if password:
                return password
    return None


def _atomic_write(path: Path, contents: str) -> None:
    """Write a mode-0600 file through a private temporary file in the same directory."""
    fd, temporary_name = tempfile.mkstemp(prefix=".local-runtime-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(contents)
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _ensure_env_setting(path: Path, key: str, setting: str) -> None:
    """Add a missing generated setting without overwriting the owner's profile choices."""
    current = path.read_text(encoding="utf-8") if path.exists() else ""
    if any(line.startswith(f"{key}=") for line in current.splitlines()):
        return
    updated = f"{current.rstrip()}\n{setting}\n" if current.strip() else f"{setting}\n"
    _atomic_write(path, updated)


def ensure_local_credentials(runtime_dir: Path = DEFAULT_RUNTIME_DIR) -> tuple[Path, Path]:
    """Create or preserve PostgreSQL credentials without reading or editing root `.env`."""
    runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(runtime_dir, 0o700)
    files_dir = runtime_dir / "files"
    files_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(files_dir, 0o700)

    database_file = runtime_dir / "local-postgres.env"
    application_file = runtime_dir / "local-app.env"
    password = _read_password(database_file) or secrets.token_urlsafe(32)

    database_contents = "\n".join(
        (
            "POSTGRES_DB=myra",
            "POSTGRES_USER=myra",
            f"POSTGRES_PASSWORD={password}",
            "",
        )
    )
    local_database_url = (
        f"postgresql+psycopg2://myra:{password}@myra-local-postgres:5432/myra"
    )
    application_contents = "\n".join(
        (
            "MYRA_RUNTIME_PROFILE=local",
            "MYRA_LOCAL_STORAGE_ROOT=/app/data/storage",
            f"DATABASE_URL={local_database_url}",
            f"MYRA_BUDGET_DATABASE_URL={local_database_url}",
            "",
        )
    )

    # Preserve an existing database password so a repeated setup cannot invalidate its volume.
    if not database_file.exists():
        _atomic_write(database_file, database_contents)
    elif not _read_password(database_file):
        raise ValueError("Local PostgreSQL credentials exist but contain no password")

    if not application_file.exists():
        _atomic_write(application_file, application_contents)
    _ensure_env_setting(
        application_file,
        "MYRA_BUDGET_DATABASE_URL",
        f"MYRA_BUDGET_DATABASE_URL={local_database_url}",
    )

    return database_file, application_file


def main() -> None:
    database_file, application_file = ensure_local_credentials()
    print(f"Local PostgreSQL credentials are ready: {database_file.relative_to(ROOT)}")
    print(f"Local application profile is ready: {application_file.relative_to(ROOT)}")
    print("Secret values were not displayed; root .env was not read or changed.")


if __name__ == "__main__":
    main()
