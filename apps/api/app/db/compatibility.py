import logging
from pathlib import Path

from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy.engine import Connection, Engine

logger = logging.getLogger("myra.db.compatibility")


class IncompatibleSchemaError(RuntimeError):
    """Raised when the database schema revision does not match the application head."""


def find_alembic_ini(custom_path: str | Path | None = None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Alembic config not found at: {custom_path}")

    candidates = [
        Path.cwd() / "alembic.ini",
        Path(__file__).resolve().parents[2] / "alembic.ini",
        Path.cwd() / "apps" / "api" / "alembic.ini",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate alembic.ini in standard search paths")


def get_schema_revisions(
    engine_or_conn: Engine | Connection,
    alembic_ini_path: str | Path | None = None,
) -> tuple[str | None, list[str]]:
    """Returns (current_db_revision, expected_head_revisions)."""
    ini_path = find_alembic_ini(alembic_ini_path)
    cfg = Config(str(ini_path))
    script_loc = cfg.get_main_option("script_location")
    if not script_loc:
        cfg.set_main_option("script_location", str(ini_path.parent / "alembic"))
    elif not Path(script_loc).is_absolute():
        cfg.set_main_option(
            "script_location",
            str((ini_path.parent / script_loc).resolve()),
        )

    script = ScriptDirectory.from_config(cfg)
    expected_heads = script.get_heads()

    if isinstance(engine_or_conn, Engine):
        with engine_or_conn.connect() as conn:
            ctx = MigrationContext.configure(conn)
            current_rev = ctx.get_current_revision()
    else:
        ctx = MigrationContext.configure(engine_or_conn)
        current_rev = ctx.get_current_revision()

    return current_rev, expected_heads


def check_schema_compatibility(
    engine_or_conn: Engine | Connection,
    alembic_ini_path: str | Path | None = None,
) -> None:
    """Verifies that the database has the expected head revision.

    Raises IncompatibleSchemaError if the current revision does not match the expected head.
    """
    current_rev, expected_heads = get_schema_revisions(engine_or_conn, alembic_ini_path)

    if not current_rev:
        raise IncompatibleSchemaError(
            "Database has no applied migrations. "
            f"Expected schema revision: {expected_heads}. "
            "Please run 'alembic upgrade head' before starting services."
        )

    if current_rev not in expected_heads:
        raise IncompatibleSchemaError(
            f"Database schema revision '{current_rev}' is incompatible with "
            f"application expected head revision {expected_heads}. "
            "Please run 'alembic upgrade head' to align schemas."
        )

    logger.info(
        "schema_compatibility_verified",
        extra={"current_revision": current_rev, "expected_heads": expected_heads},
    )
