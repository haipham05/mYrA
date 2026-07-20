"""Apply the current additive schema to the persistent local provider-budget ledger."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine

from app.config import Settings
from app.db.compatibility import check_schema_compatibility
from app.db.session import resolve_budget_database_url


def migrate_budget_database(settings: Settings | None = None) -> None:
    """Upgrade only the explicitly configured local budget database to Alembic head."""
    current_settings = settings or Settings.from_environment()
    database_url = resolve_budget_database_url(current_settings)
    if database_url is None:
        raise ValueError("Select an explicit local or cloud-data profile before migration")

    api_root = Path(__file__).resolve().parents[2]
    config = Config(str(api_root / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    command.upgrade(config, "head")

    migration_engine = create_engine(database_url, pool_pre_ping=True)
    try:
        check_schema_compatibility(migration_engine, api_root / "alembic.ini")
    finally:
        migration_engine.dispose()


def main() -> None:
    migrate_budget_database()
    print("Local provider budget ledger schema is current; no credentials were displayed.")


if __name__ == "__main__":
    main()
