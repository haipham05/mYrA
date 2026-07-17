from collections.abc import Generator
from typing import Any
from urllib.parse import urlsplit

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.db.base import Base

settings = Settings.from_environment()


def resolve_database_url(settings: Settings) -> str:
    """Resolve an explicit profile without silently crossing local/cloud boundaries."""
    if settings.runtime_profile == "local":
        if not settings.database_url:
            raise ValueError("local profile requires DATABASE_URL")
        parsed = urlsplit(settings.database_url)
        local_hosts = {"localhost", "127.0.0.1", "::1", "myra-local-postgres"}
        if not parsed.scheme.startswith("postgresql") or parsed.hostname not in local_hosts:
            raise ValueError("local profile DATABASE_URL must target the local PostgreSQL service")
        return settings.database_url

    if settings.runtime_profile == "cloud-data":
        # Settings rejects missing cloud values; never fall back to SQLite in this profile.
        if not settings.database_url:
            raise ValueError("cloud-data profile requires DATABASE_URL")
        parsed = urlsplit(settings.database_url)
        if parsed.scheme not in {"postgresql", "postgresql+psycopg2"} or not parsed.hostname:
            raise ValueError("cloud-data profile DATABASE_URL must target PostgreSQL")
        return settings.database_url

    return settings.database_url or "sqlite:///./myra_dev.db"


DATABASE_URL = resolve_database_url(settings)

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=(settings.log_level == "DEBUG"),
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    if type(dbapi_connection).__module__ == "sqlite3":
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Ensure tables exist for local development or testing."""
    Base.metadata.create_all(bind=engine)
