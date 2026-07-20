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


def resolve_budget_database_url(settings: Settings) -> str | None:
    """Keep the spend ledger on the persistent local database across data profiles."""
    if settings.runtime_profile == "auto":
        return None
    database_url = settings.budget_database_url
    if settings.runtime_profile == "local" and database_url is None:
        database_url = resolve_database_url(settings)
    if database_url is None:
        raise ValueError("cloud-data profile requires MYRA_BUDGET_DATABASE_URL")

    parsed = urlsplit(database_url)
    local_hosts = {"localhost", "127.0.0.1", "::1", "myra-local-postgres"}
    if (
        parsed.scheme not in {"postgresql", "postgresql+psycopg2"}
        or parsed.hostname not in local_hosts
    ):
        raise ValueError("MYRA_BUDGET_DATABASE_URL must target local PostgreSQL")
    return database_url


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

_budget_engine: Engine | None = None
_budget_engine_url: str | None = None
_budget_session_factory: sessionmaker | None = None


def get_budget_session_factory(settings: Settings) -> sessionmaker:
    """Return the local persistent ledger session, independent of cloud corpus selection."""
    database_url = resolve_budget_database_url(settings)
    if database_url is None or database_url == DATABASE_URL:
        return SessionLocal

    global _budget_engine, _budget_engine_url, _budget_session_factory
    if _budget_session_factory is None or _budget_engine_url != database_url:
        if _budget_engine is not None:
            _budget_engine.dispose()
        _budget_engine = create_engine(database_url, pool_pre_ping=True)
        _budget_session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_budget_engine,
        )
        _budget_engine_url = database_url
    return _budget_session_factory


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_budget_db() -> Generator[Session, None, None]:
    """Yield a session from the persistent local budget ledger database."""
    factory = get_budget_session_factory(Settings.from_environment())
    db = factory()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Ensure tables exist for local development or testing."""
    Base.metadata.create_all(bind=engine)
