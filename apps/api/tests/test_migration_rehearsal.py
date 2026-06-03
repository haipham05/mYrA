import tempfile
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text

from app.db.compatibility import check_schema_compatibility, get_schema_revisions


def get_alembic_config(db_url: str) -> Config:
    ini_path = Path.cwd() / "alembic.ini"
    if not ini_path.exists():
        ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
    cfg = Config(str(ini_path))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


def test_empty_database_migration_rehearsal():
    """Prove that an empty disposable database cleanly upgrades through all migrations to head."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)

        # 1. Upgrade from empty to head
        command.upgrade(cfg, "head")

        # 2. Verify head revision reached
        engine = create_engine(db_url)
        current_rev, heads = get_schema_revisions(engine)
        assert current_rev is not None
        assert current_rev in heads

        # 3. Compatibility check passes
        check_schema_compatibility(engine)


def test_seeded_database_migration_rehearsal_and_data_preservation():
    """Prove that upgrading an already-seeded database preserves all existing research rows."""
    with tempfile.NamedTemporaryFile(suffix=".db") as tmp:
        db_url = f"sqlite:///{tmp.name}"
        cfg = get_alembic_config(db_url)

        # 1. Migrate up to intermediate revision (e.g. conversation summary revision)
        command.upgrade(cfg, "a1b2c3d4e5f6")
        engine = create_engine(db_url)

        # 2. Seed data into projects, papers, conversations
        project_id = str(uuid4())
        paper_id = str(uuid4())
        conv_id = str(uuid4())
        with engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO projects (id, name, created_at, updated_at) "
                    "VALUES (:id, :name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"id": project_id, "name": "Seeded Rehearsal Project"},
            )
            conn.execute(
                text(
                    "INSERT INTO papers "
                    "(id, project_id, filename, storage_path, status, created_at, updated_at) "
                    "VALUES (:id, :project_id, :filename, :storage_path, 'READY', "
                    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {
                    "id": paper_id,
                    "project_id": project_id,
                    "filename": "rehearsal.pdf",
                    "storage_path": "papers/rehearsal.pdf",
                },
            )
            conn.execute(
                text(
                    "INSERT INTO conversations "
                    "(id, project_id, title, summary, created_at, updated_at) "
                    "VALUES (:id, :project_id, :title, :summary, "
                    "CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {
                    "id": conv_id,
                    "project_id": project_id,
                    "title": "Seeded Conversation",
                    "summary": "Initial summary",
                },
            )
            conn.commit()

        # 3. Run upgrade to head (additive migration e6f7a8b9c0d1)
        command.upgrade(cfg, "head")

        # 4. Verify compatibility check passes
        check_schema_compatibility(engine)

        # 5. Verify all seeded data is preserved and new column has safe defaults
        with engine.connect() as conn:
            proj_row = conn.execute(
                text("SELECT name FROM projects WHERE id = :id"), {"id": project_id}
            ).fetchone()
            assert proj_row is not None
            assert proj_row[0] == "Seeded Rehearsal Project"

            paper_row = conn.execute(
                text("SELECT filename, status FROM papers WHERE id = :id"), {"id": paper_id}
            ).fetchone()
            assert paper_row is not None
            assert paper_row[0] == "rehearsal.pdf"
            assert paper_row[1] == "READY"

            conv_row = conn.execute(
                text("SELECT title, summary, is_archived FROM conversations WHERE id = :id"),
                {"id": conv_id},
            ).fetchone()
            assert conv_row is not None
            assert conv_row[0] == "Seeded Conversation"
            assert conv_row[1] == "Initial summary"
            # In SQLite / Postgres, default is False (0)
            assert conv_row[2] in (0, False)
