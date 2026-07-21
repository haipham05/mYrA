from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text


def test_conversation_scope_migration_preserves_existing_project_wide_conversations():
    with NamedTemporaryFile(suffix=".db") as tmp:
        database_url = f"sqlite:///{tmp.name}"
        config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
        config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(config, "m3a4b5c6d7e8")
        engine = create_engine(database_url)
        project_id = uuid4().hex
        conversation_id = uuid4().hex
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO projects (id, name, created_at, updated_at) "
                    "VALUES (:id, 'Legacy', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"id": project_id},
            )
            connection.execute(
                text(
                    "INSERT INTO conversations (id, project_id, created_at, updated_at) "
                    "VALUES (:id, :project_id, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"id": conversation_id, "project_id": project_id},
            )

        command.upgrade(config, "head")
        with engine.connect() as connection:
            row = connection.execute(
                text("SELECT paper_scope, selected_paper_ids FROM conversations WHERE id = :id"),
                {"id": conversation_id},
            ).one()
        assert row.paper_scope == "project"
        assert row.selected_paper_ids is None
        engine.dispose()
