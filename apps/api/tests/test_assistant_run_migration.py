from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


def test_assistant_run_migration_is_additive_and_creates_durable_tables() -> None:
    with NamedTemporaryFile(suffix=".db") as tmp:
        database_url = f"sqlite:///{tmp.name}"
        config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
        config.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(config, "n4b6c8d0e2f3")
        engine = create_engine(database_url)
        project_id = uuid4().hex
        conversation_id = uuid4().hex
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO projects (id, name, created_at, updated_at) "
                    "VALUES (:id, 'Preserved', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
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
        inspector = inspect(engine)
        assert {
            "assistant_runs",
            "assistant_run_steps",
            "assistant_approval_actions",
        }.issubset(inspector.get_table_names())
        assert "assistant_run_id" in {
            column["name"] for column in inspector.get_columns("messages")
        }
        assert "provider_usage" in {column["name"] for column in inspector.get_columns("messages")}
        with engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT name FROM projects WHERE id = :id"), {"id": project_id}
                ).scalar_one()
                == "Preserved"
            )
            assert (
                connection.execute(
                    text("SELECT paper_scope FROM conversations WHERE id = :id"),
                    {"id": conversation_id},
                ).scalar_one()
                == "project"
            )
        engine.dispose()
