from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from alembic import command
from alembic.config import Config
from sqlalchemy import MetaData, Table, create_engine, insert, inspect, select, text


def _alembic_config(database_url: str) -> Config:
    ini_path = Path(__file__).resolve().parents[1] / "alembic.ini"
    config = Config(str(ini_path))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_paper_metadata_migration_preserves_legacy_rows_and_allows_duplicate_ids():
    with NamedTemporaryFile(suffix=".db") as tmp:
        database_url = f"sqlite:///{tmp.name}"
        config = _alembic_config(database_url)
        command.upgrade(config, "l2c4e6a8b0d1")
        engine = create_engine(database_url)
        legacy_project_id = uuid4()
        second_project_id = uuid4()
        legacy_paper_id = uuid4()
        second_paper_id = uuid4()
        metadata = MetaData()

        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO projects (id, name, created_at, updated_at) "
                    "VALUES (:id, :name, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {"id": legacy_project_id.hex, "name": "Legacy project"},
            )
            connection.execute(
                text(
                    "INSERT INTO papers (id, project_id, filename, storage_path, status, "
                    "created_at, updated_at) VALUES (:id, :project_id, :filename, :path, "
                    "'READY', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                ),
                {
                    "id": legacy_paper_id.hex,
                    "project_id": legacy_project_id.hex,
                    "filename": "legacy.pdf",
                    "path": "papers/legacy.pdf",
                },
            )

        command.upgrade(config, "head")
        papers = Table("papers", metadata, autoload_with=engine)
        columns = {column["name"] for column in inspect(engine).get_columns("papers")}
        expected_metadata_columns = {
            "title",
            "authors",
            "publication_year",
            "doi",
            "arxiv_id",
            "abstract",
            "source_url",
            "metadata_provenance",
        }
        assert expected_metadata_columns <= columns

        with engine.begin() as connection:
            legacy_row = (
                connection.execute(select(papers).where(papers.c.id == legacy_paper_id))
                .mappings()
                .one()
            )
            assert legacy_row["filename"] == "legacy.pdf"
            assert legacy_row["storage_path"] == "papers/legacy.pdf"
            assert legacy_row["status"] == "READY"
            assert all(legacy_row[name] is None for name in expected_metadata_columns)

            projects = Table("projects", metadata, autoload_with=connection)
            connection.execute(
                insert(projects).values(
                    id=second_project_id.hex,
                    name="Second project",
                    created_at=legacy_row["created_at"],
                    updated_at=legacy_row["updated_at"],
                )
            )
            connection.execute(
                insert(papers).values(
                    id=second_paper_id.hex,
                    project_id=second_project_id.hex,
                    filename="same-work.pdf",
                    storage_path="papers/same-work.pdf",
                    status="READY",
                    created_at=legacy_row["created_at"],
                    updated_at=legacy_row["updated_at"],
                    title="Attention Is All You Need",
                    authors=["Ashish Vaswani", "Noam Shazeer"],
                    publication_year=2017,
                    doi="10.48550/arXiv.1706.03762",
                    arxiv_id="1706.03762",
                    abstract="A paper about attention-based sequence models.",
                    source_url="https://arxiv.org/abs/1706.03762",
                    metadata_provenance={"title": "manual", "abstract": "catalog"},
                )
            )
            # Identifiers are not unique constraints or implicit merge instructions.
            duplicate_paper_id = uuid4()
            connection.execute(
                insert(papers).values(
                    id=duplicate_paper_id.hex,
                    project_id=legacy_project_id.hex,
                    filename="duplicate-entry.pdf",
                    storage_path="papers/duplicate-entry.pdf",
                    status="READY",
                    created_at=legacy_row["created_at"],
                    updated_at=legacy_row["updated_at"],
                    doi="10.48550/arXiv.1706.03762",
                    arxiv_id="1706.03762",
                )
            )

            stored = (
                connection.execute(select(papers).where(papers.c.id == second_paper_id))
                .mappings()
                .one()
            )
            assert stored["authors"] == ["Ashish Vaswani", "Noam Shazeer"]
            assert stored["metadata_provenance"] == {
                "title": "manual",
                "abstract": "catalog",
            }
            duplicate_count = connection.execute(
                select(papers.c.id).where(papers.c.doi == "10.48550/arXiv.1706.03762")
            ).all()
            assert len(duplicate_count) == 2

        engine.dispose()
