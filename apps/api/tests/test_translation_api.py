import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.crud.translation import claim_next_translation, save_translation_segment
from app.db.base import Base
from app.db.models import Paper, PaperPage, Project, TranslationDocument
from app.db.session import get_db
from app.main import app


@pytest.fixture
def translation_client(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'translations.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with session_factory() as db:
        project = Project(name="Translation test")
        db.add(project)
        db.flush()
        paper = Paper(
            project_id=project.id,
            filename="paper.pdf",
            storage_path="papers/paper.pdf",
            document_sha256="a" * 64,
            status="READY",
            page_count=3,
        )
        foreign_project = Project(name="Other project")
        db.add_all([paper, foreign_project])
        db.commit()
        project_id = project.id
        paper_id = paper.id
        foreign_project_id = foreign_project.id

    def override_get_db():
        with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    try:
        with TestClient(app) as client:
            yield client, project_id, paper_id, foreign_project_id, session_factory
    finally:
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def test_translation_requires_external_disclosure_and_project_ready_paper(translation_client):
    client, project_id, paper_id, foreign_project_id, session_factory = translation_client
    payload = {"project_id": str(project_id), "acknowledge_external_processing": False}
    assert client.post(f"/api/v1/papers/{paper_id}/translations", json=payload).status_code == 409

    payload["acknowledge_external_processing"] = True
    payload["idempotency_key"] = "test-request"
    payload["project_id"] = str(foreign_project_id)
    assert client.post(f"/api/v1/papers/{paper_id}/translations", json=payload).status_code == 404

    payload["project_id"] = str(project_id)
    with session_factory() as db:
        paper = db.get(Paper, paper_id)
        paper.status = "PROCESSING"
        db.commit()
    assert client.post(f"/api/v1/papers/{paper_id}/translations", json=payload).status_code == 409


def test_translation_request_is_idempotent_and_project_scoped(translation_client):
    client, project_id, paper_id, foreign_project_id, _ = translation_client
    payload = {
        "project_id": str(project_id),
        "acknowledge_external_processing": True,
        "idempotency_key": "same-request",
    }
    first = client.post(f"/api/v1/papers/{paper_id}/translations", json=payload)
    second = client.post(f"/api/v1/papers/{paper_id}/translations", json=payload)
    assert first.status_code == 202
    assert second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert first.json()["source_sha256"] == "a" * 64

    assert (
        client.get(
            f"/api/v1/translations/{first.json()['id']}", params={"project_id": foreign_project_id}
        ).status_code
        == 404
    )
    listed = client.get(
        f"/api/v1/papers/{paper_id}/translations", params={"project_id": project_id}
    )
    assert listed.status_code == 200
    assert len(listed.json()["items"]) == 1


def test_translation_checkpoints_are_idempotent_and_attempt_fenced(translation_client):
    client, project_id, paper_id, _, session_factory = translation_client
    created = client.post(
        f"/api/v1/papers/{paper_id}/translations",
        json={
            "project_id": str(project_id),
            "acknowledge_external_processing": True,
            "idempotency_key": "checkpoint-test",
        },
    )
    translation_id = created.json()["id"]
    with session_factory() as db:
        db.add(
            PaperPage(
                paper_id=paper_id,
                page_number=1,
                width=100,
                height=100,
                raw_text="Original source quote.",
            )
        )
        db.commit()
        claimed = claim_next_translation(db, worker_id="worker-a")
        assert claimed is not None
        token = claimed.attempt_token
        assert save_translation_segment(
            db,
            claimed.id,
            worker_id="worker-a",
            attempt_token=token,
            engine_checkpoint_key="1" * 64,
            ordinal=0,
            source_page_number=1,
            source_text_hash="c" * 64,
            source_quote="Original source quote.",
            translated_text="Bản dịch.",
            translated_text_hash="d" * 64,
        )
        assert not save_translation_segment(
            db,
            claimed.id,
            worker_id="worker-a",
            attempt_token="stale-attempt",
            engine_checkpoint_key="2" * 64,
            ordinal=1,
            source_page_number=1,
            source_text_hash="e" * 64,
            source_quote="Another source quote.",
            translated_text="Câu khác.",
            translated_text_hash="f" * 64,
        )
    status = client.get(f"/api/v1/translations/{translation_id}", params={"project_id": project_id})
    assert status.json()["completed_units"] == 1
    assert status.json()["segments"][0]["translated_text"] == "Bản dịch."
    source_map = client.get(
        f"/api/v1/translations/{translation_id}/source-map",
        params={"project_id": project_id},
    ).json()
    assert source_map["segments"][0]["anchor_status"] == "verified"
    assert source_map["segments"][0]["source_char_start"] == 0
    assert source_map["segments"][0]["source_char_end"] == 22
    with session_factory() as db:
        paper = db.get(Paper, paper_id)
        paper.document_sha256 = "f" * 64
        db.commit()
    stale_map = client.get(
        f"/api/v1/translations/{translation_id}/source-map",
        params={"project_id": project_id},
    ).json()
    assert stale_map["segments"][0]["anchor_status"] == "page_only"
    assert stale_map["segments"][0]["highlight_label"] == "Exact highlight unavailable"


def test_translation_glossary_is_project_scoped_and_replaced_atomically(translation_client):
    client, project_id, _, foreign_project_id, _ = translation_client
    url = f"/api/v1/projects/{project_id}/translation-glossary"
    body = {"entries": [{"source_term": "attention", "preferred_translation": "chú ý"}]}
    saved = client.put(url, json=body)
    assert saved.status_code == 200
    assert saved.json()["entries"][0]["preferred_translation"] == "chú ý"

    duplicate = client.put(
        url,
        json={
            "entries": [
                {"source_term": "attention", "preferred_translation": "chú ý"},
                {"source_term": "ATTENTION", "preferred_translation": "sự chú ý"},
            ]
        },
    )
    assert duplicate.status_code == 422
    assert (
        client.get(f"/api/v1/projects/{foreign_project_id}/translation-glossary").json()["entries"]
        == []
    )


def test_translation_pdf_supports_inline_preview_and_project_scope(translation_client, monkeypatch):
    client, project_id, paper_id, foreign_project_id, session_factory = translation_client
    with session_factory() as db:
        translation = TranslationDocument(
            project_id=project_id,
            paper_id=paper_id,
            status="COMPLETED",
            stage="COMPLETED",
            idempotency_key="completed-preview",
            acknowledge_external_processing=True,
            source_sha256="a" * 64,
            source_storage_path="papers/paper.pdf",
            source_filename='paper".pdf',
            output_storage_path="translations/result.pdf",
            output_sha256="b" * 64,
        )
        db.add(translation)
        db.commit()
        translation_id = translation.id

    class FakeStorage:
        async def exists(self, key):
            return key == "translations/result.pdf"

        async def open_stream(self, key):
            yield b"%PDF test"

    monkeypatch.setattr("app.api.v1.translations.get_storage", lambda: FakeStorage())
    base_url = f"/api/v1/translations/{translation_id}/pdf"
    download = client.get(base_url, params={"project_id": project_id})
    preview = client.get(base_url, params={"project_id": project_id, "inline": "true"})
    foreign = client.get(base_url, params={"project_id": foreign_project_id})
    assert download.status_code == preview.status_code == 200
    assert download.headers["content-disposition"].startswith("attachment;")
    assert preview.headers["content-disposition"].startswith("inline;")
    assert 'paper".pdf' not in download.headers["content-disposition"]
    assert foreign.status_code == 404


def test_migration_adds_translation_tables_and_rolls_forward(tmp_path):
    from alembic import command
    from alembic.config import Config
    from sqlalchemy import inspect

    db_path = tmp_path / "translation-migration.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(config, "head")
    engine = create_engine(f"sqlite:///{db_path}")
    tables = set(inspect(engine).get_table_names())
    assert {
        "translation_documents",
        "translation_segments",
        "project_translation_glossary_entries",
    } <= tables
    engine.dispose()
