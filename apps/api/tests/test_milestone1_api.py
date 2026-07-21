import io
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.crud.paper import create_paper, create_paper_with_job
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.services.ingestion import IngestionPipeline
from app.services.llm import FakeLLMProvider, set_llm_provider
from app.storage.factory import set_storage
from app.storage.local import MemoryStorage


def create_sample_pdf() -> bytes:
    return b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj << /Length 55 >> stream
BT
/F1 12 Tf
72 712 Td
(Attention Is All You Need. We propose the Transformer.) Tj
ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000244 00000 n 
0000000350 00000 n 
trailer << /Size 6 /Root 1 0 R >>
startxref
426
%%EOF"""


@pytest.fixture
def test_env(tmp_path):
    # Set up isolated SQLite database on tmp_path
    db_file = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    memory_storage = MemoryStorage()
    set_storage(memory_storage)
    set_llm_provider(FakeLLMProvider())

    def override_get_db():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    yield {
        "engine": engine,
        "session_maker": TestingSession,
        "storage": memory_storage,
    }

    app.dependency_overrides.clear()
    set_storage(None)
    set_llm_provider(None)


def test_projects_crud(test_env):
    with TestClient(app) as client:
        resp = client.post(
            "/api/v1/projects",
            json={"name": "NLP Research", "description": "Transformers"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "NLP Research"
        project_id = data["id"]

        # Get single
        get_resp = client.get(f"/api/v1/projects/{project_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == project_id

        # List
        list_resp = client.get("/api/v1/projects")
        assert list_resp.status_code == 200
        assert list_resp.json()["total"] >= 1
        bad_resp = client.get(f"/api/v1/projects/{uuid4()}")
        assert bad_resp.status_code == 404


def test_paper_metadata_manual_correction_round_trip(test_env):
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "Metadata"}).json()
        db = test_env["session_maker"]()
        try:
            paper, _job = create_paper_with_job(
                db, UUID(project["id"]), "unknown-title.pdf", "papers/unknown-title.pdf"
            )
        finally:
            db.close()

        updated = client.patch(
            f"/api/v1/papers/{paper.id}",
            params={"project_id": project["id"]},
            json={"title": "Corrected title", "authors": ["  Ada Lovelace  ", ""]},
        )
        assert updated.status_code == 200
        payload = updated.json()
        assert payload["title"] == "Corrected title"
        assert payload["authors"] == ["Ada Lovelace"]
        assert payload["metadata_provenance"] == {"title": "manual", "authors": "manual"}
        assert payload["status"] == "PROCESSING"
        assert payload["document_sha256"] is None

        # Partial edits leave unrelated metadata unknown; an explicit null clears a value.
        second = client.patch(
            f"/api/v1/papers/{paper.id}",
            params={"project_id": project["id"]},
            json={"title": None},
        )
        assert second.status_code == 200
        assert second.json()["title"] is None
        assert second.json()["metadata_provenance"]["title"] == "unknown"
        assert second.json()["doi"] is None

        assert (
            client.patch(
                f"/api/v1/papers/{paper.id}",
                params={"project_id": str(UUID(int=0))},
                json={"title": "Must not cross project"},
            ).status_code
            == 404
        )
        assert (
            client.patch(
                f"/api/v1/papers/{paper.id}",
                params={"project_id": project["id"]},
                json={"title": "", "publication_year": 42},
            ).status_code
            == 422
        )


def test_project_paper_search_filters_and_pagination(test_env):
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "Library"}).json()["id"]
        db = test_env["session_maker"]()
        try:
            project_uuid = UUID(project_id)
            papers = [
                create_paper(db, project_uuid, "legacy-notes.pdf", "one", status="READY"),
                create_paper(db, project_uuid, "second.pdf", "two", status="PROCESSING"),
                create_paper(db, project_uuid, "third.pdf", "three", status="READY"),
            ]
            papers[0].title = "A Study of Retrieval"
            papers[0].authors = ["Ada Lovelace", "Alan Turing"]
            papers[0].publication_year = 2024
            db.commit()
            target_paper_id = str(papers[0].id)
        finally:
            db.close()

        filename = client.get(
            f"/api/v1/projects/{project_id}/papers", params={"q": "legacy-notes"}
        ).json()
        assert [paper["filename"] for paper in filename["items"]] == ["legacy-notes.pdf"]
        title = client.get(
            f"/api/v1/projects/{project_id}/papers", params={"q": "Retrieval"}
        ).json()
        assert title["items"][0]["id"] == target_paper_id
        author = client.get(
            f"/api/v1/projects/{project_id}/papers", params={"q": "Lovelace"}
        ).json()
        assert author["items"][0]["id"] == target_paper_id

        filtered = client.get(
            f"/api/v1/projects/{project_id}/papers",
            params={"status": "READY", "year": 2024, "limit": 1, "offset": 0},
        )
        assert filtered.status_code == 200
        assert filtered.json()["total"] == 1
        assert filtered.json()["items"][0]["id"] == target_paper_id
        assert (
            client.get(f"/api/v1/projects/{project_id}/papers", params={"q": "x" * 201}).status_code
            == 422
        )
        assert (
            client.get(f"/api/v1/projects/{project_id}/papers", params={"limit": 101}).status_code
            == 422
        )


def test_conversation_scope_persists_and_rejects_foreign_or_empty_dispatch(test_env):
    with TestClient(app) as client:
        project_id = client.post("/api/v1/projects", json={"name": "Scope owner"}).json()["id"]
        foreign_project_id = client.post("/api/v1/projects", json={"name": "Other project"}).json()[
            "id"
        ]
        db = test_env["session_maker"]()
        try:
            owned_paper = create_paper(db, UUID(project_id), "owned.pdf", "owned", status="READY")
            foreign_paper = create_paper(
                db, UUID(foreign_project_id), "foreign.pdf", "foreign", status="READY"
            )
            owned_paper_id = str(owned_paper.id)
            foreign_paper_id = str(foreign_paper.id)
        finally:
            db.close()

        created = client.post(
            f"/api/v1/projects/{project_id}/conversations",
            json={
                "title": "Scoped",
                "paper_scope": "paper",
                "selected_paper_ids": [owned_paper_id],
            },
        )
        assert created.status_code == 201
        conversation_id = created.json()["id"]
        assert created.json()["paper_scope"] == "paper"
        assert created.json()["selected_paper_ids"] == [owned_paper_id]

        updated = client.patch(
            f"/api/v1/conversations/{conversation_id}?project_id={project_id}",
            json={"paper_scope": "selection", "selected_paper_ids": [owned_paper_id]},
        )
        assert updated.status_code == 200
        listed = client.get(
            f"/api/v1/projects/{project_id}/conversations?include_archived=true"
        ).json()
        assert listed["items"][0]["paper_scope"] == "selection"
        assert listed["items"][0]["selected_paper_ids"] == [owned_paper_id]

        assert (
            client.patch(
                f"/api/v1/conversations/{conversation_id}?project_id={project_id}",
                json={"paper_scope": "paper", "selected_paper_ids": [foreign_paper_id]},
            ).status_code
            == 404
        )
        assert (
            client.patch(
                f"/api/v1/conversations/{conversation_id}?project_id={project_id}",
                json={"paper_scope": "paper", "selected_paper_ids": []},
            ).status_code
            == 422
        )

        empty_selection = client.patch(
            f"/api/v1/conversations/{conversation_id}?project_id={project_id}",
            json={"paper_scope": "selection", "selected_paper_ids": []},
        )
        assert empty_selection.status_code == 200
        refused = client.post(
            f"/api/v1/conversations/{conversation_id}/messages?project_id={project_id}",
            json={"content": "Only search the selected papers"},
        )
        assert refused.status_code == 409
        assert "Select at least one paper" in refused.json()["detail"]
        assert (
            client.get(
                f"/api/v1/conversations/{conversation_id}/messages?project_id={project_id}"
            ).json()
            == []
        )


def test_paper_upload_and_validation(test_env):
    with TestClient(app) as client:
        p_resp = client.post("/api/v1/projects", json={"name": "Vision Research"})
        project_id = p_resp.json()["id"]

        # 1. Invalid extension
        txt_file = io.BytesIO(b"not a pdf")
        bad_ext = client.post(
            f"/api/v1/projects/{project_id}/papers",
            files={"file": ("test.txt", txt_file, "text/plain")},
        )
        assert bad_ext.status_code == 400
        assert "Only PDF files" in bad_ext.json()["detail"]

        # 2. Invalid PDF signature
        fake_pdf = io.BytesIO(b"random bytes that do not start with %PDF")
        bad_sig = client.post(
            f"/api/v1/projects/{project_id}/papers",
            files={"file": ("fake.pdf", fake_pdf, "application/pdf")},
        )
        assert bad_sig.status_code == 400
        assert "Invalid PDF file signature" in bad_sig.json()["detail"]

        # 3. Valid PDF upload
        valid_pdf_bytes = create_sample_pdf()
        valid_file = io.BytesIO(valid_pdf_bytes)
        upload_resp = client.post(
            f"/api/v1/projects/{project_id}/papers",
            files={"file": ("valid.pdf", valid_file, "application/pdf")},
        )
        assert upload_resp.status_code == 202
        upload_data = upload_resp.json()
        assert upload_data["status"] == "PROCESSING"
        paper_id = upload_data["paper_id"]
        job_id = upload_data["job_id"]
        assert paper_id is not None

        # Check job status
        job_resp = client.get(f"/api/v1/jobs/{job_id}")
        assert job_resp.status_code == 200
        assert job_resp.json()["status"] == "PENDING"

        # Check paper in list
        papers_resp = client.get(f"/api/v1/projects/{project_id}/papers")
        assert papers_resp.status_code == 200
        assert papers_resp.json()["total"] == 1
        listed_paper = papers_resp.json()["items"][0]
        assert listed_paper["title"] is None
        assert listed_paper["authors"] is None
        assert listed_paper["metadata_provenance"] is None

        detail_resp = client.get(f"/api/v1/papers/{paper_id}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["filename"] == "valid.pdf"
        assert detail_resp.json()["doi"] is None


@pytest.mark.anyio
async def test_full_pipeline_and_qa_loop(test_env):
    with TestClient(app) as client:
        # 1. Create project
        p_resp = client.post("/api/v1/projects", json={"name": "Attention Paper"})
        project_id = p_resp.json()["id"]

        # 2. Upload paper
        valid_pdf_bytes = create_sample_pdf()
        valid_file = io.BytesIO(valid_pdf_bytes)
        upload_resp = client.post(
            f"/api/v1/projects/{project_id}/papers",
            files={"file": ("attention.pdf", valid_file, "application/pdf")},
        )
        paper_id = upload_resp.json()["paper_id"]
        job_id = upload_resp.json()["job_id"]

        # 3. Run ingestion pipeline on paper
        session = test_env["session_maker"]()
        pipeline = IngestionPipeline()
        await pipeline.process_paper(session, paper_id=UUID(paper_id), job_id=UUID(job_id))
        session.close()

        # 4. Check paper status is READY
        paper_resp = client.get(f"/api/v1/papers/{paper_id}")
        assert paper_resp.status_code == 200
        assert paper_resp.json()["status"] == "READY"
        assert paper_resp.json()["page_count"] == 1

        # 5. Check document streaming endpoint
        doc_resp = client.get(f"/api/v1/papers/{paper_id}/document")
        assert doc_resp.status_code == 200
        assert doc_resp.headers["content-type"] == "application/pdf"
        assert doc_resp.content == valid_pdf_bytes

        # 6. Check debug elements endpoint
        elems_resp = client.get(f"/api/v1/papers/{paper_id}/elements")
        assert elems_resp.status_code == 200
        assert isinstance(elems_resp.json(), list)

        # 7. Create conversation
        conv_resp = client.post(
            f"/api/v1/projects/{project_id}/conversations",
            json={"title": "Q&A"},
        )
        assert conv_resp.status_code == 201
        conv_id = conv_resp.json()["id"]

        # 8. Send question
        msg_resp = client.post(
            f"/api/v1/conversations/{conv_id}/messages",
            json={"content": "What are the main results of the paper?"},
        )
        assert msg_resp.status_code == 200
        msg_data = msg_resp.json()
        assert msg_data["role"] == "ASSISTANT"
        assert len(msg_data["citations"]) > 0
        assert msg_data["citations"][0]["citation_index"] == 1
        assert msg_data["citations"][0]["page_number"] == 1
        assert "[1]" in msg_data["content"]
        assert len(msg_data["evidence"]) > 0

        # 9. Read conversation messages
        hist_resp = client.get(f"/api/v1/conversations/{conv_id}/messages")
        assert hist_resp.status_code == 200
        assert len(hist_resp.json()) == 2  # 1 USER, 1 ASSISTANT
