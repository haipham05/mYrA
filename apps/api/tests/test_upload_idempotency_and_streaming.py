import io
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from app.db.base import Base
from app.db.models import Paper
from app.db.session import SessionLocal, engine
from app.main import app
from app.schemas.paper import PaperStatus


def _make_pdf(text: str = "Test Content", pages: int = 1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_upload_paper_and_idempotency():
    client = TestClient(app)

    # 1. Create project
    p_res = client.post("/api/v1/projects", json={"name": "Idempotency Project"})
    assert p_res.status_code == 201
    project_id = p_res.json()["id"]

    pdf_bytes = _make_pdf("Idempotent PDF v1")
    file_payload = {"file": ("paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")}

    # Initial upload -> 202 ACCEPTED
    res1 = client.post(
        f"/api/v1/projects/{project_id}/papers",
        files=file_payload,
        headers={"Idempotency-Key": "test-key-1"},
    )
    assert res1.status_code == 202
    data1 = res1.json()
    paper_id = data1["paper_id"]
    job_id = data1["job_id"]
    assert data1["status"] == "PROCESSING"

    # Concurrent / repeated upload while PROCESSING -> 202 with SAME paper and job
    file_payload2 = {"file": ("paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    res2 = client.post(
        f"/api/v1/projects/{project_id}/papers",
        files=file_payload2,
        headers={"Idempotency-Key": "test-key-1"},
    )
    assert res2.status_code == 202
    data2 = res2.json()
    assert data2["paper_id"] == paper_id
    assert data2["job_id"] == job_id

    # Mark paper READY in database
    with SessionLocal() as db:
        paper = db.query(Paper).filter(Paper.id == UUID(paper_id)).first()
        paper.status = PaperStatus.READY
        db.commit()

    # Repeated upload after READY -> 200 OK with SAME paper_id and status READY
    file_payload3 = {"file": ("paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    res3 = client.post(
        f"/api/v1/projects/{project_id}/papers",
        files=file_payload3,
    )
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["paper_id"] == paper_id
    assert data3["status"] == "READY"


def test_upload_failed_paper_retries_idempotently():
    client = TestClient(app)
    p_res = client.post("/api/v1/projects", json={"name": "Retry Project"})
    project_id = p_res.json()["id"]

    pdf_bytes = _make_pdf("Retry PDF")
    res1 = client.post(
        f"/api/v1/projects/{project_id}/papers",
        files={"file": ("paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert res1.status_code == 202
    paper_id = res1.json()["paper_id"]
    job_id1 = res1.json()["job_id"]

    # Mark paper FAILED
    with SessionLocal() as db:
        paper = db.query(Paper).filter(Paper.id == UUID(paper_id)).first()
        paper.status = PaperStatus.FAILED
        paper.error_message = "Simulated failure"
        db.commit()

    # Re-upload should safely re-queue a new job
    res2 = client.post(
        f"/api/v1/projects/{project_id}/papers",
        files={"file": ("paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert res2.status_code == 202
    data2 = res2.json()
    assert data2["paper_id"] == paper_id
    assert data2["job_id"] != job_id1
    assert data2["status"] == "PROCESSING"


def test_same_filename_different_bytes_creates_distinct_papers():
    client = TestClient(app)
    p_res = client.post("/api/v1/projects", json={"name": "Distinct Versions"})
    project_id = p_res.json()["id"]

    pdf1 = _make_pdf("Version 1", pages=1)
    pdf2 = _make_pdf("Version 2", pages=2)

    res1 = client.post(
        f"/api/v1/projects/{project_id}/papers",
        files={"file": ("doc.pdf", io.BytesIO(pdf1), "application/pdf")},
    )
    res2 = client.post(
        f"/api/v1/projects/{project_id}/papers",
        files={"file": ("doc.pdf", io.BytesIO(pdf2), "application/pdf")},
    )
    assert res1.status_code == 202
    assert res2.status_code == 202
    assert res1.json()["paper_id"] != res2.json()["paper_id"]


def test_document_streaming_and_scoping():
    client = TestClient(app)
    p_res = client.post("/api/v1/projects", json={"name": "Streaming Project"})
    project_id = p_res.json()["id"]

    pdf_bytes = _make_pdf("Streamable PDF", pages=1)
    upload_res = client.post(
        f"/api/v1/projects/{project_id}/papers",
        files={"file": ("stream.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    paper_id = upload_res.json()["paper_id"]

    # Stream from /papers/{id}/document
    stream_res1 = client.get(f"/api/v1/papers/{paper_id}/document")
    assert stream_res1.status_code == 200
    assert stream_res1.headers["content-type"] == "application/pdf"
    assert "ETag" in stream_res1.headers
    assert stream_res1.content == pdf_bytes

    # Stream from /projects/{project_id}/papers/{paper_id}/document
    stream_res2 = client.get(f"/api/v1/projects/{project_id}/papers/{paper_id}/document")
    assert stream_res2.status_code == 200
    assert stream_res2.content == pdf_bytes

    # Stream with mismatched project_id returns 404
    wrong_project_id = str(uuid4())
    stream_wrong = client.get(f"/api/v1/projects/{wrong_project_id}/papers/{paper_id}/document")
    assert stream_wrong.status_code == 404


def test_upload_storage_compensation_on_db_failure():
    client = TestClient(app)
    p_res = client.post("/api/v1/projects", json={"name": "Compensation Project"})
    project_id = p_res.json()["id"]

    pdf_bytes = _make_pdf("Compensate PDF")
    with patch(
        "app.api.v1.projects.create_paper_with_job",
        side_effect=RuntimeError("DB Insert Failed"),
    ):
        res = client.post(
            f"/api/v1/projects/{project_id}/papers",
            files={"file": ("compensate.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert res.status_code == 500
        assert "Failed to initialize paper record" in res.json()["detail"]
