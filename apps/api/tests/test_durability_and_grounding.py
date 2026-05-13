import io
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.crud.chat import create_conversation
from app.crud.job import claim_next_job, create_job
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.models import ChunkElement, Job, PaperChunk, PaperElement
from app.db.session import SessionLocal, create_tables
from app.main import app
from app.schemas.evidence import AnchorStatus
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.chat_service import ChatService, check_claim_support
from app.services.llm import FakeLLMProvider, set_llm_provider

client = TestClient(app)


def test_claim_support_overlap_function():
    # Valid overlap
    claim = "The Transformer model uses self-attention mechanisms."
    quote = "We introduce the Transformer, architecture relying entirely on self-attention."
    assert check_claim_support(claim, quote) is True

    # Complete mismatch / hallucination
    unrelated_quote = "Baking cookies requires sugar, flour, and butter at 350 degrees."
    assert check_claim_support(claim, unrelated_quote) is False

    # Empty inputs
    assert check_claim_support("", quote) is False
    assert check_claim_support(claim, "") is False


def test_oversized_pdf_upload():
    create_tables()
    db = SessionLocal()
    proj = create_project(db, ProjectCreate(name="Oversized Test"))
    pid = proj.id
    db.close()

    # Upload file exceeding 50MB (mock using custom settings or large bytes)
    # The default limit is 50MB; let's test with a simulated small limit or check response
    from app.api.v1 import projects as proj_module

    old_limit = proj_module.settings.max_upload_size_bytes
    try:
        proj_module.settings.max_upload_size_bytes = 100  # 100 bytes limit for test
        large_bytes = b"%PDF-1.4 " + (b"A" * 200)
        res = client.post(
            f"/api/v1/projects/{pid}/papers",
            files={"file": ("large.pdf", io.BytesIO(large_bytes), "application/pdf")},
        )
        assert res.status_code == 413
    finally:
        proj_module.settings.max_upload_size_bytes = old_limit


def test_malformed_pdf_upload():
    create_tables()
    db = SessionLocal()
    proj = create_project(db, ProjectCreate(name="Malformed Test"))
    pid = proj.id
    db.close()

    res = client.post(
        f"/api/v1/projects/{pid}/papers",
        files={"file": ("corrupt.pdf", io.BytesIO(b"NOT A REAL PDF FILE"), "application/pdf")},
    )
    assert res.status_code == 400


def test_duplicate_pdf_upload():
    create_tables()
    db = SessionLocal()
    proj = create_project(db, ProjectCreate(name="Duplicate Test"))
    pid = proj.id
    db.close()

    pdf_bytes = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj << /Length 20 >> stream
BT /F1 12 Tf (Hi) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000244 00000 n
0000000315 00000 n
trailer << /Size 6 /Root 1 0 R >>
startxref
391
%%EOF"""

    # First upload
    res1 = client.post(
        f"/api/v1/projects/{pid}/papers",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert res1.status_code == 202

    # Second identical upload to the same project
    res2 = client.post(
        f"/api/v1/projects/{pid}/papers",
        files={"file": ("doc.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
    )
    assert res2.status_code == 409
    assert "already exists" in res2.json()["detail"]


def test_worker_lease_recovery():
    create_tables()
    db = SessionLocal()
    # Clean any leftover jobs to isolate test
    db.query(Job).delete()
    db.commit()

    proj = create_project(db, ProjectCreate(name="Lease Test"))
    paper = create_paper(db, proj.id, "paper.pdf", "paper.pdf")
    job = create_job(db, paper.id)

    # Claim job with worker-1
    claimed = claim_next_job(db, worker_id="worker-1", lease_timeout_seconds=60)
    assert claimed is not None
    assert claimed.id == job.id
    assert claimed.worker_id == "worker-1"
    assert claimed.retry_count == 0

    # Backdate claimed_at to simulate worker-1 crashing 10 minutes ago
    claimed.claimed_at = datetime.fromtimestamp(datetime.now(tz=UTC).timestamp() - 600, tz=UTC)
    db.commit()

    # Worker-2 attempts to claim job with lease timeout of 300s
    recovered = claim_next_job(db, worker_id="worker-2", lease_timeout_seconds=300)
    assert recovered is not None
    assert recovered.id == job.id
    assert recovered.worker_id == "worker-2"
    assert recovered.retry_count == 1
    db.close()


@pytest.mark.anyio
async def test_chat_grounding_and_unsupported_citation():
    create_tables()
    db = SessionLocal()
    proj = create_project(db, ProjectCreate(name="Grounding Test"))
    conv = create_conversation(db, proj.id, title="Test Conv")

    paper = create_paper(db, proj.id, "ground.pdf", "ground.pdf")
    paper.status = PaperStatus.READY
    paper.document_sha256 = "mock_hash_for_test"
    elem = PaperElement(
        paper_id=paper.id,
        page_number=1,
        element_index=0,
        element_type="text",
        text="Superconducting circuits enable fast qubit operations.",
        bbox_x_min=10.0,
        bbox_y_min=10.0,
        bbox_x_max=100.0,
        bbox_y_max=20.0,
        page_width=612.0,
        page_height=792.0,
        parser_version="docling-2.130.0",
    )
    db.add(elem)
    db.flush()

    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text="Superconducting circuits enable fast qubit operations.",
        token_count=10,
        embedding=[0.1] * 1024,
        embedding_vec=[0.1] * 1024,
    )
    db.add(chunk)
    db.flush()
    db.add(ChunkElement(chunk_id=chunk.id, element_id=elem.id, order_index=0))
    db.commit()

    # Fake LLM returns one supported citation [E1] and one hallucinated citation [E99]
    fake_llm = FakeLLMProvider(
        fixed_response=(
            "Superconducting circuits enable fast qubit operations [E1]. Also unicorns exist [E99]."
        )
    )
    set_llm_provider(fake_llm)

    service = ChatService()
    resp = await service.answer_question(db, conv.id, "How do qubit operations work?")

    # E99 and its unsupported claim must have been completely stripped from content
    assert "[E99]" not in resp.content
    assert "[99]" not in resp.content
    assert "unicorns exist" not in resp.content
    assert "[1]" in resp.content
    assert "Superconducting circuits enable fast qubit operations [1]." in resp.content

    # Only 1 validated citation should exist
    assert len(resp.citations) == 1
    assert resp.citations[0].citation_index == 1
    assert resp.citations[0].anchor_status == AnchorStatus.VERIFIED

    db.close()
