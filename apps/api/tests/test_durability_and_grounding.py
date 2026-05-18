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


def test_claim_support_requires_ordered_extract():
    # Shared vocabulary alone cannot establish a claim's relationship.
    claim = "The Transformer model uses self-attention mechanisms."
    quote = "We introduce the Transformer, architecture relying entirely on self-attention."
    assert check_claim_support(claim, quote) is False
    assert check_claim_support(
        "Transformer architecture relying entirely on self-attention.", quote
    )

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


def test_adversarial_claim_grounding_direct():
    """Adversarial validation tests specifically required by Milestone 1 review:

    1. Numeric contradiction: 28.4 vs 14.2
    2. Reversed conclusion / negation: 'did not improve' vs 'improved'
    3. Reversed conclusion / positive-on-negative: 'improved' vs 'did not improve'
    4. Directional opposition: 'increased' vs 'decreased'
    5. Valid grounded claim
    """
    from app.services.chat_service import check_claim_support

    # 1. Numeric contradiction must return False
    assert check_claim_support("The BLEU score was 28.4.", "The BLEU score was 14.2.") is False

    # 2. Negation contradiction must return False
    assert (
        check_claim_support(
            "The treatment did not improve survival.",
            "The treatment improved survival.",
        )
        is False
    )

    # 3. Positive claim on negated quote must return False
    assert (
        check_claim_support(
            "The treatment improved survival.",
            "The treatment did not improve survival.",
        )
        is False
    )

    # 4. Directional opposition must return False
    assert (
        check_claim_support(
            "The model increased error rate.",
            "The model decreased error rate.",
        )
        is False
    )

    # 5. Grounded valid claim must return True
    quote_text = (
        "The Transformer model achieves a state-of-the-art BLEU score of 28.4 "
        "on the English-to-German task."
    )
    assert check_claim_support("The Transformer achieves a BLEU score of 28.4.", quote_text) is True

    assert check_claim_support("Alice outperformed Bob.", "Bob outperformed Alice.") is False
    assert (
        check_claim_support(
            "Model A scored 90 and Model B scored 80.",
            "Model A scored 80 and Model B scored 90.",
        )
        is False
    )
    assert check_claim_support("The dose was 90 mg.", "The dose was 90 kg.") is False
    assert check_claim_support("The accuracy was 90%.", "The accuracy was 90.") is False


@pytest.mark.anyio
async def test_claim_grounding_rejects_parent_only_claims():
    """Ensure claims supported only by parent context (not atomic quote) are rejected."""
    create_tables()
    db = SessionLocal()
    proj = create_project(db, ProjectCreate(name="Parent Context Test"))
    conv = create_conversation(db, proj.id, title="Parent Test")

    paper = create_paper(db, proj.id, "parent_test.pdf", "parent_test.pdf")
    paper.status = PaperStatus.READY
    paper.document_sha256 = "mock_hash_parent"

    # Child atomic element
    child_elem = PaperElement(
        paper_id=paper.id,
        page_number=1,
        element_index=0,
        element_type="text",
        text="The learning rate was set to 0.001.",
        page_width=612.0,
        page_height=792.0,
        parser_version="docling-2.130.0",
    )
    db.add(child_elem)
    db.flush()

    # Chunk with broader parent text (mentioning accuracy) but atomic quote only has learning rate
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text=(
            "Parent context: Accuracy reached 95.5%. "
            "Atomic quote: The learning rate was set to 0.001."
        ),
        token_count=15,
        embedding=[0.1] * 1024,
        embedding_vec=[0.1] * 1024,
    )
    db.add(chunk)
    db.flush()
    db.add(ChunkElement(chunk_id=chunk.id, element_id=child_elem.id, order_index=0))
    db.commit()

    # LLM hallucinates claim from parent context not supported by child quote
    fake_llm = FakeLLMProvider(fixed_response="The model achieved an accuracy of 95.5% [E1].")
    set_llm_provider(fake_llm)

    service = ChatService()
    resp = await service.answer_question(db, conv.id, "What was the accuracy?")

    # Unsupported claim must be rejected and system abstains
    assert "95.5%" not in resp.content
    assert len(resp.citations) == 0
    assert "Insufficient evidence" in resp.content

    db.close()


def test_worker_lease_fencing():
    from app.crud.job import claim_next_job, create_job, renew_job_lease

    create_tables()
    db = SessionLocal()
    db.query(Job).delete()
    db.commit()
    proj = create_project(db, ProjectCreate(name="Fence Test"))
    paper = create_paper(db, proj.id, "fence.pdf", "fence.pdf")
    job = create_job(db, paper.id)

    # Worker 1 claims job
    claimed_1 = claim_next_job(db, worker_id="worker-1", lease_timeout_seconds=0)
    assert claimed_1 is not None
    assert claimed_1.worker_id == "worker-1"

    # With timeout=0, cutoff is now, so worker 2 reclaims expired job
    claimed_2 = claim_next_job(db, worker_id="worker-2", lease_timeout_seconds=0)
    assert claimed_2 is not None
    assert claimed_2.id == job.id
    assert claimed_2.worker_id == "worker-2"

    # Worker 1 heartbeat must be rejected (fenced out)
    assert renew_job_lease(db, job.id, worker_id="worker-1") is False

    # Worker 2 heartbeat succeeds
    assert renew_job_lease(db, job.id, worker_id="worker-2") is True
    db.close()


def test_job_retry_exponential_backoff():
    from datetime import UTC, datetime, timedelta

    from app.crud.job import claim_next_job, create_job

    create_tables()
    db = SessionLocal()
    db.query(Job).delete()
    db.commit()
    proj = create_project(db, ProjectCreate(name="Backoff Test"))
    paper = create_paper(db, proj.id, "backoff.pdf", "backoff.pdf")
    job = create_job(db, paper.id)

    # Simulate retried job with recent updated_at (retry_count=2, backoff = 20s)
    job.retry_count = 2
    job.updated_at = datetime.now(tz=UTC)
    db.commit()

    # Should not be claimed yet because backoff hasn't elapsed
    assert claim_next_job(db, worker_id="worker-1") is None

    # Simulate updated_at in the past (> 20s ago)
    job.updated_at = datetime.now(tz=UTC) - timedelta(seconds=25)
    db.commit()

    # Now it should be claimed
    claimed = claim_next_job(db, worker_id="worker-1")
    assert claimed is not None
    assert claimed.id == job.id
    db.close()


def test_is_transient_error_classification():
    from app.services.ingestion import is_transient_error

    # Fatal errors
    assert is_transient_error(ValueError("Invalid PDF signature")) is False
    assert is_transient_error(KeyError("missing_field")) is False
    assert is_transient_error(TypeError("bad type")) is False
    assert is_transient_error(Exception("PDF has no extractable text")) is False

    # Transient errors
    assert is_transient_error(TimeoutError("Connection timed out")) is True
    assert is_transient_error(ConnectionError("Socket closed")) is True
    assert is_transient_error(OSError("Resource temporarily unavailable")) is True


@pytest.mark.anyio
async def test_storage_open_stream():
    from app.storage.local import MemoryStorage

    storage = MemoryStorage()
    await storage.put("stream_test.pdf", b"1234567890" * 100)

    assert await storage.exists("stream_test.pdf") is True
    assert await storage.exists("non_existent.pdf") is False

    chunks = []
    async for chunk in storage.open_stream("stream_test.pdf", chunk_size=50):
        chunks.append(chunk)

    assert b"".join(chunks) == b"1234567890" * 100
    assert len(chunks) == 20
