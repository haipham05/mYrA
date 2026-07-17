import io

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.crud.chat import create_conversation
from app.crud.job import renew_job_lease
from app.crud.paper import create_paper_with_job
from app.crud.project import create_project
from app.db.base import Base
from app.schemas.evidence import AnchorStatus
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.chat_service import ChatService
from app.services.ingestion import IngestionPipeline
from app.services.llm import FakeLLMProvider, get_llm_provider, set_llm_provider
from app.services.retrieval import HybridRetriever
from app.storage.local import MemoryStorage


def create_twopage_pdf() -> bytes:
    """Generate a valid 2-page PDF with distinct selectable text on each page."""
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = {}

    def write_obj(num, data):
        offsets[num] = out.tell()
        out.write(f"{num} 0 obj\n".encode())
        out.write(data)
        out.write(b"\nendobj\n")

    write_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
    write_obj(2, b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>")

    # Page 1
    stream1 = (
        b"BT\n/F1 12 Tf\n72 700 Td\n"
        b"(Attention Is All You Need. We propose the Transformer architecture.) Tj\nET\n"
    )
    write_obj(
        3,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 7 0 R >> >> >>",
    )
    write_obj(4, f"<< /Length {len(stream1)} >>\nstream\n".encode() + stream1 + b"\nendstream")

    # Page 2
    stream2 = (
        b"BT\n/F1 12 Tf\n72 700 Td\n"
        b"(Scaled Dot-Product Attention computes softmax of query-key inner products.) Tj\nET\n"
    )
    write_obj(
        5,
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 6 0 R /Resources << /Font << /F1 7 0 R >> >> >>",
    )
    write_obj(6, f"<< /Length {len(stream2)} >>\nstream\n".encode() + stream2 + b"\nendstream")

    write_obj(7, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    xref_pos = out.tell()
    out.write(b"xref\n0 8\n0000000000 65535 f \n")
    for i in range(1, 8):
        out.write(f"{offsets[i]:010d} 00000 n \n".encode())
    out.write(f"trailer << /Size 8 /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode())
    return out.getvalue()


@pytest.fixture
def test_db_session(tmp_path):
    db_file = tmp_path / "test_provenance.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()


@pytest.mark.anyio
async def test_full_provenance_roundtrip(test_db_session):
    """Test full PDF -> Ingestion -> Retrieval -> Provenance with exact character offsets."""
    db = test_db_session
    storage = MemoryStorage()
    pdf_bytes = create_twopage_pdf()
    storage_path = "papers/test_transformer.pdf"
    await storage.put(storage_path, pdf_bytes)

    project = create_project(db, ProjectCreate(name="Provenance Project"))
    paper, job = create_paper_with_job(
        db,
        project_id=project.id,
        filename="test_transformer.pdf",
        storage_path=storage_path,
        status=PaperStatus.PROCESSING,
    )

    pipeline = IngestionPipeline()
    # Mock storage fetch inside pipeline
    from unittest.mock import patch

    with patch("app.services.ingestion.get_storage", return_value=storage):
        await pipeline.process_paper(db, paper_id=paper.id, job_id=job.id)

    db.refresh(paper)
    assert paper.status == PaperStatus.READY
    assert paper.page_count == 2
    assert paper.document_sha256 is not None

    # Retrieve evidence for page 2 query
    retriever = HybridRetriever(top_candidates=10, top_evidence=5)
    evidence = retriever.retrieve(
        db,
        project_id=project.id,
        query="What is Scaled Dot-Product Attention?",
    )

    assert len(evidence) > 0
    top_item = evidence[0]
    assert top_item.page_number == 2
    assert "Scaled Dot-Product Attention" in top_item.quote
    assert top_item.parent_context is not None

    # Verify CitationAnchor provenance
    assert len(top_item.anchors) > 0
    verified_anchor = top_item.anchors[0]
    assert verified_anchor.anchor_status == AnchorStatus.VERIFIED
    assert verified_anchor.source_char_start is not None
    assert verified_anchor.source_char_end is not None
    assert verified_anchor.source_char_end > verified_anchor.source_char_start
    assert verified_anchor.document_sha256 == paper.document_sha256

    # Verify claim support and chat generation
    fake_llm = FakeLLMProvider()
    set_llm_provider(fake_llm)

    conv = create_conversation(db, project_id=project.id, title="QA")
    chat_service = ChatService(retriever=retriever)
    resp = await chat_service.answer_question(db, conv.id, "Explain scaled attention")

    assert "[1]" in resp.content
    assert len(resp.citations) == 1
    citation = resp.citations[0]
    assert citation.citation_index == 1
    assert citation.page_number == 2
    assert citation.anchor_status == AnchorStatus.VERIFIED
    assert len(citation.bounding_boxes) > 0

    # Keep the verified extract, but do not retain unsupported generated prose.
    set_llm_provider(FakeLLMProvider(fixed_response=f'The paper explains "{top_item.quote}" [E1].'))
    extractive = await chat_service.answer_question(db, conv.id, "Explain scaled attention")
    assert top_item.quote in extractive.content
    assert "[1]" in extractive.content
    assert "The paper explains" not in extractive.content
    assert len(extractive.citations) == 1


@pytest.mark.anyio
async def test_adversarial_hallucination_and_abstention(test_db_session):
    """Test that hallucinated [E99] sentences are stripped,
    and unsupported claims cause abstention.
    """
    db = test_db_session
    project = create_project(db, ProjectCreate(name="Adversarial Project"))

    # Case 1: Fake LLM hallucinates an unsupported citation [E99]
    fake_llm = FakeLLMProvider(
        fixed_response="Superconducting processors solve quantum problems [E99]."
    )
    set_llm_provider(fake_llm)

    conv = create_conversation(db, project_id=project.id, title="Adversarial QA")
    chat_service = ChatService()
    resp = await chat_service.answer_question(db, conv.id, "Tell me about quantum")

    # Must abstain because E99 is completely invalid and sentence was removed
    assert "[E99]" not in resp.content
    assert len(resp.citations) == 0
    assert "Insufficient evidence available" in resp.content


def test_production_mode_deepseek_fail_clearly():
    """Test that DeepSeek provider fails clearly when API key is missing in production mode."""
    from app.config import Settings

    set_llm_provider(None)
    settings_no_key = Settings(deepseek_api_key=None)
    with pytest.raises(RuntimeError, match="Production mode requires DeepSeek API key"):
        get_llm_provider(settings=settings_no_key, mode="production")


def test_worker_lease_heartbeat_renewal(test_db_session):
    """Test that renew_job_lease successfully updates claimed_at timestamp on active job."""
    db = test_db_session
    project = create_project(db, ProjectCreate(name="Heartbeat Project"))
    paper, job = create_paper_with_job(
        db,
        project_id=project.id,
        filename="heartbeat.pdf",
        storage_path="papers/heartbeat.pdf",
    )

    # Job is initially PENDING
    assert not renew_job_lease(db, job.id)

    # Transition to PROCESSING
    job.status = "PROCESSING"
    db.commit()

    success = renew_job_lease(db, job.id)
    assert success is True

    db.refresh(job)
    assert job.claimed_at is not None
