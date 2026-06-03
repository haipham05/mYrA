import hashlib
import io
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.crud.job import claim_next_job, create_job
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.base import Base
from app.db.models import Job, Paper, PaperChunk, PaperPage
from app.db.session import get_db
from app.ingestion.parser import ParsedElement, ParsedPage, ParseResult
from app.main import app
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.embedding import DeterministicEmbeddingProvider, set_embedding_provider
from app.services.ingestion import IngestionPipeline
from app.storage.factory import set_storage
from app.storage.local import MemoryStorage

PDF_BYTES = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj << /Length 35 >> stream
BT /F1 12 Tf (Failure Matrix Test Text) Tj ET
endstream endobj
5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000244 00000 n
0000000330 00000 n
trailer << /Size 6 /Root 1 0 R >>
startxref
406
%%EOF"""


class MockParser:
    def parse(self, data: bytes) -> ParseResult:
        return ParseResult(
            pages=[
                ParsedPage(
                    page_number=1, width=612.0, height=792.0, raw_text="Failure Matrix Test Text"
                )
            ],
            elements=[
                ParsedElement(
                    element_index=0,
                    page_number=1,
                    element_type="paragraph",
                    text="Failure Matrix Test Text",
                    bbox_x_min=10.0,
                    bbox_y_min=10.0,
                    bbox_x_max=200.0,
                    bbox_y_max=30.0,
                    page_width=612.0,
                    page_height=792.0,
                )
            ],
        )


@pytest.fixture
def test_env(tmp_path):
    db_file = tmp_path / "test_failure_matrix.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()

    storage = MemoryStorage()
    set_storage(storage)
    set_embedding_provider(DeterministicEmbeddingProvider(dimension=1024))

    def override_get_db():
        try:
            yield session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)

    try:
        yield session, storage, client
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        set_storage(None)
        set_embedding_provider(None)
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_failure_matrix_storage_put_failure_leaves_no_rows(test_env):
    db, storage, client = test_env
    proj = create_project(db, ProjectCreate(name="Fail Put Project"))

    with patch.object(storage, "put_stream", side_effect=OSError("Storage disk failure")):
        res = client.post(
            f"/api/v1/projects/{proj.id}/papers",
            files={"file": ("fail_put.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        assert res.status_code == 500
        # Verify 0 papers and 0 jobs created
        assert db.query(Paper).filter(Paper.project_id == proj.id).count() == 0
        assert db.query(Job).count() == 0


@pytest.mark.anyio
async def test_failure_matrix_db_commit_failure_compensates_storage(test_env):
    db, storage, client = test_env
    proj = create_project(db, ProjectCreate(name="Fail Commit Project"))

    with patch(
        "app.api.v1.projects.create_paper_with_job",
        side_effect=RuntimeError("DB constraint failure"),
    ):
        res = client.post(
            f"/api/v1/projects/{proj.id}/papers",
            files={"file": ("fail_db.pdf", io.BytesIO(PDF_BYTES), "application/pdf")},
        )
        assert res.status_code == 500
        # Database rows must be empty
        assert db.query(Paper).filter(Paper.project_id == proj.id).count() == 0
        assert db.query(Job).count() == 0
        # Storage compensation must have deleted any staged PDF keys
        for key in storage._store.keys():
            assert not key.startswith(f"papers/{proj.id}")


@pytest.mark.anyio
async def test_failure_matrix_parse_failure_and_clean_recovery(test_env):
    db, storage, _ = test_env
    proj = create_project(db, ProjectCreate(name="Fail Parse Project"))
    data_sha = hashlib.sha256(PDF_BYTES).hexdigest()
    storage_key = f"papers/{proj.id}/doc.pdf"
    await storage.put(storage_key, PDF_BYTES)

    paper = create_paper(db, proj.id, "doc.pdf", storage_key, document_sha256=data_sha)
    job = create_job(db, paper.id)

    class FailingParser:
        def parse(self, _data: bytes):
            raise TimeoutError("OCR parsing timeout")

    # 1. Process with failing parser
    pipeline_fail = IngestionPipeline(parser=FailingParser())
    await pipeline_fail.process_paper(db, paper.id, job.id)
    db.refresh(job)
    db.refresh(paper)

    # Job must be transiently retried (PENDING) and paper must not be READY
    assert job.status == "PENDING"
    assert job.retry_count == 1
    assert job.is_retryable is True
    assert "[PROCESSING_TIMEOUT]" in job.error_message
    assert paper.status != PaperStatus.READY
    assert db.query(PaperChunk).filter(PaperChunk.paper_id == paper.id).count() == 0

    # 2. Recovery run with functioning parser (simulate exponential backoff elapsing)
    job.updated_at = datetime.now(tz=UTC) - timedelta(seconds=20)
    db.commit()
    pipeline_recover = IngestionPipeline(parser=MockParser())
    claimed_job = claim_next_job(db, worker_id="worker-recovery")
    assert claimed_job is not None
    await pipeline_recover.process_paper(db, paper.id, claimed_job.id, worker_id="worker-recovery")

    db.refresh(paper)
    db.refresh(claimed_job)
    assert paper.status == PaperStatus.READY
    assert paper.document_sha256 == data_sha
    assert claimed_job.status == "COMPLETED"
    assert db.query(PaperChunk).filter(PaperChunk.paper_id == paper.id).count() > 0
    assert db.query(PaperPage).filter(PaperPage.paper_id == paper.id).count() == 1


@pytest.mark.anyio
async def test_failure_matrix_embedding_failure_and_clean_recovery(test_env):
    db, storage, _ = test_env
    proj = create_project(db, ProjectCreate(name="Fail Embed Project"))
    data_sha = hashlib.sha256(PDF_BYTES).hexdigest()
    storage_key = f"papers/{proj.id}/embed_doc.pdf"
    await storage.put(storage_key, PDF_BYTES)

    paper = create_paper(db, proj.id, "embed_doc.pdf", storage_key, document_sha256=data_sha)
    job = create_job(db, paper.id)

    class FailingEmbedProvider:
        model_name = "test-model"
        model_version = "v1"

        def embed_documents(self, _texts):
            raise ConnectionError("Embedding service unreachable")

    # 1. Pipeline fails during embedding
    set_embedding_provider(FailingEmbedProvider())
    pipeline = IngestionPipeline(parser=MockParser())
    await pipeline.process_paper(db, paper.id, job.id)
    db.refresh(job)
    db.refresh(paper)

    assert job.status == "PENDING"
    assert job.retry_count == 1
    assert job.is_retryable is True
    assert "[EMBEDDING_PROVIDER_ERROR]" in job.error_message
    assert paper.status != PaperStatus.READY
    assert db.query(PaperChunk).filter(PaperChunk.paper_id == paper.id).count() == 0

    # 2. Recover with working embedding provider (simulate exponential backoff elapsing)
    job.updated_at = datetime.now(tz=UTC) - timedelta(seconds=20)
    db.commit()
    set_embedding_provider(DeterministicEmbeddingProvider(dimension=1024))
    claimed_job = claim_next_job(db, worker_id="worker-recovery-2")
    assert claimed_job is not None
    await pipeline.process_paper(db, paper.id, claimed_job.id, worker_id="worker-recovery-2")

    db.refresh(paper)
    db.refresh(claimed_job)
    assert paper.status == PaperStatus.READY
    assert claimed_job.status == "COMPLETED"
    assert db.query(PaperChunk).filter(PaperChunk.paper_id == paper.id).count() > 0
