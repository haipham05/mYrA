import hashlib
import io
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.crud.corpus import read_corpus_revision
from app.crud.graph import (
    create_or_enqueue_graph_event,
    get_active_graph_events_for_paper,
    get_graph_event,
    get_latest_completed_graph_event,
    lock_graph_publication,
)
from app.crud.job import create_job
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.base import Base
from app.db.models import GraphEvent, Job, Paper
from app.db.session import get_db
from app.ingestion.parser import ParsedElement, ParsedPage, ParseResult
from app.main import app
from app.schemas.job import JobStatus
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.embedding import DeterministicEmbeddingProvider, set_embedding_provider
from app.services.ingestion import IngestionPipeline
from app.storage.factory import set_storage
from app.storage.local import MemoryStorage

TINY_PDF_BYTES = b"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]
/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj << /Length 35 >> stream
BT /F1 12 Tf (Graph Ingestion Enqueue Test) Tj ET
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


class RecordingTelemetry:
    def __init__(self):
        self.stages = {}
        self.events = []

    @contextmanager
    def operation(self, name, **_kwargs):
        yield None

    @contextmanager
    def stage(self, name, **_kwargs):
        observation = _RecordingObservation()
        self.stages[name] = observation
        yield observation

    def event(self, name, *, metadata=None, **_kwargs):
        self.events.append((name, metadata or {}))


class _RecordingObservation:
    def __init__(self):
        self.metadata = {}

    def update(self, *, metadata=None, **_kwargs):
        self.metadata.update(metadata or {})


class MockParser:
    def parse(self, data: bytes) -> ParseResult:
        return ParseResult(
            pages=[
                ParsedPage(
                    page_number=1,
                    width=612.0,
                    height=792.0,
                    raw_text="Graph Ingestion Enqueue Test",
                )
            ],
            elements=[
                ParsedElement(
                    element_index=0,
                    page_number=1,
                    element_type="paragraph",
                    text="Graph Ingestion Enqueue Test Element",
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
def env(tmp_path):
    db_file = tmp_path / "test_graph_ingestion.db"
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
async def test_successful_ingestion_does_not_implicitly_enqueue_graph_even_when_enabled(env):
    """Graph availability does not opt ordinary paper ingestion into paid extraction."""
    db, storage, _ = env
    proj = create_project(db, ProjectCreate(name="Graph Enqueue Proj 1"))
    pdf_sha = hashlib.sha256(TINY_PDF_BYTES).hexdigest()
    storage_key = f"papers/{proj.id}/enqueue_test.pdf"
    await storage.put(storage_key, TINY_PDF_BYTES)

    paper = create_paper(db, proj.id, "enqueue_test.pdf", storage_key, document_sha256=pdf_sha)
    job = create_job(db, paper.id)

    settings = Settings(graphrag_enabled=True)
    pipeline = IngestionPipeline(parser=MockParser(), settings=settings)

    await pipeline.process_paper(db, paper.id, job.id)

    db.refresh(paper)
    db.refresh(job)
    assert paper.status == PaperStatus.READY
    assert job.status == JobStatus.COMPLETED

    assert get_active_graph_events_for_paper(db, paper.id) == []
    assert db.query(GraphEvent).filter(GraphEvent.paper_id == paper.id).count() == 0


@pytest.mark.anyio
async def test_ingestion_telemetry_records_stage_counts_without_document_text(env, monkeypatch):
    db, storage, _ = env
    proj = create_project(db, ProjectCreate(name="Telemetry Ingestion Proj"))
    pdf_sha = hashlib.sha256(TINY_PDF_BYTES).hexdigest()
    storage_key = f"papers/{proj.id}/telemetry.pdf"
    await storage.put(storage_key, TINY_PDF_BYTES)
    paper = create_paper(db, proj.id, "telemetry.pdf", storage_key, document_sha256=pdf_sha)
    job = create_job(db, paper.id)
    telemetry = RecordingTelemetry()
    monkeypatch.setattr("app.services.ingestion.get_telemetry", lambda: telemetry)

    await IngestionPipeline(parser=MockParser()).process_paper(db, paper.id, job.id)

    assert set(telemetry.stages) == {
        "ingestion.storage_read",
        "ingestion.parse",
        "ingestion.chunk",
        "ingestion.embed",
    }
    assert telemetry.stages["ingestion.parse"].metadata["page_count"] == 1
    assert telemetry.stages["ingestion.parse"].metadata["element_count"] == 1
    assert telemetry.stages["ingestion.chunk"].metadata["chunk_count"] > 0
    assert telemetry.stages["ingestion.embed"].metadata["embedding_count"] > 0
    assert all(stage.metadata["outcome"] == "success" for stage in telemetry.stages.values())
    event_names = [name for name, _ in telemetry.events]
    assert "ingestion.index_publication" in event_names
    assert "ingestion.completed" in event_names
    emitted = repr((telemetry.stages, telemetry.events))
    assert "Graph Ingestion Enqueue Test Element" not in emitted
    assert "BT /F1" not in emitted


@pytest.mark.anyio
async def test_successful_ingestion_without_graph_configuration_enqueues_zero_events(env):
    """The default graph-disabled path also completes normal ingestion unchanged."""
    db, storage, _ = env
    proj = create_project(db, ProjectCreate(name="Graph Disabled Proj"))
    pdf_sha = hashlib.sha256(TINY_PDF_BYTES).hexdigest()
    storage_key = f"papers/{proj.id}/disabled_test.pdf"
    await storage.put(storage_key, TINY_PDF_BYTES)

    paper = create_paper(db, proj.id, "disabled_test.pdf", storage_key, document_sha256=pdf_sha)
    job = create_job(db, paper.id)

    settings = Settings(graphrag_enabled=False)
    pipeline = IngestionPipeline(parser=MockParser(), settings=settings)

    await pipeline.process_paper(db, paper.id, job.id)

    db.refresh(paper)
    db.refresh(job)
    assert paper.status == PaperStatus.READY
    assert job.status == JobStatus.COMPLETED

    events = get_active_graph_events_for_paper(db, paper.id)
    assert len(events) == 0

    all_events = db.query(GraphEvent).filter(GraphEvent.paper_id == paper.id).all()
    assert len(all_events) == 0


@pytest.mark.anyio
async def test_ingestion_failure_rollback_leaves_no_graph_event(env):
    """Test 3: Ingestion failure / rollback leaves no GraphEvent in the database."""
    db, storage, _ = env
    proj = create_project(db, ProjectCreate(name="Fail Rollback Proj"))
    pdf_sha = hashlib.sha256(TINY_PDF_BYTES).hexdigest()
    storage_key = f"papers/{proj.id}/fail_doc.pdf"
    await storage.put(storage_key, TINY_PDF_BYTES)

    paper = create_paper(db, proj.id, "fail_doc.pdf", storage_key, document_sha256=pdf_sha)
    job = create_job(db, paper.id)

    class FailingParser:
        def parse(self, _data: bytes):
            raise ValueError("Corrupt PDF parse error")

    settings = Settings(graphrag_enabled=True)
    pipeline = IngestionPipeline(parser=FailingParser(), settings=settings)

    await pipeline.process_paper(db, paper.id, job.id)

    db.refresh(paper)
    assert paper.status != PaperStatus.READY
    events = db.query(GraphEvent).filter(GraphEvent.paper_id == paper.id).all()
    assert len(events) == 0

    # Also test failure during the publish transaction right at commit
    paper2 = create_paper(db, proj.id, "fail_doc2.pdf", storage_key, document_sha256=pdf_sha)
    job2 = create_job(db, paper2.id)

    pipeline_ok = IngestionPipeline(parser=MockParser(), settings=settings)
    original_commit = db.commit

    def failing_commit():
        # Raise an exception when committing the publish transaction
        raise RuntimeError("Database IO failure during publish commit")

    with patch.object(db, "commit", side_effect=failing_commit):
        await pipeline_ok.process_paper(db, paper2.id, job2.id)

    # Restore commit and check that transaction rollback dropped any staged GraphEvent
    db.commit = original_commit
    events2 = db.query(GraphEvent).filter(GraphEvent.paper_id == paper2.id).all()
    assert len(events2) == 0


@pytest.mark.anyio
async def test_reindexing_paper_does_not_schedule_graph_extraction(env):
    """Reindexing updates the corpus without implicitly scheduling graph extraction."""
    db, storage, _ = env
    proj = create_project(db, ProjectCreate(name="Reindex Proj"))
    pdf_sha = hashlib.sha256(TINY_PDF_BYTES).hexdigest()
    storage_key = f"papers/{proj.id}/reindex_doc.pdf"
    await storage.put(storage_key, TINY_PDF_BYTES)

    paper = create_paper(db, proj.id, "reindex_doc.pdf", storage_key, document_sha256=pdf_sha)
    job1 = create_job(db, paper.id)

    settings = Settings(graphrag_enabled=True)
    pipeline = IngestionPipeline(parser=MockParser(), settings=settings)

    # 1. First ingestion run
    await pipeline.process_paper(db, paper.id, job1.id)
    db.refresh(paper)
    assert paper.status == PaperStatus.READY
    assert read_corpus_revision(db, proj.id) == 1

    assert db.query(GraphEvent).filter(GraphEvent.paper_id == paper.id).count() == 0

    # 2. Second ingestion run (reindex)
    job2 = create_job(db, paper.id)
    await pipeline.process_paper(db, paper.id, job2.id)
    db.refresh(paper)
    assert paper.status == PaperStatus.READY
    assert read_corpus_revision(db, proj.id) == 2

    assert db.query(GraphEvent).filter(GraphEvent.paper_id == paper.id).count() == 0


@pytest.mark.anyio
async def test_upload_and_ingestion_with_neo4j_offline_or_unconfigured(env):
    """Test 5: Verify upload remains 100% correct when Neo4j is offline or unconfigured
    (zero calls to Neo4j during upload/ingestion).
    """
    db, storage, client = env
    proj = create_project(db, ProjectCreate(name="Neo4j Offline Proj"))

    # GraphRAG enabled in settings, but Neo4j is offline/unconfigured
    settings = Settings(
        graphrag_enabled=True,
        neo4j_uri=None,
        neo4j_user=None,
        neo4j_password=None,
    )

    with (
        patch("neo4j.GraphDatabase.driver") as mock_driver,
        patch("app.services.graphrag.neo4j_repository.Neo4jRepository") as mock_repo,
    ):
        # 1. Upload paper via API endpoint
        res = client.post(
            f"/api/v1/projects/{proj.id}/papers",
            files={"file": ("offline_neo4j.pdf", io.BytesIO(TINY_PDF_BYTES), "application/pdf")},
        )
        assert res.status_code == 202
        upload_data = res.json()
        paper_id = UUID(upload_data["paper_id"])
        job_id = UUID(upload_data["job_id"])

        # 2. Process paper through ingestion pipeline
        pipeline = IngestionPipeline(parser=MockParser(), settings=settings)
        await pipeline.process_paper(db, paper_id, job_id)

        # 3. Assert zero calls to Neo4j
        mock_driver.assert_not_called()
        mock_repo.assert_not_called()

        # 4. Verify paper is READY and graph extraction was not implicitly enqueued
        paper = db.get(Paper, paper_id)
        assert paper is not None
        assert paper.status == PaperStatus.READY

        job = db.get(Job, job_id)
        assert job is not None
        assert job.status == JobStatus.COMPLETED

        events = get_active_graph_events_for_paper(db, paper.id)
        assert events == []
        assert db.query(GraphEvent).filter(GraphEvent.paper_id == paper.id).count() == 0


def test_crud_graph_event_operations(env):
    """Test full CRUD operations on GraphEvent in app/crud/graph.py."""
    db, _, _ = env
    proj = create_project(db, ProjectCreate(name="CRUD Proj"))
    paper_id = uuid4()
    paper = Paper(
        id=paper_id,
        project_id=proj.id,
        filename="crud_test.pdf",
        storage_path="papers/test.pdf",
        status="READY",
    )
    db.add(paper)
    db.commit()

    # 1. create_or_enqueue_graph_event with default generation_id
    event = create_or_enqueue_graph_event(
        db=db,
        project_id=proj.id,
        paper_id=paper.id,
        action="UPSERT",
    )
    db.commit()

    assert event.id is not None
    assert event.project_id == proj.id
    assert event.paper_id == paper.id
    assert event.status == "PENDING"
    assert event.attempts == 0
    assert event.max_attempts == 3
    assert event.ontology_version == "1.0.0"
    assert event.extractor_version == "1.0.0"
    assert event.generation_id.startswith(f"gen_{paper.id.hex[:8]}_")

    # 2. get_graph_event by UUID and str
    retrieved = get_graph_event(db, event.id)
    assert retrieved is not None
    assert retrieved.id == event.id

    retrieved_str = get_graph_event(db, str(event.id))
    assert retrieved_str is not None
    assert retrieved_str.id == event.id

    assert get_graph_event(db, uuid4()) is None

    # 3. get_active_graph_events_for_paper
    active = get_active_graph_events_for_paper(db, paper.id)
    assert len(active) == 1
    assert active[0].id == event.id

    # String paper_id
    active_str = get_active_graph_events_for_paper(db, str(paper.id))
    assert len(active_str) == 1

    # 4. Status filtering in get_active_graph_events_for_paper
    # PROCESSING is active
    event.status = "PROCESSING"
    db.commit()
    active = get_active_graph_events_for_paper(db, paper.id)
    assert len(active) == 1

    # COMPLETED is inactive
    now = datetime.now(tz=UTC)
    event.status = "COMPLETED"
    event.completed_at = now - timedelta(minutes=10)
    db.commit()
    assert len(get_active_graph_events_for_paper(db, paper.id)) == 0

    # FAILED is inactive
    event_failed = create_or_enqueue_graph_event(
        db=db,
        project_id=proj.id,
        paper_id=paper.id,
        action="UPSERT",
    )
    event_failed.status = "FAILED"
    db.commit()
    assert len(get_active_graph_events_for_paper(db, paper.id)) == 0

    # 5. get_latest_completed_graph_event
    latest = get_latest_completed_graph_event(db, paper.id)
    assert latest is not None
    assert latest.id == event.id

    # Add a newer completed event
    event_newer = create_or_enqueue_graph_event(
        db=db,
        project_id=proj.id,
        paper_id=paper.id,
        action="UPSERT",
    )
    event_newer.status = "COMPLETED"
    event_newer.completed_at = now
    # Completion/retry time must not make the older source event current again.
    event.completed_at = now + timedelta(hours=1)
    db.commit()

    latest2 = get_latest_completed_graph_event(db, paper.id)
    assert latest2 is not None
    assert latest2.id == event_newer.id

    # String paper_id
    latest2_str = get_latest_completed_graph_event(db, str(paper.id))
    assert latest2_str is not None
    assert latest2_str.id == event_newer.id

    # Paper with no completed events
    other_paper_id = uuid4()
    assert get_latest_completed_graph_event(db, other_paper_id) is None

    # 6. Custom generation_id and versions
    custom_event = create_or_enqueue_graph_event(
        db=db,
        project_id=str(proj.id),
        paper_id=str(paper.id),
        action="DELETE",
        generation_id="custom-generation-42",
        ontology_version="2.0.0",
        extractor_version="3.0.0",
    )
    db.commit()

    assert custom_event.action == "DELETE"
    assert custom_event.generation_id == "custom-generation-42"
    assert custom_event.ontology_version == "2.0.0"
    assert custom_event.extractor_version == "3.0.0"


def test_publication_lock_refreshes_lease_after_separate_session_takeover(env):
    db, _, _ = env
    project = create_project(db, ProjectCreate(name="Lease Refresh Proj"))
    paper = create_paper(db, project.id, "lease.pdf", "lease.pdf")
    event = create_or_enqueue_graph_event(db, project.id, paper.id)
    event.status = "PROCESSING"
    event.lease_owner = "old-worker"
    event.lease_expires_at = datetime.now(tz=UTC) + timedelta(minutes=5)
    db.commit()
    event_id = event.id
    assert get_graph_event(db, event_id).lease_owner == "old-worker"

    takeover_session = sessionmaker(bind=db.get_bind(), expire_on_commit=False)()
    try:
        takeover = takeover_session.get(GraphEvent, event_id)
        takeover.lease_owner = "replacement-worker"
        takeover.lease_expires_at = datetime.now(tz=UTC) + timedelta(minutes=5)
        takeover_session.commit()
    finally:
        takeover_session.close()

    assert lock_graph_publication(db, event_id, "old-worker") is False
