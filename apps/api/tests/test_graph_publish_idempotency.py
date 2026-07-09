"""Tests for Task 5.19: Publish then acknowledge idempotently.

Verifies:
1. End-to-end publication from validated fact snapshots into Neo4j.
2. Failure before graph write: snapshots remain durable in PostgreSQL.
3. Failure after graph write / before ack: replay executes MERGE with zero duplicates.
4. Stale generation check: older generation does not retire newer facts.
5. Replay idempotency: duplicate delivery runs produce identical graph state.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from app.config import Settings
from app.crud.graph import (
    claim_next_graph_event,
    complete_graph_event,
    create_or_enqueue_graph_event,
    get_graph_event,
)
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.models import (
    ChunkElement,
    GraphEvent,
    GraphFactSnapshot,
    Job,
    Paper,
    PaperChunk,
    PaperElement,
    PaperPage,
)
from app.db.session import SessionLocal, create_tables
from app.schemas.graph import (
    EntityType,
    GraphEntitySchema,
    GraphFactCandidate,
    GraphProvenanceSchema,
    GraphQualifierSchema,
    RelationshipPredicate,
)
from app.schemas.project import ProjectCreate
from app.services.graphrag.extractor import ExtractionResult, GraphExtractionAdapter
from app.services.graphrag.input_selector import ExtractionEvidenceItem
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.processor import GraphEventProcessor, _extraction_cache_key

QUOTE_77 = "The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German dataset."


def _cache_evidence(text: str = "evidence") -> ExtractionEvidenceItem:
    from uuid import uuid4

    return ExtractionEvidenceItem(
        evidence_id="ev_1",
        chunk_id=uuid4(),
        text=text,
        page_number=1,
        parser_version="p1",
        document_sha256="paper-hash",
        chunk_index=0,
    )


def test_graph_extraction_cache_key_tracks_evidence_and_configuration():
    from types import SimpleNamespace

    extractor = SimpleNamespace(cache_version="extractor-v1")
    evidence = _cache_evidence()

    def key(*, items=None, limit=4, ontology="ont-v1", model=extractor):
        return _extraction_cache_key(
            project_id=UUID("00000000-0000-0000-0000-000000000001"),
            paper_id=UUID("00000000-0000-0000-0000-000000000002"),
            paper_sha256="paper-hash",
            evidence_items=items or [evidence],
            chunk_limit=limit,
            ontology_version=ontology,
            extractor_version="extract-v1",
            extractor=model,
        )

    baseline = key()
    assert baseline
    assert key(items=[_cache_evidence("changed")]) != baseline
    assert key(limit=5) != baseline
    assert key(ontology="ont-v2") != baseline
    assert key(model=SimpleNamespace(cache_version="extractor-v2")) != baseline
    assert key(model=SimpleNamespace()) is None

    provider_v1 = SimpleNamespace(provider_name="provider", model_name="model-v1")
    provider_v2 = SimpleNamespace(provider_name="provider", model_name="model-v2")
    adapter_v1 = GraphExtractionAdapter(provider_v1)
    adapter_v2 = GraphExtractionAdapter(provider_v2)

    def provider_key(adapter):
        return _extraction_cache_key(
            project_id=UUID("00000000-0000-0000-0000-000000000001"),
            paper_id=UUID("00000000-0000-0000-0000-000000000002"),
            paper_sha256="paper-hash",
            evidence_items=[evidence],
            chunk_limit=4,
            ontology_version="ont-v1",
            extractor_version="extract-v1",
            extractor=adapter,
        )

    assert provider_key(adapter_v1) != provider_key(adapter_v2)
    assert provider_key(GraphExtractionAdapter(SimpleNamespace())) is None


@pytest.mark.anyio
async def test_graph_extraction_cache_hit_skips_extractor_but_reverifies_candidate(monkeypatch):
    class MemoryCache:
        def __init__(self):
            self.values = {}
            self.writes = 0

        def get(self, key, validator):
            value = self.values.get(key)
            return validator(value) if value is not None else None

        def set(self, key, value, *, ttl_seconds):
            assert ttl_seconds == 24 * 60 * 60
            self.values[key] = value
            self.writes += 1
            return True

    class StableExtractor:
        cache_version = "fixture-model-v1"

        def __init__(self, result):
            self.result = result
            self.calls = 0

        async def extract(self, project_id, paper_id, evidence_items):
            self.calls += 1
            return self.result

    cache = MemoryCache()
    monkeypatch.setattr("app.services.graphrag.processor.get_cache", lambda: cache)
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Graph Cache Project"))
        paper = _setup_ready_paper(db, project.id)
        first = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        first_id, project_id, paper_id = first.id, project.id, paper.id
        chunk_id = db.query(PaperChunk).filter(PaperChunk.paper_id == paper_id).first().id

    candidate = GraphFactCandidate(
        subject=GraphEntitySchema(id="cache-subject", name="Transformer", type=EntityType.METHOD),
        predicate=RelationshipPredicate.EVALUATED_ON,
        object=GraphEntitySchema(id="cache-object", name="WMT 2014", type=EntityType.DATASET),
        provenance=GraphProvenanceSchema(
            paper_id=paper_id,
            chunk_id=chunk_id,
            page_number=1,
            exact_quote=QUOTE_77,
            char_start=0,
            char_end=len(QUOTE_77),
            document_sha256="d0c0" * 16,
        ),
    )
    extractor = StableExtractor(
        ExtractionResult(accepted_facts=[candidate], rejected_count=0, rejection_reasons={})
    )
    repo = MagicMock(spec=Neo4jRepository)
    processor = GraphEventProcessor(
        settings=Settings(graphrag_enabled=True), repo=repo, extractor=extractor
    )
    verification = iter([(True, None), (False, "UNRESOLVED_ANCHOR")])
    monkeypatch.setattr(
        "app.services.graphrag.processor.verify_candidate_fact",
        lambda *args: next(verification),
    )

    with SessionLocal() as db:
        assert claim_next_graph_event(db, worker_id="cache-first") is not None
        assert await processor.process_graph_event(db, first_id, "cache-first")
    assert extractor.calls == 1
    assert cache.writes == 1

    with SessionLocal() as db:
        second = create_or_enqueue_graph_event(db, project_id, paper_id, action="UPSERT")
        db.commit()
        second_id = second.id
    with SessionLocal() as db:
        assert claim_next_graph_event(db, worker_id="cache-second") is not None
        assert not await processor.process_graph_event(db, second_id, "cache-second")

    # The cached candidate was rechecked against the current verifier and rejected;
    # only the first attempt called the extractor.
    assert extractor.calls == 1
    with SessionLocal() as db:
        assert get_graph_event(db, second_id).error_code == "NO_VERIFIED_FACTS"


@pytest.mark.anyio
async def test_invalid_graph_extraction_result_is_not_cached(monkeypatch):
    class MemoryCache:
        writes = 0

        def get(self, key, validator):
            return None

        def set(self, key, value, *, ttl_seconds):
            self.writes += 1
            return True

    class StableExtractor:
        cache_version = "fixture-model-v1"

        async def extract(self, project_id, paper_id, evidence_items):
            return ExtractionResult(
                accepted_facts=[], rejected_count=1, rejection_reasons={"MALFORMED_JSON": 1}
            )

    cache = MemoryCache()
    monkeypatch.setattr("app.services.graphrag.processor.get_cache", lambda: cache)
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Invalid Graph Cache Project"))
        paper = _setup_ready_paper(db, project.id)
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        event_id = event.id

    repo = MagicMock(spec=Neo4jRepository)
    processor = GraphEventProcessor(
        settings=Settings(graphrag_enabled=True), repo=repo, extractor=StableExtractor()
    )
    with SessionLocal() as db:
        assert claim_next_graph_event(db, worker_id="invalid-cache") is not None
        assert not await processor.process_graph_event(db, event_id, "invalid-cache")
    assert cache.writes == 0


@pytest.fixture(autouse=True)
def clean_database():
    create_tables()
    with SessionLocal() as db:
        db.query(GraphFactSnapshot).delete()
        db.query(GraphEvent).delete()
        db.query(ChunkElement).delete()
        db.query(PaperChunk).delete()
        db.query(PaperElement).delete()
        db.query(PaperPage).delete()
        db.query(Job).delete()
        db.query(Paper).delete()
        db.commit()


def _setup_ready_paper(db, project_id: UUID) -> Paper:
    paper = create_paper(db, project_id, "attention.pdf", "attention.pdf")
    paper.status = "READY"
    paper.document_sha256 = "d0c0" * 16

    page = PaperPage(
        paper_id=paper.id,
        page_number=1,
        width=612.0,
        height=792.0,
        rotation=0,
        raw_text=QUOTE_77,
    )
    db.add(page)
    db.flush()

    element = PaperElement(
        paper_id=paper.id,
        page_number=1,
        element_index=0,
        element_type="paragraph",
        text=QUOTE_77,
        parser_version="v1",
    )
    db.add(element)
    db.flush()

    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text=QUOTE_77,
        token_count=16,
    )
    db.add(chunk)
    db.flush()

    link = ChunkElement(chunk_id=chunk.id, element_id=element.id, order_index=0)
    db.add(link)
    db.commit()
    db.refresh(paper)
    return paper


@pytest.mark.anyio
async def test_end_to_end_extraction_snapshot_and_neo4j_publish():
    """Test 1: Full pipeline: extraction -> verify -> snapshot -> Neo4j -> ack."""
    with SessionLocal() as db:
        proj = create_project(db, ProjectCreate(name="E2E Publish Proj"))
        paper = _setup_ready_paper(db, proj.id)
        event = create_or_enqueue_graph_event(db, proj.id, paper.id, action="UPSERT")
        db.commit()
        db.refresh(event)
        event_id = event.id
        paper_id = paper.id
        project_id = proj.id
        gen_id = event.generation_id

    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_extractor = AsyncMock()

    with SessionLocal() as db:
        chunk = db.query(PaperChunk).filter(PaperChunk.paper_id == paper_id).first()
        chunk_id = chunk.id

    candidate = GraphFactCandidate(
        subject=GraphEntitySchema(id="e1", name="Transformer", type=EntityType.METHOD),
        predicate=RelationshipPredicate.EVALUATED_ON,
        object=GraphEntitySchema(id="e2", name="WMT 2014", type=EntityType.DATASET),
        qualifiers=GraphQualifierSchema(metric="BLEU", result_value=28.4),
        provenance=GraphProvenanceSchema(
            paper_id=paper_id,
            chunk_id=chunk_id,
            page_number=1,
            exact_quote=QUOTE_77,
            char_start=0,
            char_end=77,
            document_sha256="d0c0" * 16,
        ),
    )
    mock_extractor.extract.return_value = ExtractionResult(
        accepted_entities=[candidate.subject, candidate.object],
        accepted_facts=[candidate],
        rejected_count=0,
        rejection_reasons={},
    )

    processor = GraphEventProcessor(repo=mock_repo, extractor=mock_extractor)

    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="worker-publish-1")
        assert claimed is not None
        assert claimed.id == event_id
        await processor.process_graph_event(db, event_id, worker_id="worker-publish-1")

    # Verify event completed and snapshots stored in PostgreSQL
    with SessionLocal() as db:
        ev = get_graph_event(db, event_id)
        assert ev.status == "COMPLETED"
        assert ev.completed_at is not None

        snaps = db.query(GraphFactSnapshot).filter(GraphFactSnapshot.paper_id == paper_id).all()
        assert len(snaps) == 1
        assert snaps[0].predicate == "EVALUATED_ON"
        assert snaps[0].generation_id == gen_id

    # Verify Neo4j upsert calls
    mock_repo.upsert_nodes.assert_called_once()
    mock_repo.upsert_facts.assert_called_once()
    mock_repo.retire_older_generations.assert_called_once_with(project_id, paper_id, gen_id)


@pytest.mark.anyio
async def test_failure_before_graph_write_reuses_snapshots():
    """Test 2: Failure before graph write leaves snapshots in PostgreSQL; retry reuses them."""
    with SessionLocal() as db:
        proj = create_project(db, ProjectCreate(name="Fail Before Write Proj"))
        paper = _setup_ready_paper(db, proj.id)
        event = create_or_enqueue_graph_event(db, proj.id, paper.id, action="UPSERT")
        db.commit()
        db.refresh(event)
        event_id = event.id
        paper_id = paper.id

    with SessionLocal() as db:
        chunk = db.query(PaperChunk).filter(PaperChunk.paper_id == paper_id).first()
        chunk_id = chunk.id

    candidate = GraphFactCandidate(
        subject=GraphEntitySchema(id="e1", name="Transformer", type=EntityType.METHOD),
        predicate=RelationshipPredicate.EVALUATED_ON,
        object=GraphEntitySchema(id="e2", name="WMT 2014", type=EntityType.DATASET),
        qualifiers=GraphQualifierSchema(metric="BLEU", result_value=28.4),
        provenance=GraphProvenanceSchema(
            paper_id=paper_id,
            chunk_id=chunk_id,
            page_number=1,
            exact_quote=QUOTE_77,
            char_start=0,
            char_end=77,
            document_sha256="d0c0" * 16,
        ),
    )

    mock_extractor = AsyncMock()
    mock_extractor.extract.return_value = ExtractionResult(
        accepted_entities=[candidate.subject, candidate.object],
        accepted_facts=[candidate],
        rejected_count=0,
        rejection_reasons={},
    )

    # First attempt: Neo4j raises connection error during upsert_nodes
    mock_repo_failing = MagicMock(spec=Neo4jRepository)
    mock_repo_failing.upsert_nodes.side_effect = ConnectionError("Bolt unavailable")

    processor_1 = GraphEventProcessor(repo=mock_repo_failing, extractor=mock_extractor)

    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="worker-fail-1")
        assert claimed is not None
        await processor_1.process_graph_event(db, event_id, worker_id="worker-fail-1")

    # Snapshots were committed to PostgreSQL BEFORE Neo4j write was attempted
    with SessionLocal() as db:
        ev = get_graph_event(db, event_id)
        assert ev.status == "PENDING"  # Retried
        assert ev.attempts == 1

        snaps = db.query(GraphFactSnapshot).filter(GraphFactSnapshot.paper_id == paper_id).all()
        assert len(snaps) == 1

    # Second attempt: Extractor must NOT be called again because snapshots are reused
    mock_repo_ok = MagicMock(spec=Neo4jRepository)
    mock_extractor_unused = AsyncMock()

    processor_2 = GraphEventProcessor(repo=mock_repo_ok, extractor=mock_extractor_unused)

    with SessionLocal() as db:
        # Reset updated_at to bypass backoff
        ev = get_graph_event(db, event_id)
        ev.updated_at = datetime.now(tz=UTC) - timedelta(seconds=120)
        db.commit()

        claimed = claim_next_graph_event(db, worker_id="worker-retry-2")
        assert claimed is not None
        await processor_2.process_graph_event(db, event_id, worker_id="worker-retry-2")

    # Extractor was never touched on retry
    mock_extractor_unused.extract.assert_not_called()

    # Event is now COMPLETED and Neo4j was updated
    with SessionLocal() as db:
        ev = get_graph_event(db, event_id)
        assert ev.status == "COMPLETED"

    mock_repo_ok.upsert_nodes.assert_called_once()
    mock_repo_ok.upsert_facts.assert_called_once()


@pytest.mark.anyio
async def test_failure_after_graph_commit_before_ack_is_replayable():
    """Test 3: Neo4j write succeeded, but failure happens before PostgreSQL ack.

    Retry executes MERGE on exact same fact IDs with zero duplicate facts.
    """
    with SessionLocal() as db:
        proj = create_project(db, ProjectCreate(name="Fail After Graph Write Proj"))
        paper = _setup_ready_paper(db, proj.id)
        event = create_or_enqueue_graph_event(db, proj.id, paper.id, action="UPSERT")
        db.commit()
        db.refresh(event)
        event_id = event.id
        paper_id = paper.id
        project_id = proj.id
        gen_id = event.generation_id

    # Seed verified snapshot directly
    with SessionLocal() as db:
        snap = GraphFactSnapshot(
            fact_id="fact-replay-test-1",
            project_id=project_id,
            paper_id=paper_id,
            generation_id=gen_id,
            event_id=event_id,
            subject_key="proj_transformer",
            subject_name="Transformer",
            subject_type="Method",
            predicate="ACHIEVES_RESULT",
            object_key="proj_wmt14",
            object_name="WMT 2014",
            object_type="Dataset",
            qualifiers={"metric": "BLEU", "result_value": 28.4},
            char_start=0,
            char_end=76,
            page_number=1,
            exact_quote=QUOTE_77,
            document_sha256="d0c0" * 16,
        )
        db.add(snap)
        db.commit()

    call_counts = {"upsert_nodes": 0, "upsert_facts": 0}

    def count_nodes(*args, **kwargs):
        call_counts["upsert_nodes"] += 1

    def count_facts(*args, **kwargs):
        call_counts["upsert_facts"] += 1

    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.upsert_nodes.side_effect = count_nodes
    mock_repo.upsert_facts.side_effect = count_facts

    processor = GraphEventProcessor(repo=mock_repo)

    # First run succeeds at Neo4j write but simulate DB crash during complete_graph_event
    with patch(
        "app.services.graphrag.processor.complete_graph_event",
        side_effect=RuntimeError("DB Ack failed"),
    ):
        with SessionLocal() as db:
            claimed = claim_next_graph_event(db, worker_id="worker-ack-fail")
            assert claimed is not None
            await processor.process_graph_event(db, event_id, worker_id="worker-ack-fail")

    assert call_counts["upsert_facts"] == 1

    # Event remains PENDING
    with SessionLocal() as db:
        ev = get_graph_event(db, event_id)
        assert ev.status == "PENDING"
        ev.updated_at = datetime.now(tz=UTC) - timedelta(seconds=120)
        db.commit()

    # Replay: normal run without failure
    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="worker-ack-retry")
        assert claimed is not None
        await processor.process_graph_event(db, event_id, worker_id="worker-ack-retry")

    assert call_counts["upsert_facts"] == 2  # MERGE executed again idempotently with same fact_id

    with SessionLocal() as db:
        ev = get_graph_event(db, event_id)
        assert ev.status == "COMPLETED"
        assert ev.completed_at is not None


@pytest.mark.anyio
async def test_stale_generation_does_not_retire_newer_graph():
    """Test 4: Older event completing after newer does NOT retire newer facts."""
    with SessionLocal() as db:
        proj = create_project(db, ProjectCreate(name="Stale Gen Proj"))
        paper = _setup_ready_paper(db, proj.id)

        # Event 1: older generation
        ev1 = GraphEvent(
            project_id=proj.id,
            paper_id=paper.id,
            action="UPSERT",
            generation_id="gen-older-v1",
            status="PENDING",
            created_at=datetime.now(tz=UTC) - timedelta(hours=2),
        )
        # Event 2: newer generation already completed
        ev2 = GraphEvent(
            project_id=proj.id,
            paper_id=paper.id,
            action="UPSERT",
            generation_id="gen-newer-v2",
            status="COMPLETED",
            created_at=datetime.now(tz=UTC) - timedelta(hours=1),
            completed_at=datetime.now(tz=UTC) - timedelta(minutes=30),
        )
        db.add(ev1)
        db.add(ev2)
        db.commit()
        db.refresh(ev1)
        db.refresh(ev2)
        ev1_id = ev1.id
        paper_id = paper.id
        project_id = proj.id

        # Snapshot for older generation
        snap1 = GraphFactSnapshot(
            fact_id="fact-older-1",
            project_id=project_id,
            paper_id=paper_id,
            generation_id="gen-older-v1",
            event_id=ev1_id,
            subject_key="proj_transformer",
            subject_name="Transformer",
            subject_type="Method",
            predicate="ACHIEVES_RESULT",
            object_key="proj_wmt14",
            object_name="WMT 2014",
            object_type="Dataset",
            char_start=0,
            char_end=76,
            page_number=1,
            exact_quote=QUOTE_77,
            document_sha256="d0c0" * 16,
        )
        db.add(snap1)
        db.commit()

    mock_repo = MagicMock(spec=Neo4jRepository)
    processor = GraphEventProcessor(repo=mock_repo)

    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="worker-stale")
        assert claimed is not None
        assert claimed.id == ev1_id
        await processor.process_graph_event(db, ev1_id, worker_id="worker-stale")

    # The stale generation is rejected before any graph writes or retirement.
    mock_repo.upsert_nodes.assert_not_called()
    mock_repo.upsert_facts.assert_not_called()
    mock_repo.retire_older_generations.assert_not_called()

    with SessionLocal() as db:
        ev1_refreshed = get_graph_event(db, ev1_id)
        assert ev1_refreshed.status == "FAILED"
        assert ev1_refreshed.error_code == "SUPERSEDED"


@pytest.mark.anyio
async def test_expired_graph_lease_cannot_publish():
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Expired Lease Proj"))
        paper = _setup_ready_paper(db, project.id)
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        event_id = event.id

    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="expired-worker")
        assert claimed is not None and claimed.id == event_id
        claimed.lease_expires_at = datetime.now(tz=UTC) - timedelta(seconds=1)
        db.commit()

    mock_repo = MagicMock(spec=Neo4jRepository)
    processor = GraphEventProcessor(repo=mock_repo)
    with SessionLocal() as db:
        await processor.process_graph_event(db, event_id, worker_id="expired-worker")

    mock_repo.upsert_nodes.assert_not_called()
    mock_repo.upsert_facts.assert_not_called()
    mock_repo.retire_older_generations.assert_not_called()


@pytest.mark.anyio
async def test_repository_unavailable_never_completes_or_extracts():
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="No Graph Repo Proj"))
        paper = _setup_ready_paper(db, project.id)
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        event_id = event.id

    extractor = AsyncMock()
    processor = GraphEventProcessor(
        settings=Settings(graphrag_enabled=True), repo=None, extractor=extractor
    )
    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="repo-missing")
        assert claimed is not None
        completed = await processor.process_graph_event(db, event_id, worker_id="repo-missing")
        assert completed is False

    with SessionLocal() as db:
        persisted = get_graph_event(db, event_id)
        assert persisted.status == "PENDING"
        assert persisted.error_code == "NEO4J_UNAVAILABLE"
    extractor.extract.assert_not_awaited()


@pytest.mark.anyio
async def test_failed_repository_connectivity_retries_without_extraction():
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Offline Graph Repo Proj"))
        paper = _setup_ready_paper(db, project.id)
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        event_id = event.id

    repo = MagicMock(spec=Neo4jRepository)
    repo.verify_connectivity.return_value = False
    extractor = AsyncMock()
    processor = GraphEventProcessor(
        settings=Settings(graphrag_enabled=True), repo=repo, extractor=extractor
    )
    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="repo-offline")
        assert claimed is not None
        await processor.process_graph_event(db, event_id, worker_id="repo-offline")

    with SessionLocal() as db:
        persisted = get_graph_event(db, event_id)
        assert persisted.status == "PENDING"
        assert persisted.error_code == "NEO4J_UNAVAILABLE"
    extractor.extract.assert_not_awaited()
    repo.upsert_facts.assert_not_called()


@pytest.mark.anyio
async def test_extraction_selector_honors_configured_batch_limit():
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Bounded Extraction Proj"))
        paper = _setup_ready_paper(db, project.id)
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        event_id = event.id

    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.verify_connectivity.return_value = True
    extractor = AsyncMock()
    extractor.extract.return_value = ExtractionResult(
        accepted_entities=[], accepted_facts=[], rejected_count=0, rejection_reasons={}
    )
    processor = GraphEventProcessor(
        settings=Settings(graphrag_enabled=True, graph_batch_limit=2),
        repo=mock_repo,
        extractor=extractor,
    )
    with patch(
        "app.services.graphrag.processor.select_extraction_inputs", return_value=[MagicMock()]
    ) as selector:
        with SessionLocal() as db:
            claimed = claim_next_graph_event(db, worker_id="bounded-extractor")
            assert claimed is not None
            await processor.process_graph_event(db, event_id, worker_id="bounded-extractor")

    assert selector.call_args.kwargs["max_chunks"] == 2


@pytest.mark.anyio
async def test_duplicate_delivery_idempotency():
    """Test 5: Replay idempotency: Multiple duplicate delivery runs on the exact same event
    produce identical graph state and identical fact counts.
    """
    with SessionLocal() as db:
        proj = create_project(db, ProjectCreate(name="Duplicate Delivery Proj"))
        paper = _setup_ready_paper(db, proj.id)
        event = create_or_enqueue_graph_event(db, proj.id, paper.id, action="UPSERT")
        db.commit()
        db.refresh(event)
        event_id = event.id
        paper_id = paper.id
        project_id = proj.id
        gen_id = event.generation_id

        snap = GraphFactSnapshot(
            fact_id="fact-dup-1",
            project_id=project_id,
            paper_id=paper_id,
            generation_id=gen_id,
            event_id=event_id,
            subject_key="proj_transformer",
            subject_name="Transformer",
            subject_type="Method",
            predicate="PROPOSES_METHOD",
            object_key="proj_self_attention",
            object_name="Self-Attention",
            object_type="Concept",
            char_start=0,
            char_end=76,
            page_number=1,
            exact_quote=QUOTE_77,
            document_sha256="d0c0" * 16,
        )
        db.add(snap)
        db.commit()

    mock_repo = MagicMock(spec=Neo4jRepository)
    processor = GraphEventProcessor(repo=mock_repo)

    # First delivery
    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="worker-dup-1")
        assert claimed is not None
        await processor.process_graph_event(db, event_id, worker_id="worker-dup-1")

    # Reset status back to PENDING to simulate duplicate delivery
    with SessionLocal() as db:
        ev = get_graph_event(db, event_id)
        assert ev.status == "COMPLETED"
        ev.status = "PENDING"
        ev.updated_at = datetime.now(tz=UTC) - timedelta(seconds=120)
        db.commit()

    # Second delivery
    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="worker-dup-2")
        assert claimed is not None
        await processor.process_graph_event(db, event_id, worker_id="worker-dup-2")

    # In PostgreSQL, snapshot count remains exactly 1
    with SessionLocal() as db:
        ev = get_graph_event(db, event_id)
        assert ev.status == "COMPLETED"
        snaps = db.query(GraphFactSnapshot).filter(GraphFactSnapshot.paper_id == paper_id).all()
        assert len(snaps) == 1
        assert snaps[0].fact_id == "fact-dup-1"


@pytest.mark.anyio
async def test_empty_reindex_retires_old_generation_before_completion():
    """An empty newer generation must not leave stale facts queryable."""
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Empty Reindex Proj"))
        paper = _setup_ready_paper(db, project.id)
        db.query(ChunkElement).filter(
            ChunkElement.chunk_id.in_(
                db.query(PaperChunk.id).filter(PaperChunk.paper_id == paper.id)
            )
        ).delete(synchronize_session=False)
        db.query(PaperChunk).filter(PaperChunk.paper_id == paper.id).delete()
        db.commit()

        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        event_id = event.id
        generation_id = event.generation_id
        project_id = project.id
        paper_id = paper.id

    repo = MagicMock(spec=Neo4jRepository)
    processor = GraphEventProcessor(repo=repo, extractor=AsyncMock())
    call_order = []
    repo.retire_older_generations.side_effect = lambda *args: call_order.append("retire")

    def record_completion(*args, **kwargs):
        call_order.append("complete")
        return complete_graph_event(*args, **kwargs)

    with patch(
        "app.services.graphrag.processor.complete_graph_event",
        side_effect=record_completion,
    ):
        with SessionLocal() as db:
            claimed = claim_next_graph_event(db, worker_id="empty-reindex")
            assert claimed is not None
            completed = await processor.process_graph_event(db, event_id, worker_id="empty-reindex")
            assert completed is True

    repo.retire_older_generations.assert_called_once_with(project_id, paper_id, generation_id)
    assert call_order == ["retire", "complete"]
    with SessionLocal() as db:
        persisted = get_graph_event(db, event_id)
        assert persisted.status == "COMPLETED"


@pytest.mark.anyio
@pytest.mark.parametrize("rejected_at", ["verification", "extraction"])
async def test_rejected_extracted_facts_fail_without_publishing_empty_generation(rejected_at):
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Rejected Facts Proj"))
        paper = _setup_ready_paper(db, project.id)
        chunk_id = db.query(PaperChunk.id).filter(PaperChunk.paper_id == paper.id).scalar()
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        event_id = event.id
        paper_id = paper.id

    candidate = GraphFactCandidate(
        subject=GraphEntitySchema(id="e1", name="Transformer", type=EntityType.METHOD),
        predicate=RelationshipPredicate.EVALUATED_ON,
        object=GraphEntitySchema(id="e2", name="WMT 2014", type=EntityType.DATASET),
        qualifiers=GraphQualifierSchema(metric="BLEU", result_value=28.4),
        provenance=GraphProvenanceSchema(
            paper_id=paper_id,
            chunk_id=chunk_id,
            page_number=1,
            exact_quote=QUOTE_77,
            char_start=0,
            char_end=len(QUOTE_77),
            document_sha256="d0c0" * 16,
        ),
    )
    extractor = AsyncMock()
    if rejected_at == "verification":
        extractor.extract.return_value = ExtractionResult(
            accepted_entities=[candidate.subject, candidate.object],
            accepted_facts=[candidate],
            rejected_count=0,
            rejection_reasons={},
        )
    else:
        extractor.extract.return_value = ExtractionResult(
            accepted_entities=[],
            accepted_facts=[],
            rejected_count=1,
            rejection_reasons={"INVALID_ENDPOINT_TYPES": 1},
        )
    repo = MagicMock(spec=Neo4jRepository)
    processor = GraphEventProcessor(repo=repo, extractor=extractor)

    with (
        patch(
            "app.services.graphrag.processor.verify_candidate_fact",
            return_value=(False, "UNRESOLVED_ANCHOR"),
        ),
        patch("app.services.graphrag.processor.logger.info") as log_info,
    ):
        with SessionLocal() as db:
            claimed = claim_next_graph_event(db, worker_id="rejected-facts")
            assert claimed is not None
            await processor.process_graph_event(db, event_id, worker_id="rejected-facts")

    with SessionLocal() as db:
        persisted = get_graph_event(db, event_id)
        assert persisted.status == "FAILED"
        assert persisted.error_code == "NO_VERIFIED_FACTS"
        assert persisted.completed_at is None
        assert db.query(GraphFactSnapshot).filter_by(paper_id=paper_id).count() == 0
        assert db.get(Paper, paper_id).status == "READY"

    repo.upsert_facts.assert_not_called()
    repo.retire_older_generations.assert_not_called()
    reason_logs = repr(log_info.call_args_list)
    expected_reason = (
        "UNRESOLVED_ANCHOR" if rejected_at == "verification" else "INVALID_ENDPOINT_TYPES"
    )
    assert expected_reason in reason_logs
    assert QUOTE_77 not in reason_logs
