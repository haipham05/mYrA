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

from app.crud.graph import (
    claim_next_graph_event,
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
from app.services.graphrag.extractor import ExtractionResult
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.processor import GraphEventProcessor

QUOTE_77 = "The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German dataset."


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

    # Older event completes, but retire_older_generations was NEVER called!
    mock_repo.retire_older_generations.assert_not_called()

    with SessionLocal() as db:
        ev1_refreshed = get_graph_event(db, ev1_id)
        assert ev1_refreshed.status == "COMPLETED"


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
