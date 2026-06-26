"""Tests for GraphRAG reconciliation, drift detection, active generation retirement,
multi-paper isolation, and snapshot rebuilds (Task 5.13).
"""

from __future__ import annotations

from collections.abc import Generator
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from neo4j import GraphDatabase

from app.crud.graph import (
    claim_next_graph_event,
    create_or_enqueue_graph_event,
    get_graph_event,
)
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.models import GraphEvent, GraphFactSnapshot, Job, Paper, Project
from app.db.session import SessionLocal, create_tables
from app.schemas.project import ProjectCreate
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.processor import GraphEventProcessor
from app.services.graphrag.reconciliation import (
    rebuild_paper_graph_from_snapshots,
    reconcile_graph_drift,
    validate_snapshot_provenance,
)


@pytest.fixture(autouse=True)
def clean_database():
    """Ensure clean PostgreSQL tables before each test."""
    create_tables()
    with SessionLocal() as db:
        db.query(GraphFactSnapshot).delete()
        db.query(GraphEvent).delete()
        db.query(Job).delete()
        db.query(Paper).delete()
        db.query(Project).delete()
        db.commit()


@pytest.fixture
def real_repo(disposable_neo4j_uri: str) -> Generator[Neo4jRepository, None, None]:
    """Connect only to the explicitly acknowledged disposable test target."""
    driver = GraphDatabase.driver(disposable_neo4j_uri, auth=None)
    driver.verify_connectivity()
    repo = Neo4jRepository(driver=driver, database="neo4j")
    repo.ensure_schema()
    yield repo
    driver.close()


def _create_snapshot(
    db,
    project_id: UUID,
    paper_id: UUID,
    generation_id: str,
    fact_id: str,
    subject_key: str,
    predicate: str,
    object_key: str,
    document_sha256: str = "abcdef0123456789" * 4,
    event_id: UUID | None = None,
    char_start: int = 10,
    char_end: int = 50,
    page_number: int = 1,
    exact_quote: str = "Test quote for grounding.",
) -> GraphFactSnapshot:
    snapshot = GraphFactSnapshot(
        fact_id=fact_id,
        project_id=project_id,
        paper_id=paper_id,
        generation_id=generation_id,
        event_id=event_id,
        subject_key=subject_key,
        subject_name=subject_key.replace("_", " ").title(),
        subject_type="Method",
        predicate=predicate,
        object_key=object_key,
        object_name=object_key.replace("_", " ").title(),
        object_type="Dataset",
        char_start=char_start,
        char_end=char_end,
        page_number=page_number,
        exact_quote=exact_quote,
        document_sha256=document_sha256,
        qualifiers={"split": "test"},
    )
    db.add(snapshot)
    return snapshot


# =============================================================================
# Test 1: Active Generation Retirement
# =============================================================================


@pytest.mark.anyio
async def test_active_generation_retirement(real_repo: Neo4jRepository) -> None:
    """Test 1: Active generation retirement: Upsert gen 1 facts, then upsert gen 2 facts.

    Verify gen 1 facts are retired from Neo4j while gen 2 facts remain active.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Gen Retirement Project"))
        paper = create_paper(db, project.id, "retire_gen.pdf", "retire_gen.pdf")
        paper.status = "READY"
        paper.document_sha256 = "hash1111" * 8
        db.commit()
        project_id = project.id
        paper_id = paper.id

    try:
        processor = GraphEventProcessor(repo=real_repo)

        # 1. Enqueue and process Gen 1
        with SessionLocal() as db:
            ev1 = create_or_enqueue_graph_event(
                db, project_id, paper_id, action="UPSERT", generation_id="gen-retire-1"
            )
            db.commit()
            db.refresh(ev1)
            ev1_id = ev1.id

            _create_snapshot(
                db,
                project_id=project_id,
                paper_id=paper_id,
                generation_id="gen-retire-1",
                fact_id=f"fact-retire-1-{uuid4().hex[:8]}",
                subject_key="transformer",
                predicate="EVALUATED_ON",
                object_key="wmt14",
                event_id=ev1_id,
                document_sha256="hash1111" * 8,
            )
            db.commit()

            claimed = claim_next_graph_event(db, worker_id="worker-gen1")
            assert claimed is not None and claimed.id == ev1_id
            await processor.process_graph_event(db, ev1_id, worker_id="worker-gen1")

        # Verify Gen 1 fact is present in Neo4j
        with SessionLocal() as db:
            ev1_persisted = get_graph_event(db, ev1_id)
            assert ev1_persisted.status == "COMPLETED"
            gen1_fact_id = ev1_persisted.snapshots[0].fact_id

        fact_gen1 = real_repo.get_fact_by_id(project_id, gen1_fact_id)
        assert fact_gen1 is not None
        assert fact_gen1["generation_id"] == "gen-retire-1"

        # 2. Enqueue and process Gen 2 for the same paper
        with SessionLocal() as db:
            ev2 = create_or_enqueue_graph_event(
                db, project_id, paper_id, action="UPSERT", generation_id="gen-retire-2"
            )
            db.commit()
            db.refresh(ev2)
            ev2_id = ev2.id

            _create_snapshot(
                db,
                project_id=project_id,
                paper_id=paper_id,
                generation_id="gen-retire-2",
                fact_id=f"fact-retire-2-{uuid4().hex[:8]}",
                subject_key="flash_attention",
                predicate="EVALUATED_ON",
                object_key="glue",
                event_id=ev2_id,
                document_sha256="hash1111" * 8,
            )
            db.commit()

            claimed2 = claim_next_graph_event(db, worker_id="worker-gen2")
            assert claimed2 is not None and claimed2.id == ev2_id
            await processor.process_graph_event(db, ev2_id, worker_id="worker-gen2")

        # Verify Gen 2 is active and Gen 1 is retired
        with SessionLocal() as db:
            ev2_persisted = get_graph_event(db, ev2_id)
            assert ev2_persisted.status == "COMPLETED"
            gen2_fact_id = ev2_persisted.snapshots[0].fact_id

        # Gen 1 fact must be retired from Neo4j
        retired_fact = real_repo.get_fact_by_id(project_id, gen1_fact_id)
        assert retired_fact is None, "Gen 1 fact should have been retired from Neo4j"

        # Gen 2 fact must remain active in Neo4j
        active_fact = real_repo.get_fact_by_id(project_id, gen2_fact_id)
        assert active_fact is not None, "Gen 2 fact should remain active in Neo4j"
        assert active_fact["generation_id"] == "gen-retire-2"

    finally:
        real_repo.delete_project_graph(project_id)


# =============================================================================
# Test 2: Multi-Paper Isolation
# =============================================================================


@pytest.mark.anyio
async def test_multi_paper_isolation(real_repo: Neo4jRepository) -> None:
    """Test 2: Multi-paper isolation: Two papers in the same project have facts in Neo4j.

    Retiring or deleting one paper's facts removes ONLY that paper's facts;
    the second paper's facts are preserved.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Multi Paper Project"))
        paper_a = create_paper(db, project.id, "paper_a.pdf", "paper_a.pdf")
        paper_a.status = "READY"
        paper_b = create_paper(db, project.id, "paper_b.pdf", "paper_b.pdf")
        paper_b.status = "READY"
        db.commit()
        project_id = project.id
        paper_a_id = paper_a.id
        paper_b_id = paper_b.id

    try:
        processor = GraphEventProcessor(repo=real_repo)

        # 1. Publish Paper A facts (gen-a-1)
        with SessionLocal() as db:
            ev_a1 = create_or_enqueue_graph_event(
                db, project_id, paper_a_id, action="UPSERT", generation_id="gen-a-1"
            )
            db.commit()
            db.refresh(ev_a1)
            ev_a1_id = ev_a1.id

            s_a = _create_snapshot(
                db,
                project_id=project_id,
                paper_id=paper_a_id,
                generation_id="gen-a-1",
                fact_id=f"fact-a1-{uuid4().hex[:8]}",
                subject_key="model_a",
                predicate="EVALUATED_ON",
                object_key="dataset_a",
                event_id=ev_a1_id,
            )
            db.commit()
            fact_a1_id = s_a.fact_id

            claimed = claim_next_graph_event(db, worker_id="worker-a")
            assert claimed is not None and claimed.id == ev_a1_id
            await processor.process_graph_event(db, ev_a1_id, worker_id="worker-a")

        # 2. Publish Paper B facts (gen-b-1)
        with SessionLocal() as db:
            ev_b1 = create_or_enqueue_graph_event(
                db, project_id, paper_b_id, action="UPSERT", generation_id="gen-b-1"
            )
            db.commit()
            db.refresh(ev_b1)
            ev_b1_id = ev_b1.id

            s_b = _create_snapshot(
                db,
                project_id=project_id,
                paper_id=paper_b_id,
                generation_id="gen-b-1",
                fact_id=f"fact-b1-{uuid4().hex[:8]}",
                subject_key="model_b",
                predicate="EVALUATED_ON",
                object_key="dataset_b",
                event_id=ev_b1_id,
            )
            db.commit()
            fact_b1_id = s_b.fact_id

            claimed = claim_next_graph_event(db, worker_id="worker-b")
            assert claimed is not None and claimed.id == ev_b1_id
            await processor.process_graph_event(db, ev_b1_id, worker_id="worker-b")

        # Both facts must exist in Neo4j
        assert real_repo.get_fact_by_id(project_id, fact_a1_id) is not None
        assert real_repo.get_fact_by_id(project_id, fact_b1_id) is not None

        # 3. Process DELETE event for Paper A
        with SessionLocal() as db:
            ev_del_a = create_or_enqueue_graph_event(
                db, project_id, paper_a_id, action="DELETE", generation_id="gen-a-del"
            )
            db.commit()
            db.refresh(ev_del_a)
            ev_del_a_id = ev_del_a.id

            claimed = claim_next_graph_event(db, worker_id="worker-del")
            assert claimed is not None and claimed.id == ev_del_a_id
            await processor.process_graph_event(db, ev_del_a_id, worker_id="worker-del")

        # Verify: Paper A fact removed, Paper B fact completely preserved!
        assert real_repo.get_fact_by_id(project_id, fact_a1_id) is None
        preserved_b = real_repo.get_fact_by_id(project_id, fact_b1_id)
        assert preserved_b is not None
        assert preserved_b["paper_id"] == str(paper_b_id)

    finally:
        real_repo.delete_project_graph(project_id)


# =============================================================================
# Test 3: Reconciliation Sweep Detects Absent Papers
# =============================================================================


def test_reconciliation_sweep_detects_absent_papers(real_repo: Neo4jRepository) -> None:
    """Test 3: Reconciliation sweep detects absent papers: Paper facts exist in Neo4j for a paper
    deleted from PostgreSQL. `reconcile_graph_drift` cleans up the orphaned paper facts from Neo4j.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Sweep Absent Project"))
        paper_live = create_paper(db, project.id, "paper_live.pdf", "paper_live.pdf")
        paper_live.status = "READY"
        paper_orphan = create_paper(db, project.id, "paper_orphan.pdf", "paper_orphan.pdf")
        paper_orphan.status = "READY"
        db.commit()
        project_id = project.id
        paper_live_id = paper_live.id
        paper_orphan_id = paper_orphan.id

        # Mark completed event for paper_live with active gen-live-2
        ev_live = create_or_enqueue_graph_event(
            db, project_id, paper_live_id, action="UPSERT", generation_id="gen-live-2"
        )
        ev_live.status = "COMPLETED"
        ev_live.completed_at = datetime.now(tz=UTC)
        db.commit()

    try:
        # Seed facts directly into Neo4j:
        # 1. Live paper: active gen-live-2 facts AND a stale gen-live-1 fact
        live_fact_active_id = f"fact-live-2-{uuid4().hex[:8]}"
        live_fact_stale_id = f"fact-live-1-{uuid4().hex[:8]}"
        real_repo.upsert_facts(
            project_id=project_id,
            paper_id=paper_live_id,
            generation_id="gen-live-2",
            facts=[
                {
                    "fact_id": live_fact_active_id,
                    "subject_key": "live_model",
                    "predicate": "USES",
                    "object_key": "live_tech",
                }
            ],
        )
        real_repo.upsert_facts(
            project_id=project_id,
            paper_id=paper_live_id,
            generation_id="gen-live-1",
            facts=[
                {
                    "fact_id": live_fact_stale_id,
                    "subject_key": "live_model",
                    "predicate": "USES_OLD",
                    "object_key": "old_tech",
                }
            ],
        )

        # 2. Orphan paper: facts in Neo4j
        orphan_fact_id = f"fact-orphan-{uuid4().hex[:8]}"
        real_repo.upsert_facts(
            project_id=project_id,
            paper_id=paper_orphan_id,
            generation_id="gen-orphan-1",
            facts=[
                {
                    "fact_id": orphan_fact_id,
                    "subject_key": "orphan_model",
                    "predicate": "EVALUATED_ON",
                    "object_key": "orphan_dataset",
                }
            ],
        )

        # Both papers have facts in Neo4j
        assert real_repo.get_fact_by_id(project_id, live_fact_active_id) is not None
        assert real_repo.get_fact_by_id(project_id, live_fact_stale_id) is not None
        assert real_repo.get_fact_by_id(project_id, orphan_fact_id) is not None

        # Now DELETE paper_orphan from PostgreSQL to create drift
        with SessionLocal() as db:
            p_to_del = db.get(Paper, paper_orphan_id)
            assert p_to_del is not None
            db.delete(p_to_del)
            db.commit()

        # Execute reconciliation sweep
        with SessionLocal() as db:
            summary = reconcile_graph_drift(db=db, repo=real_repo, project_id=project_id, limit=50)

        # Verify summary counts
        assert summary["project_id"] == str(project_id)
        assert summary["scanned_papers"] >= 2
        assert summary["orphaned_papers_cleaned"] == 1
        assert summary["retired_facts_count"] >= 1  # Stale gen-live-1 fact was retired!

        # Orphaned paper facts must be removed from Neo4j
        assert real_repo.get_fact_by_id(project_id, orphan_fact_id) is None

        # Live paper active facts must remain
        assert real_repo.get_fact_by_id(project_id, live_fact_active_id) is not None

        # Live paper stale generation must have been retired
        assert real_repo.get_fact_by_id(project_id, live_fact_stale_id) is None

    finally:
        real_repo.delete_project_graph(project_id)


# =============================================================================
# Test 4: Rebuild From Snapshots
# =============================================================================


def test_rebuild_from_snapshots(real_repo: Neo4jRepository) -> None:
    """Test 4: Rebuild from snapshots: Drops paper facts in Neo4j, replays PostgreSQL snapshots,
    and confirms exact facts and nodes are restored.
    """
    valid_sha256 = "c0ffee" * 10 + "abcd"
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Rebuild Project"))
        paper = create_paper(db, project.id, "rebuild.pdf", "rebuild.pdf")
        paper.status = "READY"
        paper.document_sha256 = valid_sha256
        db.commit()
        project_id = project.id
        paper_id = paper.id

        generation_id = "gen-rebuild-v1"

        # Create 2 valid snapshots
        f1_id = f"fact-rb-1-{uuid4().hex[:8]}"
        f2_id = f"fact-rb-2-{uuid4().hex[:8]}"
        _create_snapshot(
            db,
            project_id=project_id,
            paper_id=paper_id,
            generation_id=generation_id,
            fact_id=f1_id,
            subject_key="transformer_rb",
            predicate="PROPOSES",
            object_key="self_attention",
            document_sha256=valid_sha256,
            char_start=0,
            char_end=25,
            page_number=1,
            exact_quote="We propose the Transformer.",
        )
        _create_snapshot(
            db,
            project_id=project_id,
            paper_id=paper_id,
            generation_id=generation_id,
            fact_id=f2_id,
            subject_key="self_attention",
            predicate="USED_IN",
            object_key="translation_task",
            document_sha256=valid_sha256,
            char_start=30,
            char_end=75,
            page_number=2,
            exact_quote="Self-attention is used in translation.",
        )

        # Create 1 invalid snapshot (stale document_sha256)
        invalid_f_id = f"fact-rb-invalid-{uuid4().hex[:8]}"
        _create_snapshot(
            db,
            project_id=project_id,
            paper_id=paper_id,
            generation_id=generation_id,
            fact_id=invalid_f_id,
            subject_key="stale_model",
            predicate="EVALUATED_ON",
            object_key="stale_dataset",
            document_sha256="stale_mismatched_hash" * 3,
            char_start=10,
            char_end=50,
            page_number=1,
            exact_quote="Stale unverified quote.",
        )
        db.commit()

    try:
        # Initial state: Drop paper facts in Neo4j
        real_repo.delete_paper_facts(project_id, paper_id)
        assert real_repo.get_fact_by_id(project_id, f1_id) is None
        assert real_repo.get_fact_by_id(project_id, f2_id) is None

        # Rebuild paper graph strictly from PostgreSQL snapshots
        with SessionLocal() as db:
            result = rebuild_paper_graph_from_snapshots(
                db=db,
                repo=real_repo,
                project_id=project_id,
                paper_id=paper_id,
                generation_id=generation_id,
            )

        assert result["status"] == "SUCCESS"
        assert result["scanned_snapshots"] == 3
        assert result["rejected_snapshots"] == 1
        assert result["rebuilt_facts_count"] == 2

        # Verify fact 1 was faithfully restored in Neo4j
        f1_restored = real_repo.get_fact_by_id(project_id, f1_id)
        assert f1_restored is not None
        assert f1_restored["predicate"] == "PROPOSES"
        assert f1_restored["exact_quote"] == "We propose the Transformer."
        assert f1_restored["page_number"] == 1
        assert f1_restored["char_start"] == 0
        assert f1_restored["char_end"] == 25
        assert f1_restored["subject_key"] == "transformer_rb"
        assert f1_restored["object_key"] == "self_attention"

        # Verify fact 2 was faithfully restored in Neo4j
        f2_restored = real_repo.get_fact_by_id(project_id, f2_id)
        assert f2_restored is not None
        assert f2_restored["predicate"] == "USED_IN"
        assert f2_restored["exact_quote"] == "Self-attention is used in translation."
        assert f2_restored["page_number"] == 2

        # Verify invalid snapshot was rejected and NOT inserted into Neo4j
        assert real_repo.get_fact_by_id(project_id, invalid_f_id) is None

        # Verify entity nodes exist in Neo4j
        n1 = real_repo.get_node_by_key(project_id, "transformer_rb")
        assert n1 is not None
        assert n1["name"] == "Transformer Rb"

    finally:
        real_repo.delete_project_graph(project_id)


# =============================================================================
# Test 5: Invariant Verification (No PDF/Storage, No Postgres Paper Mutation)
# =============================================================================


def test_invariants_no_storage_no_paper_mutation() -> None:
    """Test 5: Invariant verification: Verify that rebuild and reconciliation NEVER touch
    PDF storage or modify paper records in PostgreSQL.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Invariant Project"))
        paper = create_paper(db, project.id, "invariant.pdf", "storage/invariant.pdf")
        paper.status = "READY"
        paper.document_sha256 = "cafebabe" * 8
        paper.page_count = 12
        paper.error_message = None
        db.commit()
        db.refresh(paper)
        project_id = project.id
        paper_id = paper.id

        # Snapshot initial values
        initial_status = paper.status
        initial_updated_at = paper.updated_at
        initial_filename = paper.filename
        initial_storage_path = paper.storage_path
        initial_sha256 = paper.document_sha256

        # Add a valid snapshot
        _create_snapshot(
            db,
            project_id=project_id,
            paper_id=paper_id,
            generation_id="gen-inv-1",
            fact_id=f"fact-inv-{uuid4().hex[:8]}",
            subject_key="inv_subject",
            predicate="EVALUATED_ON",
            object_key="inv_object",
            document_sha256="cafebabe" * 8,
        )
        db.commit()

    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_session = MagicMock()
    mock_session.execute_read.return_value = [str(paper_id)]
    mock_repo._get_session.return_value.__enter__.return_value = mock_session
    mock_repo.upsert_facts.return_value = 1
    mock_repo.retire_older_generations.return_value = 0
    mock_repo.delete_paper_facts.return_value = 0

    # Patch storage facilities to guarantee they are never touched
    with (
        patch("app.storage.factory.get_storage") as mock_factory,
        patch("app.storage.local.LocalStorage") as mock_local_storage,
    ):
        with SessionLocal() as db:
            reconcile_summary = reconcile_graph_drift(
                db=db, repo=mock_repo, project_id=project_id, limit=10
            )
            assert reconcile_summary["scanned_papers"] == 1

            rebuild_summary = rebuild_paper_graph_from_snapshots(
                db=db, repo=mock_repo, project_id=project_id, paper_id=paper_id
            )
            assert rebuild_summary["status"] == "SUCCESS"

        # 1. Assert PDF storage was NEVER touched
        mock_factory.assert_not_called()
        mock_local_storage.assert_not_called()

    # 2. Assert Paper record in PostgreSQL was NEVER modified
    with SessionLocal() as db:
        refreshed_paper = db.get(Paper, paper_id)
        assert refreshed_paper is not None
        assert refreshed_paper.status == initial_status == "READY"
        assert refreshed_paper.updated_at == initial_updated_at
        assert refreshed_paper.error_message is None
        assert refreshed_paper.filename == initial_filename
        assert refreshed_paper.storage_path == initial_storage_path
        assert refreshed_paper.document_sha256 == initial_sha256


# =============================================================================
# Additional Unit Tests for Provenance Validation & Error Cases
# =============================================================================


def test_validate_snapshot_provenance_edge_cases() -> None:
    """Verify validate_snapshot_provenance rejects invalid char ranges, quotes, hashes, etc."""
    valid_snap = MagicMock(spec=GraphFactSnapshot)
    valid_snap.page_number = 1
    valid_snap.char_start = 0
    valid_snap.char_end = 20
    valid_snap.exact_quote = "Valid verbatim quote."
    valid_snap.document_sha256 = "hash123"
    valid_snap.subject_key = "sub"
    valid_snap.object_key = "obj"
    valid_snap.predicate = "USES"

    # Base valid
    assert validate_snapshot_provenance(valid_snap, paper_sha256="hash123") is True

    # Page number < 1
    valid_snap.page_number = 0
    assert validate_snapshot_provenance(valid_snap, paper_sha256="hash123") is False
    valid_snap.page_number = 1

    # Negative char_start
    valid_snap.char_start = -1
    assert validate_snapshot_provenance(valid_snap, paper_sha256="hash123") is False
    valid_snap.char_start = 0

    # char_start > char_end
    valid_snap.char_start = 30
    valid_snap.char_end = 20
    assert validate_snapshot_provenance(valid_snap, paper_sha256="hash123") is False
    valid_snap.char_start = 0
    valid_snap.char_end = 20

    # Empty quote
    valid_snap.exact_quote = "   "
    assert validate_snapshot_provenance(valid_snap, paper_sha256="hash123") is False
    valid_snap.exact_quote = "Valid verbatim quote."

    # Mismatched paper sha256
    assert validate_snapshot_provenance(valid_snap, paper_sha256="different_hash") is False

    # A snapshot accepted by an older verifier must not reintroduce a false
    # achievement edge during a rebuild.
    valid_snap.predicate = "ACHIEVES_RESULT"
    valid_snap.exact_quote = "BERTLARGE (L=24, H=1024, A=16, Total Parameters=340M)"
    valid_snap.char_end = len(valid_snap.exact_quote)
    assert validate_snapshot_provenance(valid_snap, paper_sha256="hash123") is False
    valid_snap.exact_quote = "ELMo advances the state of the art for several major NLP benchmarks"
    valid_snap.char_end = len(valid_snap.exact_quote)
    assert validate_snapshot_provenance(valid_snap, paper_sha256="hash123") is True


def test_rebuild_paper_not_found_raises() -> None:
    """Verify rebuild_paper_graph_from_snapshots raises ValueError when paper does not exist."""
    mock_repo = MagicMock(spec=Neo4jRepository)
    with SessionLocal() as db:
        with pytest.raises(ValueError, match="not found in project"):
            rebuild_paper_graph_from_snapshots(
                db=db, repo=mock_repo, project_id=uuid4(), paper_id=uuid4()
            )


def test_rebuild_no_snapshots_returns_empty_summary() -> None:
    """Verify rebuild returns NO_SNAPSHOTS when paper has no snapshots."""
    mock_repo = MagicMock(spec=Neo4jRepository)
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="No Snapshots Project"))
        paper = create_paper(db, project.id, "empty.pdf", "empty.pdf")
        paper.status = "READY"
        db.commit()

        summary = rebuild_paper_graph_from_snapshots(
            db=db, repo=mock_repo, project_id=project.id, paper_id=paper.id
        )
        assert summary["status"] == "NO_SNAPSHOTS"
        assert summary["rebuilt_facts_count"] == 0
