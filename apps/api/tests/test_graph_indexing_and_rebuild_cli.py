"""Tests for GraphRAG indexing, scoped rebuild, and CLI entrypoints (Tasks 5.20 & 5.21)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.cli.graph import main, main_index, main_rebuild
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.models import GraphEvent, GraphFactSnapshot, Job, Paper, Project
from app.db.session import SessionLocal, create_tables
from app.schemas.project import ProjectCreate
from app.services.graphrag.indexing import enqueue_existing_papers_for_graph
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.reconciliation import scoped_rebuild_project_graph


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
# Test 1: Dry-Run Indexing
# =============================================================================


def test_indexing_dry_run_changes_nothing():
    """Test 1: Dry-run for existing paper indexing changes nothing in PostgreSQL."""
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Dry Run Project"))
        paper = create_paper(db, project.id, "p1.pdf", "p1.pdf")
        paper.status = "READY"
        paper.document_sha256 = "hash1111" * 8
        db.commit()
        project_id = project.id
        paper_id = paper.id

    with SessionLocal() as db:
        res = enqueue_existing_papers_for_graph(db, project_id=project_id, dry_run=True)
        assert res["dry_run"] is True
        assert res["eligible_paper_ids"] == [str(paper_id)]
        assert res["enqueued_count"] == 0
        assert res["skipped_count"] == 0
        assert res["target_project_id"] == str(project_id)

        # Confirm PostgreSQL is untouched: no graph events exist
        events = db.query(GraphEvent).all()
        assert len(events) == 0


# =============================================================================
# Test 2: Explicit Opt-In Indexing
# =============================================================================


def test_indexing_explicit_opt_in_enqueues_paper():
    """Test 2: Explicit opt-in (dry_run=False) enqueues selected READY paper once."""
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Opt-in Project"))
        paper = create_paper(db, project.id, "p2.pdf", "p2.pdf")
        paper.status = "READY"
        paper.document_sha256 = "hash2222" * 8
        db.commit()
        project_id = project.id
        paper_id = paper.id

    with SessionLocal() as db:
        res = enqueue_existing_papers_for_graph(db, project_id=project_id, dry_run=False)
        assert res["dry_run"] is False
        assert res["eligible_paper_ids"] == [str(paper_id)]
        assert res["enqueued_count"] == 1
        assert res["skipped_count"] == 0

        # Confirm GraphEvent was created and committed in PostgreSQL
        events = db.query(GraphEvent).filter(GraphEvent.paper_id == paper_id).all()
        assert len(events) == 1
        assert events[0].status == "PENDING"
        assert events[0].action == "UPSERT"
        assert events[0].project_id == project_id


# =============================================================================
# Test 3: Idempotency
# =============================================================================


def test_indexing_idempotency_skips_active_events():
    """Test 3: Idempotency: Repeating indexing on the same paper skips already-enqueued
    paper with zero duplicate events.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Idempotent Project"))
        paper = create_paper(db, project.id, "p3.pdf", "p3.pdf")
        paper.status = "READY"
        paper.document_sha256 = "hash3333" * 8
        db.commit()
        project_id = project.id
        paper_id = paper.id

    with SessionLocal() as db:
        # Run 1: Enqueue paper
        res1 = enqueue_existing_papers_for_graph(db, project_id=project_id, dry_run=False)
        assert res1["enqueued_count"] == 1
        assert res1["skipped_count"] == 0

        # Run 2: Repeat on the same project/paper
        res2 = enqueue_existing_papers_for_graph(db, project_id=project_id, dry_run=False)
        assert res2["enqueued_count"] == 0
        assert res2["skipped_count"] == 1
        assert res2["eligible_paper_ids"] == []

        # Confirm exactly 1 GraphEvent exists
        events = db.query(GraphEvent).filter(GraphEvent.paper_id == paper_id).all()
        assert len(events) == 1


# =============================================================================
# Test 4: Explicit Target Required
# =============================================================================


def test_indexing_requires_explicit_target():
    """Test 4: Requires explicit target: Running without project_id or paper_ids returns
    0 eligible papers (no silent bulk backfill).
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Target Project"))
        paper = create_paper(db, project.id, "p4.pdf", "p4.pdf")
        paper.status = "READY"
        paper.document_sha256 = "hash4444" * 8
        db.commit()

    with SessionLocal() as db:
        res = enqueue_existing_papers_for_graph(db, project_id=None, paper_ids=None, dry_run=False)
        assert res["eligible_paper_ids"] == []
        assert res["enqueued_count"] == 0
        assert res["skipped_count"] == 0
        assert res["target_project_id"] is None
        assert "Explicit opt-in target required" in res.get("notice", "")

        # Verify no events created
        events = db.query(GraphEvent).all()
        assert len(events) == 0


# =============================================================================
# Test 5: Scoped Rebuild Replays Snapshots
# =============================================================================


def test_scoped_rebuild_replays_valid_snapshots():
    """Test 5: Scoped rebuild replays valid snapshots into Neo4j without reading
    Neo4j as authority.
    """
    valid_sha256 = "c0ffee" * 10 + "abcd"
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Rebuild Test 5"))
        paper = create_paper(db, project.id, "r5.pdf", "r5.pdf")
        paper.status = "READY"
        paper.document_sha256 = valid_sha256
        db.commit()
        project_id = project.id
        paper_id = paper.id

        generation_id = "gen-r5"
        f1_id = f"fact-r5-1-{uuid4().hex[:8]}"
        f2_id = f"fact-r5-2-{uuid4().hex[:8]}"

        _create_snapshot(
            db,
            project_id=project_id,
            paper_id=paper_id,
            generation_id=generation_id,
            fact_id=f1_id,
            subject_key="model_x",
            predicate="USES_METHOD",
            object_key="method_y",
            document_sha256=valid_sha256,
            char_start=0,
            char_end=20,
            page_number=1,
            exact_quote="Model X uses method Y.",
        )
        _create_snapshot(
            db,
            project_id=project_id,
            paper_id=paper_id,
            generation_id=generation_id,
            fact_id=f2_id,
            subject_key="method_y",
            predicate="EVALUATED_ON",
            object_key="dataset_z",
            document_sha256=valid_sha256,
            char_start=25,
            char_end=55,
            page_number=2,
            exact_quote="Method Y was evaluated on dataset Z.",
        )
        db.commit()

    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.upsert_facts.return_value = 2
    mock_repo.retire_older_generations.return_value = 0

    with SessionLocal() as db:
        result = scoped_rebuild_project_graph(
            db=db,
            repo=mock_repo,
            project_id=project_id,
            dry_run=False,
        )

    assert result["project_id"] == str(project_id)
    assert result["dry_run"] is False
    assert result["papers_rebuilt"] == 1
    assert result["facts_published"] == 2
    assert result["missing_snapshots_papers"] == []
    assert result["stale_snapshots_count"] == 0

    # Verify upsert_facts was called with both snapshots
    mock_repo.upsert_facts.assert_called_once()
    call_args = mock_repo.upsert_facts.call_args[1]
    assert call_args["project_id"] == project_id
    assert call_args["paper_id"] == paper_id
    assert call_args["generation_id"] == generation_id
    assert len(call_args["facts"]) == 2

    # Invariant: Never read Neo4j as authority
    mock_repo.get_fact_by_id.assert_not_called()
    mock_repo.get_node_by_key.assert_not_called()
    mock_repo._get_session.assert_not_called()


# =============================================================================
# Test 6: Stale Source Rejection
# =============================================================================


def test_scoped_rebuild_rejects_stale_source():
    """Test 6: Rejection of stale source: Snapshot with mismatched document_sha256
    is rejected from rebuild.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Stale Test 6"))
        paper = create_paper(db, project.id, "r6.pdf", "r6.pdf")
        paper.status = "READY"
        paper.document_sha256 = "current_sha256_" * 4
        db.commit()
        project_id = project.id
        paper_id = paper.id

        generation_id = "gen-r6"
        f1_id = f"fact-r6-stale-{uuid4().hex[:8]}"

        # Snapshot has different sha256
        _create_snapshot(
            db,
            project_id=project_id,
            paper_id=paper_id,
            generation_id=generation_id,
            fact_id=f1_id,
            subject_key="stale_model",
            predicate="PROPOSES",
            object_key="stale_arch",
            document_sha256="stale_mismatched_sha256_" * 2 + "1234",
            char_start=0,
            char_end=20,
            page_number=1,
            exact_quote="We propose Stale Model.",
        )
        db.commit()

    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.upsert_facts.return_value = 0
    mock_repo.retire_older_generations.return_value = 0

    with SessionLocal() as db:
        result = scoped_rebuild_project_graph(
            db=db,
            repo=mock_repo,
            project_id=project_id,
            paper_id=paper_id,
            dry_run=False,
        )

    assert result["papers_rebuilt"] == 0
    assert result["facts_published"] == 0
    assert result["stale_snapshots_count"] == 1
    mock_repo.upsert_facts.assert_not_called()


# =============================================================================
# Test 7: Honest About Gaps
# =============================================================================


def test_scoped_rebuild_honest_about_missing_snapshots():
    """Test 7: Honest about gaps: A paper without retained snapshots reports
    missing snapshots and does not invent facts.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Missing Test 7"))
        paper = create_paper(db, project.id, "r7.pdf", "r7.pdf")
        paper.status = "READY"
        paper.document_sha256 = "hash7777" * 8
        db.commit()
        project_id = project.id
        paper_id = paper.id

    mock_repo = MagicMock(spec=Neo4jRepository)

    with SessionLocal() as db:
        result = scoped_rebuild_project_graph(
            db=db,
            repo=mock_repo,
            project_id=project_id,
            dry_run=False,
        )

    assert result["missing_snapshots_papers"] == [str(paper_id)]
    assert result["papers_rebuilt"] == 0
    assert result["facts_published"] == 0
    mock_repo.upsert_facts.assert_not_called()


# =============================================================================
# Test 8: CLI Execution Tests
# =============================================================================


def test_cli_execution_index_and_rebuild(capsys):
    """Test 8: CLI execution test: CLI arguments --dry-run and --confirm execute
    and output valid JSON.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="CLI Test Project"))
        paper = create_paper(db, project.id, "cli.pdf", "cli.pdf")
        paper.status = "READY"
        paper.document_sha256 = "cli_hash" * 8
        db.commit()
        project_id = project.id
        paper_id = paper.id

    # 1. main_index with --dry-run
    code = main_index(["--project-id", str(project_id), "--dry-run"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["dry_run"] is True
    assert data["eligible_paper_ids"] == [str(paper_id)]
    assert data["enqueued_count"] == 0

    # 2. main_index with --confirm
    code = main_index(["--project-id", str(project_id), "--confirm"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["dry_run"] is False
    assert data["enqueued_count"] == 1

    # 3. main_rebuild with --dry-run
    mock_repo = MagicMock(spec=Neo4jRepository)
    code = main_rebuild(["--project-id", str(project_id), "--dry-run"], repo=mock_repo)
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["dry_run"] is True
    assert data["project_id"] == str(project_id)

    # 4. main_rebuild with --confirm
    code = main_rebuild(["--project-id", str(project_id), "--confirm"], repo=mock_repo)
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["dry_run"] is False
    assert data["project_id"] == str(project_id)


# =============================================================================
# Additional Edge Case Tests
# =============================================================================


def test_indexing_skips_non_ready_or_missing_hash():
    """Verify papers that are PROCESSING or have missing sha256 are skipped."""
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Edge Case Project"))
        p_processing = create_paper(db, project.id, "proc.pdf", "proc.pdf")
        p_processing.status = "PROCESSING"
        p_processing.document_sha256 = "hash_proc" * 4

        p_nohash = create_paper(db, project.id, "nohash.pdf", "nohash.pdf")
        p_nohash.status = "READY"
        p_nohash.document_sha256 = None
        db.commit()

        project_id = project.id
        p_proc_id = p_processing.id
        p_nohash_id = p_nohash.id

    with SessionLocal() as db:
        res = enqueue_existing_papers_for_graph(
            db,
            project_id=project_id,
            paper_ids=[p_proc_id, p_nohash_id],
            dry_run=False,
        )
        assert res["enqueued_count"] == 0
        assert res["skipped_count"] == 2
        assert res["eligible_paper_ids"] == []


def test_indexing_bounds_limit_cap():
    """Verify limit parameter is capped at 50."""
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Limit Cap Project"))
        for i in range(55):
            p = create_paper(db, project.id, f"p{i}.pdf", f"p{i}.pdf")
            p.status = "READY"
            p.document_sha256 = f"hash_{i:04d}" * 8
        db.commit()
        project_id = project.id

    with SessionLocal() as db:
        res = enqueue_existing_papers_for_graph(
            db,
            project_id=project_id,
            limit=100,  # Request 100, should be bounded to 50
            dry_run=True,
        )
        assert len(res["eligible_paper_ids"]) == 50


def test_cli_main_subcommand_dispatch(capsys):
    """Verify main() entrypoint dispatches to index and rebuild correctly."""
    assert main([]) == 1
    captured = capsys.readouterr()
    assert "Usage:" in captured.out

    code = main(["unknown_cmd"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Unknown subcommand" in captured.out

    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Main Dispatch Project"))
        project_id = project.id

    # Dispatch to index
    code = main(["index", "--project-id", str(project_id), "--dry-run"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["dry_run"] is True

    # Dispatch to rebuild
    mock_repo = MagicMock(spec=Neo4jRepository)
    with patch("app.cli.graph.Neo4jRepository.from_settings", return_value=mock_repo):
        code = main(["rebuild", "--project-id", str(project_id), "--dry-run"])
        assert code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["dry_run"] is True


def test_cli_rebuild_requires_project_id(capsys):
    """Verify main_rebuild returns error code 1 when --project-id is omitted."""
    code = main_rebuild([])
    assert code == 1
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "--project-id is required" in data["error"]


def test_scoped_rebuild_paper_not_found():
    """Verify scoped_rebuild_project_graph raises ValueError when requested
    paper_id does not exist.
    """
    mock_repo = MagicMock(spec=Neo4jRepository)
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Not Found Project"))
        with pytest.raises(ValueError, match="not found in project"):
            scoped_rebuild_project_graph(
                db=db,
                repo=mock_repo,
                project_id=project.id,
                paper_id=uuid4(),
            )


def test_cli_invalid_uuids(capsys):
    """Verify CLI commands handle invalid UUID strings gracefully."""
    # main_index invalid project_id
    code = main_index(["--project-id", "not-a-uuid"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Invalid project_id UUID" in json.loads(captured.out)["error"]

    # main_index invalid paper_id
    code = main_index(["--paper-id", "not-a-uuid"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Invalid paper_id UUID" in json.loads(captured.out)["error"]

    # main_rebuild invalid project_id
    code = main_rebuild(["--project-id", "not-a-uuid"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Invalid project_id UUID" in json.loads(captured.out)["error"]

    # main_rebuild invalid paper_id
    code = main_rebuild(["--project-id", str(uuid4()), "--paper-id", "not-a-uuid"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Invalid paper_id UUID" in json.loads(captured.out)["error"]


def test_cli_rebuild_neo4j_not_configured_error(capsys):
    """Verify main_rebuild outputs error when Neo4j is not configured and confirm is given."""
    with patch("app.cli.graph.Neo4jRepository.from_settings", return_value=None):
        code = main_rebuild(["--project-id", str(uuid4()), "--confirm"], repo=None)
        assert code == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "Neo4j is not configured" in data["error"]


def test_scoped_rebuild_with_completed_event_and_string_uuids():
    """Verify scoped_rebuild_project_graph handles string UUIDs and uses latest completed event."""
    valid_sha256 = "beefcafe" * 8
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="String UUID Project"))
        paper = create_paper(db, project.id, "str_uuid.pdf", "str_uuid.pdf")
        paper.status = "READY"
        paper.document_sha256 = valid_sha256
        db.commit()

        project_id = str(project.id)
        paper_id = str(paper.id)

        # Create completed event with specific generation_id
        ev = GraphEvent(
            project_id=project.id,
            paper_id=paper.id,
            action="UPSERT",
            generation_id="gen-completed-event",
            status="COMPLETED",
        )
        db.add(ev)
        db.commit()

        # Add snapshot matching that generation
        _create_snapshot(
            db,
            project_id=project.id,
            paper_id=paper.id,
            generation_id="gen-completed-event",
            fact_id=f"fact-completed-{uuid4().hex[:8]}",
            subject_key="subject_a",
            predicate="USES_METHOD",
            object_key="object_b",
            document_sha256=valid_sha256,
        )
        db.commit()

    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.upsert_facts.return_value = 1
    mock_repo.retire_older_generations.return_value = 0

    with SessionLocal() as db:
        # Pass string UUIDs
        res = scoped_rebuild_project_graph(
            db=db,
            repo=mock_repo,
            project_id=project_id,
            paper_id=paper_id,
            dry_run=False,
        )

    assert res["papers_rebuilt"] == 1
    assert res["facts_published"] == 1


def test_indexing_string_uuids():
    """Verify enqueue_existing_papers_for_graph handles string UUIDs."""
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Indexing String UUID Project"))
        paper = create_paper(db, project.id, "indexing_str.pdf", "indexing_str.pdf")
        paper.status = "READY"
        paper.document_sha256 = "str_hash" * 8
        db.commit()

        project_id_str = str(project.id)
        paper_id_str = str(paper.id)

    with SessionLocal() as db:
        res = enqueue_existing_papers_for_graph(
            db,
            project_id=project_id_str,
            paper_ids=[paper_id_str],
            dry_run=False,
        )
        assert res["enqueued_count"] == 1
        assert res["eligible_paper_ids"] == [paper_id_str]


def test_scoped_rebuild_target_generation_has_no_snapshots():
    """Verify scoped_rebuild reports missing when latest event generation has no snapshots."""
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Missing Gen Project"))
        paper = create_paper(db, project.id, "missing_gen.pdf", "missing_gen.pdf")
        paper.status = "READY"
        paper.document_sha256 = "hash_gen" * 8
        db.commit()

        # Add snapshot under gen-old
        _create_snapshot(
            db,
            project_id=project.id,
            paper_id=paper.id,
            generation_id="gen-old",
            fact_id=f"fact-old-{uuid4().hex[:8]}",
            subject_key="old_sub",
            predicate="USES_METHOD",
            object_key="old_obj",
            document_sha256="hash_gen" * 8,
        )
        # Completed event under gen-new
        ev = GraphEvent(
            project_id=project.id,
            paper_id=paper.id,
            action="UPSERT",
            generation_id="gen-new",
            status="COMPLETED",
        )
        db.add(ev)
        db.commit()
        project_id = project.id
        paper_id = paper.id

    mock_repo = MagicMock(spec=Neo4jRepository)
    with SessionLocal() as db:
        res = scoped_rebuild_project_graph(
            db=db,
            repo=mock_repo,
            project_id=project_id,
            paper_id=paper_id,
            dry_run=False,
        )

    assert res["missing_snapshots_papers"] == [str(paper_id)]
    assert res["papers_rebuilt"] == 0


def test_cli_rebuild_handles_service_exception(capsys):
    """Verify main_rebuild prints error JSON when rebuild raises an exception."""
    mock_repo = MagicMock(spec=Neo4jRepository)
    with patch(
        "app.cli.graph.scoped_rebuild_project_graph",
        side_effect=RuntimeError("Database error during rebuild"),
    ):
        code = main_rebuild(["--project-id", str(uuid4()), "--confirm"], repo=mock_repo)
        assert code == 1
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "Database error during rebuild" in data["error"]
