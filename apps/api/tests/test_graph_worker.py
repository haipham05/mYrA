"""Tests for GraphEvent worker processing, lease fencing, failure isolation, and lifecycle."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Barrier
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.crud.graph import (
    claim_next_graph_event,
    create_or_enqueue_graph_event,
    get_graph_event,
    release_graph_event,
    renew_graph_event_lease,
)
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.models import GraphEvent, GraphFactSnapshot, Job, Paper
from app.db.session import SessionLocal, create_tables
from app.schemas.project import ProjectCreate
from app.services.graphrag.processor import GraphEventProcessor
from app.worker import _persisted_graph_outcome, run_worker


def _ensure_utc(dt: datetime) -> datetime:
    """Ensure datetime has UTC tzinfo for comparisons."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


@pytest.mark.parametrize(
    ("status", "owner", "current_attempts", "claimed_attempts", "expected"),
    [
        ("COMPLETED", None, 0, 0, "success"),
        ("COMPLETED", None, 1, 0, "abandoned"),
        ("PENDING", None, 1, 1, "retry_scheduled"),
        ("FAILED", None, 1, 1, "failed"),
        ("PROCESSING", "worker-2", 1, 0, "abandoned"),
        ("PROCESSING", "worker-1", 1, 1, "incomplete"),
    ],
)
def test_persisted_graph_attempt_outcome(
    status, owner, current_attempts, claimed_attempts, expected
):
    from types import SimpleNamespace

    event = SimpleNamespace(status=status, lease_owner=owner, attempts=current_attempts)
    assert _persisted_graph_outcome(event, "worker-1", claimed_attempts) == expected


@pytest.fixture(autouse=True)
def clean_database():
    """Ensure a clean database state before each test."""
    create_tables()
    with SessionLocal() as db:
        db.query(GraphFactSnapshot).delete()
        db.query(GraphEvent).delete()
        db.query(Job).delete()
        db.query(Paper).delete()
        db.commit()


def test_concurrency_isolation_two_workers_claim_same_pending_event():
    """Test 1: Concurrency isolation: Two workers attempt to claim the same pending event;
    exactly one gets it.
    """
    now = datetime.now(tz=UTC)
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Graph Concurrency Project"))
        paper = create_paper(db, project.id, "graph_claim.pdf", "graph_claim.pdf")
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        db.refresh(event)
        event_id = event.id

    barrier = Barrier(2)

    def worker_claim(worker_id: str):
        with SessionLocal() as session:
            barrier.wait()
            claimed = claim_next_graph_event(session, worker_id=worker_id)
            return claimed.id if claimed else None

    with ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(worker_claim, "worker-alpha")
        f2 = pool.submit(worker_claim, "worker-beta")
        results = [f1.result(), f2.result()]

    # Exactly one worker claimed the event, and the other got None
    assert results.count(event_id) == 1
    assert results.count(None) == 1

    with SessionLocal() as db:
        persisted = get_graph_event(db, event_id)
        assert persisted.status == "PROCESSING"
        assert persisted.lease_owner in ["worker-alpha", "worker-beta"]
        assert persisted.lease_expires_at is not None
        assert _ensure_utc(persisted.lease_expires_at) > now


def test_expired_lease_recovery():
    """Test 2: Expired lease recovery: An event with expired lease (lease_expires_at < now)
    is reclaimed by a new worker.
    """
    now = datetime.now(tz=UTC)
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Expired Lease Project"))
        paper = create_paper(db, project.id, "expired.pdf", "expired.pdf")
        event = GraphEvent(
            project_id=project.id,
            paper_id=paper.id,
            action="UPSERT",
            generation_id="gen-expired-1",
            status="PROCESSING",
            attempts=1,
            max_attempts=3,
            lease_owner="worker-stale",
            lease_expires_at=now - timedelta(minutes=5),
            updated_at=now - timedelta(minutes=5),
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        event_id = event.id

    # New worker attempts to claim next event
    with SessionLocal() as db:
        claimed = claim_next_graph_event(
            db, worker_id="worker-recovering", lease_timeout_seconds=300
        )
        assert claimed is not None
        assert claimed.id == event_id
        assert claimed.status == "PROCESSING"
        assert claimed.lease_owner == "worker-recovering"
        assert claimed.attempts == 2
        assert _ensure_utc(claimed.lease_expires_at) > now


@pytest.mark.anyio
async def test_shutdown_release(monkeypatch):
    """Test 3: Shutdown release: When worker is cancelled/shut down, the claimed event
    is released back to PENDING.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Shutdown Release Project"))
        paper = create_paper(db, project.id, "release.pdf", "release.pdf")
        paper.status = "READY"
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        db.refresh(event)
        event_id = event.id

    # Part A: Direct release_graph_event function
    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="worker-direct-release")
        assert claimed is not None
        assert claimed.status == "PROCESSING"
        assert claimed.lease_owner == "worker-direct-release"

        release_graph_event(db, event_id, worker_id="worker-direct-release")
        refreshed = get_graph_event(db, event_id)
        assert refreshed.status == "PENDING"
        assert refreshed.lease_owner is None
        assert refreshed.lease_expires_at is None

    # Part B: Worker loop task cancellation during active processing
    started_event = asyncio.Event()
    monkeypatch.setenv("MYRA_GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:17687")

    async def blocking_process(*args, **kwargs):
        started_event.set()
        await asyncio.sleep(10.0)

    ready_repo = MagicMock()
    ready_repo.verify_connectivity.return_value = True
    with (
        patch.object(GraphEventProcessor, "process_graph_event", side_effect=blocking_process),
        patch(
            "app.services.graphrag.neo4j_repository.Neo4jRepository.from_settings",
            return_value=ready_repo,
        ),
    ):
        worker_task = asyncio.create_task(
            run_worker(poll_interval=0.01, once=False, heartbeat_interval=0.1)
        )
        # Wait until the worker claims the event and starts processing
        await asyncio.wait_for(started_event.wait(), timeout=5.0)

        # Cancel the worker task simulating SIGINT/SIGTERM or runtime termination
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

    # Verify event was durably released back to PENDING
    with SessionLocal() as db:
        persisted = get_graph_event(db, event_id)
        assert persisted.status == "PENDING"
        assert persisted.lease_owner is None
        assert persisted.lease_expires_at is None


@pytest.mark.anyio
async def test_failure_isolation_paper_remains_ready(caplog):
    """Test 4: Failure isolation: Graph event error or timeout marks the GraphEvent
    as FAILED/retried, while Paper.status remains READY and paper is completely untouched.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Failure Isolation Project"))
        paper = create_paper(db, project.id, "failure_isolation.pdf", "failure_isolation.pdf")
        paper.status = "READY"
        paper.error_message = None
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        event.max_attempts = 3

        snapshot = GraphFactSnapshot(
            fact_id="fact-isolation-test-1",
            project_id=project.id,
            paper_id=paper.id,
            generation_id=event.generation_id,
            event_id=event.id,
            subject_key="s1",
            subject_name="Transformer",
            subject_type="Method",
            predicate="EVALUATED_ON",
            object_key="o1",
            object_name="WMT14",
            object_type="Dataset",
            char_start=0,
            char_end=10,
            page_number=1,
            exact_quote="Tested on WMT14.",
            document_sha256="abc12345" * 8,
        )
        db.add(snapshot)
        db.commit()
        db.refresh(event)
        event_id = event.id
        paper_id = paper.id
        project_id = project.id

    # 4a: Transient error (TimeoutError) -> retried: status becomes PENDING with attempts=1
    mock_repo = MagicMock()
    mock_repo.upsert_facts.side_effect = TimeoutError("Neo4j Bolt connection timed out")

    processor = GraphEventProcessor(repo=mock_repo)

    with SessionLocal() as db:
        claimed = claim_next_graph_event(db, worker_id="worker-fail-1")
        assert claimed is not None
        assert claimed.id == event_id
        completed = await processor.process_graph_event(db, event_id, worker_id="worker-fail-1")
        assert completed is False

    assert "Neo4j Bolt connection timed out" not in caplog.text

    with SessionLocal() as db:
        ev = get_graph_event(db, event_id)
        assert ev.status == "PENDING"
        assert ev.attempts == 1
        assert ev.error_code == "TIMEOUT"
        assert "timed out" in ev.error_message.lower()
        assert ev.lease_owner is None
        assert ev.lease_expires_at is None

        # Verify Paper record invariant: strictly READY, untouched, no error message
        p = db.get(Paper, paper_id)
        assert p.status == "READY"
        assert p.error_message is None

    # 4b: Exhaust attempts -> becomes FAILED, Paper STILL remains READY
    with SessionLocal() as db:
        # Directly set attempts to 2 and updated_at to past so backoff is bypassed
        ev = get_graph_event(db, event_id)
        ev.attempts = 2
        ev.updated_at = datetime.now(tz=UTC) - timedelta(seconds=120)
        db.commit()

        claimed = claim_next_graph_event(db, worker_id="worker-fail-2")
        assert claimed is not None
        await processor.process_graph_event(db, event_id, worker_id="worker-fail-2")

    with SessionLocal() as db:
        ev = get_graph_event(db, event_id)
        assert ev.status == "FAILED"
        assert ev.attempts == 3
        assert ev.error_code == "TIMEOUT"
        assert ev.lease_owner is None
        assert ev.lease_expires_at is None

        # Check paper invariant once again: paper is untouched
        p = db.get(Paper, paper_id)
        assert p.status == "READY"
        assert p.error_message is None

    # 4c: Non-transient failure immediately transitions to FAILED
    with SessionLocal() as db:
        non_transient_event = create_or_enqueue_graph_event(
            db, project_id, paper_id, action="INVALID_ACTION"
        )
        db.commit()
        db.refresh(non_transient_event)
        nt_id = non_transient_event.id

        claimed_nt = claim_next_graph_event(db, worker_id="worker-nt")
        assert claimed_nt is not None
        await processor.process_graph_event(db, nt_id, worker_id="worker-nt")

        ev_nt = get_graph_event(db, nt_id)
        assert ev_nt.status == "FAILED"
        assert ev_nt.error_code == "UNKNOWN_ACTION"

        # Paper remains completely untouched
        p = db.get(Paper, paper_id)
        assert p.status == "READY"
        assert p.error_message is None


@pytest.mark.anyio
async def test_event_completion_and_delete_action():
    """Test 5: Event completion: Successful processing marks event COMPLETED
    with completed_at set. Also tests DELETE action calling repo.delete_paper_facts.
    """
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Completion Project"))
        paper = create_paper(db, project.id, "complete.pdf", "complete.pdf")
        paper.status = "READY"
        upsert_event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        db.refresh(upsert_event)
        upsert_id = upsert_event.id
        paper_id = paper.id
        project_id = project.id

    mock_repo = MagicMock()
    processor = GraphEventProcessor(repo=mock_repo)

    # 5a: Process UPSERT event
    with SessionLocal() as db:
        claimed_upsert = claim_next_graph_event(db, worker_id="worker-complete")
        assert claimed_upsert is not None
        assert claimed_upsert.id == upsert_id
        completed = await processor.process_graph_event(db, upsert_id, worker_id="worker-complete")
        assert completed is True

    with SessionLocal() as db:
        ev = get_graph_event(db, upsert_id)
        assert ev.status == "COMPLETED"
        assert ev.completed_at is not None
        assert ev.lease_owner is None
        assert ev.lease_expires_at is None
        assert ev.error_code is None

    # 5b: Process DELETE event
    with SessionLocal() as db:
        delete_event = create_or_enqueue_graph_event(db, project_id, paper_id, action="DELETE")
        db.commit()
        delete_id = delete_event.id
        claimed_delete = claim_next_graph_event(db, worker_id="worker-delete")
        assert claimed_delete is not None
        assert claimed_delete.id == delete_id
        completed = await processor.process_graph_event(db, delete_id, worker_id="worker-delete")
        assert completed is True

    with SessionLocal() as db:
        ev = get_graph_event(db, delete_id)
        assert ev.status == "COMPLETED"
        assert ev.completed_at is not None
        assert ev.lease_owner is None
        assert ev.lease_expires_at is None
        # Verify mock repo delete_paper_facts was called with correct IDs
        mock_repo.delete_paper_facts.assert_called_once_with(
            project_id=project_id, paper_id=paper_id
        )


def test_lease_renewal_discipline():
    """Verify renew_graph_event_lease renews for owner and returns False if lost."""
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Renewal Project"))
        paper = create_paper(db, project.id, "renew.pdf", "renew.pdf")
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        db.refresh(event)
        event_id = event.id

        # Before claim, renew returns False
        assert not renew_graph_event_lease(db, event_id, worker_id="worker-owner")

        claimed = claim_next_graph_event(db, worker_id="worker-owner", lease_timeout_seconds=60)
        assert claimed is not None
        first_expiry = _ensure_utc(claimed.lease_expires_at)

        # Renewal by wrong worker returns False
        assert not renew_graph_event_lease(db, event_id, worker_id="worker-imposter")

        # Renewal by owner succeeds and extends expiry
        renewed = renew_graph_event_lease(
            db, event_id, worker_id="worker-owner", extend_seconds=120
        )
        assert renewed is True

        refreshed = get_graph_event(db, event_id)
        assert _ensure_utc(refreshed.lease_expires_at) > first_expiry


def test_exponential_backoff_for_retried_event():
    """Verify that retried events with attempts > 0 respect exponential backoff."""
    now = datetime.now(tz=UTC)
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Backoff Project"))
        paper = create_paper(db, project.id, "backoff.pdf", "backoff.pdf")
        event = GraphEvent(
            project_id=project.id,
            paper_id=paper.id,
            action="UPSERT",
            generation_id="gen-backoff-1",
            status="PENDING",
            attempts=2,  # 2**2 * 5 = 20 seconds backoff
            max_attempts=3,
            updated_at=now - timedelta(seconds=5),  # only 5s passed, not yet eligible
        )
        db.add(event)
        db.commit()

        # Should not be claimed yet because backoff has not elapsed
        assert claim_next_graph_event(db, worker_id="worker-backoff") is None

        # Advance elapsed time beyond 20 seconds
        event.updated_at = now - timedelta(seconds=25)
        db.commit()

        claimed = claim_next_graph_event(db, worker_id="worker-backoff")
        assert claimed is not None
        assert claimed.id == event.id


@pytest.mark.anyio
async def test_worker_run_once_processes_graph_event(monkeypatch):
    """Verify worker processes a pending graph event when run with once=True."""
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Worker Graph Run Once"))
        paper = create_paper(db, project.id, "worker_once.pdf", "worker_once.pdf")
        paper.status = "READY"
        event = create_or_enqueue_graph_event(db, project.id, paper.id, action="UPSERT")
        db.commit()
        event_id = event.id

    telemetry = MagicMock()
    monkeypatch.setattr("app.worker.get_telemetry", lambda: telemetry)

    monkeypatch.setenv("MYRA_GRAPHRAG_ENABLED", "true")
    monkeypatch.setenv("NEO4J_URI", "bolt://127.0.0.1:17687")
    ready_repo = MagicMock()
    ready_repo.verify_connectivity.return_value = True
    with patch(
        "app.services.graphrag.neo4j_repository.Neo4jRepository.from_settings",
        return_value=ready_repo,
    ):
        await run_worker(poll_interval=0.01, once=True)

    with SessionLocal() as db:
        persisted = get_graph_event(db, event_id)
        assert persisted.status == "COMPLETED"
        assert persisted.completed_at is not None

    telemetry.operation.assert_called_once()
    operation_name = telemetry.operation.call_args.args[0]
    metadata = telemetry.operation.call_args.kwargs["metadata"]
    assert operation_name == "worker.graph_event_attempt"
    assert metadata["event_id"] == str(event_id)
    assert metadata["attempt_number"] == 1
    assert metadata["queue_age_basis"] == "since_initial_enqueue"


@pytest.mark.anyio
async def test_disabled_worker_never_claims_graph_events():
    create_tables()
    with patch(
        "app.worker.Settings.from_environment",
        return_value=Settings(check_migration_compatibility=False),
    ):
        with (
            patch("app.worker.claim_next_job", return_value=None),
            patch("app.worker.claim_next_graph_event") as graph_claim,
            patch("app.worker.IngestionPipeline"),
        ):
            await run_worker(poll_interval=0.01, once=True)
    graph_claim.assert_not_called()
