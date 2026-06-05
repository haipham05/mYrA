import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from app.crud.job import release_job
from app.crud.paper import create_paper_with_job
from app.crud.project import create_project
from app.crud.reconciliation import reconcile_stranded_resources
from app.db.base import Base
from app.db.session import SessionLocal, engine
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.storage.local import MemoryStorage


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_reconcile_stuck_recoverable_job():
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Reconciler Test"))
        paper, job = create_paper_with_job(
            db,
            project_id=project.id,
            filename="paper.pdf",
            storage_path="papers/p1.pdf",
            document_sha256="abc123",
            status=PaperStatus.PROCESSING,
        )
        # Set job as stuck in processing for 10 minutes
        job.status = "PROCESSING"
        job.worker_id = "worker-stale"
        job.claimed_at = datetime.now(tz=UTC) - timedelta(seconds=600)
        job.retry_count = 0
        job.max_retries = 3
        db.commit()

        # 1. Dry run does not mutate
        report_dry = reconcile_stranded_resources(db, dry_run=True, lease_timeout_seconds=300)
        assert str(job.id) in report_dry.stuck_jobs_recovered
        db.refresh(job)
        assert job.status == "PROCESSING"

        # 2. Live reconciliation recovers job to PENDING
        report_live = reconcile_stranded_resources(db, dry_run=False, lease_timeout_seconds=300)
        assert str(job.id) in report_live.stuck_jobs_recovered
        db.refresh(job)
        assert job.status == "PENDING"
        assert job.stage == "QUEUED"
        assert job.worker_id is None
        assert job.retry_count == 1


def test_reconcile_stuck_expired_job_fails():
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Expired Test"))
        paper, job = create_paper_with_job(
            db,
            project_id=project.id,
            filename="paper.pdf",
            storage_path="papers/p2.pdf",
            document_sha256="abc456",
            status=PaperStatus.PROCESSING,
        )
        # Set job stuck at max_retries
        job.status = "PROCESSING"
        job.worker_id = "worker-dead"
        job.claimed_at = datetime.now(tz=UTC) - timedelta(seconds=600)
        job.retry_count = 3
        job.max_retries = 3
        db.commit()

        report = reconcile_stranded_resources(db, dry_run=False, lease_timeout_seconds=300)
        assert str(job.id) in report.stuck_jobs_expired
        db.refresh(job)
        db.refresh(paper)
        assert job.status == "FAILED"
        assert job.stage == "FAILED"
        assert "PROCESSING_TIMEOUT" in job.error_message
        assert paper.status == PaperStatus.FAILED


def test_reconcile_storage_orphans():
    storage = MemoryStorage()
    asyncio.run(storage.put("papers/valid.pdf", b"Valid PDF"))
    asyncio.run(storage.put("papers/orphan.pdf", b"Orphan PDF"))

    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Storage Orphan Test"))
        create_paper_with_job(
            db,
            project_id=project.id,
            filename="valid.pdf",
            storage_path="memory://papers/valid.pdf",
            document_sha256="hashvalid",
            status=PaperStatus.READY,
        )

        candidates = ["papers/valid.pdf", "papers/orphan.pdf"]

        # Dry run identifies orphan without deleting
        report_dry = reconcile_stranded_resources(
            db, storage=storage, dry_run=True, storage_candidates=candidates
        )
        assert "papers/orphan.pdf" in report_dry.orphaned_storage_keys
        assert "papers/valid.pdf" not in report_dry.orphaned_storage_keys
        assert asyncio.run(storage.exists("papers/orphan.pdf")) is True

        # Live run deletes orphan and preserves referenced paper
        report_live = reconcile_stranded_resources(
            db, storage=storage, dry_run=False, storage_candidates=candidates
        )
        assert "papers/orphan.pdf" in report_live.orphaned_storage_keys
        assert asyncio.run(storage.exists("papers/orphan.pdf")) is False
        assert asyncio.run(storage.exists("papers/valid.pdf")) is True


def test_reconcile_auto_discover_storage_orphans():
    """Prove that storage orphans can be auto-discovered under a bounded prefix,
    while in-flight uploads and referenced papers are strictly protected.
    """
    storage = MemoryStorage()
    now = datetime.now(tz=UTC)

    # 1. Active paper original (mature or recent, referenced in DB)
    asyncio.run(storage.put("papers/valid_active.pdf", b"Active Paper Content"))
    storage.set_mtime("papers/valid_active.pdf", now - timedelta(seconds=3600))

    # 2. In-flight upload (unreferenced in DB, but uploaded 30 seconds ago)
    asyncio.run(storage.put("papers/inflight_upload.pdf", b"In-Flight Upload"))
    storage.set_mtime("papers/inflight_upload.pdf", now - timedelta(seconds=30))

    # 3. Mature unreferenced orphan (uploaded 2 hours ago)
    asyncio.run(storage.put("papers/stranded_orphan.pdf", b"Mature Orphan Content"))
    storage.set_mtime("papers/stranded_orphan.pdf", now - timedelta(seconds=7200))

    # 4. Object outside approved papers namespace
    asyncio.run(storage.put("other/unrelated.pdf", b"Unrelated Folder Content"))

    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Discovery Test"))
        create_paper_with_job(
            db,
            project_id=project.id,
            filename="valid.pdf",
            storage_path="papers/valid_active.pdf",
            document_sha256="hashvalidactive",
            status=PaperStatus.READY,
        )

        # A. Bounded discovery with dry_run=True (scans papers/, min_age=900s)
        report_dry = reconcile_stranded_resources(
            db, storage=storage, dry_run=True, scan_prefix="papers/", min_age_seconds=900
        )
        assert report_dry.storage_keys_scanned == 3
        # In-flight is skipped from orphans
        assert "papers/inflight_upload.pdf" in report_dry.skipped_in_flight_keys
        assert "papers/inflight_upload.pdf" not in report_dry.orphaned_storage_keys
        # Mature orphan is identified
        assert "papers/stranded_orphan.pdf" in report_dry.orphaned_storage_keys
        # Active paper is preserved
        assert "papers/valid_active.pdf" not in report_dry.orphaned_storage_keys
        # Unrelated prefix not touched
        assert "other/unrelated.pdf" not in report_dry.orphaned_storage_keys
        assert asyncio.run(storage.exists("papers/stranded_orphan.pdf")) is True

        # B. Live run: only mature orphan is deleted; in-flight and active paper survive!
        report_live = reconcile_stranded_resources(
            db, storage=storage, dry_run=False, scan_prefix="papers/", min_age_seconds=900
        )
        assert "papers/stranded_orphan.pdf" in report_live.orphaned_storage_keys
        assert asyncio.run(storage.exists("papers/stranded_orphan.pdf")) is False
        assert asyncio.run(storage.exists("papers/inflight_upload.pdf")) is True
        assert asyncio.run(storage.exists("papers/valid_active.pdf")) is True

        # C. Rejection of illegal / broad prefixes
        with pytest.raises(ValueError, match="outside the approved 'papers/' namespace"):
            reconcile_stranded_resources(db, storage=storage, scan_prefix="other/")

        with pytest.raises(ValueError, match="non-empty path"):
            reconcile_stranded_resources(db, storage=storage, scan_prefix="")


def test_reconcile_api_endpoint(monkeypatch):
    """Prove that POST /api/v1/system/reconcile handles scan_storage flags and guards properly."""
    from starlette.testclient import TestClient

    from app.main import app

    storage = MemoryStorage()
    now = datetime.now(tz=UTC)
    asyncio.run(storage.put("papers/orphan_api.pdf", b"Orphan"))
    storage.set_mtime("papers/orphan_api.pdf", now - timedelta(seconds=3600))
    monkeypatch.setattr("app.storage.factory.get_storage", lambda: storage)

    client = TestClient(app)

    # 1. Default reconcile: job only, does not scan storage
    res1 = client.post("/api/v1/system/reconcile?dry_run=true")
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["storage_keys_scanned"] == 0
    assert data1["orphaned_storage_keys"] == []

    # 2. Reconcile with scan_storage=true (dry run): discovers mature orphan
    res2 = client.post("/api/v1/system/reconcile?dry_run=true&scan_storage=true")
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["storage_keys_scanned"] >= 1
    assert "papers/orphan_api.pdf" in data2["orphaned_storage_keys"]

    # 3. Illegal prefix rejected with 400
    res3 = client.post("/api/v1/system/reconcile?scan_storage=true&prefix=root_danger/")
    assert res3.status_code == 400

    # 4. Destructive run without confirm_destructive rejected with 400
    res4 = client.post("/api/v1/system/reconcile?dry_run=false&scan_storage=true")
    assert res4.status_code == 400
    assert "confirm_destructive=true" in res4.json()["detail"]

    # 5. Destructive run with confirmation succeeds and deletes orphan
    res5 = client.post(
        "/api/v1/system/reconcile?dry_run=false&scan_storage=true&confirm_destructive=true"
    )
    assert res5.status_code == 200
    assert asyncio.run(storage.exists("papers/orphan_api.pdf")) is False


def test_release_job_on_worker_shutdown():
    with SessionLocal() as db:
        project = create_project(db, ProjectCreate(name="Shutdown Test"))
        paper, job = create_paper_with_job(
            db,
            project_id=project.id,
            filename="paper.pdf",
            storage_path="papers/p3.pdf",
            document_sha256="hash3",
            status=PaperStatus.PROCESSING,
        )
        job.status = "PROCESSING"
        job.worker_id = "worker-graceful"
        db.commit()

        # Releasing by the owning worker resets to PENDING
        assert release_job(db, job.id, worker_id="worker-graceful") is True
        db.refresh(job)
        assert job.status == "PENDING"
        assert job.worker_id is None

        # Releasing by non-owner returns False
        assert release_job(db, job.id, worker_id="other-worker") is False
