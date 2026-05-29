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
