import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from app.db.models import Job, Paper
from app.schemas.job import JobStage, JobStatus
from app.schemas.paper import PaperStatus
from app.storage.base import ObjectStorage


@dataclass
class ReconciliationReport:
    dry_run: bool
    stuck_jobs_expired: list[str] = field(default_factory=list)
    stuck_jobs_recovered: list[str] = field(default_factory=list)
    orphaned_storage_keys: list[str] = field(default_factory=list)
    storage_keys_scanned: int = 0


def reconcile_stranded_resources(
    db: Session,
    storage: ObjectStorage | None = None,
    dry_run: bool = True,
    lease_timeout_seconds: int = 300,
    storage_candidates: list[str] | None = None,
) -> ReconciliationReport:
    """Reconciles stranded jobs and orphaned storage keys.

    - Scans for jobs stuck in PROCESSING beyond lease_timeout_seconds.
      - If retry_count >= max_retries: marks FAILED.
      - If retry_count < max_retries: releases back to PENDING/QUEUED.
    - If storage is provided and storage_candidates is supplied:
      - Verifies whether keys are referenced by any Paper record.
      - If dry_run is False: deletes unreferenced orphan keys.
      - Never deletes keys referenced by a Paper.
    """
    now = datetime.now(tz=UTC)
    cutoff = datetime.fromtimestamp(now.timestamp() - lease_timeout_seconds, tz=UTC)
    report = ReconciliationReport(dry_run=dry_run)

    # 1. Scan stuck PROCESSING jobs
    stuck_jobs = db.query(Job).filter(Job.status == "PROCESSING", Job.claimed_at < cutoff).all()

    for job in stuck_jobs:
        job_id_str = str(job.id)
        paper = db.query(Paper).filter(Paper.id == job.paper_id).first()
        if job.retry_count >= job.max_retries:
            report.stuck_jobs_expired.append(job_id_str)
            if not dry_run:
                job.status = JobStatus.FAILED
                job.stage = JobStage.FAILED
                job.error_message = (
                    "[PROCESSING_TIMEOUT] Job exceeded lease timeout and maximum retries"
                )
                job.is_retryable = False
                if paper:
                    paper.status = PaperStatus.FAILED
                    paper.error_message = (
                        "[PROCESSING_TIMEOUT] Job exceeded lease timeout and maximum retries"
                    )
        else:
            report.stuck_jobs_recovered.append(job_id_str)
            if not dry_run:
                job.status = JobStatus.PENDING
                job.stage = JobStage.QUEUED
                job.worker_id = None
                job.claimed_at = None
                job.retry_count += 1
                job.error_message = "Recovered from expired worker lease"
                job.is_retryable = True

    if not dry_run and stuck_jobs:
        db.commit()

    # 2. Check storage candidates if provided
    if storage and storage_candidates:
        report.storage_keys_scanned = len(storage_candidates)
        # Fetch all referenced storage paths from DB
        referenced_paths = set(p[0] for p in db.query(Paper.storage_path).all() if p[0])
        for key in storage_candidates:
            is_referenced = any(
                key == ref or key.endswith(ref) or ref.endswith(key) for ref in referenced_paths
            )
            if not is_referenced:
                report.orphaned_storage_keys.append(key)
                if not dry_run:
                    try:
                        asyncio.run(storage.delete(key))
                    except Exception:
                        pass

    return report
