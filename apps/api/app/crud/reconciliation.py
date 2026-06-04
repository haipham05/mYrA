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


async def reconcile_stranded_resources_async(
    db: Session,
    storage: ObjectStorage | None = None,
    dry_run: bool = True,
    lease_timeout_seconds: int = 300,
    storage_candidates: list[str] | None = None,
    scan_prefix: str | None = None,
) -> ReconciliationReport:
    """Reconciles stranded jobs and orphaned storage keys asynchronously.

    - Scans for jobs stuck in PROCESSING beyond lease_timeout_seconds.
      - If retry_count >= max_retries: marks FAILED.
      - If retry_count < max_retries: releases back to PENDING/QUEUED.
    - If storage is provided and either storage_candidates is supplied or scan_prefix is set:
      - Discovers or uses supplied storage keys.
      - Verifies whether keys are referenced by any Paper record.
      - If dry_run is False: deletes unreferenced orphan keys safely via await storage.delete.
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

    # 2. Check storage candidates or discover them
    if storage:
        candidates = storage_candidates
        if candidates is None and scan_prefix is not None:
            candidates = await storage.list_keys(prefix=scan_prefix)

        if candidates:
            report.storage_keys_scanned = len(candidates)
            referenced_paths = set(p[0] for p in db.query(Paper.storage_path).all() if p[0])
            for key in candidates:
                is_referenced = any(
                    key == ref or key.endswith(ref) or ref.endswith(key) for ref in referenced_paths
                )
                if not is_referenced:
                    report.orphaned_storage_keys.append(key)
                    if not dry_run:
                        try:
                            await storage.delete(key)
                        except Exception:
                            pass

    return report


def reconcile_stranded_resources(
    db: Session,
    storage: ObjectStorage | None = None,
    dry_run: bool = True,
    lease_timeout_seconds: int = 300,
    storage_candidates: list[str] | None = None,
    scan_prefix: str | None = None,
) -> ReconciliationReport:
    """Synchronous wrapper for reconcile_stranded_resources_async."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                asyncio.run,
                reconcile_stranded_resources_async(
                    db=db,
                    storage=storage,
                    dry_run=dry_run,
                    lease_timeout_seconds=lease_timeout_seconds,
                    storage_candidates=storage_candidates,
                    scan_prefix=scan_prefix,
                ),
            ).result()
    else:
        return asyncio.run(
            reconcile_stranded_resources_async(
                db=db,
                storage=storage,
                dry_run=dry_run,
                lease_timeout_seconds=lease_timeout_seconds,
                storage_candidates=storage_candidates,
                scan_prefix=scan_prefix,
            )
        )
