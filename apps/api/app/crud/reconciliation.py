import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.crud.corpus import bump_corpus_revision
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
    skipped_in_flight_keys: list[str] = field(default_factory=list)
    storage_keys_scanned: int = 0
    deletion_errors: list[str] = field(default_factory=list)


def normalize_storage_path(path: str) -> str:
    """Normalize stored URIs and relative paths for exact comparison."""
    p = path.strip()
    if p.startswith("memory://"):
        p = p[len("memory://") :]
    elif p.startswith("gs://"):
        parts = p.split("/", 3)
        p = parts[3] if len(parts) > 3 else p
    return p.lstrip("/")


def validate_scan_prefix(prefix: str | None) -> str:
    """Validate that scan_prefix is non-empty and strictly inside approved namespaces."""
    if not prefix or not prefix.strip():
        raise ValueError("scan_prefix must be a non-empty path within the 'papers/' namespace")
    cleaned = prefix.strip().lstrip("/")
    if not cleaned.startswith("papers/"):
        raise ValueError(f"scan_prefix '{prefix}' is outside the approved 'papers/' namespace")
    return cleaned


MIN_SAFE_ORPHAN_AGE_SECONDS = 300  # 5 minutes minimum floor to protect in-flight uploads


async def reconcile_stranded_resources_async(
    db: Session,
    storage: ObjectStorage | None = None,
    dry_run: bool = True,
    lease_timeout_seconds: int = 300,
    storage_candidates: list[str] | None = None,
    scan_prefix: str | None = None,
    min_age_seconds: int = 900,
    max_keys: int = 500,
) -> ReconciliationReport:
    """Reconciles stranded jobs and orphaned storage keys asynchronously.

    Safety controls:
    - Namespace isolation: scan_prefix must be within approved 'papers/' namespace.
    - In-flight upload protection: min_age_seconds must be >= 300s; objects modified
      < min_age_seconds or with unknown mtime are skipped.
    - Point-in-time DB recheck: unreferenced candidates are rechecked against DB before deletion.
    - Bounded discovery: capped by max_keys to prevent memory exhaustion.
    - Exact key normalization: matches paper storage paths across local/GCS/memory URIs.
    """
    if min_age_seconds < MIN_SAFE_ORPHAN_AGE_SECONDS:
        raise ValueError(
            f"min_age_seconds must be at least {MIN_SAFE_ORPHAN_AGE_SECONDS} seconds "
            f"(got {min_age_seconds}) to guarantee in-flight uploads are not deleted."
        )

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
                    was_ready = paper.status == PaperStatus.READY
                    paper.status = PaperStatus.FAILED
                    paper.error_message = (
                        "[PROCESSING_TIMEOUT] Job exceeded lease timeout and maximum retries"
                    )
                    if was_ready:
                        bump_corpus_revision(db, paper.project_id)
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

    # 2. Check storage candidates or discover them safely
    if storage:
        candidates_with_mtime: list[tuple[str, datetime | None]] = []

        if storage_candidates is not None:
            for k in storage_candidates[:max_keys]:
                candidates_with_mtime.append((k, None))
        elif scan_prefix is not None:
            validated_prefix = validate_scan_prefix(scan_prefix)
            candidates_with_mtime = await storage.list_objects(
                prefix=validated_prefix, limit=max_keys
            )

        if candidates_with_mtime:
            report.storage_keys_scanned = len(candidates_with_mtime)
            # Fetch and normalize all referenced storage paths from DB
            raw_referenced = [p[0] for p in db.query(Paper.storage_path).all() if p[0]]
            referenced_paths = {normalize_storage_path(r) for r in raw_referenced}

            for key, mtime in candidates_with_mtime:
                norm_key = normalize_storage_path(key)

                # If auto-discovering via scan_prefix and mtime is unknown, skip to protect
                # in-flight uploads
                if scan_prefix is not None and mtime is None:
                    report.skipped_in_flight_keys.append(key)
                    continue

                if mtime is not None:
                    age = (now - mtime).total_seconds()
                    if age < min_age_seconds:
                        report.skipped_in_flight_keys.append(key)
                        continue

                is_referenced = norm_key in referenced_paths

                if not is_referenced:
                    # Point-in-time DB recheck right before deleting to avoid race conditions
                    # with concurrent commits during the scan. Checks exact keys, normalized keys,
                    # canonical gs:// and memory:// URIs, and active in-flight jobs.
                    candidate_patterns = [
                        Paper.storage_path == key,
                        Paper.storage_path == norm_key,
                        Paper.storage_path == f"/{norm_key}",
                        Paper.storage_path == f"papers/{norm_key}",
                        Paper.storage_path.like(f"gs://%/{norm_key}"),
                        Paper.storage_path.like(f"memory://%/{norm_key}"),
                        Paper.storage_path.like(f"file://%/{norm_key}"),
                        Paper.storage_path.like(f"%/{norm_key}"),
                        Paper.storage_path.like(f"%/{key.lstrip('/')}"),
                    ]
                    if hasattr(storage, "bucket_name") and getattr(storage, "bucket_name"):
                        bucket = getattr(storage, "bucket_name")
                        candidate_patterns.extend(
                            [
                                Paper.storage_path == f"gs://{bucket}/{norm_key}",
                                Paper.storage_path == f"gs://{bucket}/{key.lstrip('/')}",
                            ]
                        )

                    point_in_time_ref = db.query(Paper.id).filter(or_(*candidate_patterns)).first()
                    if point_in_time_ref is None:
                        active_job_ref = (
                            db.query(Job.id)
                            .join(Paper, Job.paper_id == Paper.id)
                            .filter(
                                Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING]),
                                or_(*candidate_patterns),
                            )
                            .first()
                        )
                        if active_job_ref is not None:
                            point_in_time_ref = active_job_ref

                    if point_in_time_ref is not None:
                        report.skipped_in_flight_keys.append(key)
                        continue

                    report.orphaned_storage_keys.append(key)
                    if not dry_run:
                        try:
                            await storage.delete(key)
                        except Exception as del_err:
                            report.deletion_errors.append(f"{key}: {del_err}")

    return report


def reconcile_stranded_resources(
    db: Session,
    storage: ObjectStorage | None = None,
    dry_run: bool = True,
    lease_timeout_seconds: int = 300,
    storage_candidates: list[str] | None = None,
    scan_prefix: str | None = None,
    min_age_seconds: int = 900,
    max_keys: int = 500,
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
                    min_age_seconds=min_age_seconds,
                    max_keys=max_keys,
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
                min_age_seconds=min_age_seconds,
                max_keys=max_keys,
            )
        )
