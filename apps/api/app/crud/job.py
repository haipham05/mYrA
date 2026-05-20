from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import Job


class LostJobLeaseError(RuntimeError):
    """This worker no longer owns the job and must not publish its work."""


def create_job(db: Session, paper_id: UUID) -> Job:
    job = Job(paper_id=paper_id, status="PENDING", stage="QUEUED", progress=0.0)
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: UUID | str) -> Job | None:
    if isinstance(job_id, str):
        job_id = UUID(job_id)
    return db.query(Job).filter(Job.id == job_id).first()


def claim_next_job(db: Session, worker_id: str, lease_timeout_seconds: int = 300) -> Job | None:
    """Atomically claim the next pending job or recover an expired worker lease."""
    bind = db.get_bind()
    is_sqlite = bind.dialect.name == "sqlite"
    now = datetime.now(tz=UTC)
    cutoff = datetime.fromtimestamp(now.timestamp() - lease_timeout_seconds, tz=UTC)

    # 1. Prioritize PENDING jobs (honoring backoff for retried jobs)
    query = db.query(Job).filter(Job.status == "PENDING").order_by(Job.created_at.asc())
    if not is_sqlite:
        query = query.with_for_update(skip_locked=True)

    pending_jobs = query.all()
    job = None
    for candidate in pending_jobs:
        if candidate.retry_count == 0:
            job = candidate
            break
        backoff_seconds = min(300, (2**candidate.retry_count) * 5)
        if candidate.updated_at:
            cand_time = candidate.updated_at
            if cand_time.tzinfo is None:
                cand_time = cand_time.replace(tzinfo=UTC)
            elapsed = (now - cand_time).total_seconds()
            if elapsed >= backoff_seconds:
                job = candidate
                break
        else:
            job = candidate
            break

    # 2. Recover abandoned PROCESSING jobs whose worker lease has expired
    if not job:
        expired_query = (
            db.query(Job)
            .filter(
                Job.status == "PROCESSING",
                Job.claimed_at < cutoff,
                Job.retry_count < Job.max_retries,
            )
            .order_by(Job.claimed_at.asc())
        )
        if not is_sqlite:
            expired_query = expired_query.with_for_update(skip_locked=True)
        job = expired_query.first()
        if job:
            job.retry_count += 1

    if job:
        job.status = "PROCESSING"
        job.stage = "PARSING"
        job.worker_id = worker_id
        job.claimed_at = now
        db.commit()
        db.refresh(job)
    return job


def update_job_progress(
    db: Session,
    job_id: UUID | str,
    stage: str,
    progress: float,
    status: str = "PROCESSING",
    error_message: str | None = None,
    is_retryable: bool = False,
    worker_id: str | None = None,
) -> Job | None:
    if isinstance(job_id, str):
        job_id = UUID(job_id)
    values = {
        Job.stage: stage,
        Job.progress: progress,
        Job.status: status,
        Job.is_retryable: is_retryable,
    }
    if error_message is not None:
        values[Job.error_message] = error_message
    query = db.query(Job).filter(Job.id == job_id, Job.worker_id == worker_id)
    if worker_id is None:
        query = query.filter(Job.status.in_(["PENDING", "PROCESSING"]))
    else:
        query = query.filter(Job.status == "PROCESSING")
    if query.update(values, synchronize_session=False) != 1:
        db.rollback()
        raise LostJobLeaseError(f"Job {job_id} is no longer owned by this worker")
    db.commit()
    return get_job(db, job_id)


def fence_job_for_publish(db: Session, job_id: UUID, worker_id: str | None) -> None:
    """Lock the owned job row until the caller's final transaction commits."""
    changed = (
        db.query(Job)
        .filter(Job.id == job_id, Job.status == "PROCESSING", Job.worker_id == worker_id)
        .update({Job.claimed_at: datetime.now(tz=UTC)}, synchronize_session=False)
    )
    if changed != 1:
        db.rollback()
        raise LostJobLeaseError(f"Job {job_id} is no longer owned by this worker")


def renew_job_lease(db: Session, job_id: UUID | str, worker_id: str | None = None) -> bool:
    """Heartbeat to renew claimed_at timestamp on an actively processing job."""
    if isinstance(job_id, str):
        job_id = UUID(job_id)
    query = db.query(Job).filter(Job.id == job_id, Job.status == "PROCESSING")
    if worker_id is not None:
        query = query.filter(Job.worker_id == worker_id)
    job = query.first()
    if job:
        job.claimed_at = datetime.now(tz=UTC)
        db.commit()
        return True
    return False
