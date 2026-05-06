from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import Job


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

    # 1. Prioritize PENDING jobs
    query = db.query(Job).filter(Job.status == "PENDING").order_by(Job.created_at.asc())
    if not is_sqlite:
        query = query.with_for_update(skip_locked=True)

    job = query.first()

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
) -> Job | None:
    if isinstance(job_id, str):
        job_id = UUID(job_id)
    job = get_job(db, job_id)
    if not job:
        return None
    job.stage = stage
    job.progress = progress
    job.status = status
    if error_message is not None:
        job.error_message = error_message
    job.is_retryable = is_retryable
    db.commit()
    db.refresh(job)
    return job
