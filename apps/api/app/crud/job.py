from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from app.crud.corpus import bump_corpus_revision
from app.db.models import Job
from app.observability.context import OperationContext
from app.services.job_state_machine import validate_job_transition


class LostJobLeaseError(RuntimeError):
    """This worker no longer owns the job and must not publish its work."""


def create_job(db: Session, paper_id: UUID, trace_context: OperationContext | None = None) -> Job:
    job = Job(
        paper_id=paper_id,
        status="PENDING",
        stage="QUEUED",
        progress=0.0,
        correlation_id=trace_context.correlation_id if trace_context else None,
        trace_id=trace_context.trace_id if trace_context else None,
        parent_span_id=trace_context.span_id if trace_context else None,
        trace_sampled=trace_context.sampled if trace_context else False,
    )
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
    for candidate in pending_jobs:
        if candidate.retry_count > candidate.max_retries:
            continue
        eligible = candidate.retry_count == 0
        if not eligible and candidate.updated_at:
            backoff_seconds = min(300, (2**candidate.retry_count) * 5)
            cand_time = candidate.updated_at
            if cand_time.tzinfo is None:
                cand_time = cand_time.replace(tzinfo=UTC)
            elapsed = (now - cand_time).total_seconds()
            eligible = elapsed >= backoff_seconds
        elif not candidate.updated_at:
            eligible = True
        if not eligible:
            continue

        changed = (
            db.query(Job)
            .filter(Job.id == candidate.id, Job.status == "PENDING")
            .update(
                {
                    Job.status: "PROCESSING",
                    Job.stage: "PARSING",
                    Job.worker_id: worker_id,
                    Job.claimed_at: now,
                },
                synchronize_session=False,
            )
        )
        if changed == 1:
            db.commit()
            db.refresh(candidate)
            return candidate

    # 2. Recover abandoned PROCESSING jobs whose worker lease has expired
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
    for candidate in expired_query.all():
        changed = (
            db.query(Job)
            .filter(
                Job.id == candidate.id,
                Job.status == "PROCESSING",
                Job.worker_id == candidate.worker_id,
                Job.claimed_at < cutoff,
                Job.retry_count < Job.max_retries,
            )
            .update(
                {
                    Job.status: "PROCESSING",
                    Job.stage: "PARSING",
                    Job.worker_id: worker_id,
                    Job.claimed_at: now,
                    Job.retry_count: Job.retry_count + 1,
                },
                synchronize_session=False,
            )
        )
        if changed == 1:
            db.commit()
            db.refresh(candidate)
            return candidate
    db.rollback()
    return None


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
    current_job = get_job(db, job_id)
    if current_job:
        if worker_id is not None and current_job.worker_id != worker_id:
            raise LostJobLeaseError(f"Job {job_id} is no longer owned by this worker")
        validate_job_transition(current_job.status, current_job.stage, status, stage)
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


def release_job(db: Session, job_id: UUID | str, worker_id: str) -> bool:
    """Release an owned job back to PENDING state upon graceful worker shutdown."""
    if isinstance(job_id, str):
        job_id = UUID(job_id)
    changed = (
        db.query(Job)
        .filter(Job.id == job_id, Job.status == "PROCESSING", Job.worker_id == worker_id)
        .update(
            {
                Job.status: "PENDING",
                Job.stage: "QUEUED",
                Job.worker_id: None,
                Job.claimed_at: None,
            },
            synchronize_session=False,
        )
    )
    if changed == 1:
        db.commit()
        return True
    db.rollback()
    return False


def retry_job(db: Session, job_id: UUID | str) -> Job:
    """Manually retry a FAILED or retryable job, resetting it to PENDING/QUEUED."""
    if isinstance(job_id, str):
        job_id = UUID(job_id)
    job = get_job(db, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    if job.status == "PROCESSING":
        raise ValueError("Cannot retry a job that is currently processing")
    if job.status == "COMPLETED":
        raise ValueError("Cannot retry an already completed job")

    validate_job_transition(job.status, job.stage, "PENDING", "QUEUED")
    job.status = "PENDING"
    job.stage = "QUEUED"
    job.progress = 0.0
    job.worker_id = None
    job.claimed_at = None
    job.retry_count = 0
    job.error_message = None
    job.is_retryable = False

    if job.paper:
        was_ready = job.paper.status == "READY"
        job.paper.status = "PROCESSING"
        job.paper.error_message = None
        if was_ready:
            bump_corpus_revision(db, job.paper.project_id)

    db.commit()
    db.refresh(job)
    return job
