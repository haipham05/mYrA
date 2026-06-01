from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.crud.job import create_job, retry_job, update_job_progress
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.base import Base
from app.schemas.job import JobResponse, JobStage, JobStatus
from app.schemas.project import ProjectCreate
from app.services.job_state_machine import (
    InvalidJobTransitionError,
    validate_job_transition,
)


@pytest.fixture
def db(tmp_path):
    db_file = tmp_path / "test_state_machine.db"
    engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


def test_legal_transitions_pass():
    # Progress through happy path
    validate_job_transition(
        JobStatus.PENDING, JobStage.QUEUED, JobStatus.PROCESSING, JobStage.PARSING
    )
    validate_job_transition(
        JobStatus.PROCESSING, JobStage.PARSING, JobStatus.PROCESSING, JobStage.CHUNKING
    )
    validate_job_transition(
        JobStatus.PROCESSING, JobStage.CHUNKING, JobStatus.PROCESSING, JobStage.EMBEDDING
    )
    validate_job_transition(
        JobStatus.PROCESSING, JobStage.EMBEDDING, JobStatus.PROCESSING, JobStage.INDEXING
    )
    validate_job_transition(
        JobStatus.PROCESSING, JobStage.INDEXING, JobStatus.COMPLETED, JobStage.COMPLETED
    )

    # Identical state (progress update)
    validate_job_transition(
        JobStatus.PROCESSING, JobStage.PARSING, JobStatus.PROCESSING, JobStage.PARSING
    )

    # Retry or release to PENDING
    validate_job_transition(
        JobStatus.PROCESSING, JobStage.PARSING, JobStatus.PENDING, JobStage.QUEUED
    )
    validate_job_transition(
        JobStatus.PROCESSING, JobStage.INDEXING, JobStatus.PENDING, JobStage.QUEUED
    )

    # Terminal failure
    validate_job_transition(
        JobStatus.PROCESSING, JobStage.PARSING, JobStatus.FAILED, JobStage.FAILED
    )

    # Explicit retry from FAILED
    validate_job_transition(JobStatus.FAILED, JobStage.FAILED, JobStatus.PENDING, JobStage.QUEUED)


def test_illegal_transitions_raise_error():
    # Completed jobs cannot transition
    with pytest.raises(InvalidJobTransitionError, match="Illegal job transition"):
        validate_job_transition(
            JobStatus.COMPLETED, JobStage.COMPLETED, JobStatus.PROCESSING, JobStage.PARSING
        )

    with pytest.raises(InvalidJobTransitionError, match="Illegal job transition"):
        validate_job_transition(
            JobStatus.COMPLETED, JobStage.COMPLETED, JobStatus.FAILED, JobStage.FAILED
        )

    # Cannot jump directly to COMPLETED from PENDING
    with pytest.raises(InvalidJobTransitionError, match="Illegal job transition"):
        validate_job_transition(
            JobStatus.PENDING, JobStage.QUEUED, JobStatus.COMPLETED, JobStage.COMPLETED
        )

    # Cannot jump backwards in processing stages without going through PENDING/QUEUED
    with pytest.raises(InvalidJobTransitionError, match="Illegal job transition"):
        validate_job_transition(
            JobStatus.PROCESSING, JobStage.INDEXING, JobStatus.PROCESSING, JobStage.PARSING
        )


def test_update_job_progress_enforces_state_machine(db):
    proj = create_project(db, ProjectCreate(name="State Machine Test"))
    paper = create_paper(db, proj.id, "sm.pdf", "sm.pdf")
    job = create_job(db, paper.id)

    # Start processing
    job.status = "PROCESSING"
    job.stage = "PARSING"
    job.worker_id = "worker-sm"
    db.commit()

    # Progress update within stage succeeds
    update_job_progress(db, job.id, stage="PARSING", progress=0.2, worker_id="worker-sm")
    db.refresh(job)
    assert job.progress == 0.2

    # Advance to CHUNKING succeeds
    update_job_progress(db, job.id, stage="CHUNKING", progress=0.5, worker_id="worker-sm")
    db.refresh(job)
    assert job.stage == "CHUNKING"

    # Illegal transition: attempting to jump backwards to PARSING raises error
    with pytest.raises(InvalidJobTransitionError):
        update_job_progress(db, job.id, stage="PARSING", progress=0.1, worker_id="worker-sm")


def test_retry_job_transitions_failed_to_pending(db):
    proj = create_project(db, ProjectCreate(name="Retry Test"))
    paper = create_paper(db, proj.id, "retry.pdf", "retry.pdf")
    job = create_job(db, paper.id)

    # Set job to FAILED
    job.status = "FAILED"
    job.stage = "FAILED"
    job.progress = 1.0
    job.retry_count = 3
    job.error_message = "[CORRUPTED_PDF] Unreadable document"
    paper.status = "FAILED"
    db.commit()

    # Retry job
    retried = retry_job(db, job.id)
    assert retried.status == "PENDING"
    assert retried.stage == "QUEUED"
    assert retried.progress == 0.0
    assert retried.retry_count == 0
    assert retried.error_message is None
    assert retried.worker_id is None
    assert paper.status == "PROCESSING"

    # Retrying a pending or processing job should fail
    with pytest.raises(ValueError, match="Cannot retry a job that is currently processing"):
        retried.status = "PROCESSING"
        db.commit()
        retry_job(db, job.id)

    with pytest.raises(ValueError, match="Cannot retry an already completed job"):
        retried.status = "COMPLETED"
        retried.stage = "COMPLETED"
        db.commit()
        retry_job(db, job.id)


def test_job_response_schema_computes_attempt_and_error_code():
    # 1. New pending job
    resp_pending = JobResponse(
        id=uuid4(),
        paper_id=uuid4(),
        status=JobStatus.PENDING,
        stage=JobStage.QUEUED,
        progress=0.0,
        created_at="2026-09-26T00:00:00Z",
        updated_at="2026-09-26T00:00:00Z",
    )
    assert resp_pending.attempt_count == 0
    assert resp_pending.error_code is None

    # 2. Failed job with structured error code
    resp_failed = JobResponse(
        id=uuid4(),
        paper_id=uuid4(),
        status=JobStatus.FAILED,
        stage=JobStage.FAILED,
        progress=1.0,
        retry_count=2,
        error_message="[CORRUPTED_PDF] Failed to parse cross reference stream",
        created_at="2026-09-26T00:00:00Z",
        updated_at="2026-09-26T00:00:00Z",
    )
    assert resp_failed.attempt_count == 3  # retry_count (2) + 1
    assert resp_failed.error_code == "CORRUPTED_PDF"
