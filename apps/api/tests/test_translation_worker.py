from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.crud.translation import cancel_translation
from app.db.base import Base
from app.db.models import Paper, Project, TranslationDocument
from app.translation_worker import (
    TranslationJob,
    TranslationProcessingError,
    TranslationResult,
    TranslationWorker,
)


@pytest.fixture
def database() -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as db:
        project = Project(name="Translation worker test")
        db.add(project)
        db.flush()
        paper = Paper(
            project_id=project.id,
            filename="fixture.pdf",
            storage_path="papers/fixture.pdf",
            document_sha256=hashlib.sha256(b"fixture").hexdigest(),
            status="READY",
            page_count=1,
        )
        db.add(paper)
        db.flush()
        db.add(
            TranslationDocument(
                project_id=project.id,
                paper_id=paper.id,
                status="PENDING",
                stage="QUEUED",
                idempotency_key="worker-test",
                acknowledge_external_processing=True,
                source_sha256=paper.document_sha256,
                source_storage_path=paper.storage_path,
                source_filename=paper.filename,
                source_page_count=paper.page_count,
                glossary_snapshot=[],
            )
        )
        db.commit()
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


def _translation(factory: sessionmaker[Session]) -> TranslationDocument:
    with factory() as db:
        return db.scalar(select(TranslationDocument))


class SuccessfulProcessor:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def process(self, job: TranslationJob, *, on_progress) -> TranslationResult:
        self.started.set()
        on_progress("layout_analysis", completed_units=0, total_units=1)
        on_progress("translation", completed_units=1, total_units=1)
        return TranslationResult(
            output_storage_path=f"translations/{job.id}/result.pdf",
            output_sha256=hashlib.sha256(b"translated").hexdigest(),
            source_map=[],
        )


class BlockingProcessor:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def process(self, job: TranslationJob, *, on_progress) -> TranslationResult:
        del job, on_progress
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class FailingProcessor:
    def __init__(self, *, retryable: bool) -> None:
        self.retryable = retryable

    async def process(self, job: TranslationJob, *, on_progress) -> TranslationResult:
        del job, on_progress
        raise TranslationProcessingError(
            "PROVIDER_UNAVAILABLE",
            "The translation service is temporarily unavailable.",
            retryable=self.retryable,
        )


class UnexpectedFailingProcessor:
    async def process(self, job: TranslationJob, *, on_progress) -> TranslationResult:
        del job, on_progress
        raise RuntimeError("source quote and provider credentials must not be logged")


def test_once_mode_claims_and_completes_at_most_one_job(database):
    processor = SuccessfulProcessor()
    worker = TranslationWorker(processor, session_factory=database, worker_id="test-worker")

    asyncio.run(worker.run(once=True))

    record = _translation(database)
    assert processor.started.is_set()
    assert record.status == "COMPLETED"
    assert record.output_storage_path == f"translations/{record.id}/result.pdf"
    assert record.completed_units == 1
    assert record.total_units == 1
    assert record.attempt_count == 1
    assert asyncio.run(worker.process_one()) is False


def test_heartbeat_lease_loss_cancels_active_processor(database):
    processor = BlockingProcessor()
    worker = TranslationWorker(
        processor,
        session_factory=database,
        worker_id="test-worker",
        heartbeat_interval=0.01,
    )

    async def execute() -> None:
        run = asyncio.create_task(worker.process_one())
        await asyncio.wait_for(processor.started.wait(), timeout=1)
        with database() as db:
            record = db.scalar(select(TranslationDocument))
            cancel_translation(db, record)
        await asyncio.wait_for(run, timeout=1)

    asyncio.run(execute())

    record = _translation(database)
    assert processor.cancelled.is_set()
    assert record.status == "CANCELLED"
    assert record.output_storage_path is None


@pytest.mark.parametrize("retryable", [True, False])
def test_processor_failures_store_only_safe_classification(database, retryable):
    worker = TranslationWorker(
        FailingProcessor(retryable=retryable),
        session_factory=database,
        worker_id="test-worker",
    )

    assert asyncio.run(worker.process_one()) is True

    record = _translation(database)
    assert record.status == "FAILED"
    assert record.error_code == "PROVIDER_UNAVAILABLE"
    assert record.error_message == "The translation service is temporarily unavailable."
    assert record.is_retryable is retryable
    assert record.output_storage_path is None


def test_unexpected_exception_is_redacted_from_state_and_logs(database, caplog):
    worker = TranslationWorker(
        UnexpectedFailingProcessor(), session_factory=database, worker_id="test-worker"
    )

    asyncio.run(worker.process_one())

    record = _translation(database)
    assert record.error_code == "TRANSLATION_FAILED"
    assert "source quote" not in (record.error_message or "")
    assert "provider credentials" not in caplog.text
    assert "source quote" not in caplog.text
    assert "Traceback" not in caplog.text


def test_stale_attempt_cannot_publish_after_cancellation(database):
    processor = BlockingProcessor()
    worker = TranslationWorker(
        processor,
        session_factory=database,
        worker_id="test-worker",
        heartbeat_interval=0.01,
    )
    stale_token: str | None = None

    async def execute() -> None:
        nonlocal stale_token
        run = asyncio.create_task(worker.process_one())
        await asyncio.wait_for(processor.started.wait(), timeout=1)
        with database() as db:
            record = db.scalar(select(TranslationDocument))
            stale_token = record.attempt_token
            cancel_translation(db, record)
        await asyncio.wait_for(run, timeout=1)

    asyncio.run(execute())
    with database() as db:
        record = db.scalar(select(TranslationDocument))
        assert record.attempt_token != stale_token
        assert record.status == "CANCELLED"
    assert record.output_storage_path is None


def test_graceful_stop_cancels_active_attempt_as_retryable(database):
    processor = BlockingProcessor()
    worker = TranslationWorker(
        processor,
        session_factory=database,
        worker_id="test-worker",
        heartbeat_interval=60,
    )

    async def execute() -> None:
        run = asyncio.create_task(worker.process_one())
        await asyncio.wait_for(processor.started.wait(), timeout=1)
        worker.stop()
        await asyncio.wait_for(run, timeout=1)

    asyncio.run(execute())

    record = _translation(database)
    assert processor.cancelled.is_set()
    assert record.status == "FAILED"
    assert record.error_code == "WORKER_STOPPING"
    assert record.is_retryable is True


def test_once_mode_exits_when_queue_is_empty(database):
    with database() as db:
        db.query(TranslationDocument).delete()
        db.commit()
    worker = TranslationWorker(session_factory=database, worker_id="empty-worker")

    asyncio.run(asyncio.wait_for(worker.run(once=True), timeout=1))
