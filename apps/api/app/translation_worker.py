"""Independent queue runner for PDF translation jobs.

This worker intentionally owns a separate process and polling loop from the
ingestion/graph worker. The processor is injected so the queue state machine
can be tested without a provider, cloud storage, or PDF engine.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from app.config import Settings
from app.crud.translation import (
    claim_next_translation,
    complete_translation,
    fail_translation,
    renew_translation_lease,
    update_translation_stage,
)
from app.db.models import TranslationDocument
from app.db.session import SessionLocal
from app.logging import configure_logging
from app.observability.telemetry import get_telemetry

logger = logging.getLogger("myra.translation_worker")
LEASE_SECONDS = 300
HEARTBEAT_SECONDS = 60
MAX_JOB_SECONDS = 30 * 60

_STAGE_MAP = {
    "SOURCE_DOWNLOAD": "SOURCE_DOWNLOAD",
    "source_download": "SOURCE_DOWNLOAD",
    "layout_analysis": "LAYOUT_ANALYSIS",
    "term_extraction": "TERM_EXTRACTION",
    "translation": "TRANSLATION",
    "validation": "VALIDATION",
    "rendering": "RENDER",
    "finalizing": "PUBLICATION",
}


@dataclass(frozen=True, slots=True)
class TranslationJob:
    """Immutable, minimum job data passed to a processor."""

    id: UUID
    project_id: UUID
    paper_id: UUID
    source_sha256: str
    source_storage_path: str
    source_filename: str
    source_page_count: int | None
    glossary_snapshot: tuple[Mapping[str, str], ...]
    worker_id: str
    attempt_token: str
    attempt_count: int


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Validated artifact metadata the worker may publish to the queue row."""

    output_storage_path: str
    output_sha256: str
    source_map: list[dict[str, Any]]


class TranslationProcessingError(Exception):
    """Safe processor failure; message must be user-safe and contain no source."""

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        self.code = code if code.isupper() and len(code) <= 80 else "TRANSLATION_FAILED"
        self.safe_message = message[:500]
        self.retryable = retryable
        super().__init__(self.code)


ProgressCallback = Callable[..., None]


class TranslationProcessor(Protocol):
    async def process(
        self, job: TranslationJob, *, on_progress: ProgressCallback
    ) -> TranslationResult: ...

    async def cleanup(self, result: TranslationResult) -> None: ...


class UnconfiguredTranslationProcessor:
    """Fails clearly until the engine/storage adapter is wired into this runner."""

    async def process(
        self, job: TranslationJob, *, on_progress: ProgressCallback
    ) -> TranslationResult:
        del job, on_progress
        raise TranslationProcessingError(
            "TRANSLATION_PROCESSOR_UNAVAILABLE",
            "The translation processor is not configured.",
            retryable=False,
        )


def _snapshot(translation: TranslationDocument, *, worker_id: str) -> TranslationJob:
    if not translation.attempt_token:
        raise ValueError("Claimed translation is missing its attempt token")
    return TranslationJob(
        id=translation.id,
        project_id=translation.project_id,
        paper_id=translation.paper_id,
        source_sha256=translation.source_sha256,
        source_storage_path=translation.source_storage_path,
        source_filename=translation.source_filename,
        source_page_count=translation.source_page_count,
        glossary_snapshot=tuple(dict(item) for item in (translation.glossary_snapshot or [])),
        worker_id=worker_id,
        attempt_token=translation.attempt_token,
        attempt_count=translation.attempt_count,
    )


class TranslationWorker:
    """Claims and processes one translation at a time with lease fencing."""

    def __init__(
        self,
        processor: TranslationProcessor | None = None,
        *,
        session_factory: Callable[[], Session] = SessionLocal,
        worker_id: str | None = None,
        poll_interval: float = 2.0,
        heartbeat_interval: float = HEARTBEAT_SECONDS,
        job_timeout: float = MAX_JOB_SECONDS,
    ) -> None:
        if poll_interval < 0 or heartbeat_interval <= 0:
            raise ValueError("poll_interval must be nonnegative and heartbeat positive")
        if not 0 < job_timeout <= MAX_JOB_SECONDS:
            raise ValueError("job_timeout must be between 0 and 1800 seconds")
        if processor is None:
            from app.services.translation.processor import BabelDocTranslationProcessor

            processor = BabelDocTranslationProcessor(session_factory=session_factory)
        self.processor = processor
        self.session_factory = session_factory
        self.worker_id = worker_id or f"translation-{os.getpid()}-{uuid4().hex[:8]}"
        self.poll_interval = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.job_timeout = job_timeout
        self._stop = asyncio.Event()
        self._active_processing: asyncio.Task[TranslationResult] | None = None

    def stop(self) -> None:
        self._stop.set()
        if self._active_processing is not None and not self._active_processing.done():
            self._active_processing.cancel()

    def _claim(self) -> TranslationJob | None:
        with self.session_factory() as db:
            translation = claim_next_translation(db, worker_id=self.worker_id)
            if translation is None:
                return None
            return _snapshot(translation, worker_id=self.worker_id)

    def _update_stage(
        self,
        job: TranslationJob,
        stage: str,
        completed_units: int | None,
        total_units: int | None,
    ) -> bool:
        canonical_stage = _STAGE_MAP.get(stage)
        if canonical_stage is None:
            canonical_stage = "TRANSLATION"
        with self.session_factory() as db:
            return update_translation_stage(
                db,
                job.id,
                worker_id=self.worker_id,
                attempt_token=job.attempt_token,
                stage=canonical_stage,
                completed_units=completed_units,
                total_units=total_units,
            )

    def _heartbeat(self, job: TranslationJob) -> bool:
        with self.session_factory() as db:
            return renew_translation_lease(
                db,
                job.id,
                worker_id=self.worker_id,
                attempt_token=job.attempt_token,
            )

    def _finish(self, job: TranslationJob, result: TranslationResult) -> bool:
        with self.session_factory() as db:
            return complete_translation(
                db,
                job.id,
                worker_id=self.worker_id,
                attempt_token=job.attempt_token,
                output_storage_path=result.output_storage_path,
                output_sha256=result.output_sha256,
                source_map=result.source_map,
            )

    def _fail(
        self,
        job: TranslationJob,
        *,
        code: str,
        message: str,
        retryable: bool,
    ) -> bool:
        with self.session_factory() as db:
            return fail_translation(
                db,
                job.id,
                worker_id=self.worker_id,
                attempt_token=job.attempt_token,
                error_code=code,
                error_message=message,
                retryable=retryable,
            )

    async def process_one(self) -> bool:
        """Process one available job; return False when the queue is empty."""
        try:
            job = self._claim()
        except Exception:
            logger.error("translation_claim_failed")
            return False
        if job is None:
            return False

        logger.info("translation_claimed", extra={"translation_id": str(job.id)})
        get_telemetry().event(
            "translation.claim",
            metadata={
                "translation_id": str(job.id),
                "attempt": job.attempt_count,
                "worker_id": self.worker_id,
                "test_run": os.getenv("MYRA_TRANSLATION_TEST_RUN", "false").lower() == "true",
            },
            output={"outcome": "claimed"},
        )
        lost_lease = asyncio.Event()
        processing = asyncio.create_task(self._run_processor(job))
        self._active_processing = processing
        heartbeat = asyncio.create_task(self._heartbeat_loop(job, processing, lost_lease))
        try:
            result = await asyncio.wait_for(processing, timeout=self.job_timeout)
            if lost_lease.is_set():
                await self._cleanup_result(result)
                return True
            try:
                completed = self._finish(job, result)
            except Exception:
                await self._cleanup_result(result)
                raise
            if not completed:
                await self._cleanup_result(result)
                logger.warning(
                    "translation_completion_fenced", extra={"translation_id": str(job.id)}
                )
            else:
                logger.info("translation_completed", extra={"translation_id": str(job.id)})
        except TimeoutError:
            await self._cancel_task(processing)
            self._fail(
                job,
                code="TRANSLATION_DEADLINE_EXCEEDED",
                message="Translation exceeded the 30-minute processing limit.",
                retryable=True,
            )
            logger.warning("translation_deadline_exceeded", extra={"translation_id": str(job.id)})
        except asyncio.CancelledError:
            await self._cancel_task(processing)
            current_task = asyncio.current_task()
            stopping_worker = current_task is not None and current_task.cancelling() > 0
            stopping_requested = self._stop.is_set()
            # A shutdown leaves a clear retryable state. API cancellation or a
            # lost lease has already fenced this token, so this update is a no-op.
            if (stopping_worker or stopping_requested) and not lost_lease.is_set():
                self._fail(
                    job,
                    code="WORKER_STOPPING",
                    message="Translation stopped before completion and can be retried.",
                    retryable=True,
                )
            if stopping_worker:
                raise
            # The heartbeat cancelled only the child processing task after its
            # lease check failed (usually because the owner cancelled the job).
            # Treat that as a handled, fenced attempt rather than killing the loop.
        except TranslationProcessingError as exc:
            self._fail(
                job,
                code=exc.code,
                message=exc.safe_message,
                retryable=exc.retryable,
            )
            logger.warning(
                "translation_failed",
                extra={"translation_id": str(job.id), "error_code": exc.code},
            )
        except Exception as exc:
            self._fail(
                job,
                code="TRANSLATION_FAILED",
                message="Translation failed unexpectedly. Retry or inspect service status.",
                retryable=True,
            )
            # Deliberately omit exception text and traceback: engine/provider
            # exceptions can contain document text, URLs, or credentials.
            logger.error(
                "translation_failed: %s",
                type(exc).__name__,
                extra={"translation_id": str(job.id), "failure_class": type(exc).__name__},
            )
        finally:
            heartbeat.cancel()
            self._active_processing = None
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
        return True

    async def _cleanup_result(self, result: TranslationResult) -> None:
        cleanup = getattr(self.processor, "cleanup", None)
        if not callable(cleanup):
            return
        try:
            await cleanup(result)
        except Exception:
            logger.warning("translation_artifact_cleanup_failed")

    async def _run_processor(self, job: TranslationJob) -> TranslationResult:
        def progress(
            stage: str,
            *,
            completed_units: int | None = None,
            total_units: int | None = None,
        ) -> None:
            if not self._update_stage(job, stage, completed_units, total_units):
                raise TranslationProcessingError(
                    "TRANSLATION_ATTEMPT_REVOKED",
                    "Translation attempt was cancelled or superseded.",
                    retryable=False,
                )

        return await self.processor.process(job, on_progress=progress)

    async def _heartbeat_loop(
        self,
        job: TranslationJob,
        processing: asyncio.Task[TranslationResult],
        lost_lease: asyncio.Event,
    ) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                renewed = self._heartbeat(job)
            except Exception:
                # Transient database errors do not immediately revoke ownership;
                # the next interval retries and the database lease remains authoritative.
                logger.warning("translation_heartbeat_error", extra={"translation_id": str(job.id)})
                continue
            if not renewed:
                lost_lease.set()
                logger.warning("translation_lease_lost", extra={"translation_id": str(job.id)})
                processing.cancel()
                return

    async def _cancel_task(self, task: asyncio.Task[Any]) -> None:
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    async def run(self, *, once: bool = False) -> None:
        while not self._stop.is_set():
            processed = await self.process_one()
            if once:
                return
            if not processed:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
                except TimeoutError:
                    continue


async def run_translation_worker(*, once: bool = False, poll_interval: float = 2.0) -> None:
    settings = Settings.from_environment()
    configure_logging(settings.log_level)
    worker = TranslationWorker(poll_interval=poll_interval)
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, worker.stop)
        except (NotImplementedError, RuntimeError):
            pass
    logger.info("translation_worker_started", extra={"worker_id": worker.worker_id})
    await worker.run(once=once)
    logger.info("translation_worker_stopped", extra={"worker_id": worker.worker_id})


def main() -> None:
    parser = argparse.ArgumentParser(description="mYrA PDF translation worker")
    parser.add_argument("--once", action="store_true", help="Process at most one translation")
    parser.add_argument("--poll-interval", type=float, default=2.0)
    args = parser.parse_args()
    asyncio.run(run_translation_worker(once=args.once, poll_interval=args.poll_interval))


def _run_canonical_main() -> None:
    """Avoid duplicate exception classes when this module is executed as a module."""
    from app.translation_worker import main as canonical_main

    canonical_main()


if __name__ == "__main__":
    _run_canonical_main()
