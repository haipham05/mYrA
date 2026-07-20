import argparse
import asyncio
import logging
import os
import signal
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.config import Settings
from app.crud.graph import (
    claim_next_graph_event,
    release_graph_event,
    renew_graph_event_lease,
)
from app.crud.job import claim_next_job, release_job, renew_job_lease
from app.db.session import SessionLocal
from app.logging import configure_logging
from app.observability.context import OperationContext, use_operation_context
from app.observability.telemetry import get_telemetry
from app.services.graphrag.processor import GraphEventProcessor
from app.services.ingestion import IngestionPipeline

logger = logging.getLogger("myra.worker")


def _initial_queue_age_seconds(created_at: datetime | None) -> float | None:
    """Return age since original enqueue; retry delay is not derivable from this value."""
    if created_at is None:
        return None
    normalized = created_at if created_at.tzinfo else created_at.replace(tzinfo=UTC)
    return round(max(0.0, (datetime.now(UTC) - normalized.astimezone(UTC)).total_seconds()), 3)


def _persisted_job_outcome(job, worker_id: str, claimed_retry_count: int) -> str:
    """Map durable job state to an honest attempt outcome after processors return."""
    if job.status == "COMPLETED":
        if job.worker_id != worker_id or job.retry_count != claimed_retry_count:
            return "abandoned"
        return "success"
    if job.status == "PENDING":
        return "retry_scheduled"
    if job.status == "FAILED":
        return "failed"
    if job.status == "PROCESSING" and job.worker_id != worker_id:
        return "abandoned"
    return "incomplete"


def _persisted_graph_outcome(
    event, worker_id: str, claimed_attempts: int, completed_by_attempt: bool
) -> str:
    """Map durable graph-event state to an honest attempt outcome."""
    if event.status == "COMPLETED":
        if not completed_by_attempt or event.attempts != claimed_attempts:
            return "abandoned"
        return "success"
    if event.status == "PENDING":
        return "retry_scheduled"
    if event.status == "FAILED":
        return "failed"
    if event.status == "PROCESSING" and event.lease_owner != worker_id:
        return "abandoned"
    return "incomplete"


async def run_worker(
    poll_interval: float = 2.0, once: bool = False, heartbeat_interval: float = 60.0
) -> None:
    settings = Settings.from_environment()
    configure_logging(settings.log_level)

    if settings.check_migration_compatibility:
        from app.db.compatibility import (
            check_budget_schema_compatibility,
            check_schema_compatibility,
        )
        from app.db.session import engine

        check_schema_compatibility(engine)
        check_budget_schema_compatibility(settings)

    worker_id = f"worker-{os.getpid()}-{uuid4().hex[:6]}"
    logger.info("worker_started", extra={"worker_id": worker_id})

    pipeline = IngestionPipeline()
    processor = GraphEventProcessor(settings=settings)
    running = True
    active_processing_task: asyncio.Task | None = None

    def _sig_handler(sig, frame):
        nonlocal running
        logger.info("worker_stopping")
        running = False
        if active_processing_task and not active_processing_task.done():
            active_processing_task.cancel()

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except (ValueError, AttributeError):
        pass

    async def _heartbeat(
        job_id, worker_id: str, processing_task: asyncio.Task, interval: float = 60.0
    ):
        while True:
            await asyncio.sleep(interval)
            try:
                with SessionLocal() as h_db:
                    if not renew_job_lease(h_db, job_id, worker_id=worker_id):
                        logger.warning("job_lease_lost", extra={"job_id": str(job_id)})
                        processing_task.cancel()
                        return
            except Exception:
                pass

    async def _graph_heartbeat(
        event_id: UUID, worker_id: str, processing_task: asyncio.Task, interval: float = 60.0
    ):
        while True:
            await asyncio.sleep(interval)
            try:
                with SessionLocal() as h_db:
                    if not renew_graph_event_lease(h_db, event_id, worker_id=worker_id):
                        logger.warning("graph_event_lease_lost", extra={"event_id": str(event_id)})
                        processing_task.cancel()
                        return
            except Exception:
                pass

    async def _process_job_with_context(db, job):
        context = OperationContext.validated(
            correlation_id=job.correlation_id,
            trace_id=job.trace_id,
            span_id=job.parent_span_id,
            sampled=job.trace_sampled,
        )
        with use_operation_context(context):
            telemetry = get_telemetry()
            claimed_retry_count = int(job.retry_count)
            with telemetry.operation(
                "worker.ingestion_attempt",
                metadata={
                    "job_id": str(job.id),
                    "paper_id": str(job.paper_id),
                    "attempt_number": int(job.retry_count) + 1,
                    "initial_queue_age_seconds": _initial_queue_age_seconds(job.created_at),
                    "queue_age_basis": "since_initial_enqueue",
                },
            ) as observation:
                await pipeline.process_paper(
                    db, paper_id=job.paper_id, job_id=job.id, worker_id=worker_id
                )
                try:
                    db.refresh(job)
                    outcome = _persisted_job_outcome(job, worker_id, claimed_retry_count)
                except Exception:
                    outcome = "unknown"
                if observation is not None:
                    observation.update(metadata={"outcome": outcome})

    async def _process_graph_event_with_context(db, event):
        context = OperationContext.validated(
            correlation_id=event.correlation_id,
            trace_id=event.trace_id,
            span_id=event.parent_span_id,
            sampled=event.trace_sampled,
        )
        with use_operation_context(context):
            telemetry = get_telemetry()
            claimed_attempts = int(event.attempts)
            with telemetry.operation(
                "worker.graph_event_attempt",
                metadata={
                    "event_id": str(event.id),
                    "paper_id": str(event.paper_id),
                    "action": event.action,
                    "attempt_number": int(event.attempts) + 1,
                    "initial_queue_age_seconds": _initial_queue_age_seconds(event.created_at),
                    "queue_age_basis": "since_initial_enqueue",
                },
            ) as observation:
                completed_by_attempt = await processor.process_graph_event(
                    db, event_id=event.id, worker_id=worker_id
                )
                try:
                    db.refresh(event)
                    outcome = _persisted_graph_outcome(
                        event, worker_id, claimed_attempts, completed_by_attempt
                    )
                except Exception:
                    outcome = "unknown"
                if observation is not None:
                    observation.update(metadata={"outcome": outcome})

    while running:
        db = SessionLocal()
        try:
            job = claim_next_job(db, worker_id=worker_id)
            if job:
                logger.info(
                    "claimed_job",
                    extra={"job_id": str(job.id), "paper_id": str(job.paper_id)},
                )
                processing_task = asyncio.create_task(_process_job_with_context(db, job))
                active_processing_task = processing_task
                heartbeat_task = asyncio.create_task(
                    _heartbeat(
                        job.id,
                        worker_id=worker_id,
                        processing_task=processing_task,
                        interval=heartbeat_interval,
                    )
                )

                try:
                    await processing_task
                except asyncio.CancelledError:
                    db.rollback()
                    current_task = asyncio.current_task()
                    is_stopping = (not running) or (
                        current_task is not None
                        and getattr(current_task, "cancelling", lambda: 0)() > 0
                    )
                    if is_stopping:
                        running = False
                        with SessionLocal() as r_db:
                            release_job(r_db, job.id, worker_id=worker_id)
                        logger.info("released_job_on_shutdown", extra={"job_id": str(job.id)})
                        raise
                finally:
                    if not processing_task.done():
                        processing_task.cancel()
                        try:
                            await processing_task
                        except asyncio.CancelledError:
                            pass
                    heartbeat_task.cancel()
                    active_processing_task = None
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                logger.info("finished_job", extra={"job_id": str(job.id)})

                if once:
                    break
                continue

            # If no ingestion job, poll for graph event
            event = (
                claim_next_graph_event(db, worker_id=worker_id)
                if settings.graphrag_enabled
                else None
            )
            if event:
                logger.info(
                    "claimed_graph_event",
                    extra={
                        "event_id": str(event.id),
                        "paper_id": str(event.paper_id),
                        "action": event.action,
                    },
                )
                processing_task = asyncio.create_task(_process_graph_event_with_context(db, event))
                active_processing_task = processing_task
                heartbeat_task = asyncio.create_task(
                    _graph_heartbeat(
                        event.id,
                        worker_id=worker_id,
                        processing_task=processing_task,
                        interval=heartbeat_interval,
                    )
                )

                try:
                    await processing_task
                except asyncio.CancelledError:
                    db.rollback()
                    current_task = asyncio.current_task()
                    is_stopping = (not running) or (
                        current_task is not None
                        and getattr(current_task, "cancelling", lambda: 0)() > 0
                    )
                    if is_stopping:
                        running = False
                        with SessionLocal() as r_db:
                            release_graph_event(r_db, event.id, worker_id=worker_id)
                        logger.info(
                            "released_graph_event_on_shutdown",
                            extra={"event_id": str(event.id)},
                        )
                        raise
                finally:
                    if not processing_task.done():
                        processing_task.cancel()
                        try:
                            await processing_task
                        except asyncio.CancelledError:
                            pass
                    heartbeat_task.cancel()
                    active_processing_task = None
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                logger.info("finished_graph_event", extra={"event_id": str(event.id)})

                if once:
                    break
                continue

        except Exception as err:
            logger.error("worker_loop_error", exc_info=err)
        finally:
            db.close()

        if once:
            break
        await asyncio.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="mYrA Ingestion Worker")
    parser.add_argument("--once", action="store_true", help="Process at most one job and exit")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Seconds between polls")
    args = parser.parse_args()

    asyncio.run(run_worker(poll_interval=args.poll_interval, once=args.once))


if __name__ == "__main__":
    main()
