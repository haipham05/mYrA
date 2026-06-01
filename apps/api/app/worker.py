import argparse
import asyncio
import logging
import os
import signal
from uuid import uuid4

from app.config import Settings
from app.crud.job import claim_next_job, release_job, renew_job_lease
from app.db.session import SessionLocal
from app.logging import configure_logging
from app.services.ingestion import IngestionPipeline

logger = logging.getLogger("myra.worker")


async def run_worker(
    poll_interval: float = 2.0, once: bool = False, heartbeat_interval: float = 60.0
) -> None:
    settings = Settings.from_environment()
    configure_logging(settings.log_level)

    if settings.check_migration_compatibility:
        from app.db.compatibility import check_schema_compatibility
        from app.db.session import engine

        check_schema_compatibility(engine)

    worker_id = f"worker-{os.getpid()}-{uuid4().hex[:6]}"
    logger.info("worker_started", extra={"worker_id": worker_id})

    pipeline = IngestionPipeline()
    running = True
    active_processing_task: asyncio.Task | None = None

    def _sig_handler(sig, frame):
        nonlocal running
        logger.info("worker_stopping")
        running = False
        if active_processing_task and not active_processing_task.done():
            active_processing_task.cancel()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

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

    while running:
        db = SessionLocal()
        try:
            job = claim_next_job(db, worker_id=worker_id)
            if job:
                logger.info(
                    "claimed_job",
                    extra={"job_id": str(job.id), "paper_id": str(job.paper_id)},
                )
                processing_task = asyncio.create_task(
                    pipeline.process_paper(
                        db, paper_id=job.paper_id, job_id=job.id, worker_id=worker_id
                    )
                )
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
                    if not running:
                        with SessionLocal() as r_db:
                            release_job(r_db, job.id, worker_id=worker_id)
                        logger.info("released_job_on_shutdown", extra={"job_id": str(job.id)})
                finally:
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
