import argparse
import asyncio
import logging
import os
import signal
from uuid import uuid4

from app.config import Settings
from app.crud.job import claim_next_job, renew_job_lease
from app.db.session import SessionLocal
from app.logging import configure_logging
from app.services.ingestion import IngestionPipeline

logger = logging.getLogger("myra.worker")


async def run_worker(poll_interval: float = 2.0, once: bool = False) -> None:
    settings = Settings.from_environment()
    configure_logging(settings.log_level)

    worker_id = f"worker-{os.getpid()}-{uuid4().hex[:6]}"
    logger.info("worker_started", extra={"worker_id": worker_id})

    pipeline = IngestionPipeline()
    running = True

    def _sig_handler(sig, frame):
        nonlocal running
        logger.info("worker_stopping")
        running = False

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    async def _heartbeat(job_id, interval: float = 60.0):
        while True:
            await asyncio.sleep(interval)
            try:
                with SessionLocal() as h_db:
                    renew_job_lease(h_db, job_id)
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
                heartbeat_task = asyncio.create_task(_heartbeat(job.id))
                try:
                    await pipeline.process_paper(db, paper_id=job.paper_id, job_id=job.id)
                finally:
                    heartbeat_task.cancel()
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
