"""Graph event processor executing durability, lifecycle, and Neo4j publication.

Decoupled from paper ingestion; maintains strict failure isolation where
graph failures NEVER modify or fail the authoritative Paper record.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.config import Settings
from app.crud.graph import (
    LostGraphLeaseError,
    complete_graph_event,
    fail_graph_event,
    get_graph_event,
)
from app.db.models import GraphFactSnapshot, Paper
from app.services.graphrag.neo4j_repository import Neo4jRepository

logger = logging.getLogger("myra.graphrag.processor")


class GraphEventProcessor:
    """Processes GraphEvents from the durable queue/outbox."""

    def __init__(
        self,
        settings: Settings | None = None,
        repo: Neo4jRepository | None = None,
    ) -> None:
        self.settings = settings or Settings.from_environment()
        self.repo = repo
        if self.repo is None and self.settings.graphrag_enabled and self.settings.neo4j_uri:
            try:
                self.repo = Neo4jRepository.from_settings(self.settings)
            except Exception as exc:
                logger.warning("Could not initialize Neo4jRepository: %s", exc)
                self.repo = None

    async def process_graph_event(
        self,
        db: Session,
        event_id: UUID | str,
        worker_id: str,
    ) -> None:
        """Execute processing for a single claimed GraphEvent.

        Guarantees that Paper records are NEVER modified or marked failed.
        """
        if isinstance(event_id, str):
            event_id = UUID(event_id)

        event = get_graph_event(db, event_id)
        if not event or event.status != "PROCESSING" or event.lease_owner != worker_id:
            logger.warning(
                "Lease verification failed for event %s (worker: %s)",
                event_id,
                worker_id,
            )
            return

        now = datetime.now(tz=UTC)
        if event.lease_expires_at:
            expiry = event.lease_expires_at
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=UTC)
            if expiry < now:
                logger.warning(
                    "Lease expired for event %s (expiry: %s, now: %s)",
                    event_id,
                    expiry,
                    now,
                )
                return

        try:
            if event.action == "DELETE":
                if self.repo is not None:
                    try:
                        self.repo.delete_paper_facts(
                            project_id=event.project_id,
                            paper_id=event.paper_id,
                        )
                    except Exception as repo_err:
                        logger.error(
                            "Failed to delete paper facts in Neo4j for event %s",
                            event_id,
                            exc_info=repo_err,
                        )
                        fail_graph_event(
                            db=db,
                            event_id=event.id,
                            worker_id=worker_id,
                            error_code="NEO4J_DELETE_FAILED",
                            error_message=str(repo_err),
                            is_transient=True,
                        )
                        return
                complete_graph_event(db, event.id, worker_id=worker_id)
                logger.info("Completed DELETE graph event %s", event.id)
                return

            elif event.action == "UPSERT":
                # Validate that paper exists and is READY
                paper = db.get(Paper, event.paper_id)
                if not paper:
                    fail_graph_event(
                        db=db,
                        event_id=event.id,
                        worker_id=worker_id,
                        error_code="PAPER_NOT_FOUND",
                        error_message=f"Paper {event.paper_id} not found in database",
                        is_transient=False,
                    )
                    return

                if paper.status != "READY":
                    fail_graph_event(
                        db=db,
                        event_id=event.id,
                        worker_id=worker_id,
                        error_code="PAPER_NOT_READY",
                        error_message=(
                            f"Paper {event.paper_id} status is '{paper.status}', expected 'READY'"
                        ),
                        is_transient=True,
                    )
                    return

                # If snapshots exist, publish them to Neo4j
                snapshots = list(event.snapshots) if event.snapshots else []
                if not snapshots:
                    snapshots = (
                        db.query(GraphFactSnapshot)
                        .filter(
                            GraphFactSnapshot.project_id == event.project_id,
                            GraphFactSnapshot.paper_id == event.paper_id,
                            GraphFactSnapshot.generation_id == event.generation_id,
                        )
                        .all()
                    )

                if snapshots and self.repo is not None:
                    fact_payloads: list[dict[str, Any]] = [
                        {
                            "fact_id": s.fact_id,
                            "subject_key": s.subject_key,
                            "subject_name": s.subject_name,
                            "subject_type": s.subject_type,
                            "predicate": s.predicate,
                            "object_key": s.object_key,
                            "object_name": s.object_name,
                            "object_type": s.object_type,
                            "qualifiers": s.qualifiers,
                            "char_start": s.char_start,
                            "char_end": s.char_end,
                            "page_number": s.page_number,
                            "exact_quote": s.exact_quote,
                        }
                        for s in snapshots
                    ]
                    self.repo.upsert_facts(
                        project_id=event.project_id,
                        paper_id=event.paper_id,
                        generation_id=event.generation_id,
                        facts=fact_payloads,
                    )
                    self.repo.retire_older_generations(
                        project_id=event.project_id,
                        paper_id=event.paper_id,
                        active_generation_id=event.generation_id,
                    )

                # Complete event
                complete_graph_event(db, event.id, worker_id=worker_id)
                logger.info("Completed UPSERT graph event %s", event.id)
                return

            else:
                fail_graph_event(
                    db=db,
                    event_id=event.id,
                    worker_id=worker_id,
                    error_code="UNKNOWN_ACTION",
                    error_message=f"Unknown graph event action: {event.action}",
                    is_transient=False,
                )
                return

        except LostGraphLeaseError:
            logger.warning("Worker %s lost lease for graph event %s", worker_id, event_id)
            return
        except Exception as exc:
            logger.error("Error processing graph event %s", event_id, exc_info=exc)
            exc_str = str(exc).lower()
            is_transient = True
            error_code = "PROCESSING_ERROR"
            if "timeout" in exc_str or "timed out" in exc_str or isinstance(exc, TimeoutError):
                error_code = "TIMEOUT"
                is_transient = True
            elif "rate limit" in exc_str:
                error_code = "RATE_LIMIT"
                is_transient = True
            elif "not found" in exc_str:
                error_code = "NOT_FOUND"
                is_transient = False
            elif "connection" in exc_str:
                error_code = "CONNECTION_ERROR"
                is_transient = True

            fail_graph_event(
                db=db,
                event_id=event.id,
                worker_id=worker_id,
                error_code=error_code,
                error_message=str(exc),
                is_transient=is_transient,
            )
