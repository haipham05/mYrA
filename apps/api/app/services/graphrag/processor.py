"""Graph event processor executing durability, lifecycle, and Neo4j publication.

Decoupled from paper ingestion; maintains strict failure isolation where
graph failures NEVER modify or fail the authoritative Paper record.
"""

import json
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
    lock_graph_publication,
)
from app.db.models import Paper
from app.services.graphrag.extractor import (
    ExtractionResult,
    GraphExtractionAdapter,
    parse_and_validate_extraction,
)
from app.services.graphrag.input_selector import select_extraction_inputs
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.snapshots import (
    get_verified_fact_snapshots,
    persist_verified_fact_snapshots,
)
from app.services.graphrag.verifier import verify_candidate_fact

logger = logging.getLogger("myra.graphrag.processor")


class GraphEventProcessor:
    """Processes GraphEvents from the durable queue/outbox."""

    def __init__(
        self,
        settings: Settings | None = None,
        repo: Neo4jRepository | None = None,
        extractor: Any | None = None,
    ) -> None:
        self.settings = settings or Settings.from_environment()
        self.repo = repo
        if self.repo is None and self.settings.graphrag_enabled and self.settings.neo4j_uri:
            try:
                self.repo = Neo4jRepository.from_settings(self.settings)
            except Exception as exc:
                logger.warning("Could not initialize Neo4jRepository: %s", exc)
                self.repo = None

        self.extractor = extractor
        if self.extractor is None and self.settings.graphrag_enabled and self.repo is not None:
            try:
                from app.services.llm import get_llm_provider

                llm = get_llm_provider(self.settings)
                self.extractor = GraphExtractionAdapter(llm_provider=llm)
            except Exception as exc:
                logger.warning("Could not initialize GraphExtractionAdapter: %s", exc)
                self.extractor = None

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

        if self.repo is None:
            fail_graph_event(
                db,
                event.id,
                worker_id,
                error_code="NEO4J_UNAVAILABLE",
                error_message="Graph repository is not configured.",
                is_transient=True,
            )
            return
        try:
            repository_ready = bool(self.repo.verify_connectivity())
        except Exception:
            repository_ready = False
        if not repository_ready:
            fail_graph_event(
                db,
                event.id,
                worker_id,
                error_code="NEO4J_UNAVAILABLE",
                error_message="Graph repository connectivity check failed.",
                is_transient=True,
            )
            return

        try:
            if event.action == "DELETE":
                if not lock_graph_publication(db, event.id, worker_id):
                    return
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

                project_id = UUID(str(event.project_id))
                paper_id = UUID(str(event.paper_id))

                # Retrieve existing snapshots
                snapshots = get_verified_fact_snapshots(
                    db=db,
                    paper_id=paper_id,
                    generation_id=event.generation_id,
                )

                if not snapshots:
                    evidence_items = select_extraction_inputs(
                        db=db,
                        project_id=project_id,
                        paper_id=paper_id,
                        max_chunks=self.settings.graph_batch_limit,
                    )
                    if not evidence_items:
                        # An empty re-index is still a new authoritative
                        # generation. Fence it against newer enqueues and
                        # retire any facts published by an older generation
                        # before acknowledging the empty result.
                        if not lock_graph_publication(db, event.id, worker_id):
                            return
                        self.repo.retire_older_generations(
                            project_id,
                            paper_id,
                            event.generation_id,
                        )
                        complete_graph_event(db, event.id, worker_id=worker_id)
                        logger.info(
                            "Completed UPSERT graph event %s (no evidence items to extract)",
                            event.id,
                        )
                        return

                    if self.extractor is None:
                        raise RuntimeError("Graph extractor is not configured")

                    raw_result = await self.extractor.extract(
                        project_id,
                        paper_id,
                        evidence_items,
                    )
                    if isinstance(raw_result, ExtractionResult):
                        extraction_result = raw_result
                    elif isinstance(raw_result, str):
                        extraction_result = parse_and_validate_extraction(
                            raw_result,
                            project_id,
                            paper_id,
                            evidence_items,
                        )
                    elif isinstance(raw_result, (dict, list)):
                        extraction_result = parse_and_validate_extraction(
                            json.dumps(raw_result),
                            project_id,
                            paper_id,
                            evidence_items,
                        )
                    else:
                        extraction_result = parse_and_validate_extraction(
                            str(raw_result),
                            project_id,
                            paper_id,
                            evidence_items,
                        )

                    valid_candidates = []
                    for candidate in extraction_result.accepted_facts:
                        is_valid, _ = verify_candidate_fact(db, project_id, candidate)
                        if is_valid:
                            valid_candidates.append(candidate)

                    snapshots = persist_verified_fact_snapshots(
                        db=db,
                        project_id=project_id,
                        paper_id=paper_id,
                        generation_id=event.generation_id,
                        event_id=event.id,
                        verified_candidates=valid_candidates,
                        ontology_version=event.ontology_version or "1.0.0",
                    )
                    # Commit db transaction so snapshots are durably stored in PostgreSQL
                    # BEFORE touching Neo4j!
                    db.commit()

                # Lock/recheck after extraction and durable snapshot commit, as close
                # as possible to publication; newer enqueues serialize on Paper.
                if not lock_graph_publication(db, event.id, worker_id):
                    return

                # Publish to Neo4j (if self.repo is configured/present)
                if self.repo is not None:
                    if snapshots:
                        nodes_by_key: dict[str, dict[str, Any]] = {}
                        for s in snapshots:
                            s_name = s.subject_name or s.subject_key
                            s_type = s.subject_type or "Concept"
                            if s.subject_key not in nodes_by_key:
                                nodes_by_key[s.subject_key] = {
                                    "key": s.subject_key,
                                    "name": s_name,
                                    "type": s_type,
                                }
                            o_name = s.object_name or s.object_key
                            o_type = s.object_type or "Concept"
                            if s.object_key not in nodes_by_key:
                                nodes_by_key[s.object_key] = {
                                    "key": s.object_key,
                                    "name": o_name,
                                    "type": o_type,
                                }
                        nodes = list(nodes_by_key.values())

                        facts = [
                            {
                                "id": s.fact_id,
                                "fact_id": s.fact_id,
                                "subject_key": s.subject_key,
                                "subject_name": s.subject_name or s.subject_key,
                                "subject_type": s.subject_type or "Concept",
                                "object_key": s.object_key,
                                "object_name": s.object_name or s.object_key,
                                "object_type": s.object_type or "Concept",
                                "predicate": s.predicate,
                                "qualifiers_json": (
                                    json.dumps(s.qualifiers)
                                    if s.qualifiers and not isinstance(s.qualifiers, str)
                                    else (s.qualifiers if isinstance(s.qualifiers, str) else None)
                                ),
                                "char_start": s.char_start,
                                "char_end": s.char_end,
                                "page_number": s.page_number,
                                "exact_quote": s.exact_quote,
                            }
                            for s in snapshots
                        ]

                        self.repo.upsert_nodes(project_id, nodes)
                        self.repo.upsert_facts(
                            project_id,
                            paper_id,
                            event.generation_id,
                            facts,
                        )

                    self.repo.retire_older_generations(
                        project_id,
                        paper_id,
                        event.generation_id,
                    )

                # Checkpoint in PostgreSQL
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
            db.rollback()
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
                event_id=event.id if "event" in locals() and event else event_id,
                worker_id=worker_id,
                error_code=error_code,
                error_message=str(exc),
                is_transient=is_transient,
            )
