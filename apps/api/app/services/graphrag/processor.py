"""Graph event processor executing durability, lifecycle, and Neo4j publication.

Decoupled from paper ingestion; maintains strict failure isolation where
graph failures NEVER modify or fail the authoritative Paper record.
"""

import hashlib
import json
import logging
import re
from collections import Counter
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
from app.observability.telemetry import get_telemetry
from app.services.cache import get_cache
from app.services.graphrag.extractor import (
    SYSTEM_PROMPT,
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
GRAPH_EXTRACTION_CACHE_TTL_SECONDS = 24 * 60 * 60
GRAPH_EXTRACTION_SELECTOR_VERSION = "bounded-child-selector-v1"
GRAPH_EXTRACTION_PROMPT_FORMAT_VERSION = "evidence-prompt-v1"


def _content_hash(value: str | None) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _extractor_cache_identity(extractor: Any) -> dict[str, str] | None:
    """Return a stable configured identity, or disable caching for unknown providers."""
    if extractor is None:
        return None

    extractor_type = type(extractor)
    identity = {"extractor_class": f"{extractor_type.__module__}.{extractor_type.__qualname__}"}
    if isinstance(extractor, GraphExtractionAdapter):
        provider = extractor.llm_provider
        provider_type = type(provider)
        provider_name = getattr(provider, "provider_name", None)
        model_name = getattr(provider, "model_name", None)
        explicit_revision = getattr(provider, "model_revision", None) or getattr(
            provider, "cache_version", None
        )
        if not isinstance(provider_name, str) or not isinstance(model_name, str):
            if not isinstance(explicit_revision, str) or not explicit_revision:
                return None
            model_name = explicit_revision
            provider_name = f"{provider_type.__module__}.{provider_type.__qualname__}"
        identity.update(
            provider=provider_name,
            provider_class=f"{provider_type.__module__}.{provider_type.__qualname__}",
            model=model_name,
        )
        if isinstance(explicit_revision, str) and explicit_revision:
            identity["model_revision"] = explicit_revision
        return identity

    configured_version = getattr(extractor, "cache_version", None) or getattr(
        extractor, "version", None
    )
    if not isinstance(configured_version, str) or not configured_version.strip():
        return None
    identity["configured_version"] = configured_version.strip()
    return identity


def _extraction_cache_key(
    *,
    project_id: UUID,
    paper_id: UUID,
    paper_sha256: str,
    evidence_items: list[Any],
    chunk_limit: int,
    ontology_version: str,
    extractor_version: str,
    extractor: Any,
) -> str | None:
    """Build a content-addressed key; never include source text or credentials."""
    model_identity = _extractor_cache_identity(extractor)
    if model_identity is None:
        return None
    evidence_identity = []
    for item in evidence_items:
        evidence_identity.append(
            {
                "evidence_id": item.evidence_id,
                "chunk_id": str(item.chunk_id),
                "chunk_index": item.chunk_index,
                "page_number": item.page_number,
                "element_id": str(item.element_id) if item.element_id else None,
                "document_sha256": item.document_sha256,
                "text_sha256": _content_hash(item.text),
                "source_pages": [
                    {
                        "page_number": page.page_number,
                        "text_sha256": _content_hash(page.raw_text),
                    }
                    for page in item.source_pages
                ],
                "source_elements": [
                    {
                        "element_id": str(element.element_id),
                        "page_number": element.page_number,
                        "parser_version": element.parser_version,
                        "text_sha256": _content_hash(element.text),
                    }
                    for element in item.source_elements
                ],
            }
        )
    payload = {
        "version": 1,
        "project_id": str(project_id),
        "paper_id": str(paper_id),
        "paper_sha256": paper_sha256,
        "evidence": evidence_identity,
        "selector_version": GRAPH_EXTRACTION_SELECTOR_VERSION,
        "chunk_limit": chunk_limit,
        "ontology_version": ontology_version,
        "extractor_version": extractor_version,
        "prompt_format_version": GRAPH_EXTRACTION_PROMPT_FORMAT_VERSION,
        "system_prompt_sha256": _content_hash(SYSTEM_PROMPT),
        "model": model_identity,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"graph-extraction:v1:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def _validate_cached_extraction(value: object) -> ExtractionResult:
    if not isinstance(value, dict):
        raise ValueError("cached graph extraction must be an object")
    return ExtractionResult.model_validate(value)


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
                logger.warning(
                    "Could not initialize Neo4jRepository",
                    extra={"error_code": type(exc).__name__},
                )
                self.repo = None

        self.extractor = extractor
        if self.extractor is None and self.settings.graphrag_enabled and self.repo is not None:
            try:
                from app.services.llm import get_llm_provider

                llm = get_llm_provider(self.settings)
                self.extractor = GraphExtractionAdapter(llm_provider=llm)
            except Exception as exc:
                logger.warning(
                    "Could not initialize GraphExtractionAdapter",
                    extra={"error_code": type(exc).__name__},
                )
                self.extractor = None

    async def process_graph_event(
        self,
        db: Session,
        event_id: UUID | str,
        worker_id: str,
    ) -> bool:
        """Execute processing for a single claimed GraphEvent.

        Guarantees that Paper records are NEVER modified or marked failed.

        Returns True only after this invocation durably acknowledges completion;
        every stale, failed, unavailable, or retry path returns False.
        """
        if isinstance(event_id, str):
            event_id = UUID(event_id)

        telemetry = get_telemetry()

        event = get_graph_event(db, event_id)
        if not event or event.status != "PROCESSING" or event.lease_owner != worker_id:
            logger.warning(
                "Lease verification failed for event %s (worker: %s)",
                event_id,
                worker_id,
            )
            return False

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
                return False

        if self.repo is None:
            telemetry.event(
                "graph.event.unavailable",
                metadata={"stage": "repository", "outcome": "unavailable"},
            )
            fail_graph_event(
                db,
                event.id,
                worker_id,
                error_code="NEO4J_UNAVAILABLE",
                error_message="Graph repository is not configured.",
                is_transient=True,
            )
            return False
        try:
            repository_ready = bool(self.repo.verify_connectivity())
        except Exception:
            repository_ready = False
        if not repository_ready:
            telemetry.event(
                "graph.event.unavailable",
                metadata={"stage": "connectivity", "outcome": "unavailable"},
            )
            fail_graph_event(
                db,
                event.id,
                worker_id,
                error_code="NEO4J_UNAVAILABLE",
                error_message="Graph repository connectivity check failed.",
                is_transient=True,
            )
            return False

        try:
            if event.action == "DELETE":
                if not lock_graph_publication(db, event.id, worker_id):
                    return False
                if self.repo is not None:
                    try:
                        self.repo.delete_paper_facts(
                            project_id=event.project_id,
                            paper_id=event.paper_id,
                        )
                    except Exception as repo_err:
                        logger.error(
                            "Failed to delete paper facts in Neo4j",
                            extra={
                                "event_id": str(event_id),
                                "error_code": type(repo_err).__name__,
                            },
                        )
                        fail_graph_event(
                            db=db,
                            event_id=event.id,
                            worker_id=worker_id,
                            error_code="NEO4J_DELETE_FAILED",
                            error_message=str(repo_err),
                            is_transient=True,
                        )
                        return False
                complete_graph_event(db, event.id, worker_id=worker_id)
                telemetry.event(
                    "graph.event.completed",
                    metadata={"action": "delete", "outcome": "success"},
                )
                logger.info("Completed DELETE graph event %s", event.id)
                return True

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
                    return False

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
                    return False

                project_id = UUID(str(event.project_id))
                paper_id = UUID(str(event.paper_id))

                # Retrieve existing snapshots
                snapshots = get_verified_fact_snapshots(
                    db=db,
                    paper_id=paper_id,
                    generation_id=event.generation_id,
                )

                if not snapshots:
                    with telemetry.stage(
                        "graph.input_selection",
                        metadata={"configured_chunk_limit": self.settings.graph_batch_limit},
                    ) as selection_span:
                        evidence_items = select_extraction_inputs(
                            db=db,
                            project_id=project_id,
                            paper_id=paper_id,
                            max_chunks=self.settings.graph_batch_limit,
                        )
                        if selection_span is not None:
                            selection_span.update(
                                metadata={
                                    "configured_chunk_limit": self.settings.graph_batch_limit,
                                    "selected_chunk_count": len(evidence_items),
                                    "outcome": "selected" if evidence_items else "empty",
                                }
                            )
                    if not evidence_items:
                        # An empty re-index is still a new authoritative
                        # generation. Fence it against newer enqueues and
                        # retire any facts published by an older generation
                        # before acknowledging the empty result.
                        if not lock_graph_publication(db, event.id, worker_id):
                            return False
                        self.repo.retire_older_generations(
                            project_id,
                            paper_id,
                            event.generation_id,
                        )
                        complete_graph_event(db, event.id, worker_id=worker_id)
                        telemetry.event(
                            "graph.event.completed",
                            metadata={"action": "upsert", "outcome": "empty"},
                        )
                        logger.info(
                            "Completed UPSERT graph event %s (no evidence items to extract)",
                            event.id,
                        )
                        return True

                    if self.extractor is None:
                        raise RuntimeError("Graph extractor is not configured")

                    cache_key = _extraction_cache_key(
                        project_id=project_id,
                        paper_id=paper_id,
                        paper_sha256=paper.document_sha256 or "",
                        evidence_items=evidence_items,
                        chunk_limit=self.settings.graph_batch_limit,
                        ontology_version=event.ontology_version or "1.0.0",
                        extractor_version=event.extractor_version or "1.0.0",
                        extractor=self.extractor,
                    )
                    cache = get_cache()
                    extraction_result = None
                    if cache_key is not None:
                        try:
                            extraction_result = cache.get(cache_key, _validate_cached_extraction)
                        except Exception as exc:
                            logger.info(
                                "Graph extraction cache read failed; recomputing",
                                extra={"error_code": type(exc).__name__},
                            )

                    with telemetry.stage(
                        "graph.extraction",
                        metadata={
                            "selected_chunk_count": len(evidence_items),
                            "cache_outcome": "hit" if extraction_result is not None else "miss",
                        },
                    ) as extraction_span:
                        if extraction_result is None:
                            raw_result = await self.extractor.extract(
                                project_id,
                                paper_id,
                                evidence_items,
                            )
                            if isinstance(raw_result, ExtractionResult):
                                extraction_result = ExtractionResult.model_validate(
                                    raw_result.model_dump(mode="json")
                                )
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
                            # Rejected/malformed model output is deliberately not
                            # cached: a later attempt may produce a valid result.
                            if cache_key is not None and extraction_result.rejected_count == 0:
                                try:
                                    cache.set(
                                        cache_key,
                                        extraction_result.model_dump(mode="json"),
                                        ttl_seconds=GRAPH_EXTRACTION_CACHE_TTL_SECONDS,
                                    )
                                except Exception as exc:
                                    logger.info(
                                        "Graph extraction cache write failed; continuing",
                                        extra={"error_code": type(exc).__name__},
                                    )

                    valid_candidates = []
                    verification_rejections: Counter[str] = Counter()
                    with telemetry.stage(
                        "graph.schema_provenance_verification",
                        metadata={
                            "schema_rejected_count": extraction_result.rejected_count,
                            "candidate_fact_count": len(extraction_result.accepted_facts),
                        },
                    ) as verification_span:
                        for candidate in extraction_result.accepted_facts:
                            is_valid, rejection_reason = verify_candidate_fact(
                                db, project_id, candidate
                            )
                            if is_valid:
                                valid_candidates.append(candidate)
                            else:
                                verification_rejections[rejection_reason or "UNKNOWN"] += 1

                    if verification_rejections:
                        logger.info(
                            "Graph candidate verification for paper %s: extracted=%d, "
                            "verified=%d, rejected=%d, reasons=%s",
                            paper_id,
                            len(extraction_result.accepted_facts),
                            len(valid_candidates),
                            sum(verification_rejections.values()),
                            dict(sorted(verification_rejections.items())),
                        )

                    if extraction_result.rejected_count:
                        safe_extraction_rejections = {
                            reason: count
                            for reason, count in extraction_result.rejection_reasons.items()
                            if re.fullmatch(r"[A-Z0-9_]{1,64}", reason)
                            and isinstance(count, int)
                            and count > 0
                        }
                        logger.info(
                            "Graph extraction validation for paper %s: rejected=%d, reasons=%s",
                            paper_id,
                            extraction_result.rejected_count,
                            dict(sorted(safe_extraction_rejections.items())),
                        )

                    aggregate_rejections = Counter(verification_rejections)
                    aggregate_rejections.update(
                        {
                            reason: count
                            for reason, count in extraction_result.rejection_reasons.items()
                            if re.fullmatch(r"[A-Z0-9_]{1,64}", reason)
                            and isinstance(count, int)
                            and count > 0
                        }
                    )
                    if verification_span is not None:
                        verification_span.update(
                            metadata={
                                "candidate_fact_count": len(extraction_result.accepted_facts),
                                "verified_fact_count": len(valid_candidates),
                                "schema_rejected_count": extraction_result.rejected_count,
                                "provenance_rejected_count": sum(verification_rejections.values()),
                                "rejection_counts": dict(sorted(aggregate_rejections.items())),
                                "outcome": (
                                    "all_rejected"
                                    if extraction_result.rejected_count and not valid_candidates
                                    else "verified"
                                    if valid_candidates
                                    else "empty"
                                ),
                            }
                        )

                    if not valid_candidates and (
                        extraction_result.accepted_facts or extraction_result.rejected_count
                    ):
                        # Do not turn failed source/claim verification into a
                        # successful empty generation. Preserve the currently
                        # published generation for inspection/retry after a fix.
                        fail_graph_event(
                            db=db,
                            event_id=event.id,
                            worker_id=worker_id,
                            error_code="NO_VERIFIED_FACTS",
                            error_message=(
                                f"{len(extraction_result.accepted_facts)} extracted facts "
                                "failed source or claim verification."
                            ),
                            is_transient=False,
                        )
                        telemetry.event(
                            "graph.extraction.all_rejected",
                            metadata={
                                "candidate_fact_count": len(extraction_result.accepted_facts),
                                "schema_rejected_count": extraction_result.rejected_count,
                                "provenance_rejected_count": sum(verification_rejections.values()),
                                "outcome": "all_rejected",
                            },
                        )
                        logger.warning(
                            "Graph event %s failed: no extracted facts passed verification",
                            event.id,
                        )
                        return False

                    with telemetry.stage(
                        "graph.sql_snapshot_publication",
                        metadata={"verified_fact_count": len(valid_candidates)},
                    ) as sql_span:
                        snapshots = persist_verified_fact_snapshots(
                            db=db,
                            project_id=project_id,
                            paper_id=paper_id,
                            generation_id=event.generation_id,
                            event_id=event.id,
                            verified_candidates=valid_candidates,
                            ontology_version=event.ontology_version or "1.0.0",
                        )
                        # Commit DB snapshots durably before publishing to Neo4j.
                        db.commit()
                        if sql_span is not None:
                            sql_span.update(
                                metadata={
                                    "snapshot_count": len(snapshots),
                                    "outcome": "published",
                                }
                            )

                # Lock/recheck after extraction and durable snapshot commit, as close
                # as possible to publication; newer enqueues serialize on Paper.
                if not lock_graph_publication(db, event.id, worker_id):
                    return False

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

                        with telemetry.stage(
                            "graph.neo4j_publication",
                            metadata={
                                "node_count": len(nodes),
                                "fact_count": len(facts),
                            },
                        ) as neo4j_span:
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
                            if neo4j_span is not None:
                                neo4j_span.update(metadata={"outcome": "published"})

                    else:
                        with telemetry.stage(
                            "graph.neo4j_publication",
                            metadata={"node_count": 0, "fact_count": 0},
                        ) as neo4j_span:
                            self.repo.retire_older_generations(
                                project_id,
                                paper_id,
                                event.generation_id,
                            )
                            if neo4j_span is not None:
                                neo4j_span.update(metadata={"outcome": "published_empty"})

                # Checkpoint in PostgreSQL
                complete_graph_event(db, event.id, worker_id=worker_id)
                telemetry.event(
                    "graph.event.completed",
                    metadata={
                        "action": "upsert",
                        "snapshot_count": len(snapshots),
                        "outcome": "success",
                    },
                )
                logger.info("Completed UPSERT graph event %s", event.id)
                return True

            else:
                fail_graph_event(
                    db=db,
                    event_id=event.id,
                    worker_id=worker_id,
                    error_code="UNKNOWN_ACTION",
                    error_message=f"Unknown graph event action: {event.action}",
                    is_transient=False,
                )
                return False

        except LostGraphLeaseError:
            logger.warning("Worker %s lost lease for graph event %s", worker_id, event_id)
            return False
        except Exception as exc:
            db.rollback()
            logger.error(
                "Graph event processing failed",
                extra={"event_id": str(event_id), "error_code": type(exc).__name__},
            )
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

            telemetry.event(
                "graph.event.retry_or_failure",
                metadata={
                    "error_code": error_code,
                    "outcome": "retry_scheduled" if is_transient else "failed",
                },
            )

            fail_graph_event(
                db=db,
                event_id=event.id if "event" in locals() and event else event_id,
                worker_id=worker_id,
                error_code=error_code,
                error_message=str(exc),
                is_transient=is_transient,
            )
            return False
