"""Graph indexing service for enqueueing existing READY papers into GraphRAG processing."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.crud.graph import create_or_enqueue_graph_event, get_active_graph_events_for_paper
from app.db.models import Paper
from app.observability.context import OperationContext

logger = logging.getLogger("myra.graphrag.indexing")


def enqueue_existing_papers_for_graph(
    db: Session,
    project_id: UUID | str | None = None,
    paper_ids: list[UUID | str] | None = None,
    limit: int = 10,
    dry_run: bool = True,
    trace_context: OperationContext | None = None,
) -> dict[str, Any]:
    """Enqueue existing READY papers for GraphRAG extraction.

    Safety rules:
    - Default dry_run = True.
    - If neither project_id nor paper_ids is provided, do NOT query all papers
      (require explicit opt-in target!). Return empty eligible list with notice.
    - Enqueue currently READY papers only:
      (paper.status == "READY" and paper.document_sha256 is not None).
    - Bounded: limit to min(limit, 50).
    - For each paper:
      - Check existing active events: query graph_events for paper_id == paper.id and
        status in ("PENDING", "PROCESSING"). If active event exists, skip.
      - If dry_run is False:
        Call create_or_enqueue_graph_event(
            db, project_id=paper.project_id, paper_id=paper.id, action="UPSERT"
        ).
    - If not dry_run, db.commit().
    - Returns structured report dictionary containing:
      {"dry_run": dry_run, "eligible_paper_ids": [str(p.id) for p in eligible],
       "enqueued_count": int, "skipped_count": int,
       "target_project_id": str(project_id) if project_id else None}
    - Invariant: NEVER output raw text, quotes, or credentials in results.
    """
    if isinstance(project_id, str):
        project_id = UUID(project_id)

    parsed_paper_ids: list[UUID] | None = None
    if paper_ids is not None:
        parsed_paper_ids = [UUID(str(pid)) for pid in paper_ids]

    # Explicit target check: require opt-in project_id or paper_ids
    if project_id is None and not parsed_paper_ids:
        logger.info("enqueue_existing_papers_for_graph: No target specified, returning empty list.")
        return {
            "dry_run": dry_run,
            "eligible_paper_ids": [],
            "enqueued_count": 0,
            "skipped_count": 0,
            "target_project_id": None,
            "notice": (
                "Explicit opt-in target required. Provide project_id or paper_ids "
                "to enqueue papers for graph indexing."
            ),
        }

    # Bounded: limit to min(limit, 50)
    safe_limit = max(1, min(limit, 50))

    query = db.query(Paper)
    if project_id is not None:
        query = query.filter(Paper.project_id == project_id)
    if parsed_paper_ids is not None:
        query = query.filter(Paper.id.in_(parsed_paper_ids))
    else:
        query = query.filter(Paper.status == "READY", Paper.document_sha256.is_not(None))

    candidate_papers = query.order_by(Paper.created_at.asc()).limit(safe_limit).all()

    eligible: list[Paper] = []
    enqueued_count = 0
    skipped_count = 0

    if parsed_paper_ids is not None:
        found_ids = {p.id for p in candidate_papers}
        missing_count = sum(1 for pid in parsed_paper_ids if pid not in found_ids)
        skipped_count += missing_count

    for paper in candidate_papers:
        # Enqueue currently READY papers with non-null document_sha256 only
        if paper.status != "READY" or not paper.document_sha256:
            skipped_count += 1
            continue

        active_events = get_active_graph_events_for_paper(db, paper.id)
        if active_events:
            skipped_count += 1
            continue

        eligible.append(paper)
        if not dry_run:
            create_or_enqueue_graph_event(
                db=db,
                project_id=paper.project_id,
                paper_id=paper.id,
                action="UPSERT",
                trace_context=trace_context,
            )
            enqueued_count += 1

    if not dry_run and enqueued_count > 0:
        db.commit()

    logger.info(
        "enqueue_existing_papers_for_graph: dry_run=%s, eligible=%d, enqueued=%d, "
        "skipped=%d, target_project=%s",
        dry_run,
        len(eligible),
        enqueued_count,
        skipped_count,
        str(project_id) if project_id else None,
    )

    return {
        "dry_run": dry_run,
        "eligible_paper_ids": [str(p.id) for p in eligible],
        "enqueued_count": enqueued_count,
        "skipped_count": skipped_count,
        "target_project_id": str(project_id) if project_id else None,
    }
