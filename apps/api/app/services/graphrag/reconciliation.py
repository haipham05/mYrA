"""Graph reconciliation and rebuild services for GraphRAG.

Manages drift detection, orphaned graph fact cleanup, generation retirement,
and deterministic rebuilds strictly from PostgreSQL validated fact snapshots.
Graph failures or rebuilds NEVER modify or fail authoritative Paper records,
and NEVER touch PDF storage or read Neo4j as an authority.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.crud.graph import get_latest_completed_graph_event
from app.db.models import GraphFactSnapshot, Paper
from app.services.graphrag.neo4j_repository import Neo4jRepository

logger = logging.getLogger("myra.graphrag.reconciliation")


def validate_snapshot_provenance(
    snapshot: GraphFactSnapshot,
    paper_sha256: str | None = None,
) -> bool:
    """Validate that source provenance fields on a GraphFactSnapshot are well-formed and current.

    Ensures:
    - page_number >= 1
    - 0 <= char_start <= char_end
    - exact_quote is a non-empty string
    - document_sha256 is a non-empty string
    - if paper_sha256 is present, snapshot.document_sha256 matches paper_sha256
    - subject_key, object_key, and predicate are non-empty strings
    """
    if snapshot.page_number is None or snapshot.page_number < 1:
        return False
    if snapshot.char_start is None or snapshot.char_end is None:
        return False
    if snapshot.char_start < 0 or snapshot.char_start >= snapshot.char_end:
        return False
    if not snapshot.exact_quote or not snapshot.exact_quote.strip():
        return False
    if not snapshot.document_sha256 or not snapshot.document_sha256.strip():
        return False
    if paper_sha256 and snapshot.document_sha256.strip().lower() != paper_sha256.strip().lower():
        return False
    if not snapshot.subject_key or not snapshot.object_key or not snapshot.predicate:
        return False
    return True


def _query_distinct_paper_ids(
    repo: Neo4jRepository,
    project_id: UUID,
    limit: int = 50,
) -> list[str]:
    """Query Neo4j for paper IDs present in facts for a given project."""
    safe_limit = max(1, min(limit, 100))
    pid_str = str(project_id)
    cypher = """
    MATCH (f:Fact {project_id: $project_id})
    RETURN DISTINCT f.paper_id AS paper_id
    LIMIT $limit
    """
    params = {"project_id": pid_str, "limit": safe_limit}

    if hasattr(repo, "get_distinct_paper_ids") and callable(
        getattr(repo, "get_distinct_paper_ids")
    ):
        return repo.get_distinct_paper_ids(project_id=project_id, limit=safe_limit)

    def _tx_work(tx) -> list[str]:
        result = tx.run(cypher, params)
        return [str(record["paper_id"]) for record in result if record.get("paper_id")]

    if hasattr(repo, "_get_session"):
        with repo._get_session() as session:
            return session.execute_read(_tx_work)
    elif hasattr(repo, "driver"):
        with repo.driver.session(database=getattr(repo, "database", "neo4j")) as session:
            return session.execute_read(_tx_work)
    return []


def reconcile_graph_drift(
    db: Session,
    repo: Neo4jRepository,
    project_id: UUID,
    limit: int = 50,
) -> dict[str, Any]:
    """Reconcile drift between PostgreSQL authority and Neo4j graph projection.

    1. Queries Neo4j for paper IDs present in the graph for this project_id:
       MATCH (f:Fact {project_id: $project_id}) RETURN DISTINCT f.paper_id AS paper_id LIMIT $limit
    2. For each paper_id:
       - Checks PostgreSQL: does Paper exist with id == paper_id and project_id == project_id?
       - If paper is absent from PostgreSQL:
         - Cleans up facts for that paper via repo.delete_paper_facts(project_id, paper_id)
         - Increments orphaned_papers_cleaned
       - If paper exists:
         - Gets the latest completed GraphEvent for that paper.
         - If found, calls repo.retire_older_generations(
           project_id, paper_id, active_generation_id=latest.generation_id
         )
    3. Returns summary dict:
       {"project_id": ..., "scanned_papers": ...,
        "orphaned_papers_cleaned": ..., "retired_facts_count": ...}
    """
    if isinstance(project_id, str):
        project_id = UUID(project_id)

    paper_id_strs = _query_distinct_paper_ids(repo, project_id, limit=limit)
    scanned_papers = len(paper_id_strs)
    orphaned_papers_cleaned = 0
    retired_facts_count = 0

    for raw_id in paper_id_strs:
        try:
            paper_uuid = UUID(str(raw_id))
        except (ValueError, TypeError):
            # Non-UUID paper_id cannot exist in PostgreSQL -> orphan
            try:
                repo.delete_paper_facts(project_id=project_id, paper_id=raw_id)  # type: ignore[arg-type]
            except Exception as exc:
                logger.warning("Failed to delete orphaned paper facts for %s: %s", raw_id, exc)
            orphaned_papers_cleaned += 1
            continue

        paper = (
            db.query(Paper).filter(Paper.id == paper_uuid, Paper.project_id == project_id).first()
        )

        if not paper:
            # Paper is absent from PostgreSQL -> clean up facts
            try:
                repo.delete_paper_facts(project_id=project_id, paper_id=paper_uuid)
            except Exception as exc:
                logger.warning(
                    "Failed to delete orphaned paper facts for %s: %s",
                    paper_uuid,
                    exc,
                )
            orphaned_papers_cleaned += 1
        else:
            # Paper exists -> retire older generations for single active generation
            latest_event = get_latest_completed_graph_event(db, paper_uuid)
            if latest_event and latest_event.generation_id:
                try:
                    cnt = repo.retire_older_generations(
                        project_id=project_id,
                        paper_id=paper_uuid,
                        active_generation_id=latest_event.generation_id,
                    )
                    retired_facts_count += cnt
                except Exception as exc:
                    logger.warning(
                        "Failed to retire older generations for paper %s: %s",
                        paper_uuid,
                        exc,
                    )

    return {
        "project_id": str(project_id),
        "scanned_papers": scanned_papers,
        "orphaned_papers_cleaned": orphaned_papers_cleaned,
        "retired_facts_count": retired_facts_count,
    }


def rebuild_paper_graph_from_snapshots(
    db: Session,
    repo: Neo4jRepository,
    project_id: UUID,
    paper_id: UUID,
    generation_id: str | None = None,
) -> dict[str, Any]:
    """Rebuild Neo4j facts strictly from PostgreSQL graph_fact_snapshots.

    Guarantees:
    - Never reads Neo4j graph as authority
    - Never accesses PDF files or storage
    - Never mutates Paper or PostgreSQL authority rows
    - Validates source provenance fields before publishing
    """
    if isinstance(project_id, str):
        project_id = UUID(project_id)
    if isinstance(paper_id, str):
        paper_id = UUID(paper_id)

    paper = db.query(Paper).filter(Paper.id == paper_id, Paper.project_id == project_id).first()
    if not paper:
        raise ValueError(f"Paper {paper_id} not found in project {project_id}")

    target_generation_id = generation_id
    if target_generation_id is None:
        latest_event = get_latest_completed_graph_event(db, paper_id)
        if latest_event and latest_event.generation_id:
            target_generation_id = latest_event.generation_id
        else:
            latest_snapshot = (
                db.query(GraphFactSnapshot)
                .filter(
                    GraphFactSnapshot.project_id == project_id,
                    GraphFactSnapshot.paper_id == paper_id,
                )
                .order_by(
                    GraphFactSnapshot.created_at.desc(),
                    GraphFactSnapshot.generation_id.desc(),
                    GraphFactSnapshot.fact_id.desc(),
                )
                .first()
            )
            if latest_snapshot:
                target_generation_id = latest_snapshot.generation_id
            else:
                return {
                    "project_id": str(project_id),
                    "paper_id": str(paper_id),
                    "generation_id": None,
                    "rebuilt_facts_count": 0,
                    "scanned_snapshots": 0,
                    "rejected_snapshots": 0,
                    "status": "NO_SNAPSHOTS",
                }

    snapshots = (
        db.query(GraphFactSnapshot)
        .filter(
            GraphFactSnapshot.project_id == project_id,
            GraphFactSnapshot.paper_id == paper_id,
            GraphFactSnapshot.generation_id == target_generation_id,
        )
        .all()
    )

    valid_snapshots: list[GraphFactSnapshot] = []
    rejected_count = 0
    for s in snapshots:
        if validate_snapshot_provenance(s, paper_sha256=paper.document_sha256):
            valid_snapshots.append(s)
        else:
            rejected_count += 1
            logger.warning(
                "Rejected fact snapshot %s for paper %s due to invalid provenance",
                s.fact_id,
                paper_id,
            )

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
        for s in valid_snapshots
    ]

    rebuilt_facts_count = 0
    if fact_payloads:
        rebuilt_facts_count = repo.upsert_facts(
            project_id=project_id,
            paper_id=paper_id,
            generation_id=target_generation_id,
            facts=fact_payloads,
        )
        repo.retire_older_generations(
            project_id=project_id,
            paper_id=paper_id,
            active_generation_id=target_generation_id,
        )

    return {
        "project_id": str(project_id),
        "paper_id": str(paper_id),
        "generation_id": target_generation_id,
        "rebuilt_facts_count": rebuilt_facts_count,
        "scanned_snapshots": len(snapshots),
        "rejected_snapshots": rejected_count,
        "status": "SUCCESS",
    }


def scoped_rebuild_project_graph(
    db: Session,
    repo: Neo4jRepository,
    project_id: UUID | str,
    paper_id: UUID | str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Rebuild a project's knowledge graph deterministically from stored PostgreSQL snapshots.

    Guarantees:
    - Queries papers in project_id (filtered by paper_id if specified).
    - Checks graph_fact_snapshots in PostgreSQL for each paper.
    - If NO snapshots exist: records paper in missing_snapshots_papers with notice:
      "No snapshots retained; requires approved re-extraction".
    - If snapshots exist:
      - Rechecks snapshot validity against PostgreSQL paper row (document_sha256).
        If paper SHA-256 changed or is stale, records stale count and skips stale snapshots.
      - If not dry_run:
        Calls rebuild_paper_graph_from_snapshots(
            db, repo, project_id, paper.id
        ) with valid snapshots.
    - Returns:
      {"project_id": str(project_id), "dry_run": dry_run, "papers_rebuilt": int,
       "facts_published": int, "missing_snapshots_papers": list[str],
       "stale_snapshots_count": int}
    - Invariants:
      - Recovery is deterministic strictly from PostgreSQL snapshots.
      - NEVER reads Neo4j as authority.
      - NEVER mutates PDF storage or paper records in PostgreSQL.
    """
    if isinstance(project_id, str):
        project_id = UUID(project_id)
    if isinstance(paper_id, str):
        paper_id = UUID(paper_id)

    if paper_id is not None:
        paper = db.query(Paper).filter(Paper.id == paper_id, Paper.project_id == project_id).first()
        if not paper:
            raise ValueError(f"Paper {paper_id} not found in project {project_id}")
        papers = [paper]
    else:
        papers = (
            db.query(Paper)
            .filter(Paper.project_id == project_id)
            .order_by(Paper.created_at.asc())
            .all()
        )

    papers_rebuilt = 0
    facts_published = 0
    missing_snapshots_papers: list[str] = []
    stale_snapshots_count = 0

    for p in papers:
        snapshots = (
            db.query(GraphFactSnapshot)
            .filter(
                GraphFactSnapshot.project_id == project_id,
                GraphFactSnapshot.paper_id == p.id,
            )
            .all()
        )

        if not snapshots:
            missing_snapshots_papers.append(str(p.id))
            logger.warning(
                "Paper %s: No snapshots retained; requires approved re-extraction",
                p.id,
            )
            continue

        latest_event = get_latest_completed_graph_event(db, p.id)
        if latest_event and latest_event.generation_id:
            target_generation_id = latest_event.generation_id
        else:
            latest_snapshot = max(snapshots, key=lambda s: s.created_at)
            target_generation_id = latest_snapshot.generation_id

        target_snapshots = [s for s in snapshots if s.generation_id == target_generation_id]
        if not target_snapshots:
            missing_snapshots_papers.append(str(p.id))
            logger.warning(
                "Paper %s: No snapshots retained; requires approved re-extraction",
                p.id,
            )
            continue

        paper_stale_count = 0
        for s in target_snapshots:
            if not validate_snapshot_provenance(s, paper_sha256=p.document_sha256):
                paper_stale_count += 1

        stale_snapshots_count += paper_stale_count

        if not dry_run:
            rebuild_res = rebuild_paper_graph_from_snapshots(
                db=db,
                repo=repo,
                project_id=project_id,
                paper_id=p.id,
                generation_id=target_generation_id,
            )
            pub_count = rebuild_res.get("rebuilt_facts_count", 0)
            facts_published += pub_count
            if pub_count > 0:
                papers_rebuilt += 1

    return {
        "project_id": str(project_id),
        "dry_run": dry_run,
        "papers_rebuilt": papers_rebuilt,
        "facts_published": facts_published,
        "missing_snapshots_papers": missing_snapshots_papers,
        "stale_snapshots_count": stale_snapshots_count,
    }
