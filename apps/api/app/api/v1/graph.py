"""Project-scoped GraphRAG API endpoints for search, exploration, and indexing."""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.config import Settings
from app.crud.project import get_project
from app.db.models import GraphEvent, GraphFactSnapshot, Paper, Project
from app.db.session import get_db
from app.observability.context import get_operation_context
from app.schemas.evidence import AnchorStatus, Citation
from app.schemas.graph import (
    GraphFactDetailResponse,
    GraphIndexRequest,
    GraphIndexResponse,
    GraphNeighborListResponse,
    GraphNeighborResponse,
    GraphNodeListResponse,
    GraphNodeResponse,
    GraphRelationshipResponse,
    GraphStatusResponse,
)
from app.services.graphrag.evidence import resolve_graph_fact_to_evidence
from app.services.graphrag.indexing import enqueue_existing_papers_for_graph
from app.services.graphrag.neo4j_repository import Neo4jRepository

logger = logging.getLogger("myra.api.graph")
router = APIRouter(prefix="/projects/{project_id}/graph", tags=["graph"])


def get_graph_repo() -> Neo4jRepository | None:
    """Dependency provider returning an active Neo4jRepository or None."""
    settings = Settings.from_environment()
    try:
        return Neo4jRepository.from_settings(settings)
    except Exception as exc:
        logger.warning("Could not initialize Neo4jRepository: %s", exc)
        return None


def _check_repo_available(repo: Neo4jRepository | None) -> bool:
    """Check whether Neo4jRepository is configured and connected."""
    if repo is None:
        return False
    try:
        return bool(repo.verify_connectivity())
    except Exception as exc:
        logger.warning("Neo4j connectivity check failed: %s", exc)
        return False


def _get_project_or_404(db: Session, project_id: UUID) -> Project:
    """Validate project existence; raise 404 if absent."""
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    return project


def _format_updated_at(val: Any) -> str | None:
    """Format updated_at date or timestamp to ISO string."""
    if val is None:
        return None
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    return str(val)


def _build_fact_detail_response(
    db: Session,
    project_id: UUID,
    snapshot: GraphFactSnapshot | None,
    fact_dict: dict[str, Any] | None = None,
    citation_index: int = 1,
    evidence_id: str = "G1",
) -> GraphFactDetailResponse:
    """Resolve a single graph fact against PostgreSQL ground truth
    and construct GraphFactDetailResponse.
    """
    target = snapshot if snapshot is not None else fact_dict
    evidence_item, anchor, res_status = resolve_graph_fact_to_evidence(
        db=db,
        project_id=project_id,
        fact=target,
        evidence_id=evidence_id,
    )

    citation: Citation | None = None
    if evidence_item and anchor:
        citation = Citation(
            citation_index=citation_index,
            evidence_id=evidence_id,
            paper_id=evidence_item.paper_id,
            page_number=evidence_item.page_number,
            quote=evidence_item.quote,
            bounding_boxes=evidence_item.bounding_boxes,
            document_sha256=evidence_item.document_sha256,
            parser_version=evidence_item.parser_version,
            anchor_status=res_status,
            anchors=[anchor] if anchor else [],
        )

    if snapshot is not None:
        fid = snapshot.fact_id
        paper_id = snapshot.paper_id
        generation_id = snapshot.generation_id
        predicate = snapshot.predicate
        subject_key = snapshot.subject_key
        subject_name = snapshot.subject_name
        subject_type = snapshot.subject_type
        object_key = snapshot.object_key
        object_name = snapshot.object_name
        object_type = snapshot.object_type
        qualifiers = snapshot.qualifiers
        exact_quote = snapshot.exact_quote
        page_number = snapshot.page_number
        char_start = snapshot.char_start
        char_end = snapshot.char_end
        doc_sha = snapshot.document_sha256
        updated_at = _format_updated_at(snapshot.created_at)
    else:
        assert fact_dict is not None
        fid = str(fact_dict.get("id") or fact_dict.get("fact_id"))
        paper_id = UUID(str(fact_dict["paper_id"]))
        generation_id = str(fact_dict.get("generation_id", ""))
        predicate = str(fact_dict.get("predicate", ""))
        subject_key = str(fact_dict.get("subject_key", ""))
        subject_name = str(fact_dict.get("subject_name", subject_key))
        subject_type = str(fact_dict.get("subject_type", "Concept"))
        object_key = str(fact_dict.get("object_key", ""))
        object_name = str(fact_dict.get("object_name", object_key))
        object_type = str(fact_dict.get("object_type", "Concept"))
        qualifiers = fact_dict.get("qualifiers")
        exact_quote = str(fact_dict.get("exact_quote", ""))
        page_number = int(fact_dict.get("page_number", 1))
        char_start = int(fact_dict.get("char_start", 0))
        char_end = int(fact_dict.get("char_end", 0))
        doc_sha = str(fact_dict.get("document_sha256") or "")
        updated_at = _format_updated_at(fact_dict.get("updated_at"))

    paper_title = None
    if evidence_item and evidence_item.paper_title:
        paper_title = evidence_item.paper_title
    else:
        p_row = db.query(Paper.filename, Paper.document_sha256).filter(Paper.id == paper_id).first()
        if p_row:
            paper_title = p_row.filename
            if not doc_sha:
                doc_sha = p_row.document_sha256 or ""

    return GraphFactDetailResponse(
        id=fid,
        project_id=project_id,
        paper_id=paper_id,
        paper_title=paper_title,
        generation_id=generation_id,
        predicate=predicate,
        subject_key=subject_key,
        subject_name=subject_name,
        subject_type=subject_type,
        object_key=object_key,
        object_name=object_name,
        object_type=object_type,
        qualifiers=qualifiers,
        exact_quote=exact_quote,
        page_number=page_number,
        char_start=char_start,
        char_end=char_end,
        document_sha256=doc_sha,
        updated_at=updated_at,
        citation=citation,
        anchor_status=res_status if (evidence_item and anchor) else AnchorStatus.UNRESOLVED,
    )


@router.get("/status", response_model=GraphStatusResponse)
def get_graph_status(
    project_id: UUID,
    db: Session = Depends(get_db),
    repo: Neo4jRepository | None = Depends(get_graph_repo),
) -> GraphStatusResponse:
    """Retrieve GraphRAG status and event metrics for a project."""
    _get_project_or_404(db, project_id)

    pending_count = (
        db.query(func.count(GraphEvent.id))
        .filter(
            GraphEvent.project_id == project_id,
            GraphEvent.status.in_(["PENDING", "PROCESSING"]),
        )
        .scalar()
        or 0
    )
    completed_count = (
        db.query(func.count(GraphEvent.id))
        .filter(
            GraphEvent.project_id == project_id,
            GraphEvent.status == "COMPLETED",
        )
        .scalar()
        or 0
    )
    failed_count = (
        db.query(func.count(GraphEvent.id))
        .filter(
            GraphEvent.project_id == project_id,
            GraphEvent.status == "FAILED",
        )
        .scalar()
        or 0
    )

    neo4j_available = False
    node_count = 0
    fact_count = 0

    if repo is not None:
        neo4j_available = _check_repo_available(repo)

    if neo4j_available and repo is not None:
        try:
            counts = repo.count_project_elements(project_id)
            node_count = counts.get("nodes", counts.get("node_count", 0))
            fact_count = counts.get("facts", counts.get("fact_count", 0))
        except Exception as exc:
            logger.warning("count_project_elements error: %s", exc)
            neo4j_available = False

    settings = Settings.from_environment()
    return GraphStatusResponse(
        project_id=project_id,
        graphrag_enabled=settings.graphrag_enabled,
        neo4j_available=neo4j_available,
        node_count=node_count,
        fact_count=fact_count,
        pending_events_count=pending_count,
        completed_events_count=completed_count,
        failed_events_count=failed_count,
    )


@router.get("/nodes", response_model=GraphNodeListResponse)
def search_graph_nodes(
    project_id: UUID,
    query: str | None = None,
    entity_type: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    repo: Neo4jRepository | None = Depends(get_graph_repo),
) -> GraphNodeListResponse:
    """Search and paginate project nodes filtered by query and entity type."""
    _get_project_or_404(db, project_id)

    if repo is None or not _check_repo_available(repo):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph service unavailable",
        )

    nodes = repo.search_nodes(
        project_id=project_id,
        query=query,
        entity_type=entity_type,
        limit=limit,
        skip=skip,
    )

    items = [
        GraphNodeResponse(
            key=n["key"],
            project_id=UUID(str(n["project_id"])),
            name=n["name"],
            type=n["type"],
            description=n.get("description"),
            aliases=list(n.get("aliases") or []),
            updated_at=_format_updated_at(n.get("updated_at")),
        )
        for n in nodes
    ]

    try:
        total = repo.count_matching_nodes(
            project_id=project_id,
            query=query,
            entity_type=entity_type,
        )
    except Exception as exc:
        logger.warning("count_matching_nodes error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph service unavailable",
        ) from exc

    return GraphNodeListResponse(
        items=items,
        total=total,
        limit=limit,
        skip=skip,
    )


@router.get("/nodes/{node_key}", response_model=GraphNodeResponse)
def get_graph_node(
    project_id: UUID,
    node_key: str,
    db: Session = Depends(get_db),
    repo: Neo4jRepository | None = Depends(get_graph_repo),
) -> GraphNodeResponse:
    """Retrieve a single entity node by key within a project."""
    _get_project_or_404(db, project_id)

    if repo is None or not _check_repo_available(repo):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph service unavailable",
        )

    node = repo.get_node_by_key(project_id, node_key)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found in this project",
        )

    return GraphNodeResponse(
        key=node["key"],
        project_id=UUID(str(node["project_id"])),
        name=node["name"],
        type=node["type"],
        description=node.get("description"),
        aliases=list(node.get("aliases") or []),
        updated_at=_format_updated_at(node.get("updated_at")),
    )


@router.get("/nodes/{node_key}/neighbors", response_model=GraphNeighborListResponse)
def get_graph_node_neighbors(
    project_id: UUID,
    node_key: str,
    direction: str = Query(default="BOTH"),
    predicate: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    db: Session = Depends(get_db),
    repo: Neo4jRepository | None = Depends(get_graph_repo),
) -> GraphNeighborListResponse:
    """Retrieve 1-hop neighbors of an entity node within a project."""
    _get_project_or_404(db, project_id)

    if repo is None or not _check_repo_available(repo):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph service unavailable",
        )

    node = repo.get_node_by_key(project_id, node_key)
    if node is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Node not found in this project",
        )

    norm_dir = direction.strip().upper()
    if norm_dir not in {"OUTGOING", "INCOMING", "BOTH"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid direction '{direction}'. Must be OUTGOING, INCOMING, or BOTH.",
        )

    neighbors = repo.get_node_neighbors(
        project_id=project_id,
        key=node_key,
        direction=norm_dir,
        predicate=predicate,
        limit=limit,
    )

    items = [
        GraphNeighborResponse(
            neighbor_key=item["neighbor_key"],
            neighbor_name=item["neighbor_name"],
            neighbor_type=item["neighbor_type"],
            direction=item["direction"],
            predicate=item["predicate"],
            fact_id=item["fact_id"],
            qualifiers=item.get("qualifiers"),
        )
        for item in neighbors
    ]

    return GraphNeighborListResponse(
        node_key=node_key,
        neighbors=items,
        total=len(items),
    )


@router.get("/facts/{fact_id}", response_model=GraphFactDetailResponse)
def get_graph_fact_detail(
    project_id: UUID,
    fact_id: str,
    db: Session = Depends(get_db),
    repo: Neo4jRepository | None = Depends(get_graph_repo),
) -> GraphFactDetailResponse:
    """Retrieve a fact with durable PostgreSQL snapshot and re-resolved Citation anchor."""
    _get_project_or_404(db, project_id)

    snapshot = (
        db.query(GraphFactSnapshot)
        .filter(
            GraphFactSnapshot.fact_id == fact_id,
            GraphFactSnapshot.project_id == project_id,
        )
        .first()
    )

    fact_dict = None
    if snapshot is None:
        if _check_repo_available(repo):
            assert repo is not None
            fact_dict = repo.get_fact_by_id(project_id, fact_id)

        if fact_dict is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Fact not found in this project",
            )

    return _build_fact_detail_response(
        db=db,
        project_id=project_id,
        snapshot=snapshot,
        fact_dict=fact_dict,
        citation_index=1,
        evidence_id="G1",
    )


@router.get("/relationships", response_model=GraphRelationshipResponse)
def get_graph_relationships(
    project_id: UUID,
    subject_key: str,
    object_key: str,
    predicate: str | None = None,
    db: Session = Depends(get_db),
    repo: Neo4jRepository | None = Depends(get_graph_repo),
) -> GraphRelationshipResponse:
    """Find relationships between subject and object, resolving against PostgreSQL ground truth."""
    _get_project_or_404(db, project_id)

    if repo is None or not _check_repo_available(repo):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Graph service unavailable",
        )

    rels = repo.find_relationships_between(
        project_id=project_id,
        subject_key=subject_key,
        object_key=object_key,
        predicate=predicate,
    )

    items: list[GraphFactDetailResponse] = []
    for idx, rel in enumerate(rels, start=1):
        fid = str(rel.get("id") or rel.get("fact_id"))
        snapshot = (
            db.query(GraphFactSnapshot)
            .filter(
                GraphFactSnapshot.fact_id == fid,
                GraphFactSnapshot.project_id == project_id,
            )
            .first()
        )
        detail = _build_fact_detail_response(
            db=db,
            project_id=project_id,
            snapshot=snapshot,
            fact_dict=rel,
            citation_index=idx,
            evidence_id=f"G{idx}",
        )
        items.append(detail)

    return GraphRelationshipResponse(
        items=items,
        total=len(items),
    )


@router.post("/index", response_model=GraphIndexResponse)
def trigger_graph_index(
    project_id: UUID,
    req: GraphIndexRequest,
    db: Session = Depends(get_db),
    repo: Neo4jRepository | None = Depends(get_graph_repo),
) -> GraphIndexResponse:
    """Enqueue existing READY papers for GraphRAG extraction."""
    _get_project_or_404(db, project_id)

    settings = Settings.from_environment()
    if not req.dry_run:
        if not settings.graphrag_enabled:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Graph indexing is disabled by configuration.",
            )
        if repo is None or not _check_repo_available(repo):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Graph indexing is unavailable because Neo4j is not connected.",
            )

    result = enqueue_existing_papers_for_graph(
        db=db,
        project_id=project_id,
        paper_ids=req.paper_ids,
        limit=req.limit,
        dry_run=req.dry_run,
        trace_context=get_operation_context(),
    )

    return GraphIndexResponse(
        dry_run=result["dry_run"],
        eligible_paper_ids=result["eligible_paper_ids"],
        enqueued_count=result["enqueued_count"],
        skipped_count=result["skipped_count"],
        target_project_id=project_id,
    )
