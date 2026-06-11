import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.crud.memory import (
    MemoryVersionConflictError,
    atomic_create_and_supersede,
    delete_memory,
    get_memory,
    list_memories,
    update_memory,
)
from app.crud.project import get_project
from app.db.models import Memory
from app.db.session import get_db
from app.schemas.evidence import AnchorStatus
from app.schemas.memory import (
    MemoryAuditResponse,
    MemoryCreate,
    MemoryListResponse,
    MemoryResponse,
    MemorySourceResponse,
    MemorySourceType,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
)
from app.services.memory_service import (
    capture_conversation_memories,
    consolidate_memory_candidate,
    validate_memory_candidate,
)

logger = logging.getLogger("myra.api.memory")
router = APIRouter(prefix="/projects/{project_id}/memories", tags=["memory"])


def serialize_memory_with_resolved_sources(db: Session, mem: Memory) -> MemoryResponse:
    sources_resp: list[MemorySourceResponse] = []
    for s in mem.sources:
        s_dict: dict[str, Any] = {
            "id": s.id,
            "memory_id": s.memory_id,
            "source_type": s.source_type,
            "message_id": s.message_id,
            "conversation_id": s.conversation_id,
            "paper_id": s.paper_id,
            "page_number": s.page_number,
            "quote_text": s.quote_text,
            "document_sha256": s.document_sha256,
            "created_at": s.created_at,
            "anchor_status": AnchorStatus.UNRESOLVED,
            "bounding_boxes": [],
            "anchors": [],
        }
        if s.source_type == MemorySourceType.PAPER_CHUNK.value and s.paper_id:
            from app.services.memory_service import resolve_paper_memory_source

            ev, anchor, status = resolve_paper_memory_source(db, mem.project_id, s)
            s_dict["anchor_status"] = status
            if ev and anchor and status == AnchorStatus.VERIFIED:
                s_dict["chunk_id"] = ev.chunk_id
                s_dict["source_element_id"] = anchor.source_element_id
                s_dict["source_char_start"] = anchor.source_char_start
                s_dict["source_char_end"] = anchor.source_char_end
                s_dict["parser_version"] = anchor.parser_version
                s_dict["bounding_boxes"] = anchor.bounding_boxes
                s_dict["anchors"] = [anchor]
        sources_resp.append(MemorySourceResponse(**s_dict))

    return MemoryResponse(
        id=mem.id,
        project_id=mem.project_id,
        memory_type=mem.memory_type,
        status=mem.status,
        title=mem.title,
        content=mem.content,
        confidence=mem.confidence,
        importance=mem.importance,
        version=mem.version,
        is_pinned=mem.is_pinned,
        superseded_by_id=mem.superseded_by_id,
        created_at=mem.created_at,
        updated_at=mem.updated_at,
        last_accessed_at=mem.last_accessed_at,
        sources=sources_resp,
        history=[MemoryAuditResponse.model_validate(h, from_attributes=True) for h in mem.history],
    )


class ConsolidateRequest(BaseModel):
    conversation_id: UUID


@router.get("", response_model=MemoryListResponse)
def list_project_memories(
    project_id: UUID,
    memory_type: MemoryType | None = None,
    status_filter: MemoryStatus | None = Query(default=None, alias="status"),
    is_pinned: bool | None = None,
    search: str | None = None,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> MemoryListResponse:
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    items, total = list_memories(
        db=db,
        project_id=project_id,
        memory_type=memory_type,
        status=status_filter,
        is_pinned=is_pinned,
        search=search,
        limit=limit,
        offset=offset,
    )

    return MemoryListResponse(
        items=[serialize_memory_with_resolved_sources(db, m) for m in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
def create_project_memory(
    project_id: UUID,
    memory_in: MemoryCreate,
    db: Session = Depends(get_db),
) -> MemoryResponse:
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    try:
        mem = consolidate_memory_candidate(db, project_id=project_id, candidate=memory_in)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return serialize_memory_with_resolved_sources(db, mem)


@router.get("/{memory_id}", response_model=MemoryResponse)
def get_project_memory(
    project_id: UUID,
    memory_id: UUID,
    db: Session = Depends(get_db),
) -> MemoryResponse:
    mem = get_memory(db, memory_id=memory_id, project_id=project_id)
    if not mem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found in project {project_id}",
        )
    return serialize_memory_with_resolved_sources(db, mem)


@router.patch("/{memory_id}", response_model=MemoryResponse)
def update_project_memory(
    project_id: UUID,
    memory_id: UUID,
    memory_update: MemoryUpdate,
    db: Session = Depends(get_db),
) -> MemoryResponse:
    mem = get_memory(db, memory_id=memory_id, project_id=project_id)
    if not mem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found in project {project_id}",
        )

    try:
        updated = update_memory(db=db, memory=mem, memory_update=memory_update)
    except MemoryVersionConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return serialize_memory_with_resolved_sources(db, updated)


@router.post("/{memory_id}/supersede", response_model=MemoryResponse)
def supersede_project_memory(
    project_id: UUID,
    memory_id: UUID,
    new_memory_in: MemoryCreate,
    expected_version: int = Query(
        ...,
        description=(
            "Current version of the memory being superseded (required for atomic concurrency)"
        ),
    ),
    db: Session = Depends(get_db),
) -> MemoryResponse:
    try:
        validate_memory_candidate(db, project_id=project_id, candidate=new_memory_in)
        embedding = None
        try:
            from app.services.embedding import get_embedding_provider

            provider = get_embedding_provider()
            embedding = provider.embed_query(new_memory_in.content)
        except Exception:
            embedding = None

        _old_mem, new_mem = atomic_create_and_supersede(
            db,
            project_id=project_id,
            old_memory_id=memory_id,
            expected_version=expected_version,
            new_memory_in=new_memory_in,
            embedding=embedding,
            reason=f"Explicitly superseded by new decision: {new_memory_in.title}",
        )
    except MemoryVersionConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e
    except ValueError as e:
        detail = str(e)
        status_code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in detail.lower()
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from e

    return serialize_memory_with_resolved_sources(db, new_mem)


@router.delete("/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project_memory(
    project_id: UUID,
    memory_id: UUID,
    hard_delete: bool = Query(default=False),
    reason: str | None = None,
    db: Session = Depends(get_db),
) -> None:
    mem = get_memory(db, memory_id=memory_id, project_id=project_id)
    if not mem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found in project {project_id}",
        )
    delete_memory(
        db=db,
        memory=mem,
        hard_delete=hard_delete,
        reason=reason or "Deleted via API",
    )


@router.post("/consolidate", response_model=list[MemoryResponse])
def trigger_conversation_consolidation(
    project_id: UUID,
    body: ConsolidateRequest,
    db: Session = Depends(get_db),
) -> list[MemoryResponse]:
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    try:
        memories = capture_conversation_memories(
            db=db, project_id=project_id, conversation_id=body.conversation_id
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    return [serialize_memory_with_resolved_sources(db, m) for m in memories]
