import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.crud.memory import (
    MemoryVersionConflictError,
    delete_memory,
    get_memory,
    list_memories,
    supersede_memory,
    update_memory,
)
from app.crud.project import get_project
from app.db.session import get_db
from app.schemas.memory import (
    MemoryCreate,
    MemoryListResponse,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    MemoryUpdate,
)
from app.services.memory_service import (
    capture_conversation_memories,
    consolidate_memory_candidate,
)

logger = logging.getLogger("myra.api.memory")
router = APIRouter(prefix="/projects/{project_id}/memories", tags=["memory"])


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
        items=[MemoryResponse.model_validate(m, from_attributes=True) for m in items],
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

    return MemoryResponse.model_validate(mem, from_attributes=True)


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
    return MemoryResponse.model_validate(mem, from_attributes=True)


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

    return MemoryResponse.model_validate(updated, from_attributes=True)


@router.post("/{memory_id}/supersede", response_model=MemoryResponse)
def supersede_project_memory(
    project_id: UUID,
    memory_id: UUID,
    new_memory_in: MemoryCreate,
    db: Session = Depends(get_db),
) -> MemoryResponse:
    old_mem = get_memory(db, memory_id=memory_id, project_id=project_id)
    if not old_mem:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Memory {memory_id} not found in project {project_id}",
        )

    try:
        new_mem = consolidate_memory_candidate(db, project_id=project_id, candidate=new_memory_in)
        supersede_memory(
            db,
            old_memory=old_mem,
            new_memory=new_mem,
            reason=f"Explicitly superseded by new decision: {new_memory_in.title}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    return MemoryResponse.model_validate(new_mem, from_attributes=True)


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

    memories = capture_conversation_memories(
        db=db, project_id=project_id, conversation_id=body.conversation_id
    )
    return [MemoryResponse.model_validate(m, from_attributes=True) for m in memories]
