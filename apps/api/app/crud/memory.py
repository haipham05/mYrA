from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import or_
from sqlalchemy.orm import Session, joinedload

from app.db.models import Memory, MemoryAudit, MemorySource
from app.schemas.memory import MemoryCreate, MemoryStatus, MemoryType, MemoryUpdate


class MemoryVersionConflictError(Exception):
    """Raised when an update contains an outdated version number."""

    pass


def create_memory(
    db: Session,
    project_id: UUID,
    memory_in: MemoryCreate,
    embedding: list[float] | None = None,
) -> Memory:
    memory = Memory(
        project_id=project_id,
        memory_type=memory_in.memory_type.value,
        status=MemoryStatus.ACTIVE.value,
        title=memory_in.title,
        content=memory_in.content,
        confidence=memory_in.confidence,
        importance=memory_in.importance,
        version=1,
        is_pinned=memory_in.is_pinned,
        embedding=embedding,
        embedding_vec=embedding,
    )
    db.add(memory)
    db.flush()

    for s in memory_in.sources:
        source_rec = MemorySource(
            memory_id=memory.id,
            source_type=s.source_type.value,
            message_id=s.message_id,
            paper_id=s.paper_id,
            page_number=s.page_number,
            quote_text=s.quote_text,
            document_sha256=s.document_sha256,
        )
        db.add(source_rec)

    audit = MemoryAudit(
        memory_id=memory.id,
        action="CREATED",
        old_content=None,
        new_content=memory.content,
        reason="Initial memory creation",
    )
    db.add(audit)
    db.commit()
    db.refresh(memory)
    return memory


def get_memory(
    db: Session,
    memory_id: UUID,
    project_id: UUID | None = None,
) -> Memory | None:
    query = (
        db.query(Memory)
        .options(joinedload(Memory.sources), joinedload(Memory.history))
        .filter(Memory.id == memory_id)
    )
    if project_id is not None:
        query = query.filter(Memory.project_id == project_id)
    return query.first()


def list_memories(
    db: Session,
    project_id: UUID,
    memory_type: MemoryType | str | None = None,
    status: MemoryStatus | str | None = None,
    is_pinned: bool | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Memory], int]:
    query = (
        db.query(Memory)
        .options(joinedload(Memory.sources), joinedload(Memory.history))
        .filter(Memory.project_id == project_id)
    )

    if memory_type:
        type_val = memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
        query = query.filter(Memory.memory_type == type_val)

    if status:
        status_val = status.value if isinstance(status, MemoryStatus) else str(status)
        query = query.filter(Memory.status == status_val)

    if is_pinned is not None:
        query = query.filter(Memory.is_pinned.is_(is_pinned))

    if search and search.strip():
        term = f"%{search.strip()}%"
        query = query.filter(or_(Memory.title.ilike(term), Memory.content.ilike(term)))

    total = query.distinct().count()
    items = (
        query.order_by(
            Memory.is_pinned.desc(),
            Memory.importance.desc(),
            Memory.created_at.desc(),
        )
        .offset(offset)
        .limit(limit)
        .all()
    )
    return items, total


def update_memory(
    db: Session,
    memory: Memory,
    memory_update: MemoryUpdate,
    reason: str | None = None,
) -> Memory:
    if memory_update.version is None:
        raise MemoryVersionConflictError(
            "Version is required for updating memory to prevent silent overwrites."
        )
    if memory_update.version != memory.version:
        raise MemoryVersionConflictError(
            f"Version conflict: current version is {memory.version}, "
            f"but provided version was {memory_update.version}"
        )

    old_content = memory.content
    actions = []

    if memory_update.title is not None and memory_update.title != memory.title:
        memory.title = memory_update.title
        actions.append("TITLE_UPDATED")

    if memory_update.content is not None and memory_update.content != memory.content:
        memory.content = memory_update.content
        actions.append("CONTENT_UPDATED")

    if memory_update.status is not None:
        status_val = (
            memory_update.status.value
            if isinstance(memory_update.status, MemoryStatus)
            else str(memory_update.status)
        )
        if status_val != memory.status:
            memory.status = status_val
            actions.append(f"STATUS_{status_val}")

    if memory_update.importance is not None and memory_update.importance != memory.importance:
        memory.importance = memory_update.importance
        actions.append("IMPORTANCE_UPDATED")

    if memory_update.confidence is not None and memory_update.confidence != memory.confidence:
        memory.confidence = memory_update.confidence
        actions.append("CONFIDENCE_UPDATED")

    if memory_update.is_pinned is not None and memory_update.is_pinned != memory.is_pinned:
        memory.is_pinned = memory_update.is_pinned
        actions.append("PINNED" if memory.is_pinned else "UNPINNED")

    memory.version += 1
    memory.updated_at = datetime.now(UTC)

    action_label = ", ".join(actions) if actions else "UPDATED"
    audit_reason = memory_update.reason or reason or "Memory updated"

    audit = MemoryAudit(
        memory_id=memory.id,
        action=action_label,
        old_content=old_content,
        new_content=memory.content,
        reason=audit_reason,
    )
    db.add(audit)
    db.commit()
    db.refresh(memory)
    return memory


def supersede_memory(
    db: Session,
    old_memory: Memory,
    new_memory: Memory,
    reason: str = "Superseded by newer decision",
) -> Memory:
    if old_memory.id == new_memory.id:
        raise ValueError("Cannot supersede memory with itself.")
    if old_memory.content.strip().lower() == new_memory.content.strip().lower():
        raise ValueError("Cannot supersede memory with identical content.")
    if old_memory.status != MemoryStatus.ACTIVE.value:
        raise ValueError(
            f"Cannot supersede memory with status '{old_memory.status}'; must be ACTIVE."
        )

    old_content = old_memory.content
    old_memory.status = MemoryStatus.SUPERSEDED.value
    old_memory.superseded_by_id = new_memory.id
    old_memory.version += 1
    old_memory.updated_at = datetime.now(UTC)

    audit_old = MemoryAudit(
        memory_id=old_memory.id,
        action="SUPERSEDED",
        old_content=old_content,
        new_content=new_memory.content,
        reason=f"{reason} (new memory ID: {new_memory.id})",
    )
    db.add(audit_old)

    audit_new = MemoryAudit(
        memory_id=new_memory.id,
        action="SUPERSEDED_PRIOR",
        old_content=None,
        new_content=new_memory.content,
        reason=f"Supersedes prior memory ID: {old_memory.id}",
    )
    db.add(audit_new)

    db.commit()
    db.refresh(old_memory)
    return old_memory


def delete_memory(
    db: Session,
    memory: Memory,
    hard_delete: bool = False,
    reason: str = "Memory archived/deleted",
) -> None:
    if hard_delete:
        db.delete(memory)
        db.commit()
    else:
        old_content = memory.content
        memory.status = MemoryStatus.ARCHIVED.value
        memory.version += 1
        memory.updated_at = datetime.now(UTC)

        audit = MemoryAudit(
            memory_id=memory.id,
            action="ARCHIVED",
            old_content=old_content,
            new_content=None,
            reason=reason,
        )
        db.add(audit)
        db.commit()


def record_memory_access(db: Session, memory_ids: list[UUID]) -> None:
    if not memory_ids:
        return
    now = datetime.now(UTC)
    db.query(Memory).filter(Memory.id.in_(memory_ids)).update(
        {Memory.last_accessed_at: now}, synchronize_session=False
    )
    db.commit()
