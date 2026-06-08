from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, or_
from sqlalchemy.orm import Session, joinedload

from app.db.models import Memory, MemoryAudit, MemorySource, PaperPage
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
    base_filter = [Memory.project_id == project_id]

    if memory_type:
        type_val = memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
        base_filter.append(Memory.memory_type == type_val)

    if status:
        status_val = status.value if isinstance(status, MemoryStatus) else str(status)
        base_filter.append(Memory.status == status_val)

    if is_pinned is not None:
        base_filter.append(Memory.is_pinned.is_(is_pinned))

    if search and search.strip():
        term = f"%{search.strip()}%"
        base_filter.append(or_(Memory.title.ilike(term), Memory.content.ilike(term)))

    total = db.query(func.count(Memory.id)).filter(*base_filter).scalar() or 0
    items = (
        db.query(Memory)
        .options(joinedload(Memory.sources), joinedload(Memory.history))
        .filter(*base_filter)
        .order_by(
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

    # Lock row at write-time to ensure atomic compare-and-swap
    locked_mem = db.query(Memory).filter(Memory.id == memory.id).with_for_update().first()
    if not locked_mem:
        raise ValueError(f"Memory {memory.id} not found.")

    if memory_update.version != locked_mem.version:
        raise MemoryVersionConflictError(
            f"Version conflict: current version is {locked_mem.version}, "
            f"but provided version was {memory_update.version}"
        )

    # If updating a PAPER_FACT's content:
    if locked_mem.memory_type == MemoryType.PAPER_FACT.value:
        if (
            memory_update.content is not None
            and memory_update.content.strip().lower() != locked_mem.content.strip().lower()
        ):
            from app.services.memory_service import verify_claim_supported_by_quote

            supported = False
            for s in locked_mem.sources:
                if s.quote_text:
                    page_text = None
                    if s.paper_id and s.page_number is not None:
                        pg = (
                            db.query(PaperPage)
                            .filter(
                                PaperPage.paper_id == s.paper_id,
                                PaperPage.page_number == s.page_number,
                            )
                            .first()
                        )
                        if pg:
                            page_text = pg.raw_text
                    is_supp, _ = verify_claim_supported_by_quote(
                        memory_update.content, s.quote_text, page_text=page_text
                    )
                    if is_supp:
                        supported = True
                        break
            if not supported:
                raise ValueError(
                    "Cannot update paper fact: new content is not supported by cited source quote."
                )

            try:
                from app.services.embedding import get_embedding_provider

                provider = get_embedding_provider()
                new_vec = provider.embed_query(memory_update.content)
                locked_mem.embedding = new_vec
                locked_mem.embedding_vec = new_vec
            except Exception:
                pass

    old_content = locked_mem.content
    actions = []

    if memory_update.title is not None and memory_update.title != locked_mem.title:
        locked_mem.title = memory_update.title
        actions.append("TITLE_UPDATED")

    if memory_update.content is not None and memory_update.content != locked_mem.content:
        locked_mem.content = memory_update.content
        actions.append("CONTENT_UPDATED")

    if memory_update.status is not None:
        status_val = (
            memory_update.status.value
            if isinstance(memory_update.status, MemoryStatus)
            else str(memory_update.status)
        )
        if status_val != locked_mem.status:
            locked_mem.status = status_val
            actions.append(f"STATUS_{status_val}")

    if memory_update.importance is not None and memory_update.importance != locked_mem.importance:
        locked_mem.importance = memory_update.importance
        actions.append("IMPORTANCE_UPDATED")

    if memory_update.confidence is not None and memory_update.confidence != locked_mem.confidence:
        locked_mem.confidence = memory_update.confidence
        actions.append("CONFIDENCE_UPDATED")

    if memory_update.is_pinned is not None and memory_update.is_pinned != locked_mem.is_pinned:
        locked_mem.is_pinned = memory_update.is_pinned
        actions.append("PINNED" if locked_mem.is_pinned else "UNPINNED")

    locked_mem.version += 1
    locked_mem.updated_at = datetime.now(UTC)

    action_label = ", ".join(actions) if actions else "UPDATED"
    audit_reason = memory_update.reason or reason or "Memory updated"

    audit = MemoryAudit(
        memory_id=locked_mem.id,
        action=action_label,
        old_content=old_content,
        new_content=locked_mem.content,
        reason=audit_reason,
    )
    db.add(audit)
    db.commit()
    db.refresh(locked_mem)
    return locked_mem


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


def atomic_create_and_supersede(
    db: Session,
    project_id: UUID,
    old_memory_id: UUID,
    expected_version: int,
    new_memory_in: MemoryCreate,
    embedding: list[float] | None = None,
    reason: str = "Superseded by newer decision",
) -> tuple[Memory, Memory]:
    """Atomically create a replacement active memory and supersede the prior active memory.

    Guarantees:
    - Row-level lock on old memory preventing concurrent race conditions.
    - Write-time optimistic concurrency version verification.
    - Single transaction: flush IDs, update records and audits, single commit.
    - Rollback on failure: never leaves two active memories or partial rows.
    """
    old_mem = (
        db.query(Memory)
        .filter(Memory.id == old_memory_id, Memory.project_id == project_id)
        .with_for_update()
        .first()
    )
    if not old_mem:
        raise ValueError(f"Memory {old_memory_id} not found in project {project_id}.")

    if old_mem.status != MemoryStatus.ACTIVE.value:
        raise ValueError(f"Cannot supersede memory with status '{old_mem.status}'; must be ACTIVE.")

    if old_mem.version != expected_version:
        raise MemoryVersionConflictError(
            f"Version conflict: current version is {old_mem.version}, "
            f"but expected {expected_version}"
        )

    if new_memory_in.content.strip().lower() == old_mem.content.strip().lower():
        raise ValueError("Cannot supersede memory with identical content.")

    try:
        new_mem = Memory(
            project_id=project_id,
            memory_type=new_memory_in.memory_type.value,
            status=MemoryStatus.ACTIVE.value,
            title=new_memory_in.title,
            content=new_memory_in.content,
            confidence=new_memory_in.confidence,
            importance=new_memory_in.importance,
            version=1,
            is_pinned=new_memory_in.is_pinned,
            embedding=embedding,
            embedding_vec=embedding,
        )
        db.add(new_mem)
        db.flush()

        for s in new_memory_in.sources:
            source_rec = MemorySource(
                memory_id=new_mem.id,
                source_type=s.source_type.value,
                message_id=s.message_id,
                paper_id=s.paper_id,
                page_number=s.page_number,
                quote_text=s.quote_text,
                document_sha256=s.document_sha256,
            )
            db.add(source_rec)

        old_content = old_mem.content
        old_mem.status = MemoryStatus.SUPERSEDED.value
        old_mem.superseded_by_id = new_mem.id
        old_mem.version += 1
        old_mem.updated_at = datetime.now(UTC)

        audit_create = MemoryAudit(
            memory_id=new_mem.id,
            action="CREATED",
            old_content=None,
            new_content=new_mem.content,
            reason=f"Supersedes prior memory ID: {old_mem.id}",
        )
        audit_old = MemoryAudit(
            memory_id=old_mem.id,
            action="SUPERSEDED",
            old_content=old_content,
            new_content=new_mem.content,
            reason=f"{reason} (new memory ID: {new_mem.id})",
        )
        db.add(audit_create)
        db.add(audit_old)

        db.commit()
        db.refresh(old_mem)
        db.refresh(new_mem)
        return old_mem, new_mem
    except Exception:
        db.rollback()
        raise


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
