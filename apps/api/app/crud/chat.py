from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import Conversation, Message


def create_conversation(db: Session, project_id: UUID, title: str | None = None) -> Conversation:
    conv = Conversation(project_id=project_id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def get_conversation(
    db: Session,
    conversation_id: UUID,
    project_id: UUID | None = None,
) -> Conversation | None:
    query = db.query(Conversation).filter(Conversation.id == conversation_id)
    if project_id is not None:
        query = query.filter(Conversation.project_id == project_id)
    return query.first()


def list_conversations(
    db: Session,
    project_id: UUID,
    limit: int = 50,
    offset: int = 0,
    include_archived: bool = False,
) -> tuple[list[Conversation], int]:
    query = db.query(Conversation).filter(Conversation.project_id == project_id)
    if not include_archived:
        query = query.filter(Conversation.is_archived.is_(False))
    query = query.order_by(Conversation.updated_at.desc(), Conversation.created_at.desc())
    total = query.count()
    items = query.offset(offset).limit(limit).all()
    return items, total


def update_conversation(
    db: Session,
    conversation_id: UUID,
    title: str | None = None,
    summary: str | None = None,
    is_archived: bool | None = None,
) -> Conversation | None:
    conv = get_conversation(db, conversation_id)
    if not conv:
        return None
    if title is not None:
        conv.title = title
    if summary is not None:
        conv.summary = summary
    if is_archived is not None:
        conv.is_archived = is_archived
    conv.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(conv)
    return conv


def delete_conversation(
    db: Session,
    conversation_id: UUID,
    project_id: UUID | None = None,
) -> bool:
    conv = get_conversation(db, conversation_id, project_id=project_id)
    if not conv:
        return False
    db.delete(conv)
    db.commit()
    return True


def add_message(
    db: Session,
    conversation_id: UUID,
    role: str,
    content: str,
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    model_name: str | None = None,
    token_count: int | None = None,
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations,
        evidence=evidence,
        model_name=model_name,
        token_count=token_count,
    )
    db.add(msg)
    # Touch conversation updated_at
    conv = db.get(Conversation, conversation_id)
    if conv:
        conv.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(msg)
    return msg


def list_messages(
    db: Session,
    conversation_id: UUID,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[Message], int]:
    query = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
    )
    total = query.count()
    items = query.offset(offset).limit(limit).all()
    return items, total
