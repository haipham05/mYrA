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


def get_conversation(db: Session, conversation_id: UUID) -> Conversation | None:
    return db.query(Conversation).filter(Conversation.id == conversation_id).first()


def add_message(
    db: Session,
    conversation_id: UUID,
    role: str,
    content: str,
    citations: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> Message:
    msg = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        citations=citations,
        evidence=evidence,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg
