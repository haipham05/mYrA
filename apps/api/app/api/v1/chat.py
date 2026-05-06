import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.crud.chat import create_conversation, get_conversation
from app.crud.project import get_project
from app.db.models import Conversation
from app.db.session import get_db
from app.schemas.chat import (
    ConversationCreate,
    ConversationResponse,
    MessageCreate,
    MessageResponse,
    MessageRole,
)
from app.schemas.evidence import Citation, EvidenceItem
from app.services.chat_service import ChatService

logger = logging.getLogger("myra.api.chat")
router = APIRouter(tags=["chat"])
chat_service = ChatService()


@router.post(
    "/projects/{project_id}/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_new_conversation(
    project_id: UUID,
    conversation_in: ConversationCreate | None = None,
    db: Session = Depends(get_db),
) -> ConversationResponse:
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    title = conversation_in.title if conversation_in else None
    conv = create_conversation(db, project_id=project_id, title=title)
    return ConversationResponse.model_validate(conv, from_attributes=True)


@router.get(
    "/projects/{project_id}/conversations",
    response_model=list[ConversationResponse],
)
def list_project_conversations(
    project_id: UUID,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> list[ConversationResponse]:
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    convs = (
        db.query(Conversation)
        .filter(Conversation.project_id == project_id)
        .order_by(Conversation.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [ConversationResponse.model_validate(c, from_attributes=True) for c in convs]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
)
def get_single_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
) -> ConversationResponse:
    conv = get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return ConversationResponse.model_validate(conv, from_attributes=True)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
def get_conversation_messages(
    conversation_id: UUID,
    db: Session = Depends(get_db),
) -> list[MessageResponse]:
    conv = get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    results = []
    for msg in conv.messages:
        citations = [Citation.model_validate(c) for c in (msg.citations or [])]
        evidence = [EvidenceItem.model_validate(e) for e in (msg.evidence or [])]
        results.append(
            MessageResponse(
                id=msg.id,
                conversation_id=msg.conversation_id,
                role=MessageRole(msg.role),
                content=msg.content,
                citations=citations,
                evidence=evidence,
                created_at=msg.created_at,
            )
        )
    return results


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=MessageResponse,
)
async def send_message(
    conversation_id: UUID,
    message_in: MessageCreate,
    db: Session = Depends(get_db),
) -> MessageResponse:
    conv = get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )

    try:
        return await chat_service.answer_question(
            db=db,
            conversation_id=conversation_id,
            question=message_in.content,
        )
    except Exception as err:
        logger.error("Failed to generate answer", exc_info=err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate answer",
        ) from err
