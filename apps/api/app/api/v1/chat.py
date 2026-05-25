import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.crud.chat import (
    create_conversation,
    delete_conversation,
    get_conversation,
    list_conversations,
    update_conversation,
)
from app.crud.project import get_project
from app.db.models import Conversation
from app.db.session import get_db
from app.schemas.chat import (
    ConversationCreate,
    ConversationListResponse,
    ConversationResponse,
    ConversationUpdate,
    MessageCreate,
    MessageResponse,
    MessageRole,
)
from app.schemas.evidence import Citation, EvidenceItem
from app.services.chat_service import ChatService

logger = logging.getLogger("myra.api.chat")
router = APIRouter(tags=["chat"])
chat_service = ChatService()


def _to_conversation_response(
    conv: Conversation, message_count: int | None = None
) -> ConversationResponse:
    count = message_count if message_count is not None else len(conv.messages)
    return ConversationResponse(
        id=conv.id,
        project_id=conv.project_id,
        title=conv.title,
        is_archived=conv.is_archived,
        message_count=count,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


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
    return _to_conversation_response(conv, message_count=0)


@router.get(
    "/projects/{project_id}/conversations",
    response_model=ConversationListResponse,
)
def list_project_conversations(
    project_id: UUID,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    include_archived: bool = Query(default=False),
    db: Session = Depends(get_db),
) -> ConversationListResponse:
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    convs, total = list_conversations(
        db,
        project_id=project_id,
        limit=limit,
        offset=offset,
        include_archived=include_archived,
    )
    items = [_to_conversation_response(c) for c in convs]
    return ConversationListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


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
    return _to_conversation_response(conv)


@router.patch(
    "/conversations/{conversation_id}",
    response_model=ConversationResponse,
)
def update_single_conversation(
    conversation_id: UUID,
    conv_update: ConversationUpdate,
    db: Session = Depends(get_db),
) -> ConversationResponse:
    conv = update_conversation(
        db,
        conversation_id=conversation_id,
        title=conv_update.title,
        is_archived=conv_update.is_archived,
    )
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )
    return _to_conversation_response(conv)


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_single_conversation(
    conversation_id: UUID,
    db: Session = Depends(get_db),
) -> None:
    deleted = delete_conversation(db, conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversation {conversation_id} not found",
        )


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
                model_name=msg.model_name,
                token_count=msg.token_count,
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
