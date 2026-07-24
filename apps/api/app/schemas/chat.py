from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.evidence import Citation, ClaimEvidenceSupport, EvidenceItem


class MessageRole(StrEnum):
    USER = "USER"
    ASSISTANT = "ASSISTANT"
    SYSTEM = "SYSTEM"


class PaperScope(StrEnum):
    PAPER = "paper"
    SELECTION = "selection"
    PROJECT = "project"


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    paper_scope: PaperScope = PaperScope.PROJECT
    selected_paper_ids: list[UUID] = Field(default_factory=list, max_length=50)


class ConversationUpdate(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    summary: str | None = None
    is_archived: bool | None = None
    paper_scope: PaperScope | None = None
    selected_paper_ids: list[UUID] | None = Field(default=None, max_length=50)


class ConversationResponse(BaseModel):
    id: UUID
    project_id: UUID
    title: str | None = None
    summary: str | None = None
    paper_scope: PaperScope = PaperScope.PROJECT
    selected_paper_ids: list[UUID] = Field(default_factory=list)
    is_archived: bool = False
    message_count: int = 0
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    items: list[ConversationResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=10000)


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: MessageRole
    content: str
    citations: list[Citation] = Field(default_factory=list)
    claim_supports: list[ClaimEvidenceSupport] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    model_name: str | None = None
    token_count: int | None = None
    provider_usage: dict[str, int | None] | None = None
    created_at: datetime
