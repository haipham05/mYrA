from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MemoryType(StrEnum):
    DECISION = "DECISION"
    PREFERENCE = "PREFERENCE"
    TERMINOLOGY = "TERMINOLOGY"
    PROCEDURAL = "PROCEDURAL"
    PAPER_FACT = "PAPER_FACT"
    EPISODIC = "EPISODIC"


class MemoryStatus(StrEnum):
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    ARCHIVED = "ARCHIVED"
    DISPUTED = "DISPUTED"


class MemorySourceType(StrEnum):
    MESSAGE = "MESSAGE"
    PAPER_CHUNK = "PAPER_CHUNK"


class MemorySourceBase(BaseModel):
    source_type: MemorySourceType
    message_id: UUID | None = None
    paper_id: UUID | None = None
    page_number: int | None = None
    quote_text: str | None = None
    document_sha256: str | None = None


class MemorySourceCreate(MemorySourceBase):
    pass


class MemorySourceResponse(MemorySourceBase):
    id: UUID
    memory_id: UUID
    conversation_id: UUID | None = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class MemoryAuditResponse(BaseModel):
    id: UUID
    memory_id: UUID
    action: str
    old_content: str | None = None
    new_content: str | None = None
    reason: str | None = None
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class MemoryCreate(BaseModel):
    memory_type: MemoryType
    title: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    is_pinned: bool = False
    sources: list[MemorySourceCreate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_type_sources(self) -> "MemoryCreate":
        if self.memory_type == MemoryType.PAPER_FACT:
            valid_sources = [
                s
                for s in self.sources
                if s.source_type == MemorySourceType.PAPER_CHUNK
                and s.paper_id is not None
                and s.quote_text
                and len(s.quote_text.strip()) > 0
            ]
            if not valid_sources:
                raise ValueError(
                    "PAPER_FACT memory requires a verified paper source "
                    "with paper_id and quote_text"
                )
        return self


class MemoryUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    content: str | None = Field(default=None, min_length=1)
    status: MemoryStatus | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_pinned: bool | None = None
    version: int = Field(
        ..., description="Current version of the memory for optimistic concurrency"
    )
    reason: str | None = None


class MemoryResponse(BaseModel):
    id: UUID
    project_id: UUID
    memory_type: MemoryType
    status: MemoryStatus
    title: str
    content: str
    confidence: float
    importance: float
    version: int
    is_pinned: bool
    superseded_by_id: UUID | None = None
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    sources: list[MemorySourceResponse] = Field(default_factory=list)
    history: list[MemoryAuditResponse] = Field(default_factory=list)
    model_config = ConfigDict(from_attributes=True)


class MemoryListResponse(BaseModel):
    items: list[MemoryResponse]
    total: int
    limit: int
    offset: int
