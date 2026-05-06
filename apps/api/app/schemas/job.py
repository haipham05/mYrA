from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobStage(StrEnum):
    QUEUED = "QUEUED"
    PARSING = "PARSING"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING = "INDEXING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class JobResponse(BaseModel):
    id: UUID
    paper_id: UUID
    status: JobStatus
    stage: JobStage
    progress: float = Field(ge=0.0, le=1.0)
    error_message: str | None = None
    is_retryable: bool = False
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime
    updated_at: datetime
