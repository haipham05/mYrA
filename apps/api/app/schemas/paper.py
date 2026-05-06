from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel

from app.schemas.common import PaginatedResponse


class PaperStatus(StrEnum):
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class PaperUploadResponse(BaseModel):
    paper_id: UUID
    job_id: UUID
    status: PaperStatus = PaperStatus.PROCESSING


class PaperResponse(BaseModel):
    id: UUID
    project_id: UUID
    filename: str
    status: PaperStatus
    page_count: int | None = None
    document_sha256: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class PaperListResponse(PaginatedResponse[PaperResponse]):
    pass
