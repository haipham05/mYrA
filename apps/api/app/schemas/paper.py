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
    title: str | None = None
    authors: list[str] | None = None
    publication_year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    source_url: str | None = None
    metadata_provenance: dict[str, str] | None = None
    page_count: int | None = None
    document_sha256: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class PaperListResponse(PaginatedResponse[PaperResponse]):
    pass
