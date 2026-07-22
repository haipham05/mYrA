from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.common import PaginatedResponse


class PaperStatus(StrEnum):
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class PaperUploadResponse(BaseModel):
    paper_id: UUID
    job_id: UUID
    status: PaperStatus = PaperStatus.PROCESSING


class IngestionJobSummary(BaseModel):
    id: UUID
    status: str
    stage: str
    progress: float = Field(ge=0.0, le=1.0)
    is_retryable: bool = False
    retry_count: int = 0


class PaperResponse(BaseModel):
    id: UUID
    project_id: UUID
    filename: str
    status: PaperStatus
    latest_job: IngestionJobSummary | None = None
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


class PaperMetadataUpdate(BaseModel):
    """Explicit owner-provided bibliographic corrections; omitted fields stay unchanged."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=2000)
    authors: list[str] | None = Field(default=None, max_length=100)
    publication_year: int | None = Field(default=None, ge=1000, le=2100)
    doi: str | None = Field(default=None, max_length=255)
    arxiv_id: str | None = Field(default=None, max_length=128)
    abstract: str | None = Field(default=None, max_length=50000)
    source_url: str | None = Field(default=None, max_length=4000)

    @field_validator("title", "doi", "arxiv_id", "abstract", "source_url")
    @classmethod
    def normalize_text_fields(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("authors")
    @classmethod
    def normalize_authors(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [author.strip() for author in value if author.strip()]
        return cleaned or None
