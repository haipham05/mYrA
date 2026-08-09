from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class TranslationCreate(BaseModel):
    project_id: UUID
    acknowledge_external_processing: bool
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class TranslationGlossaryEntryInput(BaseModel):
    source_term: str = Field(min_length=1, max_length=255)
    preferred_translation: str = Field(min_length=1, max_length=255)


class TranslationGlossaryReplace(BaseModel):
    entries: list[TranslationGlossaryEntryInput] = Field(max_length=200)


class TranslationGlossaryEntry(BaseModel):
    id: UUID
    source_term: str
    preferred_translation: str
    created_at: datetime
    updated_at: datetime


class TranslationGlossaryResponse(BaseModel):
    project_id: UUID
    entries: list[TranslationGlossaryEntry]


class TranslationSegmentResponse(BaseModel):
    id: UUID
    ordinal: int
    source_page_number: int
    source_element_id: UUID | None
    source_quote: str
    translated_text: str
    output_page_number: int | None
    output_boxes: list[dict[str, float]] | None
    status: str


class TranslationResponse(BaseModel):
    id: UUID
    project_id: UUID
    paper_id: UUID
    status: str
    stage: str
    source_sha256: str
    source_filename: str
    source_page_count: int | None
    completed_units: int
    total_units: int | None
    skipped_units: int = 0
    warnings: list[str] = Field(default_factory=list)
    output_sha256: str | None
    output_available: bool
    error_code: str | None
    error_message: str | None
    is_retryable: bool
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    segments: list[TranslationSegmentResponse] = Field(default_factory=list)


class TranslationListResponse(BaseModel):
    items: list[TranslationResponse]
