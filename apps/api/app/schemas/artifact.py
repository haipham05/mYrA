from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ArtifactType(StrEnum):
    READING_BRIEF = "reading_brief"
    COMPARISON = "comparison"
    CLAIM_CHECK = "claim_check"
    REPORT = "report"
    GAP_ANALYSIS = "gap_analysis"
    EXPERIMENT_PROPOSAL = "experiment_proposal"


class ArtifactExportFormat(StrEnum):
    MARKDOWN = "markdown"
    CSV = "csv"
    BIBTEX = "bibtex"


class ArtifactRevisionCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    payload: dict[str, Any]
    scope_snapshot: dict[str, Any] = Field(default_factory=dict)
    source_manifest: list[dict[str, Any]] = Field(default_factory=list, max_length=500)
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    usage: dict[str, Any] = Field(default_factory=dict)


class ArtifactCreate(ArtifactRevisionCreate):
    artifact_type: ArtifactType


class ArtifactRevisionResponse(ArtifactRevisionCreate):
    id: UUID
    artifact_id: UUID
    revision_number: int
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)


class ArtifactResponse(BaseModel):
    id: UUID
    project_id: UUID
    artifact_type: ArtifactType
    title: str
    latest_revision: int
    created_at: datetime
    updated_at: datetime
    latest: ArtifactRevisionResponse
    model_config = ConfigDict(from_attributes=True)


class ArtifactDetailResponse(ArtifactResponse):
    revisions: list[ArtifactRevisionResponse]
