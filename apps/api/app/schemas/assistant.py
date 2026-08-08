import json
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.chat import PaperScope
from app.schemas.evidence import Citation
from app.schemas.paper import PaperUploadResponse


class AssistantIntent(StrEnum):
    HELP = "help"
    QA = "qa"
    READ_PAPER = "read_paper"
    COMPARE = "compare"
    VERIFY_CLAIM = "verify_claim"
    DISCOVER = "discover"
    NOTES = "notes"
    REPORT = "report"
    RESEARCH = "research"
    GAP_ANALYSIS = "gap_analysis"
    EXPERIMENT_PLAN = "experiment_plan"
    TRANSLATE = "translate"
    VISION = "vision"
    GRAPH = "graph"
    CLARIFY = "clarify"


class RunStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    NEEDS_INPUT = "NEEDS_INPUT"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class SourceSelection(BaseModel):
    paper_id: UUID
    page_number: int = Field(ge=1)
    quote: str = Field(min_length=1, max_length=2000)
    document_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")

    @field_validator("quote")
    @classmethod
    def quote_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("selected passage must not be blank")
        return value


class AssistantRunRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10000)
    conversation_id: UUID
    project_id: UUID
    scope: PaperScope = PaperScope.PROJECT
    selected_paper_ids: list[UUID] = Field(default_factory=list, max_length=6)
    intent_override: AssistantIntent | None = None
    parent_run_id: UUID | None = None
    source_selection: SourceSelection | None = None
    idempotency_key: str = Field(min_length=8, max_length=128)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value

    @field_validator("idempotency_key")
    @classmethod
    def idempotency_key_must_be_safe(cls, value: str) -> str:
        if not value.strip() or any(char.isspace() for char in value):
            raise ValueError("idempotency_key must be a non-blank token")
        return value

    @field_validator("selected_paper_ids")
    @classmethod
    def selected_papers_must_be_unique(cls, value: list[UUID]) -> list[UUID]:
        if len(set(value)) != len(value):
            raise ValueError("selected_paper_ids must be unique")
        return value

    @model_validator(mode="after")
    def scope_must_match_selected_papers(self) -> "AssistantRunRequest":
        if self.scope is PaperScope.PROJECT and self.selected_paper_ids:
            raise ValueError("project scope must not include selected paper IDs")
        if self.scope is PaperScope.PAPER and len(self.selected_paper_ids) != 1:
            raise ValueError("paper scope requires exactly one selected paper")
        if self.scope is PaperScope.SELECTION and not self.selected_paper_ids:
            raise ValueError("selection scope requires at least one selected paper")
        if self.source_selection is not None and (
            self.scope is not PaperScope.PAPER
            or self.selected_paper_ids != [self.source_selection.paper_id]
        ):
            raise ValueError("selected passage requires its single paper as the run scope")
        return self


class RouteDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: AssistantIntent
    standalone_question: str | None = Field(default=None, max_length=2000)
    resolved_paper_ids: list[UUID] = Field(default_factory=list, max_length=6)
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list, max_length=8)
    clarification: str | None = Field(default=None, max_length=1000)
    action_summary: str = Field(min_length=1, max_length=240)

    @field_validator("arguments")
    @classmethod
    def arguments_must_be_bounded(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 24:
            raise ValueError("route arguments exceed the allowed field count")
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("route arguments must contain JSON values") from exc
        if len(encoded.encode("utf-8")) > 4096:
            raise ValueError("route arguments exceed the allowed size")
        return value

    @field_validator("resolved_paper_ids")
    @classmethod
    def resolved_papers_must_be_unique(cls, value: list[UUID]) -> list[UUID]:
        if len(set(value)) != len(value):
            raise ValueError("resolved_paper_ids must be unique")
        return value


class RoutePaperContext(BaseModel):
    id: UUID
    title: str = Field(max_length=300)
    authors: list[str] = Field(default_factory=list, max_length=8)
    publication_year: int | None = Field(default=None, ge=1000, le=3000)


class RouteOutcome(StrEnum):
    ROUTED = "ROUTED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    UNAVAILABLE = "UNAVAILABLE"


class AssistantRouteResult(BaseModel):
    outcome: RouteOutcome
    decision: RouteDecision
    requested_model: str | None = None
    reported_model: str | None = None
    usage: dict[str, int | None] | None = None


class AssistantRunResult(BaseModel):
    result_type: str = Field(min_length=1, max_length=80)
    display_text: str = Field(max_length=20000)
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list, max_length=200)
    warnings: list[str] = Field(default_factory=list, max_length=50)
    usage: dict[str, int | float | str | None] = Field(default_factory=dict)
    available_actions: list[str] = Field(default_factory=list, max_length=20)
    artifact_ids: list[UUID] = Field(default_factory=list, max_length=50)


class AssistantRunResponse(BaseModel):
    id: UUID
    project_id: UUID
    conversation_id: UUID
    status: RunStatus
    intent: AssistantIntent | None = None
    action_summary: str | None = None
    stage: str | None = None
    result: AssistantRunResult | None = None
    safe_error: str | None = None
    usage: dict[str, int | float | str | None] | None = None
    created_at: datetime
    updated_at: datetime


class AssistantApprovalResponse(BaseModel):
    id: UUID
    run_id: UUID
    action_type: str = Field(min_length=1, max_length=80)
    arguments: dict[str, Any]
    source_fingerprint: str = Field(min_length=64, max_length=64)
    status: str = Field(min_length=1, max_length=24)
    expires_at: datetime
    decided_at: datetime | None = None
    import_result: PaperUploadResponse | None = None


class AssistantRunResumeRequest(BaseModel):
    additional_input: str = Field(min_length=1, max_length=4000)

    @field_validator("additional_input")
    @classmethod
    def resume_input_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("additional_input must not be blank")
        return value


class AssistantRunArtifactSaveRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
