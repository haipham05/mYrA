"""Source-linked comparison matrix output without inferred paper claims."""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.schemas.comparison import (
    BenchmarkContext,
    ComparabilityResult,
    ComparisonDimension,
)
from app.schemas.evidence import Citation, EvidenceItem


class ComparisonCellStatus(StrEnum):
    EVIDENCE_AVAILABLE = "evidence_available"
    NOT_FOUND = "not_found"


class ComparisonExcerpt(BaseModel):
    """A retrieved passage offered for review, not a validated factual cell."""

    evidence: EvidenceItem
    citation: Citation


class ComparisonCell(BaseModel):
    paper_id: UUID
    dimension: ComparisonDimension
    status: ComparisonCellStatus
    evidence_kind: Literal["candidate_source_excerpts_only"] = "candidate_source_excerpts_only"
    excerpts: list[ComparisonExcerpt] = Field(default_factory=list, max_length=3)
    message: str | None = None


class ComparisonMatrix(BaseModel):
    project_id: UUID
    question: str | None = None
    paper_ids: list[UUID]
    dimensions: list[ComparisonDimension]
    cells: list[ComparisonCell]
    interpretation_notice: str = (
        "These are candidate source excerpts, not extracted or fact-checked comparison claims."
    )


class BenchmarkComparison(BaseModel):
    """Side-by-side numeric results, with source-backed experimental contexts."""

    left_paper_id: UUID
    right_paper_id: UUID
    left_result: str = Field(max_length=160)
    right_result: str = Field(max_length=160)
    left_context: BenchmarkContext
    right_context: BenchmarkContext
    comparability: ComparabilityResult
    evidence_ids: list[str] = Field(min_length=2, max_length=6)
