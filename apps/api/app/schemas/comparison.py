from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ComparisonDimension(StrEnum):
    RESEARCH_QUESTION = "research_question"
    METHOD_ARCHITECTURE = "method_architecture"
    DATASET_TASK = "dataset_task"
    TRAINING_SETUP = "training_setup"
    METRIC = "metric"
    RESULTS = "results"
    LIMITATIONS = "limitations"


DEFAULT_COMPARISON_DIMENSIONS = list(ComparisonDimension)


class ComparisonRequest(BaseModel):
    """Bounded, explicit selection for a same-project paper comparison."""

    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    paper_ids: list[UUID] = Field(min_length=2, max_length=6)
    question: str | None = Field(default=None, min_length=1, max_length=1000)
    dimensions: list[ComparisonDimension] = Field(
        default_factory=lambda: list(DEFAULT_COMPARISON_DIMENSIONS),
        min_length=1,
        max_length=len(ComparisonDimension),
    )

    @field_validator("paper_ids")
    @classmethod
    def paper_ids_must_be_unique(cls, value: list[UUID]) -> list[UUID]:
        if len(set(value)) != len(value):
            raise ValueError("paper_ids must be unique")
        return value

    @field_validator("dimensions")
    @classmethod
    def dimensions_must_be_unique(
        cls, value: list[ComparisonDimension]
    ) -> list[ComparisonDimension]:
        if len(set(value)) != len(value):
            raise ValueError("dimensions must be unique")
        return value

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("question must not be blank")
        return value.strip() if value is not None else None


class BenchmarkContext(BaseModel):
    """Reported experimental context needed to compare two numeric results."""

    model_config = ConfigDict(extra="forbid")

    task: str | None = Field(default=None, max_length=200)
    dataset: str | None = Field(default=None, max_length=200)
    split: str | None = Field(default=None, max_length=100)
    metric: str | None = Field(default=None, max_length=100)
    unit: str | None = Field(default=None, max_length=50)
    comparison_condition: str | None = Field(default=None, max_length=500)

    @field_validator(
        "task", "dataset", "split", "metric", "unit", "comparison_condition", mode="before"
    )
    @classmethod
    def normalize_context_text(cls, value: object) -> object:
        if isinstance(value, str):
            return " ".join(value.split()) or None
        return value


class ComparabilityStatus(StrEnum):
    DIRECTLY_COMPARABLE = "directly_comparable"
    NOT_DIRECTLY_COMPARABLE = "not directly comparable"


class ComparabilityResult(BaseModel):
    status: ComparabilityStatus
    reasons: list[str] = Field(default_factory=list)
