"""Typed observations derived from selected figure or table images."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class VisualObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=1200)


class VisualReading(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=300)
    value: str = Field(min_length=1, max_length=200)
    unit: str | None = Field(default=None, max_length=100)
    kind: Literal["direct_reading", "plot_estimate"]


class VisualAnalysis(BaseModel):
    """Keep what is seen separate from interpretation and uncertainty."""

    model_config = ConfigDict(extra="forbid")

    observations: list[VisualObservation] = Field(default_factory=list, max_length=3)
    readings: list[VisualReading] = Field(default_factory=list, max_length=2)
    interpretation: str = Field(default="", max_length=1000)
    uncertainty_notes: list[str] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def require_analysis_content(self) -> "VisualAnalysis":
        if not (
            self.observations
            or self.readings
            or self.interpretation.strip()
            or self.uncertainty_notes
        ):
            raise ValueError("visual analysis contains no observations")
        return self


class VisualSourceReference(BaseModel):
    """A navigable visual source; intentionally cannot satisfy a text citation."""

    model_config = ConfigDict(extra="forbid")

    source_kind: Literal["visual"] = "visual"
    project_id: UUID
    paper_id: UUID
    document_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    page_number: int = Field(ge=1)
    crop_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    crop_box_normalized_top_left: dict[str, float]
    caption: str | None = Field(default=None, max_length=1000)
    text_citation: Literal[False] = False

    @classmethod
    def from_source_metadata(cls, source: dict[str, object]) -> "VisualSourceReference":
        return cls(
            project_id=source["project_id"],
            paper_id=source["paper_id"],
            document_sha256=source["document_sha256"],
            page_number=source["page_number"],
            crop_sha256=source["crop_sha256"],
            crop_box_normalized_top_left=source["crop_box_normalized_top_left"],
            caption=source.get("caption"),
        )


class VisualCropBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_order(self) -> "VisualCropBox":
        if not (self.left < self.right and self.top < self.bottom):
            raise ValueError("crop bounds must have positive width and height")
        return self


class VisualSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: UUID
    page_number: int = Field(ge=1)
    document_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    crop: VisualCropBox
