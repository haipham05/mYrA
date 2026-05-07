from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class CoordinateOrigin(StrEnum):
    TOP_LEFT = "TOP_LEFT"
    BOTTOM_LEFT = "BOTTOM_LEFT"


class AnchorStatus(StrEnum):
    VERIFIED = "verified"
    UNRESOLVED = "unresolved"
    LEGACY = "legacy"


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    page_width: float
    page_height: float
    origin: CoordinateOrigin = CoordinateOrigin.TOP_LEFT
    rotation: int = 0

    def to_normalized_top_left(self) -> "BoundingBox":
        """Convert box to normalized [0, 1] coordinates with TOP_LEFT origin."""
        if self.page_width <= 0 or self.page_height <= 0:
            return self

        if self.origin == CoordinateOrigin.TOP_LEFT:
            return BoundingBox(
                x_min=self.x_min / self.page_width,
                y_min=self.y_min / self.page_height,
                x_max=self.x_max / self.page_width,
                y_max=self.y_max / self.page_height,
                page_width=1.0,
                page_height=1.0,
                origin=CoordinateOrigin.TOP_LEFT,
                rotation=self.rotation,
            )
        else:
            # Bottom-left to top-left
            return BoundingBox(
                x_min=self.x_min / self.page_width,
                y_min=(self.page_height - self.y_max) / self.page_height,
                x_max=self.x_max / self.page_width,
                y_max=(self.page_height - self.y_min) / self.page_height,
                page_width=1.0,
                page_height=1.0,
                origin=CoordinateOrigin.TOP_LEFT,
                rotation=self.rotation,
            )


class SourceElement(BaseModel):
    id: UUID
    element_index: int
    element_type: str
    text: str
    page_number: int
    bounding_box: BoundingBox | None = None
    section_path: list[str] = Field(default_factory=list)


class CitationAnchor(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    page_number: int
    source_element_id: UUID | None = None
    exact_quote: str
    source_char_start: int | None = None
    source_char_end: int | None = None
    document_sha256: str | None = None
    parser_version: str | None = None
    anchor_status: AnchorStatus = AnchorStatus.UNRESOLVED
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_anchor(self) -> "CitationAnchor":
        if self.page_number < 1:
            raise ValueError("page_number must be >= 1")
        if not self.exact_quote.strip():
            self.anchor_status = AnchorStatus.UNRESOLVED
        return self


class EvidenceItem(BaseModel):
    id: str  # e.g. "E1"
    paper_id: UUID
    paper_title: str | None = None
    chunk_id: UUID
    quote: str
    parent_context: str | None = None
    page_number: int
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    source_element_ids: list[UUID] = Field(default_factory=list)
    document_sha256: str | None = None
    parser_version: str | None = None
    anchors: list[CitationAnchor] = Field(default_factory=list)


class Citation(BaseModel):
    citation_index: int  # e.g. 1
    evidence_id: str  # e.g. "E1"
    paper_id: UUID
    page_number: int
    quote: str
    bounding_boxes: list[BoundingBox] = Field(default_factory=list)
    document_sha256: str | None = None
    parser_version: str | None = None
    anchor_status: AnchorStatus = AnchorStatus.UNRESOLVED
    anchors: list[CitationAnchor] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def populate_legacy_and_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Backward-compatibility: if anchor_status not explicitly provided
            if "anchor_status" not in data:
                if data.get("bounding_boxes") and not data.get("anchors"):
                    data["anchor_status"] = AnchorStatus.LEGACY
                else:
                    data["anchor_status"] = AnchorStatus.UNRESOLVED
            # If quote is empty, mark unresolved
            if not data.get("quote", "").strip():
                data["anchor_status"] = AnchorStatus.UNRESOLVED
        return data
