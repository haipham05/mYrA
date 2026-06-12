"""Schemas for GraphRAG knowledge graph extraction, validation, and storage.

Defines:
- EntityType, RelationshipPredicate, and ClaimPolarity enums
- Strict endpoint pair constraints (VALID_ENDPOINT_CONSTRAINTS)
- GraphEntitySchema, GraphProvenanceSchema, GraphQualifierSchema
- GraphFactCandidate with endpoint validation
- GraphExtractionBatch with bounded sizes
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class EntityType(StrEnum):
    """Allowlisted entity types in the core GraphRAG ontology."""

    PAPER = "Paper"
    AUTHOR = "Author"
    INSTITUTION = "Institution"
    TASK = "Task"
    METHOD = "Method"
    MODEL = "Model"
    DATASET = "Dataset"
    METRIC = "Metric"
    RESULT = "Result"
    CLAIM = "Claim"
    LIMITATION = "Limitation"
    CONCEPT = "Concept"


class RelationshipPredicate(StrEnum):
    """Allowlisted relationship predicates connecting graph entities."""

    EVALUATED_ON = "EVALUATED_ON"
    ACHIEVES_RESULT = "ACHIEVES_RESULT"
    PROPOSES_METHOD = "PROPOSES_METHOD"
    USES_MODEL = "USES_MODEL"
    CONTRADICTS = "CONTRADICTS"
    EXTENDS = "EXTENDS"
    AUTHORED_BY = "AUTHORED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"


class ClaimPolarity(StrEnum):
    """Polarity of a claim or result assertion."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNCERTAIN = "UNCERTAIN"


# Valid endpoint constraints mapping each predicate to allowlisted (subject, object) type pairs.
VALID_ENDPOINT_CONSTRAINTS: dict[RelationshipPredicate, set[tuple[EntityType, EntityType]]] = {
    RelationshipPredicate.PROPOSES_METHOD: {
        (EntityType.PAPER, EntityType.METHOD),
        (EntityType.PAPER, EntityType.CONCEPT),
    },
    RelationshipPredicate.USES_MODEL: {
        (EntityType.PAPER, EntityType.MODEL),
        (EntityType.PAPER, EntityType.CONCEPT),
    },
    RelationshipPredicate.EVALUATED_ON: {
        (EntityType.METHOD, EntityType.DATASET),
        (EntityType.MODEL, EntityType.DATASET),
        (EntityType.PAPER, EntityType.DATASET),
        (EntityType.METHOD, EntityType.TASK),
        (EntityType.MODEL, EntityType.TASK),
        (EntityType.RESULT, EntityType.DATASET),
    },
    RelationshipPredicate.ACHIEVES_RESULT: {
        (EntityType.METHOD, EntityType.RESULT),
        (EntityType.MODEL, EntityType.RESULT),
        (EntityType.PAPER, EntityType.RESULT),
    },
    RelationshipPredicate.CONTRADICTS: {
        (EntityType.CLAIM, EntityType.CLAIM),
        (EntityType.RESULT, EntityType.RESULT),
        (EntityType.CLAIM, EntityType.RESULT),
        (EntityType.RESULT, EntityType.CLAIM),
    },
    RelationshipPredicate.EXTENDS: {
        (EntityType.MODEL, EntityType.MODEL),
        (EntityType.MODEL, EntityType.METHOD),
        (EntityType.METHOD, EntityType.METHOD),
    },
    RelationshipPredicate.AUTHORED_BY: {
        (EntityType.PAPER, EntityType.AUTHOR),
    },
    RelationshipPredicate.AFFILIATED_WITH: {
        (EntityType.AUTHOR, EntityType.INSTITUTION),
    },
}


def normalize_decimal_spaces(value: str) -> str:
    """Normalize OCR/parser spacing around decimal points and numeric formats.

    Examples:
        '41 . 0' -> '41.0'
        '27 . 5 %' -> '27.5 %'
        '+ 0 . 6' -> '+0.6'
        '1 , 000 . 5' -> '1,000.5'
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    # Normalize spaces around decimal points between digits
    text = re.sub(r"(\d+)\s*\.\s*(\d+)", r"\1.\2", text)
    # Normalize spaces around commas between digits (thousands separators)
    text = re.sub(r"(\d+)\s*,\s*(\d+)", r"\1,\2", text)
    # Normalize spaces after leading +/- before a digit
    text = re.sub(r"^([+-])\s+(\d)", r"\1\2", text)
    return text


def parse_numeric_value(raw: Any) -> tuple[float | None, str | None]:
    """Parse and standardize numeric values from strings, floats, or ints.

    Returns:
        tuple[float | None, str | None]: (standardized_float, normalized_string)
    """
    if raw is None:
        return None, None
    if isinstance(raw, (int, float)):
        val = float(raw)
        return val, str(raw)
    if isinstance(raw, str):
        normalized = normalize_decimal_spaces(raw)
        try:
            cleaned = normalized.replace(",", "")
            return float(cleaned), normalized
        except ValueError:
            pass

        # Try matching leading numeric component (e.g. '41.0%' or '41.0 BLEU')
        match = re.match(r"^([+-]?\d+(?:\.\d+)?)\s*(?:%|[a-zA-Z]+)?$", normalized)
        if match:
            try:
                return float(match.group(1)), normalized
            except ValueError:
                pass
        return None, normalized
    return None, str(raw)


class GraphEntitySchema(BaseModel):
    """Schema for an entity node extracted from paper text."""

    id: str = Field(
        ..., min_length=1, max_length=200, description="Project-scoped entity identifier."
    )
    name: str = Field(..., min_length=1, max_length=200, description="Display name of the entity.")
    type: EntityType = Field(..., description="Ontological entity type.")
    description: str | None = Field(
        default=None, max_length=1000, description="Brief description if available."
    )
    aliases: list[str] = Field(
        default_factory=list, max_length=50, description="Alternative names or acronyms."
    )

    @field_validator("id", mode="before")
    @classmethod
    def validate_id(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Entity id cannot be empty or whitespace-only")
        return v

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: Any) -> Any:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("Entity name cannot be empty or whitespace-only")
        return v

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip()
            for member in EntityType:
                if (
                    member.value.lower() == v_clean.lower()
                    or member.name.lower() == v_clean.lower()
                ):
                    return member
        return v


class GraphProvenanceSchema(BaseModel):
    """Verifiable provenance anchor linking an extracted fact to raw paper text."""

    paper_id: UUID = Field(..., description="UUID of the parent paper.")
    chunk_id: UUID = Field(..., description="UUID of the source chunk.")
    page_number: int = Field(
        ..., ge=1, description="1-indexed page number where the quote appears."
    )
    element_id: UUID | None = Field(
        default=None, description="UUID of the specific paper element if known."
    )
    exact_quote: str = Field(
        ..., min_length=1, max_length=2000, description="Verbatim text quote from the paper."
    )
    char_start: int = Field(
        ..., ge=0, description="0-indexed character start offset within the source context."
    )
    char_end: int = Field(..., gt=0, description="0-indexed character end offset (exclusive).")
    document_sha256: str = Field(
        ...,
        min_length=64,
        max_length=64,
        pattern=r"^[a-fA-F0-9]{64}$",
        description="SHA-256 hash of the authoritative paper PDF.",
    )
    parser_version: str = Field(default="v1", description="Parser pipeline version.")

    @field_validator("document_sha256", mode="before")
    @classmethod
    def normalize_sha256(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @model_validator(mode="after")
    def validate_provenance(self) -> GraphProvenanceSchema:
        if not self.exact_quote or not self.exact_quote.strip():
            raise ValueError("exact_quote cannot be empty or whitespace-only")
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be strictly greater than "
                f"char_start ({self.char_start})"
            )
        span_len = self.char_end - self.char_start
        quote_len = len(self.exact_quote)
        if span_len != quote_len:
            raise ValueError(
                f"Character span mismatch: char_end - char_start is {span_len}, "
                f"but exact_quote length is {quote_len}"
            )
        return self


class GraphQualifierSchema(BaseModel):
    """Typed qualifiers characterizing a fact, measurement, or claim."""

    result_value: float | None = Field(
        default=None, description="Standardized numeric result value."
    )
    numeric_value: float | None = Field(
        default=None, description="Synchronized alias for result_value for manifest compatibility."
    )
    raw_value: str | None = Field(
        default=None, description="Original raw representation of the result value."
    )
    unit: str | None = Field(
        default=None, max_length=50, description="Measurement unit (e.g., '%', 'BLEU', 'ms')."
    )
    metric: str | None = Field(
        default=None,
        max_length=100,
        description="Evaluation metric name (e.g., 'BLEU-4', 'Accuracy').",
    )
    dataset: str | None = Field(default=None, max_length=200, description="Target dataset name.")
    split: str | None = Field(
        default=None, max_length=100, description="Dataset split (e.g., 'test', 'dev', 'val')."
    )
    task: str | None = Field(
        default=None, max_length=200, description="Task name (e.g., 'Machine Translation')."
    )
    comparison_condition: str | None = Field(
        default=None, max_length=500, description="Experimental condition or setting."
    )
    polarity: ClaimPolarity = Field(default=ClaimPolarity.POSITIVE, description="Claim polarity.")
    uncertainty: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Uncertainty score in [0.0, 1.0]."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_qualifier_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Sync numeric_value and result_value
            if "numeric_value" in data and "result_value" not in data:
                data["result_value"] = data["numeric_value"]
            elif "result_value" in data and "numeric_value" not in data:
                data["numeric_value"] = data["result_value"]

            raw_in = data.get("raw_value")
            res_in = data.get("result_value")

            if res_in is not None:
                parsed_float, norm_str = parse_numeric_value(res_in)
                data["result_value"] = parsed_float
                data["numeric_value"] = parsed_float
                if raw_in is None and norm_str is not None:
                    data["raw_value"] = norm_str
                elif raw_in is not None and isinstance(raw_in, str):
                    data["raw_value"] = normalize_decimal_spaces(raw_in)
            elif raw_in is not None:
                if isinstance(raw_in, str):
                    norm_str = normalize_decimal_spaces(raw_in)
                    data["raw_value"] = norm_str
                    parsed_float, _ = parse_numeric_value(norm_str)
                    if parsed_float is not None:
                        data["result_value"] = parsed_float
                        data["numeric_value"] = parsed_float

            # Strip whitespace on metadata strings
            for field in ("unit", "metric", "dataset", "split", "task", "comparison_condition"):
                if field in data and isinstance(data[field], str):
                    cleaned = data[field].strip()
                    data[field] = cleaned if cleaned else None

            # Normalize polarity strings
            if "polarity" in data and isinstance(data["polarity"], str):
                p_upper = data["polarity"].strip().upper()
                try:
                    data["polarity"] = ClaimPolarity(p_upper)
                except ValueError:
                    pass

        return data

    def is_comparable_with(self, other: GraphQualifierSchema) -> bool:
        """Check if two qualifiers can be meaningfully compared.

        Two qualifiers are comparable only when their contextual dimensions
        (task, dataset, split, metric, unit) are compatible and neither has a
        missing unit or metric when the other specifies one.
        """
        if not isinstance(other, GraphQualifierSchema):
            return False

        # Unit compatibility: if either has a unit, both must have it and they must match
        if (self.unit is None) != (other.unit is None):
            return False
        if self.unit is not None and other.unit is not None:
            if self.unit.strip().lower() != other.unit.strip().lower():
                return False

        # Metric compatibility: if either has a metric, both must have it and they must match
        if (self.metric is None) != (other.metric is None):
            return False
        if self.metric is not None and other.metric is not None:
            if self.metric.strip().lower() != other.metric.strip().lower():
                return False

        # Must have at least a metric or dataset to be comparable (unit alone is insufficient)
        if self.metric is None and self.dataset is None:
            return False

        # Dataset compatibility
        if (self.dataset is None) != (other.dataset is None):
            return False
        if self.dataset is not None and other.dataset is not None:
            if self.dataset.strip().lower() != other.dataset.strip().lower():
                return False

        # Split compatibility
        if (self.split is None) != (other.split is None):
            return False
        if self.split is not None and other.split is not None:
            if self.split.strip().lower() != other.split.strip().lower():
                return False

        # Task compatibility
        if (self.task is None) != (other.task is None):
            return False
        if self.task is not None and other.task is not None:
            if self.task.strip().lower() != other.task.strip().lower():
                return False

        return True

    def is_contradiction_with(self, other: GraphQualifierSchema, tolerance: float = 1e-4) -> bool:
        """Check if two qualifiers contradict each other.

        Must be comparable first. A contradiction occurs if:
        1. Both share the same experimental condition (or both are unconditioned).
        2. Under that condition:
           a. Polarities oppose (e.g. POSITIVE vs NEGATIVE)
           b. Numeric result values differ beyond tolerance
        """
        if not self.is_comparable_with(other):
            return False

        # Must evaluate under compatible experimental conditions
        same_condition = (
            self.comparison_condition is None and other.comparison_condition is None
        ) or (
            self.comparison_condition is not None
            and other.comparison_condition is not None
            and self.comparison_condition.strip().lower()
            == other.comparison_condition.strip().lower()
        )
        if not same_condition:
            return False

        # Opposing polarities under same condition
        if (
            self.polarity == ClaimPolarity.POSITIVE and other.polarity == ClaimPolarity.NEGATIVE
        ) or (self.polarity == ClaimPolarity.NEGATIVE and other.polarity == ClaimPolarity.POSITIVE):
            return True

        # Differing numeric values under same condition
        if self.result_value is not None and other.result_value is not None:
            if abs(self.result_value - other.result_value) > tolerance:
                return True

        return False


class GraphFactCandidate(BaseModel):
    """Candidate relationship extracted between two entities, backed by provenance."""

    subject: GraphEntitySchema = Field(..., description="Subject entity.")
    predicate: RelationshipPredicate = Field(..., description="Relationship predicate.")
    object: GraphEntitySchema = Field(..., description="Object entity.")
    qualifiers: GraphQualifierSchema | None = Field(
        default=None, description="Optional measurement qualifiers."
    )
    provenance: GraphProvenanceSchema = Field(
        ..., description="Verifiable source quote and coordinates."
    )

    @field_validator("predicate", mode="before")
    @classmethod
    def validate_predicate(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_clean = v.strip().upper()
            if v_clean in RelationshipPredicate.__members__:
                return RelationshipPredicate[v_clean]
        return v

    @model_validator(mode="after")
    def validate_endpoints(self) -> GraphFactCandidate:
        allowed_pairs = VALID_ENDPOINT_CONSTRAINTS.get(self.predicate)
        if allowed_pairs is None:
            raise ValueError(f"Unknown predicate '{self.predicate}'")

        endpoint_pair = (self.subject.type, self.object.type)
        if endpoint_pair not in allowed_pairs:
            allowed_repr = ", ".join(
                f"({s.value}, {o.value})"
                for s, o in sorted(allowed_pairs, key=lambda p: (p[0].value, p[1].value))
            )
            raise ValueError(
                f"Invalid endpoint types for predicate '{self.predicate.value}': "
                f"({self.subject.type.value}, {self.object.type.value}) is not permitted. "
                f"Allowed pairs: [{allowed_repr}]"
            )
        return self


class GraphExtractionBatch(BaseModel):
    """Bounded batch of entities and fact candidates extracted from a paper."""

    project_id: UUID = Field(..., description="Project UUID.")
    paper_id: UUID = Field(..., description="Parent paper UUID.")
    entities: list[GraphEntitySchema] = Field(
        default_factory=list,
        max_length=100,
        description="Extracted entities (max 100).",
    )
    facts: list[GraphFactCandidate] = Field(
        default_factory=list,
        max_length=100,
        description="Extracted fact candidates (max 100).",
    )
