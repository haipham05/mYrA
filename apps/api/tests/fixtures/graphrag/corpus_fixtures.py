"""Synthetic GraphRAG test corpus fixtures.

Provides realistic, provenance-grounded fixtures for:
- Project A: NLP Calibration (Paper A1, Paper A2, optional Paper A3)
- Project B: Speech Recognition (Paper B1, Paper B2)

Covers:
1. Method -> Dataset relations (AURC on ImageNet, Conformer on LibriSpeech)
2. Cross-project shared acronyms with distinct semantics (ASR, ECE)
3. Comparable opposing claims (Temperature scaling ECE 2.1% vs 8.5% on ImageNet)
4. Non-contradicting different-dataset claims (ECE 2.1% on ImageNet vs 5.4% on CIFAR-100)
5. Model extension relations (Branchformer extends Conformer)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session

from app.db.models import (
    ChunkElement,
    Paper,
    PaperChunk,
    PaperElement,
    PaperPage,
    Project,
)
from app.ingestion.parser import find_verbatim_span

ONTOLOGY_VERSION = "1.0.0"

ALLOWED_ENTITIES: list[str] = [
    "Paper",
    "Author",
    "Institution",
    "Task",
    "Method",
    "Model",
    "Dataset",
    "Metric",
    "Result",
    "Claim",
    "Limitation",
    "Concept",
]

ALLOWED_PREDICATES: list[str] = [
    "EVALUATED_ON",
    "ACHIEVES_RESULT",
    "PROPOSES_METHOD",
    "USES_MODEL",
    "CONTRADICTS",
    "EXTENDS",
    "AUTHORED_BY",
    "AFFILIATED_WITH",
]

ALLOWED_QUALIFIERS: list[str] = [
    "dataset",
    "split",
    "metric",
    "unit",
    "numeric_value",
    "polarity",
    "comparison_condition",
    "task",
    "uncertainty",
]

# Stable Deterministic IDs
PROJECT_A_ID = UUID("a0000000-0000-0000-0000-000000000001")
PROJECT_B_ID = UUID("b0000000-0000-0000-0000-000000000002")

PAPER_A1_ID = UUID("a1111111-1111-1111-1111-111111111111")
PAPER_A2_ID = UUID("a2222222-2222-2222-2222-222222222222")
PAPER_A3_ID = UUID("a3333333-3333-3333-3333-333333333333")

PAPER_B1_ID = UUID("b1111111-1111-1111-1111-111111111111")
PAPER_B2_ID = UUID("b2222222-2222-2222-2222-222222222222")


# Pydantic Manifest Models
class ManifestEntity(BaseModel):
    id: str
    name: str
    type: str
    acronym: str | None = None
    expansion: str | None = None


class ManifestProvenance(BaseModel):
    paper_id: UUID
    page_number: int
    chunk_id: UUID
    element_id: UUID
    exact_quote: str
    char_start: int
    char_end: int

    @model_validator(mode="after")
    def validate_provenance(self) -> ManifestProvenance:
        if not self.exact_quote or not self.exact_quote.strip():
            raise ValueError("Provenance exact_quote cannot be empty")
        if self.page_number < 1:
            raise ValueError("page_number must be >= 1")
        if self.char_start < 0:
            raise ValueError("char_start must be >= 0")
        if self.char_end <= self.char_start:
            raise ValueError(f"char_end ({self.char_end}) must be > char_start ({self.char_start})")
        if self.char_end - self.char_start != len(self.exact_quote):
            raise ValueError(
                f"char span length ({self.char_end - self.char_start}) does not match "
                f"quote length ({len(self.exact_quote)})"
            )
        return self


class ManifestRelationship(BaseModel):
    fact_id: str
    subject: ManifestEntity
    predicate: str
    object: ManifestEntity
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    provenance: ManifestProvenance


class ManifestPaper(BaseModel):
    id: UUID
    filename: str
    title: str
    document_sha256: str
    entities: list[ManifestEntity] = Field(default_factory=list)
    relationships: list[ManifestRelationship] = Field(default_factory=list)


class ManifestProject(BaseModel):
    id: UUID
    name: str
    description: str
    papers: list[ManifestPaper] = Field(default_factory=list)


class QualityGates(BaseModel):
    local_checks: list[str]
    coverage_threshold: float
    cloud_checks: dict[str, Any]


class GraphRagManifest(BaseModel):
    ontology_version: str
    allowed_entities: list[str]
    allowed_predicates: list[str]
    allowed_qualifiers: list[str]
    quality_gates: QualityGates
    projects: dict[str, ManifestProject]
    cross_paper_benchmarks: dict[str, Any]

    @model_validator(mode="after")
    def validate_ontology_and_predicates(self) -> GraphRagManifest:
        if self.ontology_version != ONTOLOGY_VERSION:
            raise ValueError(
                f"Unsupported ontology_version: '{self.ontology_version}'. "
                f"Expected '{ONTOLOGY_VERSION}'"
            )

        for ent in ALLOWED_ENTITIES:
            if ent not in self.allowed_entities:
                raise ValueError(f"Missing required allowed entity: '{ent}'")
        for pred in ALLOWED_PREDICATES:
            if pred not in self.allowed_predicates:
                raise ValueError(f"Missing required allowed predicate: '{pred}'")

        for proj in self.projects.values():
            for paper in proj.papers:
                for entity in paper.entities:
                    if entity.type not in self.allowed_entities:
                        raise ValueError(
                            f"Entity '{entity.id}' in paper '{paper.id}' has "
                            f"invalid type '{entity.type}'. Allowed: {self.allowed_entities}"
                        )
                for rel in paper.relationships:
                    if rel.predicate not in self.allowed_predicates:
                        raise ValueError(
                            f"Relationship '{rel.fact_id}' in paper '{paper.id}' has "
                            f"invalid predicate '{rel.predicate}'. "
                            f"Allowed: {self.allowed_predicates}"
                        )
                    if rel.subject.type not in self.allowed_entities:
                        raise ValueError(
                            f"Subject of '{rel.fact_id}' has invalid type: '{rel.subject.type}'"
                        )
                    if rel.object.type not in self.allowed_entities:
                        raise ValueError(
                            f"Object of '{rel.fact_id}' has invalid type: '{rel.object.type}'"
                        )
                    for q in rel.qualifiers:
                        if q not in self.allowed_qualifiers:
                            raise ValueError(
                                f"Relationship '{rel.fact_id}' in paper '{paper.id}' has "
                                f"invalid qualifier '{q}'. "
                                f"Allowed: {self.allowed_qualifiers}"
                            )
        return self


# Detailed Document Definitions
SYNTHETIC_DATA: dict[str, Any] = {
    "project_a": {
        "id": PROJECT_A_ID,
        "name": "NLP Calibration",
        "description": (
            "Investigation of confidence calibration, expected calibration error (ECE), "
            "and adversarial attack robustness in language and vision representations."
        ),
        "papers": {
            "paper_a1": {
                "id": PAPER_A1_ID,
                "project_id": PROJECT_A_ID,
                "filename": "calibration_neural_networks.pdf",
                "storage_path": (
                    f"projects/{PROJECT_A_ID}/papers/{PAPER_A1_ID}/calibration_neural_networks.pdf"
                ),
                "document_sha256": "a1" * 32,
                "status": "READY",
                "page_count": 2,
                "title": "On Calibration of Modern Neural Networks",
                "authors": ["Chuan Guo", "Geoff Pleiss"],
                "institutions": ["Cornell University"],
                "pages": [
                    {
                        "id": UUID("a1111111-1111-1111-1111-000000000001"),
                        "page_number": 1,
                        "width": 612.0,
                        "height": 792.0,
                        "rotation": 0,
                        "crop_box": {"x": 0.0, "y": 0.0, "w": 612.0, "h": 792.0},
                        "raw_text": (
                            "On Calibration of Modern Neural Networks\n\n"
                            "Confidence calibration is essential for reliable decision "
                            "making in deep learning systems. "
                            "Expected Calibration Error (ECE) is measured across all "
                            "benchmarks to assess post-processing probability alignment. "
                            "In this work, we analyze post-processing calibration "
                            "techniques across NLP and vision domains. "
                            "Adversarial perturbations degrade calibration without "
                            "reducing the attack success rate (ASR).\n\n"
                            "1. Introduction\n\n"
                            "Modern deep neural networks are prone to overconfident "
                            "misclassifications. We investigate post-hoc calibration "
                            "strategies to mitigate probability distortion."
                        ),
                        "elements": [
                            {
                                "id": UUID("a1111111-1111-1111-1111-e00000000001"),
                                "element_index": 0,
                                "element_type": "title",
                                "text": "On Calibration of Modern Neural Networks",
                                "bbox": (72.0, 72.0, 540.0, 100.0),
                                "section_path": [],
                            },
                            {
                                "id": UUID("a1111111-1111-1111-1111-e00000000002"),
                                "element_index": 1,
                                "element_type": "abstract",
                                "text": (
                                    "Confidence calibration is essential for reliable decision "
                                    "making in deep learning systems. "
                                    "Expected Calibration Error (ECE) is measured across all "
                                    "benchmarks to assess post-processing probability alignment. "
                                    "In this work, we analyze post-processing calibration "
                                    "techniques across NLP and vision domains. "
                                    "Adversarial perturbations degrade calibration without "
                                    "reducing the attack success rate (ASR)."
                                ),
                                "bbox": (72.0, 110.0, 540.0, 220.0),
                                "section_path": ["Abstract"],
                            },
                            {
                                "id": UUID("a1111111-1111-1111-1111-e00000000003"),
                                "element_index": 2,
                                "element_type": "heading",
                                "text": "1. Introduction",
                                "bbox": (72.0, 240.0, 540.0, 260.0),
                                "section_path": ["1. Introduction"],
                            },
                            {
                                "id": UUID("a1111111-1111-1111-1111-e00000000004"),
                                "element_index": 3,
                                "element_type": "paragraph",
                                "text": (
                                    "Modern deep neural networks are prone to overconfident "
                                    "misclassifications. We investigate post-hoc calibration "
                                    "strategies to mitigate probability distortion."
                                ),
                                "bbox": (72.0, 270.0, 540.0, 360.0),
                                "section_path": ["1. Introduction"],
                            },
                        ],
                    },
                    {
                        "id": UUID("a1111111-1111-1111-1111-000000000002"),
                        "page_number": 2,
                        "width": 612.0,
                        "height": 792.0,
                        "rotation": 0,
                        "crop_box": {"x": 0.0, "y": 0.0, "w": 612.0, "h": 792.0},
                        "raw_text": (
                            "2. Experiments and Benchmarks\n\n"
                            "We evaluate AURC on ImageNet calibration. "
                            "Across standard validation protocols, temperature scaling "
                            "demonstrates consistent probability recalibration.\n\n"
                            "3. Results\n\n"
                            "Temperature scaling achieves an ECE of 2.1% on ImageNet. "
                            "This confirms that simple single-parameter scaling effectively "
                            "mitigates overconfidence."
                        ),
                        "elements": [
                            {
                                "id": UUID("a1111111-1111-1111-1111-e00000000005"),
                                "element_index": 4,
                                "element_type": "heading",
                                "text": "2. Experiments and Benchmarks",
                                "bbox": (72.0, 72.0, 540.0, 95.0),
                                "section_path": ["2. Experiments and Benchmarks"],
                            },
                            {
                                "id": UUID("a1111111-1111-1111-1111-e00000000006"),
                                "element_index": 5,
                                "element_type": "paragraph",
                                "text": (
                                    "We evaluate AURC on ImageNet calibration. "
                                    "Across standard validation protocols, temperature scaling "
                                    "demonstrates consistent probability recalibration."
                                ),
                                "bbox": (72.0, 105.0, 540.0, 180.0),
                                "section_path": ["2. Experiments and Benchmarks"],
                            },
                            {
                                "id": UUID("a1111111-1111-1111-1111-e00000000007"),
                                "element_index": 6,
                                "element_type": "heading",
                                "text": "3. Results",
                                "bbox": (72.0, 200.0, 540.0, 220.0),
                                "section_path": ["3. Results"],
                            },
                            {
                                "id": UUID("a1111111-1111-1111-1111-e00000000008"),
                                "element_index": 7,
                                "element_type": "paragraph",
                                "text": (
                                    "Temperature scaling achieves an ECE of 2.1% on ImageNet. "
                                    "This confirms that simple single-parameter scaling "
                                    "effectively mitigates overconfidence."
                                ),
                                "bbox": (72.0, 230.0, 540.0, 310.0),
                                "section_path": ["3. Results"],
                            },
                        ],
                    },
                ],
                "chunks": [
                    {
                        "id": UUID("a1111111-1111-1111-1111-c00000000001"),
                        "chunk_type": "child",
                        "chunk_index": 0,
                        "page_number": 1,
                        "text": (
                            "On Calibration of Modern Neural Networks. "
                            "Confidence calibration is essential for reliable decision "
                            "making in deep learning systems. "
                            "Expected Calibration Error (ECE) is measured across all "
                            "benchmarks to assess post-processing probability alignment. "
                            "In this work, we analyze post-processing calibration "
                            "techniques across NLP and vision domains. "
                            "Adversarial perturbations degrade calibration without "
                            "reducing the attack success rate (ASR)."
                        ),
                        "element_ids": [
                            UUID("a1111111-1111-1111-1111-e00000000001"),
                            UUID("a1111111-1111-1111-1111-e00000000002"),
                        ],
                    },
                    {
                        "id": UUID("a1111111-1111-1111-1111-c00000000002"),
                        "chunk_type": "child",
                        "chunk_index": 1,
                        "page_number": 2,
                        "text": (
                            "We evaluate AURC on ImageNet calibration. "
                            "Across standard validation protocols, temperature scaling "
                            "demonstrates consistent probability recalibration."
                        ),
                        "element_ids": [
                            UUID("a1111111-1111-1111-1111-e00000000005"),
                            UUID("a1111111-1111-1111-1111-e00000000006"),
                        ],
                    },
                    {
                        "id": UUID("a1111111-1111-1111-1111-c00000000003"),
                        "chunk_type": "child",
                        "chunk_index": 2,
                        "page_number": 2,
                        "text": (
                            "Temperature scaling achieves an ECE of 2.1% on ImageNet. "
                            "This confirms that simple single-parameter scaling effectively "
                            "mitigates overconfidence."
                        ),
                        "element_ids": [
                            UUID("a1111111-1111-1111-1111-e00000000007"),
                            UUID("a1111111-1111-1111-1111-e00000000008"),
                        ],
                    },
                ],
            },
            "paper_a2": {
                "id": PAPER_A2_ID,
                "project_id": PROJECT_A_ID,
                "filename": "limitations_temperature_scaling.pdf",
                "storage_path": (
                    f"projects/{PROJECT_A_ID}/papers/{PAPER_A2_ID}/"
                    "limitations_temperature_scaling.pdf"
                ),
                "document_sha256": "a2" * 32,
                "status": "READY",
                "page_count": 2,
                "title": ("Limitations of Temperature Scaling under Severe Distribution Shift"),
                "authors": ["Dan Hendrycks", "Thomas Dietterich"],
                "institutions": ["UC Berkeley", "Oregon State University"],
                "pages": [
                    {
                        "id": UUID("a2222222-2222-2222-2222-000000000001"),
                        "page_number": 1,
                        "width": 612.0,
                        "height": 792.0,
                        "rotation": 0,
                        "crop_box": {"x": 0.0, "y": 0.0, "w": 612.0, "h": 792.0},
                        "raw_text": (
                            "Limitations of Temperature Scaling under Severe "
                            "Distribution Shift\n\n"
                            "While temperature scaling is widely adopted for probability "
                            "calibration, its stability under domain shifts remains "
                            "questionable. We evaluate the attack success rate (ASR) "
                            "across adversarial validation splits to quantify "
                            "vulnerability. Empirical evaluations show marked "
                            "divergence under complex distributions.\n\n"
                            "1. Distributional Shift Analysis\n\n"
                            "We investigate parametric scaling methods under "
                            "non-stationary distributions where assumptions of uniform "
                            "confidence fail."
                        ),
                        "elements": [
                            {
                                "id": UUID("a2222222-2222-2222-2222-e00000000001"),
                                "element_index": 0,
                                "element_type": "title",
                                "text": (
                                    "Limitations of Temperature Scaling under "
                                    "Severe Distribution Shift"
                                ),
                                "bbox": (72.0, 72.0, 540.0, 100.0),
                                "section_path": [],
                            },
                            {
                                "id": UUID("a2222222-2222-2222-2222-e00000000002"),
                                "element_index": 1,
                                "element_type": "abstract",
                                "text": (
                                    "While temperature scaling is widely adopted for "
                                    "probability calibration, its stability under domain "
                                    "shifts remains questionable. We evaluate the attack "
                                    "success rate (ASR) across adversarial validation splits "
                                    "to quantify vulnerability. Empirical evaluations show "
                                    "marked divergence under complex distributions."
                                ),
                                "bbox": (72.0, 110.0, 540.0, 220.0),
                                "section_path": ["Abstract"],
                            },
                            {
                                "id": UUID("a2222222-2222-2222-2222-e00000000003"),
                                "element_index": 2,
                                "element_type": "heading",
                                "text": "1. Distributional Shift Analysis",
                                "bbox": (72.0, 240.0, 540.0, 260.0),
                                "section_path": ["1. Distributional Shift Analysis"],
                            },
                            {
                                "id": UUID("a2222222-2222-2222-2222-e00000000004"),
                                "element_index": 3,
                                "element_type": "paragraph",
                                "text": (
                                    "We investigate parametric scaling methods under "
                                    "non-stationary distributions where assumptions of "
                                    "uniform confidence fail."
                                ),
                                "bbox": (72.0, 270.0, 540.0, 360.0),
                                "section_path": ["1. Distributional Shift Analysis"],
                            },
                        ],
                    },
                    {
                        "id": UUID("a2222222-2222-2222-2222-000000000002"),
                        "page_number": 2,
                        "width": 612.0,
                        "height": 792.0,
                        "rotation": 0,
                        "crop_box": {"x": 0.0, "y": 0.0, "w": 612.0, "h": 792.0},
                        "raw_text": (
                            "2. Calibration Experiments\n\n"
                            "Temperature scaling fails to converge, yielding an ECE of "
                            "8.5% on ImageNet. Under unconstrained optimization, the "
                            "learned temperature parameter overfits severely.\n\n"
                            "3. Cross-Domain Transfer\n\n"
                            "Temperature scaling achieves an ECE of 5.4% on CIFAR-100. "
                            "This demonstrates that performance differs substantially "
                            "across distinct datasets."
                        ),
                        "elements": [
                            {
                                "id": UUID("a2222222-2222-2222-2222-e00000000005"),
                                "element_index": 4,
                                "element_type": "heading",
                                "text": "2. Calibration Experiments",
                                "bbox": (72.0, 72.0, 540.0, 95.0),
                                "section_path": ["2. Calibration Experiments"],
                            },
                            {
                                "id": UUID("a2222222-2222-2222-2222-e00000000006"),
                                "element_index": 5,
                                "element_type": "paragraph",
                                "text": (
                                    "Temperature scaling fails to converge, yielding an ECE "
                                    "of 8.5% on ImageNet. Under unconstrained optimization, "
                                    "the learned temperature parameter overfits severely."
                                ),
                                "bbox": (72.0, 105.0, 540.0, 180.0),
                                "section_path": ["2. Calibration Experiments"],
                            },
                            {
                                "id": UUID("a2222222-2222-2222-2222-e00000000007"),
                                "element_index": 6,
                                "element_type": "heading",
                                "text": "3. Cross-Domain Transfer",
                                "bbox": (72.0, 200.0, 540.0, 220.0),
                                "section_path": ["3. Cross-Domain Transfer"],
                            },
                            {
                                "id": UUID("a2222222-2222-2222-2222-e00000000008"),
                                "element_index": 7,
                                "element_type": "paragraph",
                                "text": (
                                    "Temperature scaling achieves an ECE of 5.4% on CIFAR-100. "
                                    "This demonstrates that performance differs substantially "
                                    "across distinct datasets."
                                ),
                                "bbox": (72.0, 230.0, 540.0, 310.0),
                                "section_path": ["3. Cross-Domain Transfer"],
                            },
                        ],
                    },
                ],
                "chunks": [
                    {
                        "id": UUID("a2222222-2222-2222-2222-c00000000001"),
                        "chunk_type": "child",
                        "chunk_index": 0,
                        "page_number": 1,
                        "text": (
                            "Limitations of Temperature Scaling under Severe "
                            "Distribution Shift. While temperature scaling is widely "
                            "adopted for probability calibration, its stability under "
                            "domain shifts remains questionable. We evaluate the attack "
                            "success rate (ASR) across adversarial validation splits to "
                            "quantify vulnerability."
                        ),
                        "element_ids": [
                            UUID("a2222222-2222-2222-2222-e00000000001"),
                            UUID("a2222222-2222-2222-2222-e00000000002"),
                        ],
                    },
                    {
                        "id": UUID("a2222222-2222-2222-2222-c00000000002"),
                        "chunk_type": "child",
                        "chunk_index": 1,
                        "page_number": 2,
                        "text": (
                            "Temperature scaling fails to converge, yielding an ECE of "
                            "8.5% on ImageNet. Under unconstrained optimization, the "
                            "learned temperature parameter overfits severely."
                        ),
                        "element_ids": [
                            UUID("a2222222-2222-2222-2222-e00000000005"),
                            UUID("a2222222-2222-2222-2222-e00000000006"),
                        ],
                    },
                    {
                        "id": UUID("a2222222-2222-2222-2222-c00000000003"),
                        "chunk_type": "child",
                        "chunk_index": 2,
                        "page_number": 2,
                        "text": (
                            "Temperature scaling achieves an ECE of 5.4% on CIFAR-100. "
                            "This demonstrates that performance differs substantially "
                            "across distinct datasets."
                        ),
                        "element_ids": [
                            UUID("a2222222-2222-2222-2222-e00000000007"),
                            UUID("a2222222-2222-2222-2222-e00000000008"),
                        ],
                    },
                ],
            },
            "paper_a3": {
                "id": PAPER_A3_ID,
                "project_id": PROJECT_A_ID,
                "filename": "cifar_calibration_benchmarks.pdf",
                "storage_path": (
                    f"projects/{PROJECT_A_ID}/papers/{PAPER_A3_ID}/cifar_calibration_benchmarks.pdf"
                ),
                "document_sha256": "a3" * 32,
                "status": "READY",
                "page_count": 1,
                "title": "Calibration Benchmarks on Vision and Language Datasets",
                "authors": ["Dan Hendrycks"],
                "institutions": ["UC Berkeley"],
                "pages": [
                    {
                        "id": UUID("a3333333-3333-3333-3333-000000000001"),
                        "page_number": 1,
                        "width": 612.0,
                        "height": 792.0,
                        "rotation": 0,
                        "crop_box": {"x": 0.0, "y": 0.0, "w": 612.0, "h": 792.0},
                        "raw_text": (
                            "Calibration Benchmarks on Vision and Language Datasets\n\n"
                            "We provide extensive benchmark numbers for post-hoc "
                            "confidence calibration across diverse datasets. "
                            "Temperature scaling achieves an ECE of 5.4% on CIFAR-100."
                        ),
                        "elements": [
                            {
                                "id": UUID("a3333333-3333-3333-3333-e00000000001"),
                                "element_index": 0,
                                "element_type": "title",
                                "text": ("Calibration Benchmarks on Vision and Language Datasets"),
                                "bbox": (72.0, 72.0, 540.0, 100.0),
                                "section_path": [],
                            },
                            {
                                "id": UUID("a3333333-3333-3333-3333-e00000000002"),
                                "element_index": 1,
                                "element_type": "paragraph",
                                "text": (
                                    "We provide extensive benchmark numbers for post-hoc "
                                    "confidence calibration across diverse datasets. "
                                    "Temperature scaling achieves an ECE of 5.4% on CIFAR-100."
                                ),
                                "bbox": (72.0, 110.0, 540.0, 220.0),
                                "section_path": ["Results"],
                            },
                        ],
                    }
                ],
                "chunks": [
                    {
                        "id": UUID("a3333333-3333-3333-3333-c00000000001"),
                        "chunk_type": "child",
                        "chunk_index": 0,
                        "page_number": 1,
                        "text": (
                            "Calibration Benchmarks on Vision and Language Datasets. "
                            "We provide extensive benchmark numbers for post-hoc "
                            "confidence calibration across diverse datasets. "
                            "Temperature scaling achieves an ECE of 5.4% on CIFAR-100."
                        ),
                        "element_ids": [
                            UUID("a3333333-3333-3333-3333-e00000000001"),
                            UUID("a3333333-3333-3333-3333-e00000000002"),
                        ],
                    }
                ],
            },
        },
    },
    "project_b": {
        "id": PROJECT_B_ID,
        "name": "Speech Recognition",
        "description": (
            "End-to-end automatic speech recognition (ASR) with conformer and branchformer "
            "architectures, energy-based confidence estimation, and LibriSpeech benchmarking."
        ),
        "papers": {
            "paper_b1": {
                "id": PAPER_B1_ID,
                "project_id": PROJECT_B_ID,
                "filename": "conformer_speech_recognition.pdf",
                "storage_path": (
                    f"projects/{PROJECT_B_ID}/papers/{PAPER_B1_ID}/conformer_speech_recognition.pdf"
                ),
                "document_sha256": "b1" * 32,
                "status": "READY",
                "page_count": 2,
                "title": ("Conformer: Convolution-augmented Transformer for Speech Recognition"),
                "authors": ["Anmol Gulati", "James Qin"],
                "institutions": ["Google LLC"],
                "pages": [
                    {
                        "id": UUID("b1111111-1111-1111-1111-000000000001"),
                        "page_number": 1,
                        "width": 612.0,
                        "height": 792.0,
                        "rotation": 0,
                        "crop_box": {"x": 0.0, "y": 0.0, "w": 612.0, "h": 792.0},
                        "raw_text": (
                            "Conformer: Convolution-augmented Transformer for "
                            "Speech Recognition\n\n"
                            "Automatic speech recognition (ASR) models have seen dramatic "
                            "improvements with Transformer and CNN architectures. "
                            "Energy-based Confidence Estimation (ECE) is applied to prune "
                            "acoustic hypothesis search in beam decoding. "
                            "In this work, we combine self-attention with convolution.\n\n"
                            "1. Model Overview\n\n"
                            "The Conformer architecture captures both local and global "
                            "dependencies by interleaving depthwise separable convolutions "
                            "with multi-head self-attention."
                        ),
                        "elements": [
                            {
                                "id": UUID("b1111111-1111-1111-1111-e00000000001"),
                                "element_index": 0,
                                "element_type": "title",
                                "text": (
                                    "Conformer: Convolution-augmented Transformer for "
                                    "Speech Recognition"
                                ),
                                "bbox": (72.0, 72.0, 540.0, 100.0),
                                "section_path": [],
                            },
                            {
                                "id": UUID("b1111111-1111-1111-1111-e00000000002"),
                                "element_index": 1,
                                "element_type": "abstract",
                                "text": (
                                    "Automatic speech recognition (ASR) models have seen "
                                    "dramatic improvements with Transformer and CNN "
                                    "architectures. Energy-based Confidence Estimation (ECE) "
                                    "is applied to prune acoustic hypothesis search in beam "
                                    "decoding. In this work, we combine self-attention with "
                                    "convolution."
                                ),
                                "bbox": (72.0, 110.0, 540.0, 220.0),
                                "section_path": ["Abstract"],
                            },
                            {
                                "id": UUID("b1111111-1111-1111-1111-e00000000003"),
                                "element_index": 2,
                                "element_type": "heading",
                                "text": "1. Model Overview",
                                "bbox": (72.0, 240.0, 540.0, 260.0),
                                "section_path": ["1. Model Overview"],
                            },
                            {
                                "id": UUID("b1111111-1111-1111-1111-e00000000004"),
                                "element_index": 3,
                                "element_type": "paragraph",
                                "text": (
                                    "The Conformer architecture captures both local and global "
                                    "dependencies by interleaving depthwise separable "
                                    "convolutions with multi-head self-attention."
                                ),
                                "bbox": (72.0, 270.0, 540.0, 360.0),
                                "section_path": ["1. Model Overview"],
                            },
                        ],
                    },
                    {
                        "id": UUID("b1111111-1111-1111-1111-000000000002"),
                        "page_number": 2,
                        "width": 612.0,
                        "height": 792.0,
                        "rotation": 0,
                        "crop_box": {"x": 0.0, "y": 0.0, "w": 612.0, "h": 792.0},
                        "raw_text": (
                            "2. Speech Experiments\n\n"
                            "We evaluate Conformer on LibriSpeech. "
                            "Training utilizes 80-channel log-mel filterbank energies with "
                            "SpecAugment data augmentation.\n\n"
                            "3. Recognition Results\n\n"
                            "Conformer achieves a WER of 4.3% on LibriSpeech test-clean. "
                            "This outperforms prior Transformer baselines by a "
                            "significant margin."
                        ),
                        "elements": [
                            {
                                "id": UUID("b1111111-1111-1111-1111-e00000000005"),
                                "element_index": 4,
                                "element_type": "heading",
                                "text": "2. Speech Experiments",
                                "bbox": (72.0, 72.0, 540.0, 95.0),
                                "section_path": ["2. Speech Experiments"],
                            },
                            {
                                "id": UUID("b1111111-1111-1111-1111-e00000000006"),
                                "element_index": 5,
                                "element_type": "paragraph",
                                "text": (
                                    "We evaluate Conformer on LibriSpeech. "
                                    "Training utilizes 80-channel log-mel filterbank "
                                    "energies with SpecAugment data augmentation."
                                ),
                                "bbox": (72.0, 105.0, 540.0, 180.0),
                                "section_path": ["2. Speech Experiments"],
                            },
                            {
                                "id": UUID("b1111111-1111-1111-1111-e00000000007"),
                                "element_index": 6,
                                "element_type": "heading",
                                "text": "3. Recognition Results",
                                "bbox": (72.0, 200.0, 540.0, 220.0),
                                "section_path": ["3. Recognition Results"],
                            },
                            {
                                "id": UUID("b1111111-1111-1111-1111-e00000000008"),
                                "element_index": 7,
                                "element_type": "paragraph",
                                "text": (
                                    "Conformer achieves a WER of 4.3% on LibriSpeech "
                                    "test-clean. This outperforms prior Transformer baselines "
                                    "by a significant margin."
                                ),
                                "bbox": (72.0, 230.0, 540.0, 310.0),
                                "section_path": ["3. Recognition Results"],
                            },
                        ],
                    },
                ],
                "chunks": [
                    {
                        "id": UUID("b1111111-1111-1111-1111-c00000000001"),
                        "chunk_type": "child",
                        "chunk_index": 0,
                        "page_number": 1,
                        "text": (
                            "Conformer: Convolution-augmented Transformer for Speech "
                            "Recognition. Automatic speech recognition (ASR) models have "
                            "seen dramatic improvements with Transformer and CNN "
                            "architectures. Energy-based Confidence Estimation (ECE) is "
                            "applied to prune acoustic hypothesis search in beam decoding."
                        ),
                        "element_ids": [
                            UUID("b1111111-1111-1111-1111-e00000000001"),
                            UUID("b1111111-1111-1111-1111-e00000000002"),
                        ],
                    },
                    {
                        "id": UUID("b1111111-1111-1111-1111-c00000000002"),
                        "chunk_type": "child",
                        "chunk_index": 1,
                        "page_number": 2,
                        "text": (
                            "We evaluate Conformer on LibriSpeech. "
                            "Training utilizes 80-channel log-mel filterbank energies with "
                            "SpecAugment data augmentation."
                        ),
                        "element_ids": [
                            UUID("b1111111-1111-1111-1111-e00000000005"),
                            UUID("b1111111-1111-1111-1111-e00000000006"),
                        ],
                    },
                    {
                        "id": UUID("b1111111-1111-1111-1111-c00000000003"),
                        "chunk_type": "child",
                        "chunk_index": 2,
                        "page_number": 2,
                        "text": (
                            "Conformer achieves a WER of 4.3% on LibriSpeech test-clean. "
                            "This outperforms prior Transformer baselines by a "
                            "significant margin."
                        ),
                        "element_ids": [
                            UUID("b1111111-1111-1111-1111-e00000000007"),
                            UUID("b1111111-1111-1111-1111-e00000000008"),
                        ],
                    },
                ],
            },
            "paper_b2": {
                "id": PAPER_B2_ID,
                "project_id": PROJECT_B_ID,
                "filename": "branchformer_speech_recognition.pdf",
                "storage_path": (
                    f"projects/{PROJECT_B_ID}/papers/{PAPER_B2_ID}/"
                    "branchformer_speech_recognition.pdf"
                ),
                "document_sha256": "b2" * 32,
                "status": "READY",
                "page_count": 2,
                "title": (
                    "Branchformer: Parallel MLP-Attention Architectures for Speech Recognition"
                ),
                "authors": ["Yifan Peng", "Shinji Watanabe"],
                "institutions": ["Carnegie Mellon University"],
                "pages": [
                    {
                        "id": UUID("b2222222-2222-2222-2222-000000000001"),
                        "page_number": 1,
                        "width": 612.0,
                        "height": 792.0,
                        "rotation": 0,
                        "crop_box": {"x": 0.0, "y": 0.0, "w": 612.0, "h": 792.0},
                        "raw_text": (
                            "Branchformer: Parallel MLP-Attention Architectures for "
                            "Speech Recognition\n\n"
                            "Recent advances in speech recognition demonstrate the benefits "
                            "of parallel branch design. The proposed streaming ASR baseline "
                            "achieves low latency under 100ms. In this paper, we explore "
                            "parallel branches for self-attention and depthwise "
                            "convolution.\n\n"
                            "1. Architecture Design\n\n"
                            "Branchformer extends Conformer with parallel attention and "
                            "convolutional branches. This decouples local extraction from "
                            "global interaction."
                        ),
                        "elements": [
                            {
                                "id": UUID("b2222222-2222-2222-2222-e00000000001"),
                                "element_index": 0,
                                "element_type": "title",
                                "text": (
                                    "Branchformer: Parallel MLP-Attention Architectures for "
                                    "Speech Recognition"
                                ),
                                "bbox": (72.0, 72.0, 540.0, 100.0),
                                "section_path": [],
                            },
                            {
                                "id": UUID("b2222222-2222-2222-2222-e00000000002"),
                                "element_index": 1,
                                "element_type": "abstract",
                                "text": (
                                    "Recent advances in speech recognition demonstrate the "
                                    "benefits of parallel branch design. The proposed "
                                    "streaming ASR baseline achieves low latency under 100ms. "
                                    "In this paper, we explore parallel branches for "
                                    "self-attention and depthwise convolution."
                                ),
                                "bbox": (72.0, 110.0, 540.0, 220.0),
                                "section_path": ["Abstract"],
                            },
                            {
                                "id": UUID("b2222222-2222-2222-2222-e00000000003"),
                                "element_index": 2,
                                "element_type": "heading",
                                "text": "1. Architecture Design",
                                "bbox": (72.0, 240.0, 540.0, 260.0),
                                "section_path": ["1. Architecture Design"],
                            },
                            {
                                "id": UUID("b2222222-2222-2222-2222-e00000000004"),
                                "element_index": 3,
                                "element_type": "paragraph",
                                "text": (
                                    "Branchformer extends Conformer with parallel attention "
                                    "and convolutional branches. This decouples local "
                                    "extraction from global interaction."
                                ),
                                "bbox": (72.0, 270.0, 540.0, 360.0),
                                "section_path": ["1. Architecture Design"],
                            },
                        ],
                    },
                    {
                        "id": UUID("b2222222-2222-2222-2222-000000000002"),
                        "page_number": 2,
                        "width": 612.0,
                        "height": 792.0,
                        "rotation": 0,
                        "crop_box": {"x": 0.0, "y": 0.0, "w": 612.0, "h": 792.0},
                        "raw_text": (
                            "2. Empirical Evaluation\n\n"
                            "We evaluate Branchformer on LibriSpeech. "
                            "Experiments are conducted using standard ESPnet recipes with "
                            "joint CTC-attention decoding.\n\n"
                            "3. Results Comparison\n\n"
                            "Branchformer achieves a WER of 4.1% on LibriSpeech test-clean. "
                            "This confirms the efficacy of parallel branching over "
                            "sequential layers."
                        ),
                        "elements": [
                            {
                                "id": UUID("b2222222-2222-2222-2222-e00000000005"),
                                "element_index": 4,
                                "element_type": "heading",
                                "text": "2. Empirical Evaluation",
                                "bbox": (72.0, 72.0, 540.0, 95.0),
                                "section_path": ["2. Empirical Evaluation"],
                            },
                            {
                                "id": UUID("b2222222-2222-2222-2222-e00000000006"),
                                "element_index": 5,
                                "element_type": "paragraph",
                                "text": (
                                    "We evaluate Branchformer on LibriSpeech. "
                                    "Experiments are conducted using standard ESPnet "
                                    "recipes with joint CTC-attention decoding."
                                ),
                                "bbox": (72.0, 105.0, 540.0, 180.0),
                                "section_path": ["2. Empirical Evaluation"],
                            },
                            {
                                "id": UUID("b2222222-2222-2222-2222-e00000000007"),
                                "element_index": 6,
                                "element_type": "heading",
                                "text": "3. Results Comparison",
                                "bbox": (72.0, 200.0, 540.0, 220.0),
                                "section_path": ["3. Results Comparison"],
                            },
                            {
                                "id": UUID("b2222222-2222-2222-2222-e00000000008"),
                                "element_index": 7,
                                "element_type": "paragraph",
                                "text": (
                                    "Branchformer achieves a WER of 4.1% on LibriSpeech "
                                    "test-clean. This confirms the efficacy of parallel "
                                    "branching over sequential layers."
                                ),
                                "bbox": (72.0, 230.0, 540.0, 310.0),
                                "section_path": ["3. Results Comparison"],
                            },
                        ],
                    },
                ],
                "chunks": [
                    {
                        "id": UUID("b2222222-2222-2222-2222-c00000000001"),
                        "chunk_type": "child",
                        "chunk_index": 0,
                        "page_number": 1,
                        "text": (
                            "Branchformer: Parallel MLP-Attention Architectures for "
                            "Speech Recognition. Recent advances in speech recognition "
                            "demonstrate the benefits of parallel branch design. "
                            "The proposed streaming ASR baseline achieves low latency "
                            "under 100ms."
                        ),
                        "element_ids": [
                            UUID("b2222222-2222-2222-2222-e00000000001"),
                            UUID("b2222222-2222-2222-2222-e00000000002"),
                        ],
                    },
                    {
                        "id": UUID("b2222222-2222-2222-2222-c00000000002"),
                        "chunk_type": "child",
                        "chunk_index": 1,
                        "page_number": 1,
                        "text": (
                            "Branchformer extends Conformer with parallel attention and "
                            "convolutional branches. This decouples local extraction "
                            "from global interaction."
                        ),
                        "element_ids": [
                            UUID("b2222222-2222-2222-2222-e00000000003"),
                            UUID("b2222222-2222-2222-2222-e00000000004"),
                        ],
                    },
                    {
                        "id": UUID("b2222222-2222-2222-2222-c00000000003"),
                        "chunk_type": "child",
                        "chunk_index": 2,
                        "page_number": 2,
                        "text": (
                            "We evaluate Branchformer on LibriSpeech. "
                            "Experiments are conducted using standard ESPnet recipes "
                            "with joint CTC-attention decoding."
                        ),
                        "element_ids": [
                            UUID("b2222222-2222-2222-2222-e00000000005"),
                            UUID("b2222222-2222-2222-2222-e00000000006"),
                        ],
                    },
                    {
                        "id": UUID("b2222222-2222-2222-2222-c00000000004"),
                        "chunk_type": "child",
                        "chunk_index": 3,
                        "page_number": 2,
                        "text": (
                            "Branchformer achieves a WER of 4.1% on LibriSpeech "
                            "test-clean. This confirms the efficacy of parallel "
                            "branching over sequential layers."
                        ),
                        "element_ids": [
                            UUID("b2222222-2222-2222-2222-e00000000007"),
                            UUID("b2222222-2222-2222-2222-e00000000008"),
                        ],
                    },
                ],
            },
        },
    },
}


def load_manifest() -> dict[str, Any]:
    """Read manifest.json from disk."""
    manifest_path = Path(__file__).parent / "manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def build_manifest_dict() -> dict[str, Any]:
    """Construct complete manifest dictionary from disk manifest."""
    return load_manifest()


def validate_manifest(
    manifest_data: dict[str, Any] | str | Path | None = None,
) -> GraphRagManifest:
    """Validate manifest data structure and return typed GraphRagManifest."""
    if manifest_data is None:
        manifest_data = load_manifest()
    elif isinstance(manifest_data, (str, Path)):
        with open(manifest_data, encoding="utf-8") as f:
            manifest_data = json.load(f)
    return GraphRagManifest.model_validate(manifest_data)


def create_synthetic_models(
    project_key: str | None = None,
    include_a3: bool = False,
) -> dict[str, Any]:
    """Instantiate in-memory SQLAlchemy models for synthetic projects,
    papers, pages, elements, chunks.
    """
    projects_to_build = [project_key] if project_key else ["project_a", "project_b"]

    models: dict[str, Any] = {
        "projects": [],
        "papers": [],
        "pages": [],
        "elements": [],
        "chunks": [],
        "chunk_elements": [],
    }

    for p_key in projects_to_build:
        p_data = SYNTHETIC_DATA[p_key]
        db_project = Project(
            id=p_data["id"],
            name=p_data["name"],
            description=p_data["description"],
        )
        models["projects"].append(db_project)

        paper_keys = list(p_data["papers"].keys())
        if p_key == "project_a" and not include_a3:
            paper_keys = ["paper_a1", "paper_a2"]

        for pap_key in paper_keys:
            pap_data = p_data["papers"][pap_key]
            db_paper = Paper(
                id=pap_data["id"],
                project_id=pap_data["project_id"],
                filename=pap_data["filename"],
                storage_path=pap_data["storage_path"],
                document_sha256=pap_data["document_sha256"],
                status=pap_data["status"],
                page_count=pap_data["page_count"],
            )
            models["papers"].append(db_paper)

            elem_id_map: dict[UUID, PaperElement] = {}

            for page_data in pap_data["pages"]:
                db_page = PaperPage(
                    id=page_data["id"],
                    paper_id=pap_data["id"],
                    page_number=page_data["page_number"],
                    width=page_data["width"],
                    height=page_data["height"],
                    rotation=page_data["rotation"],
                    crop_box=page_data["crop_box"],
                    raw_text=page_data["raw_text"],
                )
                models["pages"].append(db_page)

                for elem_data in page_data["elements"]:
                    bbox = elem_data["bbox"]
                    db_element = PaperElement(
                        id=elem_data["id"],
                        paper_id=pap_data["id"],
                        page_number=page_data["page_number"],
                        element_index=elem_data["element_index"],
                        element_type=elem_data["element_type"],
                        text=elem_data["text"],
                        bbox_x_min=bbox[0],
                        bbox_y_min=bbox[1],
                        bbox_x_max=bbox[2],
                        bbox_y_max=bbox[3],
                        page_width=page_data["width"],
                        page_height=page_data["height"],
                        coordinate_origin="TOP_LEFT",
                        rotation=0,
                        section_path=elem_data["section_path"],
                        parser_version="1.0.0",
                    )
                    models["elements"].append(db_element)
                    elem_id_map[elem_data["id"]] = db_element

            for chunk_data in pap_data["chunks"]:
                db_chunk = PaperChunk(
                    id=chunk_data["id"],
                    paper_id=pap_data["id"],
                    chunk_type=chunk_data["chunk_type"],
                    chunk_index=chunk_data["chunk_index"],
                    text=chunk_data["text"],
                    token_count=len(chunk_data["text"].split()),
                    embedding=[0.05] * 1024,
                    embedding_vec=[0.05] * 1024,
                    embedding_model="bge-large-en-v1.5",
                    embedding_version="1.0.0",
                )
                models["chunks"].append(db_chunk)

                for idx, eid in enumerate(chunk_data["element_ids"]):
                    db_chunk_elem = ChunkElement(
                        chunk_id=db_chunk.id,
                        element_id=eid,
                        order_index=idx,
                    )
                    models["chunk_elements"].append(db_chunk_elem)

    return models


def seed_synthetic_corpus(
    db: Session,
    project_key: str | None = None,
    include_a3: bool = False,
) -> dict[str, Any]:
    """Persist synthetic corpus into database session and return created models."""
    models = create_synthetic_models(project_key=project_key, include_a3=include_a3)
    for p in models["projects"]:
        db.merge(p)
    db.flush()
    for pap in models["papers"]:
        db.merge(pap)
    db.flush()
    for page in models["pages"]:
        db.merge(page)
    db.flush()
    for elem in models["elements"]:
        db.merge(elem)
    db.flush()
    for chunk in models["chunks"]:
        db.merge(chunk)
    db.flush()
    for ce in models["chunk_elements"]:
        db.merge(ce)
    db.commit()
    return models


def verify_fixture_text_provenance(paper_data: dict[str, Any]) -> list[str]:
    """Sanity-check that every element and chunk text belongs to the paper's pages."""
    errors: list[str] = []
    page_texts = {p["page_number"]: p["raw_text"] for p in paper_data["pages"]}

    for page in paper_data["pages"]:
        for elem in page["elements"]:
            span = find_verbatim_span(page["raw_text"], elem["text"])
            if span is None and elem["text"] not in page["raw_text"]:
                errors.append(f"Element {elem['id']} text not found in page {page['page_number']}")

    for chunk in paper_data["chunks"]:
        p_num = chunk["page_number"]
        raw = page_texts.get(p_num, "")
        if not raw:
            errors.append(f"Chunk {chunk['id']} declared invalid page {p_num}")

    return errors
