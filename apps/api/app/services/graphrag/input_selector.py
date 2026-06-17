"""Graph extraction input selector.

Selects approved child chunks and their linked primary elements from READY papers
within a bounded project scope for knowledge graph extraction.
"""

from __future__ import annotations

import logging
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.db.models import ChunkElement, Paper, PaperChunk, PaperElement

logger = logging.getLogger("myra.graphrag.input_selector")


class ExtractionEvidenceItem(BaseModel):
    """Evidence item payload for graph extraction.

    Carries bounded opaque evidence ID and provenance metadata to ground
    graph candidate extraction back to authoritative document elements.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    chunk_id: UUID
    text: str
    page_number: int
    element_id: UUID | None = None
    parser_version: str
    document_sha256: str
    chunk_index: int


def select_extraction_inputs(
    db: Session,
    project_id: UUID,
    paper_id: UUID,
    max_chunks: int = 50,
) -> list[ExtractionEvidenceItem]:
    """Select approved, source-linked child chunks from a paper for graph extraction.

    Checks:
    - Paper must exist in DB and belong to project_id.
    - Paper must have status == 'READY'.
    - Paper must have document_sha256.
    - Chunks must have chunk_type == 'child' (parent-only chunks strictly excluded).
    - Chunks are ordered by chunk_index ASC and limited to max_chunks.
    - Chunks must have non-empty, non-whitespace text.
    - Chunks must link to at least one PaperElement via ChunkElement (ordered by order_index ASC).
      The first linked element is selected as the primary element.
    - Each evidence item is assigned an opaque, 1-indexed local evidence ID (ev_1, ev_2, ...).

    Returns:
        List of ExtractionEvidenceItem instances, or empty list if checks fail.
    """
    if max_chunks <= 0:
        return []

    if isinstance(project_id, str):
        project_id = UUID(project_id)
    if isinstance(paper_id, str):
        paper_id = UUID(paper_id)

    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if paper is None:
        logger.warning("Paper %s does not exist", paper_id)
        return []

    if paper.project_id != project_id:
        logger.warning(
            "Paper %s belongs to project %s, but extraction requested for project %s",
            paper_id,
            paper.project_id,
            project_id,
        )
        return []

    if paper.status != "READY":
        logger.info(
            "Paper %s has status %s (expected READY); skipping extraction inputs",
            paper_id,
            paper.status,
        )
        return []

    if not paper.document_sha256:
        logger.warning(
            "Paper %s has no document_sha256; skipping extraction inputs",
            paper_id,
        )
        return []

    chunks = (
        db.query(PaperChunk)
        .filter(
            PaperChunk.paper_id == paper_id,
            PaperChunk.chunk_type == "child",
        )
        .order_by(PaperChunk.chunk_index.asc())
        .limit(max_chunks)
        .all()
    )

    evidence_items: list[ExtractionEvidenceItem] = []
    for chunk in chunks:
        if not chunk.text or not chunk.text.strip():
            logger.warning("Chunk %s has empty or whitespace text; skipping", chunk.id)
            continue

        primary_element = (
            db.query(PaperElement)
            .join(ChunkElement, ChunkElement.element_id == PaperElement.id)
            .filter(ChunkElement.chunk_id == chunk.id)
            .order_by(ChunkElement.order_index.asc())
            .first()
        )
        if primary_element is None:
            logger.warning("Chunk %s has no linked elements; skipping", chunk.id)
            continue

        ev_id = f"ev_{len(evidence_items) + 1}"
        evidence_items.append(
            ExtractionEvidenceItem(
                evidence_id=ev_id,
                chunk_id=chunk.id,
                text=chunk.text,
                page_number=primary_element.page_number,
                element_id=primary_element.id,
                parser_version=primary_element.parser_version or "v1",
                document_sha256=paper.document_sha256,
                chunk_index=chunk.chunk_index,
            )
        )

        if len(evidence_items) >= max_chunks:
            break

    return evidence_items
