"""Shared, deterministic resolution of exact paper quotes to current source rows."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.orm import Session

from app.db.models import ChunkElement, Paper, PaperChunk, PaperElement, PaperPage
from app.ingestion.parser import find_verbatim_span
from app.schemas.evidence import (
    AnchorStatus,
    BoundingBox,
    CitationAnchor,
    CoordinateOrigin,
    EvidenceItem,
)


def resolve_exact_source_anchor(
    db: Session,
    *,
    project_id: UUID,
    paper_id: UUID,
    page_number: int,
    exact_quote: str,
    document_sha256: str | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    parser_version: str | None = None,
) -> CitationAnchor | None:
    """Verify a quote against the current READY paper and canonical page text.

    This establishes source existence only. It deliberately does not decide whether
    the quote semantically supports a generated claim. Offsets are accepted only
    when the exact slice matches; otherwise the quote must resolve unambiguously.
    """
    quote = exact_quote.strip() if isinstance(exact_quote, str) else ""
    if not quote or page_number < 1:
        return None
    if char_start is not None and char_start < 0:
        return None
    if char_end is not None and char_end <= 0:
        return None
    if char_start is not None and char_end is not None and char_end <= char_start:
        return None

    paper = (
        db.query(Paper)
        .filter(
            Paper.id == paper_id,
            Paper.project_id == project_id,
            Paper.status == "READY",
        )
        .first()
    )
    if not paper or not paper.document_sha256:
        return None
    if document_sha256 and document_sha256.strip().lower() != paper.document_sha256.lower():
        return None

    page = (
        db.query(PaperPage)
        .filter(PaperPage.paper_id == paper.id, PaperPage.page_number == page_number)
        .first()
    )
    if not page or not page.raw_text:
        return None

    span = find_verbatim_span(page.raw_text, quote, preferred_char_start=char_start)
    if span is None:
        return None
    start, end = span
    if char_start is not None and start != char_start:
        return None
    if char_end is not None and end != char_end:
        return None
    if char_start is not None and char_end is not None:
        if page.raw_text[char_start:char_end] != quote:
            return None

    elements = (
        db.query(PaperElement)
        .filter(PaperElement.paper_id == paper.id, PaperElement.page_number == page_number)
        .order_by(PaperElement.element_index)
        .all()
    )
    overlapping: list[PaperElement] = []
    for element in elements:
        if not element.text:
            continue
        element_span = find_verbatim_span(page.raw_text, element.text)
        if element_span and element_span[0] < end and element_span[1] > start:
            overlapping.append(element)

    boxes: list[BoundingBox] = []
    for element in overlapping:
        if (
            element.bbox_x_min is None
            or element.page_width is None
            or element.page_height is None
            or element.page_width <= 0
            or element.page_height <= 0
        ):
            continue
        try:
            origin = CoordinateOrigin(element.coordinate_origin)
        except ValueError:
            origin = CoordinateOrigin.TOP_LEFT
        boxes.append(
            BoundingBox(
                x_min=float(element.bbox_x_min),
                y_min=float(element.bbox_y_min or 0.0),
                x_max=float(element.bbox_x_max or 0.0),
                y_max=float(element.bbox_y_max or 0.0),
                page_width=float(element.page_width),
                page_height=float(element.page_height),
                origin=origin,
                rotation=int(element.rotation or 0),
            )
        )

    current_parser_versions = list(
        dict.fromkeys(element.parser_version for element in overlapping if element.parser_version)
    )
    if parser_version and current_parser_versions and parser_version not in current_parser_versions:
        return None
    resolved_parser = current_parser_versions[0] if current_parser_versions else None
    return CitationAnchor(
        page_number=page_number,
        source_element_id=overlapping[0].id if overlapping else None,
        exact_quote=quote,
        source_char_start=start,
        source_char_end=end,
        document_sha256=paper.document_sha256,
        parser_version=resolved_parser,
        anchor_status=AnchorStatus.VERIFIED,
        bounding_boxes=boxes,
    )


def build_selected_passage_evidence(
    db: Session,
    *,
    project_id: UUID,
    paper_id: UUID,
    anchor: CitationAnchor,
) -> EvidenceItem | None:
    """Attach a verified page quote to its indexed chunk for grounded generation."""
    if (
        anchor.anchor_status != AnchorStatus.VERIFIED
        or anchor.document_sha256 is None
        or anchor.source_char_start is None
        or anchor.source_char_end is None
        or anchor.source_element_id is None
    ):
        return None

    paper = (
        db.query(Paper)
        .filter(
            Paper.id == paper_id,
            Paper.project_id == project_id,
            Paper.status == "READY",
            Paper.document_sha256 == anchor.document_sha256,
        )
        .first()
    )
    page = (
        db.query(PaperPage)
        .filter(
            PaperPage.paper_id == paper_id,
            PaperPage.page_number == anchor.page_number,
        )
        .first()
    )
    chunk = (
        db.query(PaperChunk)
        .join(ChunkElement, ChunkElement.chunk_id == PaperChunk.id)
        .filter(
            PaperChunk.paper_id == paper_id,
            PaperChunk.chunk_type == "child",
            ChunkElement.element_id == anchor.source_element_id,
        )
        .order_by(PaperChunk.chunk_index)
        .first()
    )
    if paper is None or page is None or not page.raw_text or chunk is None:
        return None
    start, end = anchor.source_char_start, anchor.source_char_end
    if page.raw_text[start:end] != anchor.exact_quote:
        return None

    context_start = max(0, start - 300)
    context_end = min(len(page.raw_text), end + 300)
    return EvidenceItem(
        id="selected-passage",
        paper_id=paper.id,
        paper_title=paper.title,
        chunk_id=chunk.id,
        quote=anchor.exact_quote,
        parent_context=page.raw_text[context_start:context_end],
        page_number=anchor.page_number,
        source_element_ids=[anchor.source_element_id],
        document_sha256=anchor.document_sha256,
        parser_version=anchor.parser_version,
        anchors=[anchor],
    )
