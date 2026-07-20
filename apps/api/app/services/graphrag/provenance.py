"""Reusable M1 provenance resolver for GraphRAG.

Resolves extracted graph fact provenance against authoritative Postgres paper,
page, chunk, and element rows, ensuring verifiable verbatim citations.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.models import ChunkElement, Paper, PaperChunk, PaperElement
from app.ingestion.parser import normalize_text
from app.schemas.evidence import AnchorStatus, BoundingBox, CitationAnchor, CoordinateOrigin
from app.schemas.graph import GraphProvenanceSchema
from app.services.source_resolution import resolve_exact_source_anchor


def resolve_graph_source_anchor(
    db: Session,
    project_id: UUID,
    provenance: GraphProvenanceSchema | dict[str, Any] | BaseModel,
) -> tuple[CitationAnchor | None, AnchorStatus]:
    """Resolve a GraphRAG fact provenance anchor against Postgres ground truth.

    Checks:
    - Paper must exist in DB with paper.project_id == project_id, paper.status == 'READY',
      and paper.document_sha256.
    - If document_sha256 in provenance, must match paper.document_sha256.
    - Page PaperPage must exist for page_number with paper_id.
    - Quote must match page.raw_text via find_verbatim_span from app.ingestion.parser.
      If char_start and char_end are provided, verify
      page.raw_text[char_start:char_end] == exact_quote.
    - Linked child chunk: PaperChunk must exist, have chunk_type == 'child', belong to
      paper_id, and exact_quote.lower() must be contained within chunk.text.lower().
    - Linked element: PaperElement must exist on that page, belong to paper, be linked
      to the chunk, and independently contain the exact quote.

    Returns:
    - (CitationAnchor(...), AnchorStatus.VERIFIED) if all checks pass.
    - (None, AnchorStatus.UNRESOLVED) if any check fails.
    """
    # 1. Extract and sanitize input fields
    if isinstance(provenance, BaseModel):
        prov_dict = provenance.model_dump()
    elif isinstance(provenance, dict):
        prov_dict = provenance
    else:
        return (None, AnchorStatus.UNRESOLVED)

    p_id_raw = prov_dict.get("paper_id")
    c_id_raw = prov_dict.get("chunk_id")
    e_id_raw = prov_dict.get("element_id")
    page_number = prov_dict.get("page_number")
    exact_quote = prov_dict.get("exact_quote")
    char_start = prov_dict.get("char_start")
    char_end = prov_dict.get("char_end")
    doc_sha = prov_dict.get("document_sha256")
    parser_ver = prov_dict.get("parser_version")

    try:
        paper_id = UUID(str(p_id_raw)) if p_id_raw is not None else None
    except (ValueError, AttributeError):
        return (None, AnchorStatus.UNRESOLVED)

    try:
        chunk_id = UUID(str(c_id_raw)) if c_id_raw is not None else None
    except (ValueError, AttributeError):
        return (None, AnchorStatus.UNRESOLVED)

    try:
        element_id = UUID(str(e_id_raw)) if e_id_raw is not None else None
    except (ValueError, AttributeError):
        return (None, AnchorStatus.UNRESOLVED)

    if paper_id is None or chunk_id is None:
        return (None, AnchorStatus.UNRESOLVED)

    if not isinstance(page_number, int) or page_number < 1:
        return (None, AnchorStatus.UNRESOLVED)

    if not exact_quote or not isinstance(exact_quote, str) or not exact_quote.strip():
        return (None, AnchorStatus.UNRESOLVED)

    if char_start is not None and (not isinstance(char_start, int) or char_start < 0):
        return (None, AnchorStatus.UNRESOLVED)

    if char_end is not None and (
        not isinstance(char_end, int) or (char_start is not None and char_end <= char_start)
    ):
        return (None, AnchorStatus.UNRESOLVED)

    # 2. Verify paper ownership, status, and document SHA-256
    paper = db.query(Paper).filter(Paper.id == paper_id, Paper.project_id == project_id).first()
    if not paper or paper.status != "READY" or not paper.document_sha256:
        return (None, AnchorStatus.UNRESOLVED)

    if doc_sha:
        if str(doc_sha).strip().lower() != paper.document_sha256.strip().lower():
            return (None, AnchorStatus.UNRESOLVED)

    # 3. Resolve exact source existence centrally. Graph-specific chunk and
    # element linkage checks below remain a separate provenance policy.
    resolved_anchor = resolve_exact_source_anchor(
        db,
        project_id=project_id,
        paper_id=paper.id,
        page_number=page_number,
        exact_quote=exact_quote,
        document_sha256=paper.document_sha256,
        char_start=char_start,
        char_end=char_end,
        parser_version=parser_ver,
    )
    if resolved_anchor is None:
        return (None, AnchorStatus.UNRESOLVED)
    start_char = resolved_anchor.source_char_start
    end_char = resolved_anchor.source_char_end
    if start_char is None or end_char is None:
        return (None, AnchorStatus.UNRESOLVED)

    # 4. Locate and verify linked child chunk
    chunk = (
        db.query(PaperChunk)
        .filter(PaperChunk.id == chunk_id, PaperChunk.paper_id == paper.id)
        .first()
    )
    if not chunk or chunk.chunk_type != "child":
        return (None, AnchorStatus.UNRESOLVED)

    if not chunk.text or normalize_text(exact_quote) not in normalize_text(chunk.text):
        return (None, AnchorStatus.UNRESOLVED)

    def element_supports_quote(text: str | None) -> bool:
        return bool(text and normalize_text(exact_quote) in normalize_text(text))

    # 5. Locate and verify linked element
    element: PaperElement | None = None
    if element_id is not None:
        element = (
            db.query(PaperElement)
            .filter(
                PaperElement.id == element_id,
                PaperElement.paper_id == paper.id,
                PaperElement.page_number == page_number,
            )
            .first()
        )
        if not element:
            return (None, AnchorStatus.UNRESOLVED)

        contains_quote = element_supports_quote(element.text)
        is_linked = (
            db.query(ChunkElement)
            .filter(
                ChunkElement.chunk_id == chunk.id,
                ChunkElement.element_id == element.id,
            )
            .first()
            is not None
        )
        if not (contains_quote and is_linked):
            return (None, AnchorStatus.UNRESOLVED)
    else:
        candidate_elements = (
            db.query(PaperElement)
            .filter(
                PaperElement.paper_id == paper.id,
                PaperElement.page_number == page_number,
            )
            .order_by(PaperElement.element_index)
            .all()
        )
        if not candidate_elements:
            return (None, AnchorStatus.UNRESOLVED)

        chunk_elem_ids = {
            ce.element_id
            for ce in db.query(ChunkElement.element_id)
            .filter(ChunkElement.chunk_id == chunk.id)
            .all()
        }

        matching_elements = [
            elem
            for elem in candidate_elements
            if elem.id in chunk_elem_ids and element_supports_quote(elem.text)
        ]
        if len(matching_elements) != 1:
            return (None, AnchorStatus.UNRESOLVED)
        element = matching_elements[0]

    # 6. Extract bounding box from element
    bboxes: list[BoundingBox] = []
    if (
        element.bbox_x_min is not None
        and element.page_width
        and element.page_height
        and element.page_width > 0
        and element.page_height > 0
    ):
        origin_str = (element.coordinate_origin or "TOP_LEFT").upper()
        try:
            origin = (
                CoordinateOrigin(origin_str)
                if origin_str in CoordinateOrigin.__members__
                else CoordinateOrigin.TOP_LEFT
            )
        except ValueError:
            origin = CoordinateOrigin.TOP_LEFT

        bboxes.append(
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

    effective_parser_ver = element.parser_version

    anchor = CitationAnchor(
        page_number=page_number,
        source_element_id=element.id,
        exact_quote=exact_quote,
        source_char_start=start_char,
        source_char_end=end_char,
        document_sha256=paper.document_sha256,
        parser_version=effective_parser_ver,
        anchor_status=AnchorStatus.VERIFIED,
        bounding_boxes=bboxes,
    )
    return (anchor, AnchorStatus.VERIFIED)
