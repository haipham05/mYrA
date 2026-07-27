"""Build a predictable, source-linked comparison evidence matrix."""

from collections import defaultdict
from collections.abc import Iterable

from app.schemas.comparison import ComparisonRequest
from app.schemas.comparison_result import (
    ComparisonCell,
    ComparisonCellStatus,
    ComparisonExcerpt,
    ComparisonMatrix,
)
from app.schemas.evidence import AnchorStatus, Citation, EvidenceItem
from app.services.comparison_retrieval import ComparisonEvidence

_MAX_EXCERPTS_PER_CELL = 3
_MAX_EXCERPTS_PER_MATRIX = 24
_NOT_FOUND_MESSAGE = "Not reported in retrieved evidence"


def _citation_anchor_status(evidence: EvidenceItem) -> AnchorStatus:
    """Mark an excerpt verified only when an anchor exactly matches its quote."""
    if any(
        anchor.anchor_status is AnchorStatus.VERIFIED
        and anchor.page_number == evidence.page_number
        and anchor.exact_quote == evidence.quote
        and anchor.document_sha256 == evidence.document_sha256
        for anchor in evidence.anchors
    ):
        return AnchorStatus.VERIFIED
    return AnchorStatus.UNRESOLVED


def build_comparison_matrix(
    request: ComparisonRequest,
    retrieved: Iterable[ComparisonEvidence],
) -> ComparisonMatrix:
    """Create every requested paper/dimension cell from only matching retrievals.

    Retrieval IDs are local to each search and can collide, so result IDs and
    citation indexes are assigned once across the complete matrix. Excerpts
    remain candidate evidence; this function makes no semantic support claims.
    """
    allowed_pairs = {
        (paper_id, dimension.value)
        for paper_id in request.paper_ids
        for dimension in request.dimensions
    }
    by_pair: dict[tuple[object, str], list[EvidenceItem]] = defaultdict(list)
    for result in retrieved:
        pair = (result.paper_id, result.dimension)
        if pair not in allowed_pairs:
            continue
        by_pair[pair].extend(item for item in result.evidence if item.paper_id == result.paper_id)

    # Keep the entire response bounded at the maximum paper/dimension count.
    # Round-robin across cells so early papers do not consume the full budget.
    selected_by_pair: dict[tuple[object, str], list[EvidenceItem]] = defaultdict(list)
    selected_count = 0
    for excerpt_index in range(_MAX_EXCERPTS_PER_CELL):
        for paper_id in request.paper_ids:
            for dimension in request.dimensions:
                pair = (paper_id, dimension.value)
                candidates = by_pair.get(pair, [])
                if excerpt_index < len(candidates) and selected_count < _MAX_EXCERPTS_PER_MATRIX:
                    selected_by_pair[pair].append(candidates[excerpt_index])
                    selected_count += 1

    cells: list[ComparisonCell] = []
    evidence_number = 1
    for paper_id in request.paper_ids:
        for dimension in request.dimensions:
            candidates = selected_by_pair.get((paper_id, dimension.value), [])
            excerpts: list[ComparisonExcerpt] = []
            for candidate in candidates:
                evidence_id = f"C{evidence_number}"
                citation_index = evidence_number
                evidence_number += 1
                evidence = candidate.model_copy(update={"id": evidence_id})
                citation = Citation(
                    citation_index=citation_index,
                    evidence_id=evidence_id,
                    paper_id=candidate.paper_id,
                    page_number=candidate.page_number,
                    quote=candidate.quote,
                    bounding_boxes=candidate.bounding_boxes,
                    document_sha256=candidate.document_sha256,
                    parser_version=candidate.parser_version,
                    anchor_status=_citation_anchor_status(candidate),
                    anchors=candidate.anchors,
                )
                excerpts.append(ComparisonExcerpt(evidence=evidence, citation=citation))

            cells.append(
                ComparisonCell(
                    paper_id=paper_id,
                    dimension=dimension,
                    status=(
                        ComparisonCellStatus.EVIDENCE_AVAILABLE
                        if excerpts
                        else ComparisonCellStatus.NOT_FOUND
                    ),
                    excerpts=excerpts,
                    message=None if excerpts else _NOT_FOUND_MESSAGE,
                )
            )

    return ComparisonMatrix(
        project_id=request.project_id,
        question=request.question,
        paper_ids=request.paper_ids,
        dimensions=request.dimensions,
        cells=cells,
    )
