from uuid import uuid4

from app.schemas.comparison import ComparisonDimension, ComparisonRequest
from app.schemas.comparison_result import ComparisonCellStatus
from app.schemas.evidence import AnchorStatus, CitationAnchor, EvidenceItem
from app.services.comparison_matrix import build_comparison_matrix
from app.services.comparison_retrieval import ComparisonEvidence


def _request(paper_ids, dimensions=None):
    return ComparisonRequest(
        project_id=uuid4(),
        paper_ids=paper_ids,
        dimensions=dimensions or [ComparisonDimension.METHOD_ARCHITECTURE],
    )


def _evidence(paper_id, *, evidence_id="E1", quote="A source excerpt", page=2):
    return EvidenceItem(
        id=evidence_id,
        paper_id=paper_id,
        chunk_id=uuid4(),
        quote=quote,
        page_number=page,
        document_sha256="a" * 64,
        parser_version="docling-test",
        anchors=[
            CitationAnchor(
                page_number=page,
                exact_quote=quote,
                document_sha256="a" * 64,
                anchor_status=AnchorStatus.VERIFIED,
            )
        ],
    )


def _retrieved(paper_id, dimension, *evidence):
    return ComparisonEvidence(
        paper_id=paper_id,
        dimension=dimension.value,
        query="method",
        evidence=list(evidence),
    )


def test_matrix_has_every_requested_cell_and_explicit_missing_state():
    paper_a, paper_b = uuid4(), uuid4()
    request = _request(
        [paper_a, paper_b],
        [ComparisonDimension.METHOD_ARCHITECTURE, ComparisonDimension.RESULTS],
    )

    matrix = build_comparison_matrix(
        request,
        [_retrieved(paper_a, ComparisonDimension.METHOD_ARCHITECTURE, _evidence(paper_a))],
    )

    assert [(cell.paper_id, cell.dimension) for cell in matrix.cells] == [
        (paper_a, ComparisonDimension.METHOD_ARCHITECTURE),
        (paper_a, ComparisonDimension.RESULTS),
        (paper_b, ComparisonDimension.METHOD_ARCHITECTURE),
        (paper_b, ComparisonDimension.RESULTS),
    ]
    missing = [cell for cell in matrix.cells if cell.status is ComparisonCellStatus.NOT_FOUND]
    assert len(missing) == 3
    assert all(cell.message == "Not reported in retrieved evidence" for cell in missing)
    assert all(cell.evidence_kind == "candidate_source_excerpts_only" for cell in matrix.cells)
    assert "not extracted or fact-checked" in matrix.interpretation_notice


def test_evidence_from_one_paper_cannot_populate_another_papers_cell():
    requested, foreign = uuid4(), uuid4()
    matrix = build_comparison_matrix(
        _request([requested, uuid4()]),
        [_retrieved(requested, ComparisonDimension.METHOD_ARCHITECTURE, _evidence(foreign))],
    )

    assert matrix.cells[0].status is ComparisonCellStatus.NOT_FOUND
    assert matrix.cells[0].excerpts == []


def test_citations_preserve_source_page_hash_and_verified_exact_anchor():
    paper_id = uuid4()
    source = _evidence(paper_id, quote="Original exact passage", page=4)

    cell = build_comparison_matrix(
        _request([paper_id, uuid4()]),
        [_retrieved(paper_id, ComparisonDimension.METHOD_ARCHITECTURE, source)],
    ).cells[0]

    excerpt = cell.excerpts[0]
    assert excerpt.evidence.paper_id == paper_id
    assert excerpt.evidence.quote == source.quote
    assert excerpt.citation.paper_id == paper_id
    assert excerpt.citation.page_number == 4
    assert excerpt.citation.quote == source.quote
    assert excerpt.citation.document_sha256 == source.document_sha256
    assert excerpt.citation.anchors == source.anchors
    assert excerpt.citation.anchor_status is AnchorStatus.VERIFIED


def test_duplicate_local_retrieval_ids_are_remapped_uniquely_across_cells():
    paper_a, paper_b = uuid4(), uuid4()
    matrix = build_comparison_matrix(
        _request([paper_a, paper_b]),
        [
            _retrieved(
                paper_a,
                ComparisonDimension.METHOD_ARCHITECTURE,
                _evidence(paper_a, evidence_id="E1"),
            ),
            _retrieved(
                paper_b,
                ComparisonDimension.METHOD_ARCHITECTURE,
                _evidence(paper_b, evidence_id="E1"),
            ),
        ],
    )

    excerpts = [excerpt for cell in matrix.cells for excerpt in cell.excerpts]
    assert [excerpt.evidence.id for excerpt in excerpts] == ["C1", "C2"]
    assert [excerpt.citation.citation_index for excerpt in excerpts] == [1, 2]
    assert [excerpt.citation.evidence_id for excerpt in excerpts] == ["C1", "C2"]


def test_each_cell_is_bounded_to_three_candidate_excerpts():
    paper_id = uuid4()
    evidence = [_evidence(paper_id, evidence_id=f"E{index}") for index in range(5)]
    matrix = build_comparison_matrix(
        _request([paper_id, uuid4()]),
        [_retrieved(paper_id, ComparisonDimension.METHOD_ARCHITECTURE, *evidence)],
    )

    assert len(matrix.cells[0].excerpts) == 3


def test_maximum_matrix_keeps_all_cells_and_bounds_total_excerpts():
    paper_ids = [uuid4() for _ in range(6)]
    dimensions = list(ComparisonDimension)
    request = _request(paper_ids, dimensions)
    retrieved = [
        _retrieved(
            paper_id,
            dimension,
            *[_evidence(paper_id, evidence_id=f"{paper_id}-{index}") for index in range(3)],
        )
        for paper_id in paper_ids
        for dimension in dimensions
    ]

    matrix = build_comparison_matrix(request, retrieved)

    assert len(matrix.cells) == 6 * len(dimensions)
    assert sum(len(cell.excerpts) for cell in matrix.cells) == 24
    assert sum(bool(cell.excerpts) for cell in matrix.cells) == 24
