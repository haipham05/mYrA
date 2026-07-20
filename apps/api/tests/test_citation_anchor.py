from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.evidence import (
    AnchorStatus,
    BoundingBox,
    Citation,
    CitationAnchor,
    CoordinateOrigin,
    EvidenceItem,
)
from app.services.chat_service import matching_verified_anchor, resolve_claim_anchor


def test_valid_citation_anchor():
    anchor = CitationAnchor(
        page_number=1,
        source_element_id=uuid4(),
        exact_quote="Attention Is All You Need.",
        document_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        parser_version="pypdf-v1",
        anchor_status=AnchorStatus.VERIFIED,
        bounding_boxes=[
            BoundingBox(
                x_min=10.0,
                y_min=20.0,
                x_max=100.0,
                y_max=40.0,
                page_width=612.0,
                page_height=792.0,
                origin=CoordinateOrigin.TOP_LEFT,
            )
        ],
    )
    assert anchor.page_number == 1
    assert anchor.anchor_status == AnchorStatus.VERIFIED
    assert len(anchor.bounding_boxes) == 1


def test_anchor_rejects_page_zero_or_negative():
    with pytest.raises(ValidationError):
        CitationAnchor(page_number=0, exact_quote="Hello")

    with pytest.raises(ValidationError):
        CitationAnchor(page_number=-1, exact_quote="Hello")


def test_anchor_marks_unresolved_on_empty_quote():
    anchor = CitationAnchor(page_number=1, exact_quote="   ", anchor_status=AnchorStatus.VERIFIED)
    # Empty quote forces unresolved status
    assert anchor.anchor_status == AnchorStatus.UNRESOLVED


def test_citation_backward_compatibility_legacy():
    # Legacy citation payload without anchor_status or anchors, but with bounding_boxes
    legacy_data = {
        "citation_index": 1,
        "evidence_id": "E1",
        "paper_id": str(uuid4()),
        "page_number": 2,
        "bounding_boxes": [
            {
                "x_min": 10.0,
                "y_min": 20.0,
                "x_max": 50.0,
                "y_max": 30.0,
                "page_width": 612.0,
                "page_height": 792.0,
                "origin": "TOP_LEFT",
                "rotation": 0,
            }
        ],
        "quote": "Some text",
    }
    citation = Citation.model_validate(legacy_data)
    assert citation.anchor_status == AnchorStatus.LEGACY
    assert len(citation.bounding_boxes) == 1
    assert citation.anchors == []


def test_citation_backward_compatibility_empty():
    # Citation with empty boxes and empty quote
    data = {
        "citation_index": 2,
        "evidence_id": "E2",
        "paper_id": str(uuid4()),
        "page_number": 1,
        "quote": "",
    }
    citation = Citation.model_validate(data)
    assert citation.anchor_status == AnchorStatus.UNRESOLVED


def test_citation_verified_with_anchors():
    pid = uuid4()
    anchor = CitationAnchor(
        page_number=3,
        exact_quote="Transformer model uses multi-head attention.",
        anchor_status=AnchorStatus.VERIFIED,
    )
    citation = Citation(
        citation_index=1,
        evidence_id="E1",
        paper_id=pid,
        page_number=3,
        quote="Transformer model uses multi-head attention.",
        anchors=[anchor],
        anchor_status=AnchorStatus.VERIFIED,
        document_sha256="abc12345",
    )
    assert citation.anchor_status == AnchorStatus.VERIFIED
    assert len(citation.anchors) == 1
    assert citation.anchors[0].page_number == 3


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("anchor_status", AnchorStatus.UNRESOLVED),
        ("page_number", 2),
        ("exact_quote", "Other quote"),
        ("document_sha256", "other-hash"),
        ("parser_version", "other-parser"),
        ("source_element_id", None),
        ("source_char_start", None),
        ("source_char_end", None),
        ("source_char_end", 1),
    ],
)
def test_only_matching_verified_span_can_support_citation(change, value):
    paper_id = uuid4()
    anchor = CitationAnchor(
        page_number=1,
        source_element_id=uuid4(),
        exact_quote="Alice outperformed Bob.",
        source_char_start=1,
        source_char_end=24,
        document_sha256="paper-hash",
        parser_version="docling-2",
        anchor_status=AnchorStatus.VERIFIED,
    )
    evidence = EvidenceItem(
        id="E1",
        paper_id=paper_id,
        chunk_id=uuid4(),
        quote=anchor.exact_quote,
        page_number=1,
        document_sha256=anchor.document_sha256,
        parser_version=anchor.parser_version,
        anchors=[anchor],
    )
    assert matching_verified_anchor(evidence) == anchor
    bad_anchor = anchor.model_copy(update={change: value})
    assert matching_verified_anchor(evidence.model_copy(update={"anchors": [bad_anchor]})) is None


def test_resolve_claim_anchor_direct_monotonic():
    from unittest.mock import MagicMock

    mock_db = MagicMock()
    paper_id = uuid4()
    anchor = CitationAnchor(
        page_number=2,
        source_element_id=uuid4(),
        exact_quote="The Transformer model relies entirely on self-attention.",
        source_char_start=100,
        source_char_end=156,
        document_sha256="test-sha",
        parser_version="docling-2",
        anchor_status=AnchorStatus.VERIFIED,
    )
    evidence = EvidenceItem(
        id="E1",
        paper_id=paper_id,
        chunk_id=uuid4(),
        quote="Different primary quote",
        page_number=2,
        document_sha256="test-sha",
        parser_version="docling-2",
        anchors=[anchor],
    )

    # Monotonic claim supported by anchor in anchors list even if not matching primary quote
    matched, phrase = resolve_claim_anchor(
        mock_db,
        evidence,
        "The Transformer model relies entirely on self-attention.",
        cite_count=1,
        project_id=uuid4(),
    )
    assert matched == anchor
    assert phrase is None


def test_resolve_claim_anchor_multi_element_verbatim_quote():
    from unittest.mock import MagicMock

    from app.db.models import Paper, PaperElement, PaperPage

    mock_db = MagicMock()
    paper_id = uuid4()
    project_id = uuid4()
    raw_page_text = (
        "We propose a new simple network architecture, the Transformer, "
        "based solely on attention mechanisms, dispensing with recurrence "
        "and convolutions entirely."
    )

    page_rec = PaperPage(
        paper_id=paper_id,
        page_number=1,
        width=612.0,
        height=792.0,
        raw_text=raw_page_text,
    )

    elem1 = PaperElement(
        id=uuid4(),
        paper_id=paper_id,
        page_number=1,
        element_index=0,
        element_type="paragraph",
        text="We propose a new simple network architecture, the Transformer,",
        bbox_x_min=10.0,
        bbox_y_min=10.0,
        bbox_x_max=100.0,
        bbox_y_max=20.0,
        page_width=612.0,
        page_height=792.0,
        coordinate_origin="TOP_LEFT",
        rotation=0,
        parser_version="docling-2",
    )
    elem2 = PaperElement(
        id=uuid4(),
        paper_id=paper_id,
        page_number=1,
        element_index=1,
        element_type="paragraph",
        text=(
            "based solely on attention mechanisms, "
            "dispensing with recurrence and convolutions entirely."
        ),
        bbox_x_min=10.0,
        bbox_y_min=25.0,
        bbox_x_max=100.0,
        bbox_y_max=35.0,
        page_width=612.0,
        page_height=792.0,
        coordinate_origin="TOP_LEFT",
        rotation=0,
        parser_version="docling-2",
    )

    # Set up mock_db queries
    def query_mock(model):
        q = MagicMock()
        if model is Paper:
            q.filter.return_value.first.return_value = Paper(
                id=paper_id,
                project_id=project_id,
                filename="source.pdf",
                storage_path="local/source.pdf",
                document_sha256="doc-hash-123",
                status="READY",
            )
        elif model is PaperPage:
            q.filter.return_value.first.return_value = page_rec
        elif model is PaperElement:
            q.filter.return_value.order_by.return_value.all.return_value = [elem1, elem2]
        return q

    mock_db.query.side_effect = query_mock

    evidence = EvidenceItem(
        id="E1",
        paper_id=paper_id,
        chunk_id=uuid4(),
        quote=elem1.text,
        page_number=1,
        document_sha256="doc-hash-123",
        parser_version="docling-2",
        anchors=[
            CitationAnchor(
                page_number=1,
                source_element_id=elem1.id,
                exact_quote=elem1.text,
                source_char_start=0,
                source_char_end=len(elem1.text),
                document_sha256="doc-hash-123",
                parser_version="docling-2",
                anchor_status=AnchorStatus.VERIFIED,
            )
        ],
    )

    # Claim quoting a multi-element sentence
    claim = (
        'The authors "propose a new simple network architecture, the Transformer, '
        "based solely on attention mechanisms, dispensing with recurrence "
        'and convolutions entirely."'
    )
    matched, phrase = resolve_claim_anchor(
        mock_db, evidence, claim, cite_count=1, project_id=project_id
    )

    assert matched is not None
    assert matched.anchor_status == AnchorStatus.VERIFIED
    assert matched.page_number == 1
    assert "propose a new simple network architecture" in matched.exact_quote
    assert matched.source_char_start is not None
    assert matched.source_char_end is not None
    assert len(matched.bounding_boxes) == 2  # Overlaps both elem1 and elem2
    assert phrase is not None


def test_resolve_claim_anchor_rejects_hallucination():
    from unittest.mock import MagicMock

    from app.db.models import Paper, PaperElement, PaperPage

    mock_db = MagicMock()
    paper_id = uuid4()
    project_id = uuid4()
    page_rec = PaperPage(
        paper_id=paper_id,
        page_number=1,
        width=612.0,
        height=792.0,
        raw_text="Real facts only.",
    )

    def query_mock(model):
        q = MagicMock()
        if model is Paper:
            q.filter.return_value.first.return_value = Paper(
                id=paper_id,
                project_id=project_id,
                filename="source.pdf",
                storage_path="local/source.pdf",
                document_sha256="doc-hash-123",
                status="READY",
            )
        elif model is PaperPage:
            q.filter.return_value.first.return_value = page_rec
        elif model is PaperElement:
            q.filter.return_value.order_by.return_value.all.return_value = []
        return q

    mock_db.query.side_effect = query_mock

    evidence = EvidenceItem(
        id="E1",
        paper_id=paper_id,
        chunk_id=uuid4(),
        quote="Real facts only.",
        page_number=1,
        document_sha256="doc-hash-123",
        parser_version="docling-2",
        anchors=[],
    )

    claim = 'The authors "invented time travel and infinite energy".'
    matched, phrase = resolve_claim_anchor(
        mock_db, evidence, claim, cite_count=1, project_id=project_id
    )
    assert matched is None
    assert phrase is None
