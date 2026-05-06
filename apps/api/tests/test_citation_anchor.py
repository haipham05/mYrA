from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.evidence import (
    AnchorStatus,
    BoundingBox,
    Citation,
    CitationAnchor,
    CoordinateOrigin,
)


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
