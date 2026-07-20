from uuid import uuid4

from app.schemas.evidence import AnchorStatus, CitationAnchor, EvidenceItem
from app.services.evidence_assembly import assemble_evidence_items


def _item(
    paper_id,
    *,
    chunk_id=None,
    quote="Source text.",
    start=0,
    context="Parent source context for understanding.",
):
    document_hash = f"{paper_id.hex:0<64}"[:64]
    return EvidenceItem(
        id="temporary",
        paper_id=paper_id,
        paper_title=f"Paper {paper_id.hex[:4]}",
        chunk_id=chunk_id or uuid4(),
        quote=quote,
        parent_context=context,
        page_number=1,
        document_sha256=document_hash,
        parser_version="test-parser",
        anchors=[
            CitationAnchor(
                page_number=1,
                exact_quote=quote,
                source_char_start=start,
                source_char_end=start + len(quote),
                document_sha256=document_hash,
                parser_version="test-parser",
                anchor_status=AnchorStatus.VERIFIED,
            )
        ],
    )


def test_assembly_deduplicates_only_same_verified_source_span():
    first_paper = uuid4()
    other_paper = uuid4()
    first = _item(first_paper, start=0)
    same_span = _item(first_paper, start=0, quote=first.quote)
    repeated_quote = _item(first_paper, start=40, quote=first.quote)
    other_source = _item(other_paper, start=0, quote=first.quote)

    assembled = assemble_evidence_items([first, same_span, repeated_quote, other_source])

    assert len(assembled) == 3
    assert [item.id for item in assembled] == ["E1", "E2", "E3"]
    assert {str(item.paper_id) for item in assembled} == {
        str(first_paper),
        str(other_paper),
    }
    assert sum(item.paper_id == first_paper for item in assembled) == 2


def test_assembly_balances_papers_and_bounds_parent_context():
    first_paper = uuid4()
    second_paper = uuid4()
    items = [
        _item(first_paper, quote="A.", start=0, context="x" * 500),
        _item(first_paper, quote="B.", start=20, context="context B"),
        _item(second_paper, quote="C.", start=0, context="context C"),
    ]

    assembled = assemble_evidence_items(items, max_items=2, context_char_budget=300)

    assert [item.paper_id for item in assembled] == [first_paper, second_paper]
    assert len(assembled[0].parent_context) < len(items[0].parent_context)
    assert assembled[0].quote == items[0].quote
    assert assembled[1].parent_context == items[2].parent_context


def test_assembly_empty_and_invalid_budget_return_no_evidence():
    assert assemble_evidence_items([]) == []
    assert assemble_evidence_items([_item(uuid4())], max_items=0) == []
    assert assemble_evidence_items([_item(uuid4())], context_char_budget=0) == []
