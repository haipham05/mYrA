"""Tests for GraphRAG source anchor resolution (Task 5.05) and deterministic identities (Task 5.06).

Verifies:
1. Valid quote with chunk and element returns VERIFIED with exact coordinates and char spans.
2. Stale document_sha256 returns UNRESOLVED.
3. Parent chunk or reindexed chunk returns UNRESOLVED.
4. Quote not present in chunk text returns UNRESOLVED.
5. Quote outside page text returns UNRESOLVED.
6. Wrong project ID returns UNRESOLVED.
7. Repeated ambiguous quote without offsets returns UNRESOLVED.
8. Identity determinism and canonicalization.
9. Cross-project isolation (same name/external ID across projects produces distinct keys).
10. Multi-paper provenance (same relation from two papers yields distinct fact IDs).
"""

from __future__ import annotations

from collections.abc import Generator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from tests.fixtures.graphrag.corpus_fixtures import (
    PAPER_A1_ID,
    PAPER_A2_ID,
    PROJECT_A_ID,
    PROJECT_B_ID,
    load_manifest,
    seed_synthetic_corpus,
    validate_manifest,
)

from app.db.base import Base
from app.db.models import ChunkElement, Paper, PaperChunk, PaperElement, PaperPage
from app.schemas.evidence import AnchorStatus, CitationAnchor
from app.schemas.graph import (
    EntityType,
    GraphProvenanceSchema,
    GraphQualifierSchema,
    RelationshipPredicate,
)
from app.services.graphrag import (
    canonicalize_name,
    generate_entity_key,
    generate_fact_id,
    resolve_graph_source_anchor,
)
from app.services.graphrag.identity import canonicalize_qualifiers


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide an isolated, in-memory SQLite database populated with the synthetic corpus."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        seed_synthetic_corpus(session)
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


# ============================================================================
# Task 5.05: Source Anchor Resolution Tests
# ============================================================================


def test_valid_quote_with_chunk_and_element_returns_verified(db_session: Session):
    """Verify that a valid quote grounded in chunk and element returns VERIFIED
    with exact coordinates and char spans in CitationAnchor."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    prov = rel.provenance

    # Construct schema object with document_sha256
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()
    schema_prov = GraphProvenanceSchema(
        paper_id=prov.paper_id,
        chunk_id=prov.chunk_id,
        page_number=prov.page_number,
        element_id=prov.element_id,
        exact_quote=prov.exact_quote,
        char_start=prov.char_start,
        char_end=prov.char_end,
        document_sha256=paper.document_sha256,
        parser_version="1.0.0",
    )

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=schema_prov,
    )

    assert status == AnchorStatus.VERIFIED
    assert anchor is not None
    assert isinstance(anchor, CitationAnchor)
    assert anchor.page_number == prov.page_number
    assert anchor.source_element_id == prov.element_id
    assert anchor.exact_quote == prov.exact_quote
    assert anchor.source_char_start == prov.char_start
    assert anchor.source_char_end == prov.char_end
    assert anchor.document_sha256 == paper.document_sha256
    assert anchor.anchor_status == AnchorStatus.VERIFIED

    # Check bounding boxes exist and match element
    assert len(anchor.bounding_boxes) == 1
    bbox = anchor.bounding_boxes[0]
    element = db_session.query(PaperElement).filter(PaperElement.id == prov.element_id).one()
    assert bbox.x_min == float(element.bbox_x_min)
    assert bbox.y_min == float(element.bbox_y_min)
    assert bbox.x_max == float(element.bbox_x_max)
    assert bbox.y_max == float(element.bbox_y_max)
    assert bbox.page_width == float(element.page_width)
    assert bbox.page_height == float(element.page_height)


def test_graph_source_rejects_stale_parser_version(db_session: Session):
    manifest = validate_manifest(load_manifest())
    provenance = manifest.projects["project_a"].papers[0].relationships[0].provenance
    paper = db_session.query(Paper).filter(Paper.id == provenance.paper_id).one()
    supplied = GraphProvenanceSchema(
        paper_id=provenance.paper_id,
        chunk_id=provenance.chunk_id,
        page_number=provenance.page_number,
        element_id=provenance.element_id,
        exact_quote=provenance.exact_quote,
        char_start=provenance.char_start,
        char_end=provenance.char_end,
        document_sha256=paper.document_sha256,
        parser_version="stale-parser-version",
    )

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=supplied,
    )

    assert anchor is None
    assert status == AnchorStatus.UNRESOLVED


def test_graph_source_preserves_unknown_legacy_parser_version(db_session: Session):
    manifest = validate_manifest(load_manifest())
    provenance = manifest.projects["project_a"].papers[0].relationships[0].provenance
    paper = db_session.query(Paper).filter(Paper.id == provenance.paper_id).one()
    element = db_session.query(PaperElement).filter(PaperElement.id == provenance.element_id).one()
    element.parser_version = None
    supplied = GraphProvenanceSchema(
        paper_id=provenance.paper_id,
        chunk_id=provenance.chunk_id,
        page_number=provenance.page_number,
        element_id=provenance.element_id,
        exact_quote=provenance.exact_quote,
        char_start=provenance.char_start,
        char_end=provenance.char_end,
        document_sha256=paper.document_sha256,
        parser_version=None,
    )

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=supplied,
    )

    assert status == AnchorStatus.VERIFIED
    assert anchor is not None
    assert anchor.parser_version is None


def test_valid_quote_from_dict_without_element_id(db_session: Session):
    """Verify that provenance dict without element_id automatically resolves
    an overlapping element on the page."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    prov = rel.provenance

    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()
    prov_dict = {
        "paper_id": str(prov.paper_id),
        "chunk_id": str(prov.chunk_id),
        "page_number": prov.page_number,
        "exact_quote": prov.exact_quote,
        "char_start": prov.char_start,
        "char_end": prov.char_end,
        "document_sha256": paper.document_sha256,
    }

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov_dict,
    )

    assert status == AnchorStatus.VERIFIED
    assert anchor is not None
    assert anchor.source_element_id is not None
    assert len(anchor.bounding_boxes) == 1


def test_stale_document_sha256_returns_unresolved(db_session: Session):
    """Verify that mismatched document_sha256 returns UNRESOLVED."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    prov = rel.provenance

    schema_prov = GraphProvenanceSchema(
        paper_id=prov.paper_id,
        chunk_id=prov.chunk_id,
        page_number=prov.page_number,
        element_id=prov.element_id,
        exact_quote=prov.exact_quote,
        char_start=prov.char_start,
        char_end=prov.char_end,
        document_sha256="00" * 32,  # Stale / wrong hash
        parser_version="1.0.0",
    )

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=schema_prov,
    )

    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_parent_chunk_returns_unresolved(db_session: Session):
    """Verify that a chunk with chunk_type == 'parent' returns UNRESOLVED."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    prov = rel.provenance
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()

    # Create a parent chunk containing the quote
    parent_chunk = PaperChunk(
        id=uuid4(),
        paper_id=paper.id,
        chunk_type="parent",
        chunk_index=999,
        text=f"Prefix text {prov.exact_quote} suffix text",
    )
    db_session.add(parent_chunk)
    db_session.commit()

    schema_prov = GraphProvenanceSchema(
        paper_id=prov.paper_id,
        chunk_id=parent_chunk.id,
        page_number=prov.page_number,
        element_id=prov.element_id,
        exact_quote=prov.exact_quote,
        char_start=prov.char_start,
        char_end=prov.char_end,
        document_sha256=paper.document_sha256,
        parser_version="1.0.0",
    )

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=schema_prov,
    )

    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_reindexed_or_missing_chunk_returns_unresolved(db_session: Session):
    """Verify that a non-existent or retired chunk ID returns UNRESOLVED."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    prov = rel.provenance
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()

    schema_prov = GraphProvenanceSchema(
        paper_id=prov.paper_id,
        chunk_id=uuid4(),  # Reindexed / missing chunk
        page_number=prov.page_number,
        element_id=prov.element_id,
        exact_quote=prov.exact_quote,
        char_start=prov.char_start,
        char_end=prov.char_end,
        document_sha256=paper.document_sha256,
        parser_version="1.0.0",
    )

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=schema_prov,
    )

    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_quote_not_present_in_chunk_returns_unresolved(db_session: Session):
    """Verify that quote not present within chunk text returns UNRESOLVED."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    prov = rel.provenance
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()

    # Find another child chunk in paper A1 that does not contain this quote
    other_chunk = (
        db_session.query(PaperChunk)
        .filter(
            PaperChunk.paper_id == paper.id,
            PaperChunk.id != prov.chunk_id,
            PaperChunk.chunk_type == "child",
        )
        .first()
    )
    assert other_chunk is not None

    schema_prov = GraphProvenanceSchema(
        paper_id=prov.paper_id,
        chunk_id=other_chunk.id,
        page_number=prov.page_number,
        element_id=prov.element_id,
        exact_quote=prov.exact_quote,
        char_start=prov.char_start,
        char_end=prov.char_end,
        document_sha256=paper.document_sha256,
        parser_version="1.0.0",
    )

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=schema_prov,
    )

    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_quote_outside_page_text_returns_unresolved(db_session: Session):
    """Verify that quote not present in page raw text returns UNRESOLVED."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    prov = rel.provenance
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()

    # Provide a quote that does not appear on page 1
    invented_quote = "This non-existent quote does not exist on page one of paper."
    prov_dict = {
        "paper_id": prov.paper_id,
        "chunk_id": prov.chunk_id,
        "page_number": prov.page_number,
        "element_id": prov.element_id,
        "exact_quote": invented_quote,
        "char_start": 0,
        "char_end": len(invented_quote),
        "document_sha256": paper.document_sha256,
    }

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov_dict,
    )

    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_chunk_link_does_not_verify_quote_absent_from_linked_element(db_session: Session):
    paper = db_session.query(Paper).filter(Paper.id == PAPER_A1_ID).one()
    quote = "Supported by page text and chunk only."
    page = PaperPage(
        id=uuid4(),
        paper_id=paper.id,
        page_number=98,
        width=612,
        height=792,
        raw_text=quote,
    )
    element = PaperElement(
        id=uuid4(),
        paper_id=paper.id,
        page_number=98,
        element_index=998,
        element_type="paragraph",
        text="Different text in linked element",
    )
    chunk = PaperChunk(
        id=uuid4(),
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=887,
        text=quote,
    )
    link = ChunkElement(
        id=uuid4(),
        chunk_id=chunk.id,
        element_id=element.id,
        order_index=0,
    )
    db_session.add_all([page, element, chunk, link])
    db_session.commit()

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance={
            "paper_id": paper.id,
            "chunk_id": chunk.id,
            "page_number": page.page_number,
            "element_id": element.id,
            "exact_quote": quote,
            "char_start": 0,
            "char_end": len(quote),
            "document_sha256": paper.document_sha256,
        },
    )

    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_wrong_project_id_returns_unresolved(db_session: Session):
    """Verify that resolving provenance under a different project ID returns UNRESOLVED."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    prov = rel.provenance
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()

    schema_prov = GraphProvenanceSchema(
        paper_id=prov.paper_id,
        chunk_id=prov.chunk_id,
        page_number=prov.page_number,
        element_id=prov.element_id,
        exact_quote=prov.exact_quote,
        char_start=prov.char_start,
        char_end=prov.char_end,
        document_sha256=paper.document_sha256,
        parser_version="1.0.0",
    )

    # Resolve using Project B instead of Project A
    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_B_ID,
        provenance=schema_prov,
    )

    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_repeated_ambiguous_quote_without_offsets_returns_unresolved(db_session: Session):
    """Verify that repeated ambiguous text without character offsets returns UNRESOLVED."""
    paper = db_session.query(Paper).filter(Paper.id == PAPER_A1_ID).one()

    # Create a page with repeated text
    repeated_phrase = "deep learning calibration"
    page_text = (
        f"Intro: {repeated_phrase} is important. "
        f"Later discussion: {repeated_phrase} is also critical."
    )
    test_page = PaperPage(
        id=uuid4(),
        paper_id=paper.id,
        page_number=99,
        width=612.0,
        height=792.0,
        raw_text=page_text,
    )
    test_elem = PaperElement(
        id=uuid4(),
        paper_id=paper.id,
        page_number=99,
        element_index=999,
        element_type="paragraph",
        text=page_text,
        bbox_x_min=72.0,
        bbox_y_min=72.0,
        bbox_x_max=500.0,
        bbox_y_max=200.0,
        page_width=612.0,
        page_height=792.0,
        parser_version="1.0.0",
    )
    test_chunk = PaperChunk(
        id=uuid4(),
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=888,
        text=page_text,
    )
    chunk_element = ChunkElement(
        id=uuid4(),
        chunk_id=test_chunk.id,
        element_id=test_elem.id,
        order_index=0,
    )
    db_session.add_all([test_page, test_elem, test_chunk, chunk_element])
    db_session.commit()

    # Provenance without offsets
    prov_dict = {
        "paper_id": paper.id,
        "chunk_id": test_chunk.id,
        "page_number": 99,
        "element_id": test_elem.id,
        "exact_quote": repeated_phrase,
        "document_sha256": paper.document_sha256,
        # char_start and char_end omitted
    }

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov_dict,
    )

    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None

    # Now verify that providing exact offset disambiguates it successfully!
    first_occurrence_offset = page_text.find(repeated_phrase)
    prov_dict["char_start"] = first_occurrence_offset
    prov_dict["char_end"] = first_occurrence_offset + len(repeated_phrase)

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov_dict,
    )
    assert status == AnchorStatus.VERIFIED
    assert anchor is not None
    assert anchor.source_char_start == first_occurrence_offset


def test_quote_slice_mismatch_returns_unresolved(db_session: Session):
    """Verify that char_start and char_end pointing to different text than quote
    returns UNRESOLVED."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    prov = rel.provenance
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()

    prov_dict = {
        "paper_id": prov.paper_id,
        "chunk_id": prov.chunk_id,
        "page_number": prov.page_number,
        "element_id": prov.element_id,
        "exact_quote": prov.exact_quote,
        "char_start": prov.char_start + 5,  # Shifted slice
        "char_end": prov.char_end + 5,
        "document_sha256": paper.document_sha256,
    }

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov_dict,
    )
    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_paper_status_not_ready_returns_unresolved(db_session: Session):
    """Verify that paper with status != 'READY' returns UNRESOLVED."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    prov = rel.provenance
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()

    paper.status = "PROCESSING"
    db_session.commit()

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov.model_dump(),
    )
    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


# ============================================================================
# Task 5.06: Deterministic Identity Tests
# ============================================================================


def test_canonicalize_name():
    """Verify canonicalize_name lowercases, strips, and collapses multiple whitespace runs."""
    assert (
        canonicalize_name("   Convolutional   Neural   Network   ")
        == "convolutional neural network"
    )
    assert canonicalize_name("BERT\n\tModel") == "bert model"
    assert canonicalize_name("  ASR  ") == "asr"
    assert canonicalize_name("") == ""
    assert canonicalize_name("   ") == ""


def test_entity_key_determinism():
    """Verify that repeated generation with identical arguments produces identical keys."""
    key1 = generate_entity_key(PROJECT_A_ID, EntityType.METHOD, "Temperature Scaling")
    key2 = generate_entity_key(PROJECT_A_ID, EntityType.METHOD, "  Temperature   Scaling  ")
    key3 = generate_entity_key(PROJECT_A_ID, "Method", "temperature scaling")

    assert key1 == key2 == key3
    assert key1.startswith("method_")
    assert len(key1) == len("method_") + 16


def test_entity_key_cross_project_isolation():
    """Verify that the same entity name in two different projects produces different keys."""
    # Cross-project acronyms: ASR means "Attack Success Rate" in Project A,
    # and "Automatic Speech Recognition" in Project B.
    key_a = generate_entity_key(PROJECT_A_ID, EntityType.CONCEPT, "ASR")
    key_b = generate_entity_key(PROJECT_B_ID, EntityType.CONCEPT, "ASR")

    assert key_a != key_b
    assert key_a.startswith("concept_")
    assert key_b.startswith("concept_")


def test_entity_key_with_external_id():
    """Verify external ID validation and project-scoped key generation."""
    # Valid external ID
    ext_key_a = generate_entity_key(
        PROJECT_A_ID,
        EntityType.MODEL,
        "Conformer",
        external_id="arxiv:2005.08100",
    )
    assert ext_key_a.startswith(f"proj_{PROJECT_A_ID.hex}_model_arxiv:2005.08100")

    # Cross-project isolation with external ID
    ext_key_b = generate_entity_key(
        PROJECT_B_ID,
        EntityType.MODEL,
        "Conformer",
        external_id="arxiv:2005.08100",
    )
    assert ext_key_a != ext_key_b

    # Invalid external ID formats raise ValueError
    with pytest.raises(ValueError, match="Invalid external_id format"):
        generate_entity_key(PROJECT_A_ID, EntityType.PAPER, "Paper", external_id="no_colon_id")

    with pytest.raises(ValueError, match="Invalid external_id format"):
        generate_entity_key(PROJECT_A_ID, EntityType.PAPER, "Paper", external_id="doi:")

    with pytest.raises(ValueError, match="Invalid external_id format"):
        generate_entity_key(PROJECT_A_ID, EntityType.PAPER, "Paper", external_id="doi:with space")


def test_entity_key_cross_project_isolation_sequential_uuids():
    """Verify that two sequential project UUIDs referencing the same external ID
    produce different keys.
    """
    proj_1 = UUID("00000000-0000-0000-0000-000000000001")
    proj_2 = UUID("00000000-0000-0000-0000-000000000002")
    key_1 = generate_entity_key(
        proj_1,
        EntityType.MODEL,
        "Conformer",
        external_id="arxiv:2005.08100",
    )
    key_2 = generate_entity_key(
        proj_2,
        EntityType.MODEL,
        "Conformer",
        external_id="arxiv:2005.08100",
    )
    assert key_1 != key_2
    assert key_1 == f"proj_{proj_1.hex}_model_arxiv:2005.08100"
    assert key_2 == f"proj_{proj_2.hex}_model_arxiv:2005.08100"


def test_entity_key_empty_name_without_external_id_raises():
    """Verify that empty entity name raises ValueError when external_id is omitted."""
    with pytest.raises(ValueError, match="Entity name cannot be empty"):
        generate_entity_key(PROJECT_A_ID, EntityType.TASK, "   ")


def test_canonicalize_qualifiers_order_and_normalization():
    """Verify qualifier dictionary canonicalization is sorted and normalized."""
    q1 = {"dataset": "ImageNet", "metric": "ECE", "result_value": 0.021}
    q2 = {"result_value": 0.021, "Metric": "  ECE  ", "DATASET": "imagenet"}
    assert canonicalize_qualifiers(q1) == canonicalize_qualifiers(q2)
    assert canonicalize_qualifiers(None) == ""
    assert canonicalize_qualifiers({}) == ""


def test_fact_id_determinism_and_multi_paper_provenance():
    """Verify fact ID determinism and multi-paper distinct fact IDs."""
    # Same relation asserted in Paper A1 vs Paper A2
    fact_id_a1 = generate_fact_id(
        project_id=PROJECT_A_ID,
        paper_id=PAPER_A1_ID,
        predicate=RelationshipPredicate.EVALUATED_ON,
        subject_key="method_1234567890abcdef",
        object_key="dataset_abcdef1234567890",
        char_start=100,
        char_end=150,
        qualifiers={"metric": "ECE", "split": "test"},
    )

    # Repeating with identical inputs produces identical fact ID
    fact_id_a1_repeat = generate_fact_id(
        project_id=PROJECT_A_ID,
        paper_id=PAPER_A1_ID,
        predicate="EVALUATED_ON",
        subject_key="method_1234567890abcdef",
        object_key="dataset_abcdef1234567890",
        char_start=100,
        char_end=150,
        qualifiers={"split": "test", "metric": "ece"},  # Reordered and lowercased
    )
    assert fact_id_a1 == fact_id_a1_repeat
    assert fact_id_a1.startswith("fact_")
    assert len(fact_id_a1) == len("fact_") + 24

    # Two papers asserting the SAME relation MUST produce distinct fact IDs
    fact_id_a2 = generate_fact_id(
        project_id=PROJECT_A_ID,
        paper_id=PAPER_A2_ID,
        predicate=RelationshipPredicate.EVALUATED_ON,
        subject_key="method_1234567890abcdef",
        object_key="dataset_abcdef1234567890",
        char_start=100,
        char_end=150,
        qualifiers={"metric": "ECE", "split": "test"},
    )
    assert fact_id_a1 != fact_id_a2

    # Different source_generation produces distinct fact IDs
    fact_id_gen = generate_fact_id(
        project_id=PROJECT_A_ID,
        paper_id=PAPER_A1_ID,
        predicate="EVALUATED_ON",
        subject_key="method_1234567890abcdef",
        object_key="dataset_abcdef1234567890",
        char_start=100,
        char_end=150,
        qualifiers={"metric": "ECE", "split": "test"},
        source_generation="generation_2",
    )
    assert fact_id_a1 != fact_id_gen


@pytest.mark.parametrize(
    ("bad_field", "bad_val"),
    [
        ("paper_id", "not-a-uuid"),
        ("chunk_id", "not-a-uuid"),
        ("element_id", "not-a-uuid"),
        ("paper_id", None),
        ("chunk_id", None),
        ("page_number", 0),
        ("page_number", -1),
        ("page_number", "1"),
        ("exact_quote", ""),
        ("exact_quote", "   "),
        ("exact_quote", None),
        ("exact_quote", 123),
        ("char_start", -1),
        ("char_end", 0),
        ("page_number", 999),  # Non-existent page
    ],
)
def test_resolve_provenance_malformed_inputs(db_session: Session, bad_field: str, bad_val):
    """Verify that malformed provenance fields safely return UNRESOLVED."""
    manifest = validate_manifest(load_manifest())
    prov = manifest.projects["project_a"].papers[0].relationships[0].provenance
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()

    prov_dict = {
        "paper_id": prov.paper_id,
        "chunk_id": prov.chunk_id,
        "page_number": prov.page_number,
        "element_id": prov.element_id,
        "exact_quote": prov.exact_quote,
        "char_start": prov.char_start,
        "char_end": prov.char_end,
        "document_sha256": paper.document_sha256,
    }
    prov_dict[bad_field] = bad_val

    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov_dict,
    )
    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_resolve_provenance_non_dict_non_model(db_session: Session):
    """Verify non-model non-dict input returns UNRESOLVED."""
    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance="invalid_provenance_type",  # type: ignore
    )
    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_resolve_provenance_partial_char_offsets(db_session: Session):
    """Verify single offset (only char_start or only char_end) resolves correctly."""
    manifest = validate_manifest(load_manifest())
    prov = manifest.projects["project_a"].papers[0].relationships[0].provenance
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()

    # Only char_start
    prov_dict_start = {
        "paper_id": prov.paper_id,
        "chunk_id": prov.chunk_id,
        "page_number": prov.page_number,
        "exact_quote": prov.exact_quote,
        "char_start": prov.char_start,
        "document_sha256": paper.document_sha256,
    }
    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov_dict_start,
    )
    assert status == AnchorStatus.VERIFIED
    assert anchor is not None
    assert anchor.source_char_start == prov.char_start

    # Only char_end
    prov_dict_end = {
        "paper_id": prov.paper_id,
        "chunk_id": prov.chunk_id,
        "page_number": prov.page_number,
        "exact_quote": prov.exact_quote,
        "char_end": prov.char_end,
        "document_sha256": paper.document_sha256,
    }
    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov_dict_end,
    )
    assert status == AnchorStatus.VERIFIED
    assert anchor is not None
    assert anchor.source_char_end == prov.char_end


def test_resolve_provenance_unmatched_element(db_session: Session):
    """Verify that an element not containing the quote and not linked to the chunk
    returns UNRESOLVED."""
    manifest = validate_manifest(load_manifest())
    prov = manifest.projects["project_a"].papers[0].relationships[0].provenance
    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()

    # Create an unrelated element on the same page
    unrelated_elem = PaperElement(
        id=uuid4(),
        paper_id=paper.id,
        page_number=prov.page_number,
        element_index=888,
        element_type="paragraph",
        text="Completely unrelated text without quote.",
        bbox_x_min=10.0,
        bbox_y_min=10.0,
        bbox_x_max=100.0,
        bbox_y_max=50.0,
        page_width=612.0,
        page_height=792.0,
        parser_version="1.0.0",
    )
    db_session.add(unrelated_elem)
    db_session.commit()

    prov_dict = {
        "paper_id": prov.paper_id,
        "chunk_id": prov.chunk_id,
        "page_number": prov.page_number,
        "element_id": unrelated_elem.id,
        "exact_quote": prov.exact_quote,
        "char_start": prov.char_start,
        "char_end": prov.char_end,
        "document_sha256": paper.document_sha256,
    }
    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov_dict,
    )
    assert status == AnchorStatus.UNRESOLVED
    assert anchor is None


def test_resolve_provenance_bottom_left_coordinate_origin(db_session: Session):
    """Verify coordinate origin handling for BOTTOM_LEFT elements."""
    manifest = validate_manifest(load_manifest())
    prov = manifest.projects["project_a"].papers[0].relationships[0].provenance
    element = db_session.query(PaperElement).filter(PaperElement.id == prov.element_id).one()
    element.coordinate_origin = "BOTTOM_LEFT"
    db_session.commit()

    paper = db_session.query(Paper).filter(Paper.id == prov.paper_id).one()
    prov_dict = {
        "paper_id": prov.paper_id,
        "chunk_id": prov.chunk_id,
        "page_number": prov.page_number,
        "element_id": prov.element_id,
        "exact_quote": prov.exact_quote,
        "char_start": prov.char_start,
        "char_end": prov.char_end,
        "document_sha256": paper.document_sha256,
    }
    anchor, status = resolve_graph_source_anchor(
        db=db_session,
        project_id=PROJECT_A_ID,
        provenance=prov_dict,
    )
    assert status == AnchorStatus.VERIFIED
    assert anchor is not None
    assert len(anchor.bounding_boxes) == 1
    assert anchor.bounding_boxes[0].origin.value == "BOTTOM_LEFT"


def test_canonicalize_qualifiers_rich_types():
    """Verify qualifier canonicalization with int, float, bool, and whitespace keys."""
    q = {
        "float_val": 42.1234567,
        "int_val": 10,
        "bool_val": True,
        "   ": "should_be_skipped",
        "nested_name": "  Some   Label  ",
    }
    result = canonicalize_qualifiers(q)
    assert "float_val=42.1235" in result
    assert "int_val=10" in result
    assert "bool_val=true" in result
    assert "nested_name=some label" in result


def test_canonicalize_qualifiers_with_pydantic_schema():
    """Verify that canonicalize_qualifiers accepts a Pydantic BaseModel (GraphQualifierSchema)."""
    schema_qual = GraphQualifierSchema(
        metric="ECE",
        dataset="ImageNet",
        result_value=0.021,
    )
    canon_str = canonicalize_qualifiers(schema_qual)
    assert isinstance(canon_str, str)
    assert "dataset=imagenet" in canon_str
    assert "metric=ece" in canon_str
    assert "result_value=0.021" in canon_str
    assert "polarity=positive" in canon_str

    # Verify generate_fact_id also accepts GraphQualifierSchema without raising AttributeError
    fact_id = generate_fact_id(
        project_id=PROJECT_A_ID,
        paper_id=PAPER_A1_ID,
        predicate=RelationshipPredicate.EVALUATED_ON,
        subject_key="method_1234567890abcdef",
        object_key="dataset_abcdef1234567890",
        char_start=100,
        char_end=150,
        qualifiers=schema_qual,
    )
    assert fact_id.startswith("fact_")


def test_generate_fact_id_empty_endpoint_keys_raises():
    """Verify that empty or whitespace-only subject_key and object_key raise ValueError."""
    with pytest.raises(ValueError, match="subject_key cannot be empty"):
        generate_fact_id(
            project_id=PROJECT_A_ID,
            paper_id=PAPER_A1_ID,
            predicate=RelationshipPredicate.EVALUATED_ON,
            subject_key="   ",
            object_key="dataset_abcdef1234567890",
            char_start=100,
            char_end=150,
        )

    with pytest.raises(ValueError, match="object_key cannot be empty"):
        generate_fact_id(
            project_id=PROJECT_A_ID,
            paper_id=PAPER_A1_ID,
            predicate=RelationshipPredicate.EVALUATED_ON,
            subject_key="method_1234567890abcdef",
            object_key="   ",
            char_start=100,
            char_end=150,
        )
