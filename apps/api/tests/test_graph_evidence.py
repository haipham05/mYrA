"""Tests for GraphRAG evidence resolution into M1 EvidenceItem and CitationAnchor (Task 5.26).

Verifies:
1. Verified exact quote: Valid snapshot with matching paper, child chunk, element,
   page text and hash produces EvidenceItem with AnchorStatus.VERIFIED and CitationAnchor.
2. Stale chunk: Chunk is missing, marked non-child, or text modified -> returns UNRESOLVED.
3. Changed PDF: paper.document_sha256 changed -> drops fact, returns UNRESOLVED.
4. Graph-vs-text disagreement: Snapshot quote does not match page text -> returns UNRESOLVED.
5. Wrong project: Snapshot or candidate belongs to Project B while querying in Project A
   -> returns UNRESOLVED.
6. Bare Neo4j edge without PostgreSQL ground truth: Fact ID not in DB -> returns UNRESOLVED.
7. Batch resolution & deduplication: Sequential IDs (G1, G2...), quote & fact ID deduplication.
8. Candidate helper & integration: Extracting fact IDs from relationship candidates,
   contradiction pairs, and corpus themes resolves them into verified EvidenceItems
   with live anchors.
"""

from __future__ import annotations

from collections.abc import Generator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from tests.fixtures.graphrag.corpus_fixtures import (
    PAPER_A1_ID,
    PAPER_B1_ID,
    PROJECT_A_ID,
    PROJECT_B_ID,
    ManifestRelationship,
    load_manifest,
    seed_synthetic_corpus,
    validate_manifest,
)

from app.db.base import Base
from app.db.models import ChunkElement, GraphFactSnapshot, Paper, PaperChunk, PaperElement
from app.schemas.evidence import AnchorStatus, CitationAnchor, EvidenceItem
from app.services.graphrag.evidence import (
    extract_fact_ids_from_candidates,
    resolve_graph_fact_to_evidence,
    resolve_graph_facts_to_evidence,
)


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


def _create_snapshot(
    db: Session,
    project_id: UUID,
    rel: ManifestRelationship,
    document_sha256: str,
    fact_id: str | None = None,
    exact_quote: str | None = None,
    char_start: int | None = None,
    char_end: int | None = None,
    chunk_id: UUID | None = None,
    generation_id: str = "gen-evidence-01",
) -> GraphFactSnapshot:
    """Helper to persist a GraphFactSnapshot from a manifest relationship."""
    prov = rel.provenance
    q = exact_quote if exact_quote is not None else prov.exact_quote
    cs = char_start if char_start is not None else prov.char_start
    ce = char_end if char_end is not None else prov.char_end
    cid = chunk_id if chunk_id is not None else prov.chunk_id
    fid = fact_id if fact_id is not None else rel.fact_id

    snapshot = GraphFactSnapshot(
        fact_id=fid,
        project_id=project_id,
        paper_id=prov.paper_id,
        generation_id=generation_id,
        subject_key=f"{rel.subject.type.lower()}:{rel.subject.name.lower().replace(' ', '_')}",
        subject_name=rel.subject.name,
        subject_type=rel.subject.type,
        predicate=rel.predicate,
        object_key=f"{rel.object.type.lower()}:{rel.object.name.lower().replace(' ', '_')}",
        object_name=rel.object.name,
        object_type=rel.object.type,
        qualifiers=rel.qualifiers or None,
        chunk_id=cid,
        page_number=prov.page_number,
        element_id=prov.element_id,
        char_start=cs,
        char_end=ce,
        exact_quote=q,
        document_sha256=document_sha256,
        validation_version="1.0.0",
    )
    db.add(snapshot)
    db.flush()
    return snapshot


# ============================================================================
# Test 1: Verified Exact Quote
# ============================================================================


def test_verified_exact_quote(db_session: Session):
    """Test 1: Valid snapshot with matching paper, child chunk, element, page text and hash

    produces EvidenceItem with AnchorStatus.VERIFIED and CitationAnchor pointing to exact
    character offsets and bboxes, and tests parent context expansion.
    """
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    paper = db_session.query(Paper).filter(Paper.id == PAPER_A1_ID).one()
    source_element = (
        db_session.query(PaperElement).filter(PaperElement.id == rel.provenance.element_id).one()
    )
    source_element.parser_version = "docling-v3"
    db_session.commit()

    snapshot = _create_snapshot(db_session, PROJECT_A_ID, rel, paper.document_sha256)

    # Resolution via fact_id str
    evidence_item, anchor, status = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, rel.fact_id, evidence_id="G1"
    )

    assert status == AnchorStatus.VERIFIED
    assert anchor is not None
    assert evidence_item is not None

    # Check EvidenceItem attributes
    assert evidence_item.id == "G1"
    assert evidence_item.paper_id == paper.id
    assert evidence_item.paper_title == paper.filename
    assert evidence_item.chunk_id == rel.provenance.chunk_id
    assert evidence_item.quote == rel.provenance.exact_quote
    assert evidence_item.page_number == rel.provenance.page_number
    assert evidence_item.document_sha256 == paper.document_sha256
    assert evidence_item.source_element_ids == [rel.provenance.element_id]
    assert len(evidence_item.bounding_boxes) > 0
    assert evidence_item.anchors == [anchor]

    # Without parent chunk, parent_context defaults to chunk.text
    chunk = db_session.query(PaperChunk).filter(PaperChunk.id == rel.provenance.chunk_id).one()
    assert evidence_item.parent_context == chunk.text

    # Check CitationAnchor attributes
    assert isinstance(anchor, CitationAnchor)
    assert anchor.page_number == rel.provenance.page_number
    assert anchor.source_element_id == rel.provenance.element_id
    assert anchor.exact_quote == rel.provenance.exact_quote
    assert anchor.source_char_start == rel.provenance.char_start
    assert anchor.source_char_end == rel.provenance.char_end
    assert anchor.document_sha256 == paper.document_sha256
    assert anchor.anchor_status == AnchorStatus.VERIFIED
    assert anchor.parser_version == "docling-v3"
    assert len(anchor.bounding_boxes) > 0

    # Resolution via GraphFactSnapshot object directly
    item_from_snap, anchor_from_snap, status_from_snap = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, snapshot, evidence_id="G2"
    )
    assert status_from_snap == AnchorStatus.VERIFIED
    assert item_from_snap is not None and item_from_snap.id == "G2"
    assert (
        anchor_from_snap is not None and anchor_from_snap.exact_quote == rel.provenance.exact_quote
    )
    assert anchor_from_snap is not None and anchor_from_snap.parser_version == "docling-v3"

    # Resolution via dict candidate with fact_id
    item_from_dict, _, status_from_dict = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, {"fact_id": rel.fact_id}, evidence_id="G3"
    )
    assert status_from_dict == AnchorStatus.VERIFIED
    assert item_from_dict is not None and item_from_dict.id == "G3"

    # Now verify Parent Context Expansion when parent chunk is present
    parent_chunk = PaperChunk(
        id=uuid4(),
        paper_id=paper.id,
        chunk_type="parent",
        chunk_index=99,
        text="Parent section context with broader surrounding paragraphs.",
    )
    db_session.add(parent_chunk)
    db_session.flush()

    db_session.add(
        ChunkElement(
            chunk_id=parent_chunk.id,
            element_id=rel.provenance.element_id,
            order_index=0,
        )
    )
    db_session.flush()

    expanded_item, _, exp_status = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, rel.fact_id, evidence_id="G4"
    )
    assert exp_status == AnchorStatus.VERIFIED
    assert expanded_item is not None
    assert expanded_item.parent_context == parent_chunk.text


# ============================================================================
# Test 2: Stale Chunk
# ============================================================================


def test_stale_chunk(db_session: Session):
    """Test 2: Chunk is missing, marked non-child, or chunk text modified so it doesn't contain

    the quote -> drops fact, returns (None, None, UNRESOLVED).
    """
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    paper = db_session.query(Paper).filter(Paper.id == PAPER_A1_ID).one()

    # Subcase 2a: Missing chunk (non-existent chunk_id)
    non_existent_chunk_id = uuid4()
    snap_missing = _create_snapshot(
        db_session,
        PROJECT_A_ID,
        rel,
        paper.document_sha256,
        fact_id="fact-missing-chunk",
        chunk_id=non_existent_chunk_id,
    )
    item, anchor, status = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, snap_missing, evidence_id="G1"
    )
    assert status == AnchorStatus.UNRESOLVED
    assert item is None
    assert anchor is None

    # Subcase 2b: Chunk marked non-child (chunk_type == "parent")
    chunk = db_session.query(PaperChunk).filter(PaperChunk.id == rel.provenance.chunk_id).one()
    orig_type = chunk.chunk_type
    chunk.chunk_type = "parent"
    db_session.flush()

    snap_non_child = _create_snapshot(
        db_session,
        PROJECT_A_ID,
        rel,
        paper.document_sha256,
        fact_id="fact-non-child-chunk",
    )
    item, anchor, status = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, snap_non_child, evidence_id="G1"
    )
    assert status == AnchorStatus.UNRESOLVED
    assert item is None
    assert anchor is None

    # Restore chunk_type
    chunk.chunk_type = orig_type
    db_session.flush()

    # Subcase 2c: Chunk text modified so quote is no longer present
    orig_text = chunk.text
    chunk.text = "This chunk text has been replaced and does not include the target quote."
    db_session.flush()

    item, anchor, status = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, snap_non_child, evidence_id="G1"
    )
    assert status == AnchorStatus.UNRESOLVED
    assert item is None
    assert anchor is None

    # Restore chunk text
    chunk.text = orig_text
    db_session.flush()


# ============================================================================
# Test 3: Changed PDF Document Hash
# ============================================================================


def test_changed_pdf_hash(db_session: Session):
    """Test 3: paper.document_sha256 changed (re-uploaded or updated) -> drops fact,

    returns (None, None, UNRESOLVED).
    """
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    paper = db_session.query(Paper).filter(Paper.id == PAPER_A1_ID).one()

    snapshot = _create_snapshot(db_session, PROJECT_A_ID, rel, paper.document_sha256)

    # Simulate paper re-upload or update with new PDF hash
    paper.document_sha256 = "f" * 64
    db_session.flush()

    item, anchor, status = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, snapshot, evidence_id="G1"
    )
    assert status == AnchorStatus.UNRESOLVED
    assert item is None
    assert anchor is None


# ============================================================================
# Test 4: Graph-vs-Text Disagreement
# ============================================================================


def test_graph_vs_text_disagreement(db_session: Session):
    """Test 4: Snapshot quote does not match page text -> drops fact,

    returns (None, None, UNRESOLVED).
    """
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    paper = db_session.query(Paper).filter(Paper.id == PAPER_A1_ID).one()

    # Snapshot with quote that does not appear in page text
    snap_disagree = _create_snapshot(
        db_session,
        PROJECT_A_ID,
        rel,
        paper.document_sha256,
        fact_id="fact-quote-mismatch",
        exact_quote="Completely bogus quote that does not appear on page one at all.",
        char_start=0,
        char_end=62,
    )

    item, anchor, status = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, snap_disagree, evidence_id="G1"
    )
    assert status == AnchorStatus.UNRESOLVED
    assert item is None
    assert anchor is None


# ============================================================================
# Test 5: Wrong Project Isolation
# ============================================================================


def test_wrong_project_isolation(db_session: Session):
    """Test 5: Snapshot or candidate belongs to Project B while querying in Project A

    -> drops fact, returns (None, None, UNRESOLVED).
    """
    manifest = validate_manifest(load_manifest())
    rel_b = manifest.projects["project_b"].papers[0].relationships[0]
    paper_b = db_session.query(Paper).filter(Paper.id == PAPER_B1_ID).one()

    snap_b = _create_snapshot(db_session, PROJECT_B_ID, rel_b, paper_b.document_sha256)

    # Subcase 5a: Snapshot object belonging to Project B queried with Project A
    item_a, anchor_a, status_a = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, snap_b, evidence_id="G1"
    )
    assert status_a == AnchorStatus.UNRESOLVED
    assert item_a is None
    assert anchor_a is None

    # Subcase 5b: Fact ID belonging to Project B snapshot queried in Project A
    item_fid, anchor_fid, status_fid = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, snap_b.fact_id, evidence_id="G1"
    )
    assert status_fid == AnchorStatus.UNRESOLVED
    assert item_fid is None
    assert anchor_fid is None

    # Subcase 5c: Candidate dict with explicit project_id=PROJECT_B_ID queried in Project A
    cand_b = {
        "fact_id": snap_b.fact_id,
        "project_id": str(PROJECT_B_ID),
    }
    item_dict, anchor_dict, status_dict = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, cand_b, evidence_id="G1"
    )
    assert status_dict == AnchorStatus.UNRESOLVED
    assert item_dict is None
    assert anchor_dict is None

    # Subcase 5d: Candidate dict with Paper B provenance queried in Project A
    prov_b_dict = {
        "paper_id": str(PAPER_B1_ID),
        "chunk_id": str(rel_b.provenance.chunk_id),
        "element_id": str(rel_b.provenance.element_id),
        "page_number": rel_b.provenance.page_number,
        "exact_quote": rel_b.provenance.exact_quote,
        "document_sha256": paper_b.document_sha256,
    }
    item_prov, _, status_prov = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, prov_b_dict, evidence_id="G1"
    )
    assert status_prov == AnchorStatus.UNRESOLVED
    assert item_prov is None


# ============================================================================
# Test 6: Bare Neo4j Edge Without PostgreSQL Ground Truth
# ============================================================================


def test_bare_neo4j_edge_dropped(db_session: Session):
    """Test 6: A fact ID that does not exist in graph_fact_snapshots and has no

    PostgreSQL backing -> drops fact, returns (None, None, UNRESOLVED) (never cites bare edge!).
    """
    # Bare ID string with no snapshot in DB
    item_str, anchor_str, status_str = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, "bare:neo4j:edge:unbacked_999", evidence_id="G1"
    )
    assert status_str == AnchorStatus.UNRESOLVED
    assert item_str is None
    assert anchor_str is None

    # Bare candidate dict with no snapshot in DB and incomplete provenance
    item_dict, anchor_dict, status_dict = resolve_graph_fact_to_evidence(
        db_session,
        PROJECT_A_ID,
        {"fact_id": "bare:neo4j:edge:unbacked_999", "predicate": "EVALUATED_ON"},
        evidence_id="G1",
    )
    assert status_dict == AnchorStatus.UNRESOLVED
    assert item_dict is None
    assert anchor_dict is None

    # Bare candidate dict with partial provenance missing exact_quote or sha256
    partial_prov_dict = {
        "fact_id": "bare:neo4j:edge:partial",
        "paper_id": str(PAPER_A1_ID),
        "chunk_id": str(uuid4()),
        "page_number": 1,
    }
    item_partial, _, status_partial = resolve_graph_fact_to_evidence(
        db_session, PROJECT_A_ID, partial_prov_dict, evidence_id="G1"
    )
    assert status_partial == AnchorStatus.UNRESOLVED
    assert item_partial is None


# ============================================================================
# Test 7: Batch Resolution & Deduplication
# ============================================================================


def test_batch_resolution_and_deduplication(db_session: Session):
    """Test 7: Sequential IDs (G1, G2...), deduplication of identical quotes or fact IDs,

    and omission of unverified facts.
    """
    manifest = validate_manifest(load_manifest())
    paper_a1 = manifest.projects["project_a"].papers[0]
    paper_row = db_session.query(Paper).filter(Paper.id == PAPER_A1_ID).one()

    rel_0 = paper_a1.relationships[0]  # quote 1: "On Calibration of Modern Neural Networks"
    rel_1 = paper_a1.relationships[1]  # quote 2: "We evaluate AURC on ImageNet calibration."
    rel_2 = paper_a1.relationships[
        2
    ]  # quote 3: "Temperature scaling achieves 2.1% ECE on ImageNet."

    snap_0 = _create_snapshot(db_session, PROJECT_A_ID, rel_0, paper_row.document_sha256)
    snap_1 = _create_snapshot(db_session, PROJECT_A_ID, rel_1, paper_row.document_sha256)
    snap_2 = _create_snapshot(db_session, PROJECT_A_ID, rel_2, paper_row.document_sha256)

    # Snapshot 3 has a different fact_id but duplicates quote 1
    snap_dup_quote = _create_snapshot(
        db_session,
        PROJECT_A_ID,
        rel_0,
        paper_row.document_sha256,
        fact_id="fact-duplicate-quote-0",
    )

    batch_inputs = [
        snap_0.fact_id,  # 1. Valid fact 0 -> resolves to G1
        snap_0.fact_id,  # 2. Duplicate fact ID -> deduplicated / skipped
        "bare-neo4j-edge-unbacked",  # 3. Bare edge -> dropped (unresolved)
        snap_dup_quote.fact_id,  # 4. Duplicate quote -> deduplicated / skipped
        snap_1.fact_id,  # 5. Valid fact 1 -> resolves to G2
        "non-existent-fact-id-404",  # 6. Unbacked ID -> dropped
        snap_2,  # 7. Valid snapshot 2 -> resolves to G3
    ]

    evidence_items = resolve_graph_facts_to_evidence(
        db=db_session,
        project_id=PROJECT_A_ID,
        facts=batch_inputs,
        prefix="G",
        start_index=1,
    )

    assert len(evidence_items) == 3

    # IDs must be strictly sequential: G1, G2, G3
    assert [e.id for e in evidence_items] == ["G1", "G2", "G3"]

    # Verify quotes match distinct valid relationships
    assert evidence_items[0].quote == rel_0.provenance.exact_quote
    assert evidence_items[1].quote == rel_1.provenance.exact_quote
    assert evidence_items[2].quote == rel_2.provenance.exact_quote

    # Each evidence item contains a verified live CitationAnchor
    for item in evidence_items:
        assert isinstance(item, EvidenceItem)
        assert len(item.anchors) == 1
        assert item.anchors[0].anchor_status == AnchorStatus.VERIFIED
        assert item.anchors[0].source_char_start is not None
        assert item.anchors[0].source_char_end is not None
        assert item.document_sha256 == paper_row.document_sha256


# ============================================================================
# Test 8: Candidate Helper & Integration
# ============================================================================


def test_candidate_helper_and_integration(db_session: Session):
    """Test 8: Candidate helper extracts fact IDs from relationship candidates,

    contradiction pairs, and corpus themes, which resolve into verified EvidenceItems
    with live CitationAnchors.
    """
    manifest = validate_manifest(load_manifest())
    paper_a1 = manifest.projects["project_a"].papers[0]
    paper_row = db_session.query(Paper).filter(Paper.id == PAPER_A1_ID).one()

    rel_0 = paper_a1.relationships[0]
    rel_1 = paper_a1.relationships[1]
    rel_2 = paper_a1.relationships[2]

    _create_snapshot(db_session, PROJECT_A_ID, rel_0, paper_row.document_sha256)
    _create_snapshot(db_session, PROJECT_A_ID, rel_1, paper_row.document_sha256)
    _create_snapshot(db_session, PROJECT_A_ID, rel_2, paper_row.document_sha256)

    # 1. Relationship candidate
    rel_candidate = {
        "fact_id": rel_0.fact_id,
        "predicate": rel_0.predicate,
        "subject_name": rel_0.subject.name,
        "object_name": rel_0.object.name,
    }

    # 2. Contradiction candidate (contains fact_a and fact_b)
    contradiction_candidate = {
        "fact_a": {"fact_id": rel_0.fact_id, "subject_name": rel_0.subject.name},
        "fact_b": {"fact_id": rel_1.fact_id, "subject_name": rel_1.subject.name},
        "conflict_type": "NUMERIC_VALUE",
        "comparison_basis": {"dataset": "ImageNet", "metric": "ECE"},
    }

    # 3. Corpus theme candidate (contains contributing_fact_ids)
    theme_candidate = {
        "theme_type": "ENTITY",
        "name": "ImageNet",
        "contributing_fact_ids": [rel_1.fact_id, rel_2.fact_id],
        "paper_count": 2,
    }

    candidates = [rel_candidate, contradiction_candidate, theme_candidate]

    # Extract fact IDs
    fact_ids = extract_fact_ids_from_candidates(candidates)

    # Must preserve discovery order and deduplicate: rel_0, rel_1, rel_2
    assert fact_ids == [rel_0.fact_id, rel_1.fact_id, rel_2.fact_id]

    # Resolve extracted fact IDs into verified EvidenceItems
    evidence_items = resolve_graph_facts_to_evidence(
        db=db_session,
        project_id=PROJECT_A_ID,
        facts=fact_ids,
        prefix="G",
        start_index=1,
    )

    assert len(evidence_items) == 3
    assert [e.id for e in evidence_items] == ["G1", "G2", "G3"]

    for idx, e in enumerate(evidence_items):
        assert e.anchors[0].anchor_status == AnchorStatus.VERIFIED
        assert e.document_sha256 == paper_row.document_sha256
        assert len(e.bounding_boxes) > 0
        assert e.quote == [rel_0, rel_1, rel_2][idx].provenance.exact_quote
