"""Unit and integration tests for GraphRAG verification (Task 5.17),
deterministic canonicalization (Task 5.18), and verified snapshot persistence (Task 5.18a).
"""

from __future__ import annotations

from collections.abc import Generator
from uuid import uuid4

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
from app.db.models import GraphFactSnapshot, Paper
from app.schemas.graph import (
    ClaimPolarity,
    EntityType,
    GraphEntitySchema,
    GraphFactCandidate,
    GraphProvenanceSchema,
    GraphQualifierSchema,
    RelationshipPredicate,
)
from app.services.graphrag.identity import generate_entity_key, generate_fact_id
from app.services.graphrag.snapshots import (
    get_verified_fact_snapshots,
    persist_verified_fact_snapshots,
)
from app.services.graphrag.verifier import verify_candidate_fact


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide an isolated in-memory SQLite database populated with the synthetic corpus."""
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
# Test 1: Real supported fact passes verification and persists
# ============================================================================


def test_real_supported_fact_passes_verification_and_persists(db_session: Session):
    """Test 1: Real supported fact with matching quote, entities, and numeric values
    passes verification and persists to graph_fact_snapshots.
    """
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[2]
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).one()

    candidate = GraphFactCandidate(
        subject=GraphEntitySchema(
            id=rel.subject.id,
            name=rel.subject.name,
            type=EntityType(rel.subject.type),
        ),
        predicate=RelationshipPredicate(rel.predicate),
        object=GraphEntitySchema(
            id=rel.object.id,
            name=rel.object.name,
            type=EntityType(rel.object.type),
        ),
        qualifiers=GraphQualifierSchema(**rel.qualifiers) if rel.qualifiers else None,
        provenance=GraphProvenanceSchema(
            paper_id=rel.provenance.paper_id,
            chunk_id=rel.provenance.chunk_id,
            element_id=rel.provenance.element_id,
            page_number=rel.provenance.page_number,
            exact_quote=rel.provenance.exact_quote,
            char_start=rel.provenance.char_start,
            char_end=rel.provenance.char_end,
            document_sha256=paper.document_sha256,
        ),
    )

    # 1. Verification succeeds
    is_valid, reason = verify_candidate_fact(db_session, PROJECT_A_ID, candidate)
    assert is_valid is True
    assert reason is None

    # 2. Persistence creates snapshot
    generation_id = "gen_a1_001"
    snapshots = persist_verified_fact_snapshots(
        db=db_session,
        project_id=PROJECT_A_ID,
        paper_id=paper.id,
        generation_id=generation_id,
        event_id=None,
        verified_candidates=[candidate],
        ontology_version="1.0.0",
    )

    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.fact_id.startswith("fact_")
    assert snap.project_id == PROJECT_A_ID
    assert snap.paper_id == paper.id
    assert snap.generation_id == generation_id
    assert snap.subject_name == "Temperature Scaling"
    assert snap.subject_type == "Method"
    assert snap.predicate == "ACHIEVES_RESULT"
    assert snap.object_name == "2.1% ECE"
    assert snap.object_type == "Result"
    assert snap.qualifiers["numeric_value"] == 2.1
    assert snap.qualifiers["metric"] == "ECE"
    assert snap.qualifiers["polarity"] == "POSITIVE"
    assert snap.exact_quote == rel.provenance.exact_quote
    assert snap.document_sha256 == paper.document_sha256
    assert snap.validation_version == "1.0.0"

    # Query via getter
    queried = get_verified_fact_snapshots(db_session, paper.id, generation_id)
    assert len(queried) == 1
    assert queried[0].fact_id == snap.fact_id


# ============================================================================
# Test 2: Reversed actor / entity not mentioned in quote is rejected
# ============================================================================


def test_reversed_actor_or_missing_entity_rejected(db_session: Session):
    """Test 2: Reversed actor and entity not mentioned in quote are rejected."""
    manifest = validate_manifest(load_manifest())

    # Case A: Entity not mentioned in quote is rejected
    rel_a = manifest.projects["project_a"].papers[0].relationships[1]
    paper_a = db_session.query(Paper).filter(Paper.id == rel_a.provenance.paper_id).one()

    cand_missing_entity = GraphFactCandidate(
        subject=GraphEntitySchema(
            id="method_aurc",
            name="AURC",
            type=EntityType.METHOD,
        ),
        predicate=RelationshipPredicate.EVALUATED_ON,
        object=GraphEntitySchema(
            id="dataset_unmentioned",
            name="NonExistentBenchmarkDataset",
            type=EntityType.DATASET,
        ),
        provenance=GraphProvenanceSchema(
            paper_id=rel_a.provenance.paper_id,
            chunk_id=rel_a.provenance.chunk_id,
            element_id=rel_a.provenance.element_id,
            page_number=rel_a.provenance.page_number,
            exact_quote=rel_a.provenance.exact_quote,  # "We evaluate AURC on ImageNet calibration."
            char_start=rel_a.provenance.char_start,
            char_end=rel_a.provenance.char_end,
            document_sha256=paper_a.document_sha256,
        ),
    )
    is_valid, reason = verify_candidate_fact(db_session, PROJECT_A_ID, cand_missing_entity)
    assert is_valid is False
    assert reason == "ENTITIES_NOT_IN_QUOTE"

    # Case B: Reversed actor in directional relation (Branchformer extends Conformer)
    rel_b = manifest.projects["project_b"].papers[1].relationships[0]  # b2_extends_conformer
    paper_b = db_session.query(Paper).filter(Paper.id == rel_b.provenance.paper_id).one()

    # The quote states: "Branchformer extends Conformer with parallel attention..."
    # If candidate asserts: Conformer EXTENDS Branchformer (reversed actor)
    cand_reversed_actor = GraphFactCandidate(
        subject=GraphEntitySchema(
            id=rel_b.object.id,
            name=rel_b.object.name,  # "Conformer" as subject
            type=EntityType(rel_b.object.type),
        ),
        predicate=RelationshipPredicate.EXTENDS,
        object=GraphEntitySchema(
            id=rel_b.subject.id,
            name=rel_b.subject.name,  # "Branchformer" as object
            type=EntityType(rel_b.subject.type),
        ),
        provenance=GraphProvenanceSchema(
            paper_id=rel_b.provenance.paper_id,
            chunk_id=rel_b.provenance.chunk_id,
            element_id=rel_b.provenance.element_id,
            page_number=rel_b.provenance.page_number,
            exact_quote=rel_b.provenance.exact_quote,
            char_start=rel_b.provenance.char_start,
            char_end=rel_b.provenance.char_end,
            document_sha256=paper_b.document_sha256,
        ),
    )
    is_valid, reason = verify_candidate_fact(db_session, PROJECT_B_ID, cand_reversed_actor)
    assert is_valid is False
    assert reason == "REVERSED_ACTOR"


# ============================================================================
# Test 3: Negation / polarity mismatch is rejected
# ============================================================================


def test_negation_polarity_mismatch_rejected(db_session: Session):
    """Test 3: Negation / polarity mismatch (quote says 'fails to converge' but
    candidate claims positive result) is rejected.
    """
    manifest = validate_manifest(load_manifest())
    # paper_a2 rel 1 quote:
    # "Temperature scaling fails to converge, yielding an ECE of 8.5% on ImageNet."
    rel = manifest.projects["project_a"].papers[1].relationships[1]
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).one()

    # Candidate asserts POSITIVE polarity despite the quote's explicit "fails to converge" negation
    cand_positive = GraphFactCandidate(
        subject=GraphEntitySchema(
            id=rel.subject.id,
            name=rel.subject.name,
            type=EntityType(rel.subject.type),
        ),
        predicate=RelationshipPredicate(rel.predicate),
        object=GraphEntitySchema(
            id=rel.object.id,
            name=rel.object.name,
            type=EntityType(rel.object.type),
        ),
        qualifiers=GraphQualifierSchema(
            dataset="ImageNet",
            metric="ECE",
            numeric_value=8.5,
            unit="%",
            polarity=ClaimPolarity.POSITIVE,  # Positive claim on negated text
        ),
        provenance=GraphProvenanceSchema(
            paper_id=rel.provenance.paper_id,
            chunk_id=rel.provenance.chunk_id,
            element_id=rel.provenance.element_id,
            page_number=rel.provenance.page_number,
            exact_quote=rel.provenance.exact_quote,
            char_start=rel.provenance.char_start,
            char_end=rel.provenance.char_end,
            document_sha256=paper.document_sha256,
        ),
    )

    is_valid, reason = verify_candidate_fact(db_session, PROJECT_A_ID, cand_positive)
    assert is_valid is False
    assert reason == "POLARITY_CONTRADICTION"

    # Candidate asserting default (implicit POSITIVE) polarity is also rejected
    cand_default = GraphFactCandidate(
        subject=GraphEntitySchema(
            id=rel.subject.id,
            name=rel.subject.name,
            type=EntityType(rel.subject.type),
        ),
        predicate=RelationshipPredicate(rel.predicate),
        object=GraphEntitySchema(
            id=rel.object.id,
            name=rel.object.name,
            type=EntityType(rel.object.type),
        ),
        qualifiers=None,
        provenance=cand_positive.provenance,
    )
    is_valid, reason = verify_candidate_fact(db_session, PROJECT_A_ID, cand_default)
    assert is_valid is False
    assert reason == "POLARITY_CONTRADICTION"


# ============================================================================
# Test 4: Swapped / mismatched numbers are rejected
# ============================================================================


def test_swapped_numeric_values_rejected(db_session: Session):
    """Test 4: Swapped/mismatched numbers (candidate claims 8.5% ECE but quote states
    2.1% ECE) is rejected.
    """
    manifest = validate_manifest(load_manifest())
    # paper_a1 rel 2: quote states "Temperature scaling achieves an ECE of 2.1% on ImageNet."
    rel = manifest.projects["project_a"].papers[0].relationships[2]
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).one()

    # Candidate falsely claims 8.5% ECE
    cand_mismatch_number = GraphFactCandidate(
        subject=GraphEntitySchema(
            id=rel.subject.id,
            name=rel.subject.name,
            type=EntityType(rel.subject.type),
        ),
        predicate=RelationshipPredicate(rel.predicate),
        object=GraphEntitySchema(
            id=rel.object.id,
            name="8.5% ECE",
            type=EntityType(rel.object.type),
        ),
        qualifiers=GraphQualifierSchema(
            dataset="ImageNet",
            metric="ECE",
            numeric_value=8.5,  # Mismatched number: quote has 2.1
            unit="%",
            polarity=ClaimPolarity.POSITIVE,
        ),
        provenance=GraphProvenanceSchema(
            paper_id=rel.provenance.paper_id,
            chunk_id=rel.provenance.chunk_id,
            element_id=rel.provenance.element_id,
            page_number=rel.provenance.page_number,
            exact_quote=rel.provenance.exact_quote,
            char_start=rel.provenance.char_start,
            char_end=rel.provenance.char_end,
            document_sha256=paper.document_sha256,
        ),
    )

    is_valid, reason = verify_candidate_fact(db_session, PROJECT_A_ID, cand_mismatch_number)
    assert is_valid is False
    assert reason == "NUMERIC_VALUE_MISMATCH"


# ============================================================================
# Test 5: Stale document hash or unresolved chunk is rejected
# ============================================================================


def test_stale_hash_or_unresolved_chunk_rejected(db_session: Session):
    """Test 5: Stale document hash or unresolved chunk is rejected with UNRESOLVED_ANCHOR."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[2]
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).one()

    # Case A: Stale document SHA-256
    cand_stale_hash = GraphFactCandidate(
        subject=GraphEntitySchema(
            id=rel.subject.id,
            name=rel.subject.name,
            type=EntityType(rel.subject.type),
        ),
        predicate=RelationshipPredicate(rel.predicate),
        object=GraphEntitySchema(
            id=rel.object.id,
            name=rel.object.name,
            type=EntityType(rel.object.type),
        ),
        provenance=GraphProvenanceSchema(
            paper_id=rel.provenance.paper_id,
            chunk_id=rel.provenance.chunk_id,
            element_id=rel.provenance.element_id,
            page_number=rel.provenance.page_number,
            exact_quote=rel.provenance.exact_quote,
            char_start=rel.provenance.char_start,
            char_end=rel.provenance.char_end,
            document_sha256="deadbeef" * 8,  # Does not match paper.document_sha256
        ),
    )
    is_valid, reason = verify_candidate_fact(db_session, PROJECT_A_ID, cand_stale_hash)
    assert is_valid is False
    assert reason == "UNRESOLVED_ANCHOR"

    # Case B: Non-existent / unresolved chunk ID
    cand_unresolved_chunk = GraphFactCandidate(
        subject=cand_stale_hash.subject,
        predicate=cand_stale_hash.predicate,
        object=cand_stale_hash.object,
        provenance=GraphProvenanceSchema(
            paper_id=rel.provenance.paper_id,
            chunk_id=uuid4(),  # Random chunk that does not exist in DB
            element_id=rel.provenance.element_id,
            page_number=rel.provenance.page_number,
            exact_quote=rel.provenance.exact_quote,
            char_start=rel.provenance.char_start,
            char_end=rel.provenance.char_end,
            document_sha256=paper.document_sha256,
        ),
    )
    is_valid, reason = verify_candidate_fact(db_session, PROJECT_A_ID, cand_unresolved_chunk)
    assert is_valid is False
    assert reason == "UNRESOLVED_ANCHOR"


# ============================================================================
# Test 6: Deterministic canonicalization & cross-project isolation
# ============================================================================


def test_deterministic_canonicalization_and_project_scoping():
    """Test 6: Deterministic canonicalization: External ID deduplication within project;
    homonyms and shared acronyms across projects do NOT collide.
    """
    # 1. External ID deduplication within the same project
    # Two entities with distinct names but same external ID resolve to the same node key
    key1 = generate_entity_key(
        project_id=PROJECT_A_ID,
        entity_type=EntityType.MODEL,
        name="BERT",
        external_id="arxiv:1810.04805",
    )
    key2 = generate_entity_key(
        project_id=PROJECT_A_ID,
        entity_type=EntityType.MODEL,
        name="Bidirectional Encoder Representations from Transformers",
        external_id="arxiv:1810.04805",
    )
    assert key1 == key2
    assert key1 == f"proj_{PROJECT_A_ID.hex}_model_arxiv:1810.04805"

    # 2. Homonyms / shared acronyms across different projects do NOT collide
    # Project A uses "ASR" for "Attack Success Rate" (Concept)
    # Project B uses "ASR" for "Automatic Speech Recognition" (Concept)
    key_asr_project_a = generate_entity_key(
        project_id=PROJECT_A_ID,
        entity_type=EntityType.CONCEPT,
        name="ASR",
    )
    key_asr_project_b = generate_entity_key(
        project_id=PROJECT_B_ID,
        entity_type=EntityType.CONCEPT,
        name="ASR",
    )
    assert key_asr_project_a != key_asr_project_b
    assert key_asr_project_a.startswith("concept_")
    assert key_asr_project_b.startswith("concept_")

    # 3. Same external ID across different projects produces different keys (strict isolation)
    ext_key_proj_a = generate_entity_key(
        project_id=PROJECT_A_ID,
        entity_type=EntityType.MODEL,
        name="Conformer",
        external_id="arxiv:2005.08100",
    )
    ext_key_proj_b = generate_entity_key(
        project_id=PROJECT_B_ID,
        entity_type=EntityType.MODEL,
        name="Conformer",
        external_id="arxiv:2005.08100",
    )
    assert ext_key_proj_a != ext_key_proj_b


# ============================================================================
# Test 7: Multi-paper provenance produces distinct fact IDs
# ============================================================================


def test_multi_paper_provenance_distinct_fact_ids(db_session: Session):
    """Test 7: Multi-paper provenance: Two different papers asserting the same relation
    produce distinct fact IDs and both persist without collision.
    """
    paper_1 = db_session.query(Paper).filter(Paper.id == PAPER_A1_ID).one()
    paper_2 = db_session.query(Paper).filter(Paper.id == PAPER_A2_ID).one()

    # Both papers assert that Temperature Scaling was evaluated on ImageNet
    subject_key = generate_entity_key(PROJECT_A_ID, EntityType.METHOD, "Temperature Scaling")
    object_key = generate_entity_key(PROJECT_A_ID, EntityType.DATASET, "ImageNet")

    # Derive fact IDs for both papers
    fact_id_1 = generate_fact_id(
        project_id=PROJECT_A_ID,
        paper_id=paper_1.id,
        predicate=RelationshipPredicate.EVALUATED_ON,
        subject_key=subject_key,
        object_key=object_key,
        char_start=0,
        char_end=41,
        source_generation="gen_multi_1",
    )
    fact_id_2 = generate_fact_id(
        project_id=PROJECT_A_ID,
        paper_id=paper_2.id,
        predicate=RelationshipPredicate.EVALUATED_ON,
        subject_key=subject_key,
        object_key=object_key,
        char_start=0,
        char_end=41,
        source_generation="gen_multi_2",
    )

    # Invariant: Distinct papers produce distinct fact IDs
    assert fact_id_1 != fact_id_2

    # Construct snapshots for both papers
    snap_1 = GraphFactSnapshot(
        fact_id=fact_id_1,
        project_id=PROJECT_A_ID,
        paper_id=paper_1.id,
        generation_id="gen_multi_1",
        subject_key=subject_key,
        subject_name="Temperature Scaling",
        subject_type="Method",
        predicate="EVALUATED_ON",
        object_key=object_key,
        object_name="ImageNet",
        object_type="Dataset",
        char_start=0,
        char_end=41,
        page_number=1,
        exact_quote="We evaluate Temperature Scaling on ImageNet",
        document_sha256=paper_1.document_sha256,
        validation_version="1.0.0",
    )
    snap_2 = GraphFactSnapshot(
        fact_id=fact_id_2,
        project_id=PROJECT_A_ID,
        paper_id=paper_2.id,
        generation_id="gen_multi_2",
        subject_key=subject_key,
        subject_name="Temperature Scaling",
        subject_type="Method",
        predicate="EVALUATED_ON",
        object_key=object_key,
        object_name="ImageNet",
        object_type="Dataset",
        char_start=0,
        char_end=41,
        page_number=1,
        exact_quote="We evaluate Temperature Scaling on ImageNet",
        document_sha256=paper_2.document_sha256,
        validation_version="1.0.0",
    )

    db_session.add(snap_1)
    db_session.add(snap_2)
    db_session.flush()

    # Both snapshots coexist without collision
    retrieved_1 = db_session.get(GraphFactSnapshot, fact_id_1)
    retrieved_2 = db_session.get(GraphFactSnapshot, fact_id_2)
    assert retrieved_1 is not None
    assert retrieved_2 is not None
    assert retrieved_1.paper_id == paper_1.id
    assert retrieved_2.paper_id == paper_2.id


# ============================================================================
# Test 8: Retry idempotency
# ============================================================================


def test_retry_idempotency_returns_existing_snapshots(db_session: Session):
    """Test 8: Retry idempotency: Re-running snapshot persistence on the same generation
    returns the existing snapshots without duplicate database rows.
    """
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[2]
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).one()

    candidate = GraphFactCandidate(
        subject=GraphEntitySchema(
            id=rel.subject.id,
            name=rel.subject.name,
            type=EntityType(rel.subject.type),
        ),
        predicate=RelationshipPredicate(rel.predicate),
        object=GraphEntitySchema(
            id=rel.object.id,
            name=rel.object.name,
            type=EntityType(rel.object.type),
        ),
        qualifiers=GraphQualifierSchema(**rel.qualifiers) if rel.qualifiers else None,
        provenance=GraphProvenanceSchema(
            paper_id=rel.provenance.paper_id,
            chunk_id=rel.provenance.chunk_id,
            element_id=rel.provenance.element_id,
            page_number=rel.provenance.page_number,
            exact_quote=rel.provenance.exact_quote,
            char_start=rel.provenance.char_start,
            char_end=rel.provenance.char_end,
            document_sha256=paper.document_sha256,
        ),
    )

    generation_id = "gen_retry_idempotency_001"

    # First persistence run
    first_run = persist_verified_fact_snapshots(
        db=db_session,
        project_id=PROJECT_A_ID,
        paper_id=paper.id,
        generation_id=generation_id,
        event_id=None,
        verified_candidates=[candidate],
    )
    assert len(first_run) == 1
    initial_fact_id = first_run[0].fact_id

    # Verify 1 row in DB
    assert (
        db_session.query(GraphFactSnapshot)
        .filter(
            GraphFactSnapshot.paper_id == paper.id,
            GraphFactSnapshot.generation_id == generation_id,
        )
        .count()
        == 1
    )

    # Second persistence run (retry with identical generation_id)
    retry_run = persist_verified_fact_snapshots(
        db=db_session,
        project_id=PROJECT_A_ID,
        paper_id=paper.id,
        generation_id=generation_id,
        event_id=None,
        verified_candidates=[candidate],
    )
    assert len(retry_run) == 1
    assert retry_run[0].fact_id == initial_fact_id

    # Row count remains exactly 1 (no duplicate rows)
    assert (
        db_session.query(GraphFactSnapshot)
        .filter(
            GraphFactSnapshot.paper_id == paper.id,
            GraphFactSnapshot.generation_id == generation_id,
        )
        .count()
        == 1
    )


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


def test_batch_deduplication_and_empty_candidates(db_session: Session):
    """Verify empty candidate lists and intra-batch candidate deduplication."""
    # Empty candidate list returns empty list without error
    empty_result = persist_verified_fact_snapshots(
        db=db_session,
        project_id=PROJECT_A_ID,
        paper_id=PAPER_A1_ID,
        generation_id="gen_empty_001",
        event_id=None,
        verified_candidates=[],
    )
    assert empty_result == []

    # Intra-batch deduplication: identical candidates in same list
    # do not cause primary key conflicts
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[2]
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).one()

    candidate = GraphFactCandidate(
        subject=GraphEntitySchema(
            id=rel.subject.id,
            name=rel.subject.name,
            type=EntityType(rel.subject.type),
        ),
        predicate=RelationshipPredicate(rel.predicate),
        object=GraphEntitySchema(
            id=rel.object.id,
            name=rel.object.name,
            type=EntityType(rel.object.type),
        ),
        qualifiers=GraphQualifierSchema(**rel.qualifiers) if rel.qualifiers else None,
        provenance=GraphProvenanceSchema(
            paper_id=rel.provenance.paper_id,
            chunk_id=rel.provenance.chunk_id,
            element_id=rel.provenance.element_id,
            page_number=rel.provenance.page_number,
            exact_quote=rel.provenance.exact_quote,
            char_start=rel.provenance.char_start,
            char_end=rel.provenance.char_end,
            document_sha256=paper.document_sha256,
        ),
    )

    # Pass the duplicate candidate twice in the same batch
    snaps = persist_verified_fact_snapshots(
        db=db_session,
        project_id=PROJECT_A_ID,
        paper_id=paper.id,
        generation_id="gen_batch_dedup_001",
        event_id=None,
        verified_candidates=[candidate, candidate],
    )
    assert len(snaps) == 1


def test_snapshot_external_id_preservation(db_session: Session):
    """Verify that external identifiers on candidates are preserved in derived node keys."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_b"].papers[0].relationships[1]  # Conformer on LibriSpeech
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).one()

    candidate = GraphFactCandidate(
        subject=GraphEntitySchema(
            id="arxiv:2005.08100",  # External ID format
            name="Conformer",
            type=EntityType.MODEL,
        ),
        predicate=RelationshipPredicate.EVALUATED_ON,
        object=GraphEntitySchema(
            id="corpus:librispeech",  # External ID format
            name="LibriSpeech",
            type=EntityType.DATASET,
        ),
        provenance=GraphProvenanceSchema(
            paper_id=rel.provenance.paper_id,
            chunk_id=rel.provenance.chunk_id,
            element_id=rel.provenance.element_id,
            page_number=rel.provenance.page_number,
            exact_quote=rel.provenance.exact_quote,
            char_start=rel.provenance.char_start,
            char_end=rel.provenance.char_end,
            document_sha256=paper.document_sha256,
        ),
    )

    snaps = persist_verified_fact_snapshots(
        db=db_session,
        project_id=PROJECT_B_ID,
        paper_id=paper.id,
        generation_id="gen_ext_id_001",
        event_id=None,
        verified_candidates=[candidate],
    )
    assert len(snaps) == 1
    assert snaps[0].subject_key == f"proj_{PROJECT_B_ID.hex}_model_arxiv:2005.08100"
    assert snaps[0].object_key == f"proj_{PROJECT_B_ID.hex}_dataset_corpus:librispeech"


def test_negative_claim_on_positive_quote_rejected(db_session: Session):
    """Verify that candidate asserting NEGATIVE claim on an affirmative quote is rejected."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[2]
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).one()

    # The quote states affirmative result:
    # "Temperature scaling achieves an ECE of 2.1% on ImageNet."
    # Candidate falsely claims NEGATIVE polarity
    cand_falsely_negative = GraphFactCandidate(
        subject=GraphEntitySchema(
            id=rel.subject.id,
            name=rel.subject.name,
            type=EntityType(rel.subject.type),
        ),
        predicate=RelationshipPredicate(rel.predicate),
        object=GraphEntitySchema(
            id=rel.object.id,
            name=rel.object.name,
            type=EntityType(rel.object.type),
        ),
        qualifiers=GraphQualifierSchema(
            dataset="ImageNet",
            metric="ECE",
            numeric_value=2.1,
            unit="%",
            polarity=ClaimPolarity.NEGATIVE,  # Opposing polarity on affirmative text
        ),
        provenance=GraphProvenanceSchema(
            paper_id=rel.provenance.paper_id,
            chunk_id=rel.provenance.chunk_id,
            element_id=rel.provenance.element_id,
            page_number=rel.provenance.page_number,
            exact_quote=rel.provenance.exact_quote,
            char_start=rel.provenance.char_start,
            char_end=rel.provenance.char_end,
            document_sha256=paper.document_sha256,
        ),
    )

    is_valid, reason = verify_candidate_fact(db_session, PROJECT_A_ID, cand_falsely_negative)
    assert is_valid is False
    assert reason == "POLARITY_CONTRADICTION"


def test_verifier_alias_and_short_names(db_session: Session):
    """Verify entity mention via alias and word-boundary rules on short names."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[2]
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).one()

    # Subject name differs from quote, but alias matches exactly
    candidate_alias = GraphFactCandidate(
        subject=GraphEntitySchema(
            id=rel.subject.id,
            name="TS Algorithm",  # Name not in quote
            aliases=["Temperature Scaling"],  # Alias is in quote
            type=EntityType(rel.subject.type),
        ),
        predicate=RelationshipPredicate(rel.predicate),
        object=GraphEntitySchema(
            id=rel.object.id,
            name=rel.object.name,
            type=EntityType(rel.object.type),
        ),
        qualifiers=GraphQualifierSchema(
            dataset="ImageNet",
            metric="ECE",
            numeric_value=2.1,
            unit="%",
            polarity=ClaimPolarity.POSITIVE,
        ),
        provenance=GraphProvenanceSchema(
            paper_id=rel.provenance.paper_id,
            chunk_id=rel.provenance.chunk_id,
            element_id=rel.provenance.element_id,
            page_number=rel.provenance.page_number,
            exact_quote=rel.provenance.exact_quote,
            char_start=rel.provenance.char_start,
            char_end=rel.provenance.char_end,
            document_sha256=paper.document_sha256,
        ),
    )
    is_valid, reason = verify_candidate_fact(db_session, PROJECT_A_ID, candidate_alias)
    assert is_valid is True
    assert reason is None


def test_verifier_raw_value_and_integer_numeric_support(db_session: Session):
    """Verify raw_value matching and integer formatted numeric results."""
    from app.services.graphrag.verifier import _check_numeric_support

    # 1. Matching raw_value string
    q1 = GraphQualifierSchema(raw_value="2.1%")
    assert _check_numeric_support(q1, [], "achieves an ECE of 2.1% on ImageNet") is True

    # 2. Integer float (e.g. 5.0 represented as '5')
    q2 = GraphQualifierSchema(result_value=5.0)
    assert _check_numeric_support(q2, [], "improved by 5 points across runs") is True

    # 3. Numeric mismatch on raw_value
    q3 = GraphQualifierSchema(raw_value="99.9%", result_value=99.9)
    assert _check_numeric_support(q3, [], "achieves an ECE of 2.1% on ImageNet") is False


def test_numeric_support_reconciles_every_value_with_boundaries():
    from app.services.graphrag.verifier import _check_numeric_support

    supported_spaced_decimal = GraphQualifierSchema(result_value=2.1, raw_value="2 . 1")
    assert _check_numeric_support(supported_spaced_decimal, [], "result was 2 . 1") is True

    inconsistent = GraphQualifierSchema.model_construct(
        result_value=999.0, numeric_value=999.0, raw_value="2.1"
    )
    assert _check_numeric_support(inconsistent, [], "result was 2.1") is False

    assert _check_numeric_support(GraphQualifierSchema(result_value=2), [], "value 12") is False
    assert _check_numeric_support(GraphQualifierSchema(), [], "no numeric claim") is True

    # The claimed result 2% accuracy must not be borrowed from the unrelated
    # "2 GPUs" count elsewhere in the quote.
    unrelated_measure = GraphQualifierSchema(
        result_value=2, unit="%", metric="accuracy", dataset="ImageNet"
    )
    assert (
        _check_numeric_support(
            unrelated_measure,
            [],
            "Method X achieves 90% accuracy on ImageNet with 2 GPUs.",
        )
        is False
    )

    from app.services.graphrag.verifier import _check_qualifier_text_support

    unsupported_conditions = GraphQualifierSchema(
        result_value=2.1,
        unit="count",
        metric="ECE",
        dataset="ImageNet",
        split="train",
        comparison_condition="augmented setup",
    )
    assert not _check_qualifier_text_support(
        unsupported_conditions,
        "Method X achieves 2.1% ECE on ImageNet test split under standard setup.",
    )

    from app.services.graphrag.verifier import _check_numeric_support

    multi_metric_quote = (
        "BERT reports GLUE 80.5% and MultiNLI 86.7% on test split for classification "
        "under standard setup."
    )
    assert (
        _check_numeric_support(
            GraphQualifierSchema(result_value=86.7, raw_value="86.7%", unit="%", metric="GLUE"),
            [],
            multi_metric_quote,
        )
        is False
    )
    assert (
        _check_numeric_support(
            GraphQualifierSchema(result_value=80.5, raw_value="80.5%", unit="%", metric="GLUE"),
            [],
            multi_metric_quote,
        )
        is True
    )

    multi_dataset_quote = (
        "Method X reports ImageNet accuracy 90% and CIFAR accuracy 80% on test split "
        "for classification under standard setup."
    )
    assert (
        _check_numeric_support(
            GraphQualifierSchema(
                result_value=80, raw_value="80%", unit="%", metric="accuracy", dataset="ImageNet"
            ),
            [],
            multi_dataset_quote,
        )
        is False
    )

    numeric_entity = GraphEntitySchema(id="e", name="999 BLEU", type=EntityType.RESULT)
    correct_raw_value = GraphQualifierSchema(raw_value="2.1 BLEU")
    assert (
        _check_numeric_support(
            correct_raw_value,
            [numeric_entity],
            "Method X achieves 2.1 BLEU on Dataset Y.",
        )
        is False
    )


def test_snapshots_extract_external_id_coverage():
    """Verify external ID extraction edge cases."""
    from app.services.graphrag.snapshots import _extract_external_id

    # Internal generated keys are omitted
    assert _extract_external_id("method_1234abcd", EntityType.METHOD) is None
    assert _extract_external_id("proj_abcdef_model_1", EntityType.MODEL) is None

    # None and empty strings
    assert _extract_external_id(None, EntityType.METHOD) is None
    assert _extract_external_id("", EntityType.METHOD) is None

    # Valid external ID
    assert _extract_external_id("arxiv:2005.08100", EntityType.MODEL) == "arxiv:2005.08100"

    # Invalid external ID format
    assert _extract_external_id("invalid:id with space", EntityType.MODEL) is None
