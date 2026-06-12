"""Tests for GraphRAG synthetic test fixtures and manifest.

Verifies:
1. Manifest structure, ontology version, allowed entities, and allowed predicates.
2. Every declared quote exists verbatim in PaperChunk text and PaperPage raw_text.
3. Character span offsets match exact string slices in page raw text.
4. Validation rejects malformed manifests (unknown predicates, entity labels, quotes).
5. Grounding of method-to-dataset relations, cross-project acronyms, and contradictions.
6. SQLAlchemy models instantiation and DB seeding.
"""

from __future__ import annotations

import copy
from typing import Any
from uuid import UUID

import pytest
from pydantic import ValidationError
from tests.fixtures.graphrag.corpus_fixtures import (
    ALLOWED_ENTITIES,
    ALLOWED_PREDICATES,
    ONTOLOGY_VERSION,
    PAPER_A1_ID,
    PAPER_A2_ID,
    PAPER_B1_ID,
    PAPER_B2_ID,
    PROJECT_A_ID,
    PROJECT_B_ID,
    SYNTHETIC_DATA,
    create_synthetic_models,
    load_manifest,
    seed_synthetic_corpus,
    validate_manifest,
    verify_fixture_text_provenance,
)

from app.db.base import Base
from app.db.models import Paper, PaperChunk, PaperElement, PaperPage, Project
from app.ingestion.parser import find_verbatim_span


def test_manifest_loads_and_validates_successfully():
    """Verify that manifest.json loads from disk and passes all strict validators."""
    raw_manifest = load_manifest()
    manifest = validate_manifest(raw_manifest)

    assert manifest.ontology_version == ONTOLOGY_VERSION
    assert manifest.allowed_entities == ALLOWED_ENTITIES
    assert manifest.allowed_predicates == ALLOWED_PREDICATES

    # Check projects
    assert "project_a" in manifest.projects
    assert "project_b" in manifest.projects

    proj_a = manifest.projects["project_a"]
    assert proj_a.id == PROJECT_A_ID
    assert proj_a.name == "NLP Calibration"
    assert len(proj_a.papers) == 2

    proj_b = manifest.projects["project_b"]
    assert proj_b.id == PROJECT_B_ID
    assert proj_b.name == "Speech Recognition"
    assert len(proj_b.papers) == 2


def test_quotes_exist_verbatim_in_chunk_and_page_raw_text():
    """Sanity-check that every declared quote in the fixture exists verbatim in
    its declared PaperChunk text, PaperElement text, and PaperPage raw_text."""
    raw_manifest = load_manifest()
    manifest = validate_manifest(raw_manifest)

    for proj_key, proj in manifest.projects.items():
        paper_lookup = SYNTHETIC_DATA[proj_key]["papers"]

        for paper in proj.papers:
            # Find matching synthetic paper data
            matching_key = next(k for k, p_data in paper_lookup.items() if p_data["id"] == paper.id)
            paper_data = paper_lookup[matching_key]

            pages_by_num = {p["page_number"]: p for p in paper_data["pages"]}
            chunks_by_id = {c["id"]: c for c in paper_data["chunks"]}

            elements_by_id: dict[UUID, dict[str, Any]] = {}
            for page in paper_data["pages"]:
                for elem in page["elements"]:
                    elements_by_id[elem["id"]] = elem

            assert len(paper.relationships) > 0, f"Paper {paper.id} must declare relationships"

            for rel in paper.relationships:
                prov = rel.provenance
                quote = prov.exact_quote

                # 1. Quote exists in declared page raw_text
                assert prov.page_number in pages_by_num
                page = pages_by_num[prov.page_number]
                assert quote in page["raw_text"], (
                    f"Quote '{quote}' missing from page {prov.page_number} raw_text in {paper.id}"
                )

                # 2. Quote exists in declared chunk text
                assert prov.chunk_id in chunks_by_id, (
                    f"Chunk {prov.chunk_id} not found in paper {paper.id}"
                )
                chunk = chunks_by_id[prov.chunk_id]
                assert quote in chunk["text"], (
                    f"Quote '{quote}' missing from chunk {prov.chunk_id} text in {paper.id}"
                )

                # 3. Quote exists in declared element text
                assert prov.element_id in elements_by_id, (
                    f"Element {prov.element_id} not found in paper {paper.id}"
                )
                element = elements_by_id[prov.element_id]
                assert quote in element["text"], (
                    f"Quote '{quote}' missing from element {prov.element_id} text in {paper.id}"
                )


def test_character_span_offsets_match_exact_slice():
    """Verify that character span offsets match the exact string slice in page text,
    and find_verbatim_span returns the identical offset pair."""
    manifest = validate_manifest()

    for proj_key, proj in manifest.projects.items():
        paper_lookup = SYNTHETIC_DATA[proj_key]["papers"]

        for paper in proj.papers:
            matching_key = next(k for k, p_data in paper_lookup.items() if p_data["id"] == paper.id)
            paper_data = paper_lookup[matching_key]
            pages_by_num = {p["page_number"]: p for p in paper_data["pages"]}

            for rel in paper.relationships:
                prov = rel.provenance
                page = pages_by_num[prov.page_number]
                raw_text = page["raw_text"]

                # Exact slice match
                slice_extracted = raw_text[prov.char_start : prov.char_end]
                assert slice_extracted == prov.exact_quote, (
                    f"Slice mismatch in paper {paper.id}: "
                    f"'{slice_extracted}' != '{prov.exact_quote}'"
                )

                # find_verbatim_span match
                computed_span = find_verbatim_span(raw_text, prov.exact_quote)
                assert computed_span == (prov.char_start, prov.char_end), (
                    f"Span mismatch in {paper.id}: computed {computed_span} vs "
                    f"declared ({prov.char_start}, {prov.char_end})"
                )


def test_validation_rejects_unknown_entity_label():
    """Verify that an unknown entity label is rejected with a clear error."""
    manifest_data = copy.deepcopy(load_manifest())
    manifest_data["projects"]["project_a"]["papers"][0]["entities"][0]["type"] = (
        "BiologicalOrganism"
    )

    with pytest.raises((ValueError, ValidationError)) as exc_info:
        validate_manifest(manifest_data)

    assert "BiologicalOrganism" in str(exc_info.value) or "invalid type" in str(exc_info.value)


def test_validation_rejects_unknown_predicate():
    """Verify that an unknown relationship predicate is rejected with a clear error."""
    manifest_data = copy.deepcopy(load_manifest())
    manifest_data["projects"]["project_a"]["papers"][0]["relationships"][0]["predicate"] = (
        "CORRELATES_WITH"
    )

    with pytest.raises((ValueError, ValidationError)) as exc_info:
        validate_manifest(manifest_data)

    assert "CORRELATES_WITH" in str(exc_info.value) or "invalid predicate" in str(exc_info.value)


def test_validation_rejects_missing_or_empty_quote():
    """Verify that empty or whitespace-only provenance quotes fail validation."""
    manifest_data = copy.deepcopy(load_manifest())
    manifest_data["projects"]["project_a"]["papers"][0]["relationships"][0]["provenance"][
        "exact_quote"
    ] = ""

    with pytest.raises((ValueError, ValidationError)) as exc_info:
        validate_manifest(manifest_data)

    assert "exact_quote cannot be empty" in str(exc_info.value)


def test_validation_rejects_mismatched_char_offsets():
    """Verify that offset length mismatch against quote length fails validation."""
    manifest_data = copy.deepcopy(load_manifest())
    prov = manifest_data["projects"]["project_a"]["papers"][0]["relationships"][0]["provenance"]
    prov["char_end"] = prov["char_start"] + 5

    with pytest.raises((ValueError, ValidationError)) as exc_info:
        validate_manifest(manifest_data)

    assert "does not match quote length" in str(exc_info.value)


def test_validation_rejects_inverted_char_offsets():
    """Verify that char_end <= char_start fails validation."""
    manifest_data = copy.deepcopy(load_manifest())
    prov = manifest_data["projects"]["project_a"]["papers"][0]["relationships"][0]["provenance"]
    prov["char_start"] = 100
    prov["char_end"] = 50

    with pytest.raises((ValueError, ValidationError)) as exc_info:
        validate_manifest(manifest_data)

    assert "must be > char_start" in str(exc_info.value)


def test_validation_rejects_invalid_ontology_version():
    """Verify that an unsupported ontology version fails validation."""
    manifest_data = copy.deepcopy(load_manifest())
    manifest_data["ontology_version"] = "2.0.0-unsupported"

    with pytest.raises((ValueError, ValidationError)) as exc_info:
        validate_manifest(manifest_data)

    assert "Unsupported ontology_version" in str(exc_info.value)


def test_method_to_dataset_grounding():
    """Verify Method -> Dataset relations are grounded in exact quotes across both projects."""
    manifest = validate_manifest()
    benchmarks = manifest.cross_paper_benchmarks["method_dataset_relations"]

    # Project A: AURC evaluated on ImageNet
    item_a = next(b for b in benchmarks if b["project_id"] == str(PROJECT_A_ID))
    assert item_a["method"] == "AURC"
    assert item_a["dataset"] == "ImageNet"
    assert item_a["exact_quote"] == "We evaluate AURC on ImageNet calibration."

    # Project B: Conformer evaluated on LibriSpeech
    item_b1 = next(
        b for b in benchmarks if b["project_id"] == str(PROJECT_B_ID) and b["method"] == "Conformer"
    )
    assert item_b1["method"] == "Conformer"
    assert item_b1["dataset"] == "LibriSpeech"
    assert item_b1["exact_quote"] == "We evaluate Conformer on LibriSpeech."

    # Project B: Branchformer evaluated on LibriSpeech
    item_b2 = next(
        b
        for b in benchmarks
        if b["project_id"] == str(PROJECT_B_ID) and b["method"] == "Branchformer"
    )
    assert item_b2["method"] == "Branchformer"
    assert item_b2["dataset"] == "LibriSpeech"
    assert item_b2["exact_quote"] == "We evaluate Branchformer on LibriSpeech."


def test_cross_project_shared_acronym_contexts():
    """Verify cross-project shared acronyms ('ASR', 'ECE') maintain distinct semantics
    and project-scoped identities without unsafe global merging."""
    manifest = validate_manifest()
    acronyms = manifest.cross_paper_benchmarks["acronym_disambiguations"]

    # 1. ASR
    asr = next(a for a in acronyms if a["acronym"] == "ASR")
    assert asr["project_a_expansion"] == "Attack Success Rate"
    assert asr["project_b_expansion"] == "Automatic Speech Recognition"
    assert asr["project_a_entity_id"] != asr["project_b_entity_id"]
    assert asr["must_merge"] is False

    # 2. ECE
    ece = next(a for a in acronyms if a["acronym"] == "ECE")
    assert ece["project_a_expansion"] == "Expected Calibration Error"
    assert ece["project_b_expansion"] == "Energy-based Confidence Estimation"
    assert ece["project_a_entity_id"] != ece["project_b_entity_id"]
    assert ece["must_merge"] is False


def test_comparable_opposing_claims_contradiction():
    """Verify that Paper A1 and Paper A2 define a true contradiction on the same dataset (ImageNet)
    and metric (ECE) with opposing outcomes (2.1% vs 8.5% failure)."""
    manifest = validate_manifest()
    contra_pairs = manifest.cross_paper_benchmarks["contradiction_pairs"]
    assert len(contra_pairs) >= 1

    pair = contra_pairs[0]
    assert pair["paper_1_id"] == str(PAPER_A1_ID)
    assert pair["paper_2_id"] == str(PAPER_A2_ID)
    assert pair["common_method"] == "Temperature Scaling"
    assert pair["common_dataset"] == "ImageNet"
    assert pair["common_metric"] == "ECE"
    assert pair["value_1"] == 2.1
    assert pair["value_2"] == 8.5
    assert pair["is_contradiction"] is True
    assert "Temperature scaling achieves an ECE of 2.1% on ImageNet." in pair["claim_1"]
    assert (
        "Temperature scaling fails to converge, yielding an ECE of 8.5% on ImageNet."
        in pair["claim_2"]
    )


def test_non_contradicting_different_dataset_claims():
    """Verify that claims evaluating different datasets (ImageNet vs CIFAR-100) are recognized
    as non-contradicting despite reporting the same metric."""
    manifest = validate_manifest()
    non_contra_pairs = manifest.cross_paper_benchmarks["non_contradiction_pairs"]
    assert len(non_contra_pairs) >= 1

    pair = non_contra_pairs[0]
    assert pair["paper_1_id"] == str(PAPER_A1_ID)
    assert pair["paper_2_id"] == str(PAPER_A2_ID)
    assert pair["dataset_1"] == "ImageNet"
    assert pair["dataset_2"] == "CIFAR-100"
    assert pair["is_contradiction"] is False


def test_model_extension_relation():
    """Verify that Paper B2 declares an EXTENDS relation (Branchformer extends Conformer)."""
    manifest = validate_manifest()
    proj_b = manifest.projects["project_b"]
    paper_b2 = next(p for p in proj_b.papers if p.id == PAPER_B2_ID)

    extends_rel = next(r for r in paper_b2.relationships if r.predicate == "EXTENDS")
    assert extends_rel.subject.name == "Branchformer"
    assert extends_rel.object.name == "Conformer"
    assert extends_rel.provenance.exact_quote == (
        "Branchformer extends Conformer with parallel attention and convolutional branches."
    )


def test_fixture_text_provenance_sanity():
    """Run provenance verification helper across all papers in the fixture."""
    for proj in SYNTHETIC_DATA.values():
        for paper_data in proj["papers"].values():
            errors = verify_fixture_text_provenance(paper_data)
            assert errors == [], f"Errors in {paper_data['title']}: {errors}"


def test_sqlalchemy_models_instantiation():
    """Verify that create_synthetic_models creates valid, populated in-memory ORM objects."""
    models = create_synthetic_models()

    assert len(models["projects"]) == 2
    assert len(models["papers"]) == 4
    assert len(models["pages"]) == 8  # 2 pages each
    assert len(models["elements"]) == 32  # 8 elements per paper
    assert len(models["chunks"]) == 13  # total child chunks across 4 papers
    assert len(models["chunk_elements"]) > 0

    for pap in models["papers"]:
        assert isinstance(pap, Paper)
        assert pap.status == "READY"
        assert pap.document_sha256 is not None
        assert pap.page_count == 2

    for page in models["pages"]:
        assert isinstance(page, PaperPage)
        assert page.page_number in (1, 2)
        assert page.raw_text is not None

    for elem in models["elements"]:
        assert isinstance(elem, PaperElement)
        assert elem.text is not None
        assert elem.bbox_x_min is not None

    for chunk in models["chunks"]:
        assert isinstance(chunk, PaperChunk)
        assert chunk.chunk_type == "child"
        assert len(chunk.embedding) == 1024


def test_reject_invalid_qualifier_in_manifest():
    """Verify that a relationship with an unknown qualifier key is rejected."""
    raw = copy.deepcopy(load_manifest())
    first_proj_key = next(iter(raw["projects"]))
    raw["projects"][first_proj_key]["papers"][0]["relationships"][0]["qualifiers"][
        "invented_qualifier"
    ] = "invalid_value"
    with pytest.raises(
        (ValueError, ValidationError), match="invalid qualifier 'invented_qualifier'"
    ):
        validate_manifest(raw)


def test_database_seeding_with_synthetic_corpus():
    """Verify that seed_synthetic_corpus commits records into an isolated DB session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)
    db = TestingSession()
    try:
        seed_synthetic_corpus(db)

        # Query back to verify persistence
        projects = db.query(Project).filter(Project.id.in_([PROJECT_A_ID, PROJECT_B_ID])).all()
        assert len(projects) == 2

        papers = (
            db.query(Paper)
            .filter(Paper.id.in_([PAPER_A1_ID, PAPER_A2_ID, PAPER_B1_ID, PAPER_B2_ID]))
            .all()
        )
        assert len(papers) == 4

        # Verify chunks have linked elements
        chunks = db.query(PaperChunk).filter(PaperChunk.paper_id == PAPER_A1_ID).all()
        assert len(chunks) == 3
        for c in chunks:
            assert len(c.elements) > 0
    finally:
        db.close()
        Base.metadata.drop_all(engine)
