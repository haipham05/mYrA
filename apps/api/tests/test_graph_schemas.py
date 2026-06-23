"""Tests for GraphRAG schemas (EntityType, RelationshipPredicate, GraphQualifierSchema,
GraphProvenanceSchema, GraphEntitySchema, GraphFactCandidate, GraphExtractionBatch).

Verifies:
1. Valid candidates parse and validate correctly.
2. Invalid entity label fails.
3. Invalid predicate fails.
4. Endpoint mismatch (e.g. Dataset USES_MODEL Author) raises ValidationError.
5. Empty or mismatched provenance quote raises ValidationError.
6. Qualifier normalization ('41 . 0' vs '41.0', opposing polarities, unit comparisons).
7. Batch size limits reject oversized batches (>100 entities or >100 facts).
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.schemas import (
    VALID_ENDPOINT_CONSTRAINTS,
    ClaimPolarity,
    EntityType,
    GraphEntitySchema,
    GraphExtractionBatch,
    GraphFactCandidate,
    GraphProvenanceSchema,
    GraphQualifierSchema,
    RelationshipPredicate,
    normalize_decimal_spaces,
    parse_numeric_value,
)

SAMPLE_PAPER_ID = UUID("11111111-1111-1111-1111-111111111111")
SAMPLE_CHUNK_ID = UUID("22222222-2222-2222-2222-222222222222")
SAMPLE_ELEMENT_ID = UUID("33333333-3333-3333-3333-333333333333")
SAMPLE_SHA256 = "a" * 64


def make_provenance(
    exact_quote: str = "Our Transformer achieves 28.4 BLEU.",
    char_start: int = 0,
    char_end: int | None = None,
    page_number: int = 1,
    sha256: str = SAMPLE_SHA256,
) -> GraphProvenanceSchema:
    if char_end is None:
        char_end = char_start + len(exact_quote)
    return GraphProvenanceSchema(
        paper_id=SAMPLE_PAPER_ID,
        chunk_id=SAMPLE_CHUNK_ID,
        element_id=SAMPLE_ELEMENT_ID,
        page_number=page_number,
        exact_quote=exact_quote,
        char_start=char_start,
        char_end=char_end,
        document_sha256=sha256,
        parser_version="v1",
    )


# ==============================================================================
# 1. Entity Schema and Type Tests
# ==============================================================================


def test_entity_schema_valid():
    """Verify GraphEntitySchema with valid types and aliases."""
    entity = GraphEntitySchema(
        id="proj:method:transformer",
        name="Transformer",
        type=EntityType.METHOD,
        description="Attention-based architecture",
        aliases=["Attention Network", "Self-Attention"],
    )
    assert entity.id == "proj:method:transformer"
    assert entity.name == "Transformer"
    assert entity.type == EntityType.METHOD
    assert len(entity.aliases) == 2


def test_entity_schema_case_insensitive_type():
    """Verify string inputs parse cleanly to EntityType enum."""
    e1 = GraphEntitySchema(id="e1", name="Attention", type="concept")
    assert e1.type == EntityType.CONCEPT

    e2 = GraphEntitySchema(id="e2", name="Dataset", type="DATASET")
    assert e2.type == EntityType.DATASET


def test_invalid_entity_label_fails():
    """Verify unknown entity label raises ValidationError."""
    with pytest.raises(ValidationError) as exc:
        GraphEntitySchema(id="e1", name="Something", type="NonExistentType")
    assert "type" in str(exc.value)


def test_entity_empty_id_or_name_fails():
    """Verify empty or whitespace-only name and id fail validation."""
    with pytest.raises(ValidationError):
        GraphEntitySchema(id="", name="Valid", type=EntityType.METHOD)

    with pytest.raises(ValidationError):
        GraphEntitySchema(id="   ", name="Valid", type=EntityType.METHOD)

    with pytest.raises(ValidationError):
        GraphEntitySchema(id="e1", name="", type=EntityType.METHOD)

    with pytest.raises(ValidationError):
        GraphEntitySchema(id="e1", name="   ", type=EntityType.METHOD)


# ==============================================================================
# 2. Relationship Predicate and Endpoint Constraints Tests
# ==============================================================================


def test_all_defined_predicates_have_constraints():
    """Verify every RelationshipPredicate member is mapped in VALID_ENDPOINT_CONSTRAINTS."""
    for predicate in RelationshipPredicate:
        assert predicate in VALID_ENDPOINT_CONSTRAINTS
        allowed = VALID_ENDPOINT_CONSTRAINTS[predicate]
        assert len(allowed) > 0
        for subj, obj in allowed:
            assert isinstance(subj, EntityType)
            assert isinstance(obj, EntityType)


def test_invalid_predicate_fails():
    """Verify invalid predicate in fact candidate raises ValidationError."""
    subj = GraphEntitySchema(id="s1", name="Paper", type=EntityType.PAPER)
    obj = GraphEntitySchema(id="o1", name="Method", type=EntityType.METHOD)
    prov = make_provenance()

    with pytest.raises(ValidationError) as exc:
        GraphFactCandidate(
            subject=subj,
            predicate="INVALID_PREDICATE",  # type: ignore
            object=obj,
            provenance=prov,
        )
    assert "predicate" in str(exc.value)


@pytest.mark.parametrize(
    ("predicate", "subj_type", "obj_type"),
    [
        (RelationshipPredicate.PROPOSES_METHOD, EntityType.PAPER, EntityType.METHOD),
        (RelationshipPredicate.PROPOSES_METHOD, EntityType.PAPER, EntityType.CONCEPT),
        (RelationshipPredicate.USES_MODEL, EntityType.PAPER, EntityType.MODEL),
        (RelationshipPredicate.USES_MODEL, EntityType.PAPER, EntityType.CONCEPT),
        (RelationshipPredicate.EVALUATED_ON, EntityType.METHOD, EntityType.DATASET),
        (RelationshipPredicate.EVALUATED_ON, EntityType.MODEL, EntityType.DATASET),
        (RelationshipPredicate.EVALUATED_ON, EntityType.PAPER, EntityType.DATASET),
        (RelationshipPredicate.EVALUATED_ON, EntityType.METHOD, EntityType.TASK),
        (RelationshipPredicate.EVALUATED_ON, EntityType.MODEL, EntityType.TASK),
        (RelationshipPredicate.EVALUATED_ON, EntityType.RESULT, EntityType.DATASET),
        (RelationshipPredicate.ACHIEVES_RESULT, EntityType.METHOD, EntityType.RESULT),
        (RelationshipPredicate.ACHIEVES_RESULT, EntityType.MODEL, EntityType.RESULT),
        (RelationshipPredicate.ACHIEVES_RESULT, EntityType.PAPER, EntityType.RESULT),
        (RelationshipPredicate.CONTRADICTS, EntityType.CLAIM, EntityType.CLAIM),
        (RelationshipPredicate.CONTRADICTS, EntityType.RESULT, EntityType.RESULT),
        (RelationshipPredicate.CONTRADICTS, EntityType.CLAIM, EntityType.RESULT),
        (RelationshipPredicate.EXTENDS, EntityType.MODEL, EntityType.MODEL),
        (RelationshipPredicate.EXTENDS, EntityType.MODEL, EntityType.METHOD),
        (RelationshipPredicate.EXTENDS, EntityType.METHOD, EntityType.METHOD),
        (RelationshipPredicate.AUTHORED_BY, EntityType.PAPER, EntityType.AUTHOR),
        (RelationshipPredicate.AFFILIATED_WITH, EntityType.AUTHOR, EntityType.INSTITUTION),
    ],
)
def test_valid_endpoint_combinations(
    predicate: RelationshipPredicate, subj_type: EntityType, obj_type: EntityType
):
    """Verify each allowlisted endpoint combination passes validation."""
    candidate = GraphFactCandidate(
        subject=GraphEntitySchema(id="s", name="Subj", type=subj_type),
        predicate=predicate,
        object=GraphEntitySchema(id="o", name="Obj", type=obj_type),
        provenance=make_provenance(),
    )
    assert candidate.predicate == predicate
    assert candidate.subject.type == subj_type
    assert candidate.object.type == obj_type


def test_endpoint_mismatch_dataset_uses_model_author_fails():
    """Verify specific required test: Dataset USES_MODEL Author raises ValidationError."""
    subj = GraphEntitySchema(id="d1", name="ImageNet", type=EntityType.DATASET)
    obj = GraphEntitySchema(id="a1", name="Vaswani", type=EntityType.AUTHOR)
    prov = make_provenance()

    with pytest.raises(ValidationError) as exc:
        GraphFactCandidate(
            subject=subj,
            predicate=RelationshipPredicate.USES_MODEL,
            object=obj,
            provenance=prov,
        )
    assert "Invalid endpoint types for predicate 'USES_MODEL'" in str(exc.value)
    assert "(Dataset, Author) is not permitted" in str(exc.value)


@pytest.mark.parametrize(
    ("predicate", "subj_type", "obj_type"),
    [
        (RelationshipPredicate.AUTHORED_BY, EntityType.AUTHOR, EntityType.PAPER),  # reversed
        (RelationshipPredicate.AFFILIATED_WITH, EntityType.PAPER, EntityType.INSTITUTION),
        (RelationshipPredicate.PROPOSES_METHOD, EntityType.METHOD, EntityType.PAPER),
        (RelationshipPredicate.ACHIEVES_RESULT, EntityType.DATASET, EntityType.RESULT),
        (RelationshipPredicate.EXTENDS, EntityType.DATASET, EntityType.DATASET),
    ],
)
def test_endpoint_mismatch_general_failures(
    predicate: RelationshipPredicate, subj_type: EntityType, obj_type: EntityType
):
    """Verify various unpermitted endpoint combinations raise ValidationError."""
    with pytest.raises(ValidationError) as exc:
        GraphFactCandidate(
            subject=GraphEntitySchema(id="s", name="S", type=subj_type),
            predicate=predicate,
            object=GraphEntitySchema(id="o", name="O", type=obj_type),
            provenance=make_provenance(),
        )
    assert f"Invalid endpoint types for predicate '{predicate.value}'" in str(exc.value)


# ==============================================================================
# 3. Provenance Schema Tests
# ==============================================================================


def test_provenance_valid():
    """Verify valid provenance constructs properly."""
    quote = "Our model achieves state of the art."
    prov = make_provenance(exact_quote=quote, char_start=15)
    assert prov.char_start == 15
    assert prov.char_end == 15 + len(quote)
    assert prov.exact_quote == quote
    assert prov.document_sha256 == SAMPLE_SHA256


def test_provenance_char_span_mismatch_fails():
    """Verify char_end - char_start != len(exact_quote) raises ValidationError."""
    quote = "Short quote"  # length 11
    with pytest.raises(ValidationError) as exc:
        make_provenance(exact_quote=quote, char_start=0, char_end=15)  # 15 - 0 = 15 != 11
    assert "Character span mismatch" in str(exc.value)


def test_provenance_empty_or_whitespace_quote_fails():
    """Verify empty or whitespace-only quote raises ValidationError."""
    with pytest.raises(ValidationError):
        make_provenance(exact_quote="", char_start=0, char_end=0)

    with pytest.raises(ValidationError):
        make_provenance(exact_quote="   ", char_start=0, char_end=3)


def test_provenance_char_end_less_than_start_fails():
    """Verify char_end <= char_start raises ValidationError."""
    with pytest.raises(ValidationError) as exc:
        make_provenance(exact_quote="test", char_start=10, char_end=5)
    assert "char_end" in str(exc.value)


def test_provenance_invalid_page_number_fails():
    """Verify page_number < 1 raises ValidationError."""
    with pytest.raises(ValidationError):
        make_provenance(page_number=0)

    with pytest.raises(ValidationError):
        make_provenance(page_number=-1)


def test_provenance_invalid_sha256_fails():
    """Verify document_sha256 must be 64-char hex string."""
    with pytest.raises(ValidationError):
        make_provenance(sha256="too_short")

    with pytest.raises(ValidationError):
        make_provenance(sha256="z" * 64)  # non-hex chars


def test_provenance_sha256_normalized():
    """Verify uppercase hex sha256 is converted to lowercase."""
    upper_sha = ("A" * 32) + ("B" * 32)
    prov = make_provenance(sha256=upper_sha)
    assert prov.document_sha256 == upper_sha.lower()


# ==============================================================================
# 4. Qualifier Schema and Normalization Tests
# ==============================================================================


def test_decimal_spaces_normalization_helper():
    """Verify normalize_decimal_spaces collapses OCR/parser whitespace."""
    assert normalize_decimal_spaces("41 . 0") == "41.0"
    assert normalize_decimal_spaces("  41 . 0  ") == "41.0"
    assert normalize_decimal_spaces("27 . 5 %") == "27.5 %"
    assert normalize_decimal_spaces("+ 0 . 6") == "+0.6"
    assert normalize_decimal_spaces("1 , 000 . 5") == "1,000.5"


def test_parse_numeric_value_helper():
    """Verify parse_numeric_value handles ints, floats, strings, and messy OCR."""
    assert parse_numeric_value(41.0) == (41.0, "41.0")
    assert parse_numeric_value(41) == (41.0, "41")
    assert parse_numeric_value("41 . 0") == (41.0, "41.0")
    assert parse_numeric_value("41.0%") == (41.0, "41.0%")
    assert parse_numeric_value("41.0 BLEU") == (41.0, "41.0 BLEU")
    assert parse_numeric_value(None) == (None, None)
    assert parse_numeric_value("non-numeric") == (None, "non-numeric")


def test_qualifier_normalization_decimal_spaces_consistency():
    """Verify '41 . 0' and '41.0' normalize consistently in GraphQualifierSchema."""
    q_spaced = GraphQualifierSchema(result_value="41 . 0", unit="BLEU")
    q_standard = GraphQualifierSchema(result_value=41.0, unit="BLEU")

    assert q_spaced.result_value == 41.0
    assert q_standard.result_value == 41.0
    assert q_spaced.result_value == q_standard.result_value
    assert q_spaced.numeric_value == q_standard.numeric_value
    assert q_spaced.raw_value == "41.0"

    # Also test via raw_value
    q_raw = GraphQualifierSchema(raw_value="41 . 0", unit="BLEU")
    assert q_raw.result_value == 41.0
    assert q_raw.raw_value == "41.0"

    # Also test via numeric_value alias
    q_alias = GraphQualifierSchema(numeric_value="41 . 0", unit="BLEU")
    assert q_alias.result_value == 41.0
    assert q_alias.numeric_value == 41.0


def test_qualifier_schema_rejects_conflicting_numeric_representations():
    with pytest.raises(ValidationError, match="Conflicting numeric qualifier representations"):
        GraphQualifierSchema(result_value=999, numeric_value=999, raw_value="2.1")

    compatible_precision = GraphQualifierSchema(result_value=2.10, raw_value="2 . 1 %")
    assert compatible_precision.result_value == 2.1
    assert compatible_precision.numeric_value == 2.1
    assert compatible_precision.raw_value == "2.1 %"


def test_qualifier_unit_comparisons_and_missing_units():
    """Verify that different units or missing units do not silently become equivalent."""
    q_bleu = GraphQualifierSchema(result_value=41.0, unit="BLEU", dataset="WMT14")
    q_rouge = GraphQualifierSchema(result_value=41.0, unit="ROUGE", dataset="WMT14")
    q_no_unit = GraphQualifierSchema(result_value=41.0, dataset="WMT14")

    # Distinct objects
    assert q_bleu != q_rouge
    assert q_bleu != q_no_unit

    # Comparable check: different units are NOT comparable
    assert not q_bleu.is_comparable_with(q_rouge)

    # Missing unit does not silently match
    assert not q_bleu.is_comparable_with(q_no_unit)
    assert not q_no_unit.is_comparable_with(q_bleu)

    # Matching units ARE comparable
    q_bleu_2 = GraphQualifierSchema(result_value=42.5, unit="bleu", dataset="WMT14")
    assert q_bleu.is_comparable_with(q_bleu_2)


def test_qualifier_opposing_polarities_and_contradiction_detection():
    """Verify opposing polarities and different values on the same setting detect contradictions."""
    # Comparable setup: same dataset, metric, and comparison condition
    q_positive = GraphQualifierSchema(
        result_value=2.1,
        unit="%",
        metric="ECE",
        dataset="ImageNet",
        comparison_condition="temperature_scaling",
        polarity=ClaimPolarity.POSITIVE,
    )
    q_negative = GraphQualifierSchema(
        result_value=8.5,
        unit="%",
        metric="ECE",
        dataset="ImageNet",
        comparison_condition="temperature_scaling",
        polarity=ClaimPolarity.NEGATIVE,
    )

    assert q_positive.polarity == ClaimPolarity.POSITIVE
    assert q_negative.polarity == ClaimPolarity.NEGATIVE
    assert q_positive.is_comparable_with(q_negative)
    assert q_positive.is_contradiction_with(q_negative)

    # Opposing values under same condition even if polarity is same
    q_differing_value = GraphQualifierSchema(
        result_value=8.5,
        unit="%",
        metric="ECE",
        dataset="ImageNet",
        comparison_condition="temperature_scaling",
        polarity=ClaimPolarity.POSITIVE,
    )
    assert q_positive.is_contradiction_with(q_differing_value)

    # Same condition, same value -> NOT a contradiction
    q_same = GraphQualifierSchema(
        result_value=2.1,
        unit="%",
        metric="ECE",
        dataset="ImageNet",
        comparison_condition="temperature_scaling",
        polarity=ClaimPolarity.POSITIVE,
    )
    assert not q_positive.is_contradiction_with(q_same)


def test_qualifier_different_datasets_are_not_contradictions():
    """Verify that different datasets are not comparable and cannot be contradictions."""
    q_imagenet = GraphQualifierSchema(
        result_value=2.1,
        unit="%",
        metric="ECE",
        dataset="ImageNet",
        polarity=ClaimPolarity.POSITIVE,
    )
    q_cifar = GraphQualifierSchema(
        result_value=5.4,
        unit="%",
        metric="ECE",
        dataset="CIFAR-100",
        polarity=ClaimPolarity.POSITIVE,
    )

    assert not q_imagenet.is_comparable_with(q_cifar)
    assert not q_imagenet.is_contradiction_with(q_cifar)


# ==============================================================================
# 5. Extraction Batch Limits Tests
# ==============================================================================


def test_extraction_batch_valid():
    """Verify valid extraction batch with <=100 items passes."""
    entities = [
        GraphEntitySchema(id=f"e{i}", name=f"Entity {i}", type=EntityType.METHOD) for i in range(10)
    ]
    facts = [
        GraphFactCandidate(
            subject=entities[0],
            predicate=RelationshipPredicate.EVALUATED_ON,
            object=GraphEntitySchema(id=f"d{i}", name=f"Dataset {i}", type=EntityType.DATASET),
            provenance=make_provenance(),
        )
        for i in range(5)
    ]
    batch = GraphExtractionBatch(
        project_id=uuid4(),
        paper_id=uuid4(),
        entities=entities,
        facts=facts,
    )
    assert len(batch.entities) == 10
    assert len(batch.facts) == 5


def test_extraction_batch_oversized_entities_rejected():
    """Verify batch with >100 entities raises ValidationError."""
    too_many_entities = [
        GraphEntitySchema(id=f"e{i}", name=f"Entity {i}", type=EntityType.METHOD)
        for i in range(101)
    ]
    with pytest.raises(ValidationError) as exc:
        GraphExtractionBatch(
            project_id=uuid4(),
            paper_id=uuid4(),
            entities=too_many_entities,
            facts=[],
        )
    assert "entities" in str(exc.value)


def test_extraction_batch_oversized_facts_rejected():
    """Verify batch with >100 facts raises ValidationError."""
    subj = GraphEntitySchema(id="s", name="Subj", type=EntityType.METHOD)
    obj = GraphEntitySchema(id="o", name="Obj", type=EntityType.DATASET)
    prov = make_provenance()

    too_many_facts = [
        GraphFactCandidate(
            subject=subj,
            predicate=RelationshipPredicate.EVALUATED_ON,
            object=obj,
            provenance=prov,
        )
        for _ in range(101)
    ]

    with pytest.raises(ValidationError) as exc:
        GraphExtractionBatch(
            project_id=uuid4(),
            paper_id=uuid4(),
            entities=[],
            facts=too_many_facts,
        )
    assert "facts" in str(exc.value)


# ==============================================================================
# 6. Edge Cases and Coverage Boost Tests
# ==============================================================================


def test_helper_edge_cases():
    """Verify normalize_decimal_spaces and parse_numeric_value handle non-string and edge inputs."""
    assert normalize_decimal_spaces(123) == 123  # type: ignore
    obj = object()
    assert parse_numeric_value(obj) == (None, str(obj))


def test_qualifier_edge_cases_and_comparisons():
    """Verify string cleaning, split/task/metric discrepancies, and non-qualifier comparisons."""
    # String metadata cleaning with whitespace
    q = GraphQualifierSchema(
        result_value="50.0",
        raw_value=" 50.0 ",
        unit=" % ",
        metric=" WER ",
        dataset=" LibriSpeech ",
        split=" test-clean ",
        task=" ASR ",
        comparison_condition=" beam_search ",
        polarity="uncertain",
    )
    assert q.unit == "%"
    assert q.metric == "WER"
    assert q.dataset == "LibriSpeech"
    assert q.split == "test-clean"
    assert q.task == "ASR"
    assert q.comparison_condition == "beam_search"
    assert q.polarity == ClaimPolarity.UNCERTAIN

    # Invalid polarity string raises ValidationError
    with pytest.raises(ValidationError):
        GraphQualifierSchema(result_value=1.0, polarity="UNKNOWN_POL")  # type: ignore

    # Non-qualifier comparison
    assert not q.is_comparable_with("not_a_qualifier")  # type: ignore

    # Insufficient context (no unit, metric, or dataset)
    q_empty = GraphQualifierSchema(result_value=1.0)
    q_empty_2 = GraphQualifierSchema(result_value=1.0)
    assert not q_empty.is_comparable_with(q_empty_2)

    # Metric mismatch
    q_m1 = GraphQualifierSchema(result_value=1.0, metric="WER", dataset="D")
    q_m2 = GraphQualifierSchema(result_value=1.0, metric="CER", dataset="D")
    assert not q_m1.is_comparable_with(q_m2)

    # Split mismatch
    q_s1 = GraphQualifierSchema(result_value=1.0, unit="%", dataset="D", split="train")
    q_s2 = GraphQualifierSchema(result_value=1.0, unit="%", dataset="D", split="test")
    assert not q_s1.is_comparable_with(q_s2)

    # Task mismatch
    q_t1 = GraphQualifierSchema(result_value=1.0, unit="%", dataset="D", task="TaskA")
    q_t2 = GraphQualifierSchema(result_value=1.0, unit="%", dataset="D", task="TaskB")
    assert not q_t1.is_comparable_with(q_t2)


def test_fact_candidate_with_qualifiers():
    """Verify GraphFactCandidate correctly includes qualifiers."""
    subj = GraphEntitySchema(id="m1", name="Model", type=EntityType.MODEL)
    obj = GraphEntitySchema(id="r1", name="Result", type=EntityType.RESULT)
    q = GraphQualifierSchema(result_value=28.4, unit="BLEU")
    prov = make_provenance()

    candidate = GraphFactCandidate(
        subject=subj,
        predicate=RelationshipPredicate.ACHIEVES_RESULT,
        object=obj,
        qualifiers=q,
        provenance=prov,
    )
    assert candidate.qualifiers is not None
    assert candidate.qualifiers.result_value == 28.4
