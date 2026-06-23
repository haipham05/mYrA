"""Tests for GraphRAG query engine (Tasks 5.22, 5.23, 5.24, 5.25)."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from uuid import uuid4

import pytest
from neo4j import GraphDatabase

from app.db.models import GraphFactSnapshot, Job, Paper, Project
from app.db.session import SessionLocal, create_tables
from app.schemas.graph import EntityType, RelationshipPredicate
from app.services.graphrag.identity import generate_entity_key, generate_fact_id
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.query_engine import (
    MAX_CONTRADICTION_LIMIT,
    MAX_RELATIONSHIP_LIMIT,
    MAX_THEMES_LIMIT,
    build_contradiction_candidates,
    build_corpus_themes,
    build_relationship_candidates,
)
from conftest import disposable_neo4j_test_uri

skip_if_no_neo4j = pytest.mark.skipif(
    disposable_neo4j_test_uri() is None,
    reason=(
        "Neo4j integration checks require MYRA_TEST_NEO4J_URI and "
        "MYRA_ALLOW_DISPOSABLE_NEO4J_TESTS=1"
    ),
)


@pytest.fixture(autouse=True)
def clean_database():
    """Ensure clean PostgreSQL tables before each test."""
    create_tables()
    with SessionLocal() as db:
        db.query(GraphFactSnapshot).delete()
        db.query(Job).delete()
        db.query(Paper).delete()
        db.query(Project).delete()
        db.commit()


@pytest.fixture
def real_repo(disposable_neo4j_uri: str) -> Generator[Neo4jRepository, None, None]:
    """Connect only to the explicitly acknowledged disposable test target."""
    driver = GraphDatabase.driver(disposable_neo4j_uri, auth=None)
    driver.verify_connectivity()
    repo = Neo4jRepository(driver=driver, database="neo4j")
    repo.ensure_schema()
    yield repo
    driver.close()


@pytest.fixture
def db_session():
    """Provide a database session."""
    with SessionLocal() as session:
        yield session


# =============================================================================
# Test 1: Relationship candidates: finds labeled relation and sources between
# Method X and Dataset Y; queries for another project's entities return empty.
# =============================================================================


@skip_if_no_neo4j
def test_relationship_candidates_finds_relation_and_enforces_isolation(
    real_repo: Neo4jRepository, db_session
) -> None:
    project_a = uuid4()
    project_b = uuid4()
    paper_1 = uuid4()

    try:
        # Create Method X and Dataset Y in Project A
        node_method_a = {
            "key": generate_entity_key(project_a, EntityType.METHOD, "Method X"),
            "name": "Method X",
            "type": EntityType.METHOD.value,
        }
        node_dataset_a = {
            "key": generate_entity_key(project_a, EntityType.DATASET, "Dataset Y"),
            "name": "Dataset Y",
            "type": EntityType.DATASET.value,
        }
        real_repo.upsert_nodes(project_a, [node_method_a, node_dataset_a])

        fact_1_id = generate_fact_id(
            project_a,
            paper_1,
            node_method_a["key"],
            RelationshipPredicate.EVALUATED_ON.value,
            node_dataset_a["key"],
            char_start=0,
            char_end=35,
        )
        real_repo.upsert_facts(
            project_a,
            paper_1,
            "gen_001",
            [
                {
                    "fact_id": fact_1_id,
                    "subject_key": node_method_a["key"],
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": node_dataset_a["key"],
                    "object_name": "Dataset Y",
                    "object_type": "Dataset",
                    "predicate": RelationshipPredicate.EVALUATED_ON.value,
                    "qualifiers": {"metric": "Accuracy", "dataset": "Dataset Y"},
                    "exact_quote": "Method X is evaluated on Dataset Y.",
                    "page_number": 3,
                }
            ],
        )

        # Create Method X and Dataset Z in Project B (different dataset)
        node_method_b = {
            "key": generate_entity_key(project_b, EntityType.METHOD, "Method X"),
            "name": "Method X",
            "type": EntityType.METHOD.value,
        }
        node_dataset_b = {
            "key": generate_entity_key(project_b, EntityType.DATASET, "Dataset Z"),
            "name": "Dataset Z",
            "type": EntityType.DATASET.value,
        }
        real_repo.upsert_nodes(project_b, [node_method_b, node_dataset_b])

        # Query Project A for Method X and Dataset Y
        candidates_a = build_relationship_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_a,
            entity_a_name="Method X",
            entity_b_name="Dataset Y",
        )
        assert len(candidates_a) == 1
        cand = candidates_a[0]
        assert cand["fact_id"] == fact_1_id
        assert cand["predicate"] == "EVALUATED_ON"
        assert cand["subject_name"] == "Method X"
        assert cand["object_name"] == "Dataset Y"
        assert cand["exact_quote"] == "Method X is evaluated on Dataset Y."
        assert cand["page_number"] == 3
        assert cand["paper_id"] == str(paper_1)

        # Query Project B for Dataset Y (does not exist in Project B) -> returns []
        candidates_b = build_relationship_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_b,
            entity_a_name="Method X",
            entity_b_name="Dataset Y",
        )
        assert candidates_b == []

        # Query for nonexistent entity returns []
        candidates_nonexistent = build_relationship_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_a,
            entity_a_name="Method X",
            entity_b_name="Unknown Dataset",
        )
        assert candidates_nonexistent == []

    finally:
        real_repo.delete_project_graph(project_a)
        real_repo.delete_project_graph(project_b)


# =============================================================================
# Test 2: Preserves multiple assertions: Two papers evaluating same method
# on same dataset return both facts with their distinct quotes/paper_ids.
# =============================================================================


@skip_if_no_neo4j
def test_relationship_candidates_preserves_multiple_paper_assertions(
    real_repo: Neo4jRepository, db_session
) -> None:
    project_id = uuid4()
    paper_1 = uuid4()
    paper_2 = uuid4()

    try:
        s_key = generate_entity_key(project_id, EntityType.METHOD, "Method X")
        o_key = generate_entity_key(project_id, EntityType.DATASET, "Dataset Y")

        real_repo.upsert_nodes(
            project_id,
            [
                {"key": s_key, "name": "Method X", "type": "Method"},
                {"key": o_key, "name": "Dataset Y", "type": "Dataset"},
            ],
        )

        fact_1_id = f"fact_p1_{uuid4().hex[:8]}"
        fact_2_id = f"fact_p2_{uuid4().hex[:8]}"

        # Paper 1 assertion
        real_repo.upsert_facts(
            project_id,
            paper_1,
            "gen_001",
            [
                {
                    "fact_id": fact_1_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key,
                    "object_name": "Dataset Y",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {
                        "metric": "Accuracy",
                        "dataset": "Dataset Y",
                        "result_value": 92.5,
                    },
                    "exact_quote": "Paper 1 achieves 92.5% accuracy on Dataset Y.",
                    "page_number": 4,
                }
            ],
        )

        # Paper 2 assertion
        real_repo.upsert_facts(
            project_id,
            paper_2,
            "gen_002",
            [
                {
                    "fact_id": fact_2_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key,
                    "object_name": "Dataset Y",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {
                        "metric": "Accuracy",
                        "dataset": "Dataset Y",
                        "result_value": 88.0,
                    },
                    "exact_quote": "Paper 2 replicates Method X with 88.0% on Dataset Y.",
                    "page_number": 7,
                }
            ],
        )

        candidates = build_relationship_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_id,
            entity_a_name="Method X",
            entity_b_name="Dataset Y",
        )

        # Both source assertions must be preserved without collapsing
        assert len(candidates) == 2
        fids = {c["fact_id"] for c in candidates}
        assert fids == {fact_1_id, fact_2_id}

        pids = {c["paper_id"] for c in candidates}
        assert pids == {str(paper_1), str(paper_2)}

        quotes = {c["exact_quote"] for c in candidates}
        assert "Paper 1 achieves 92.5% accuracy on Dataset Y." in quotes
        assert "Paper 2 replicates Method X with 88.0% on Dataset Y." in quotes

    finally:
        real_repo.delete_project_graph(project_id)


# =============================================================================
# Test 3: True contradiction candidate: Opposing claims on ImageNet with
# ECE metric (2.1% vs 8.5%) detected as NUMERIC_VALUE contradiction.
# =============================================================================


@skip_if_no_neo4j
def test_contradiction_candidate_numeric_value_conflict(
    real_repo: Neo4jRepository, db_session
) -> None:
    project_id = uuid4()
    paper_1 = uuid4()
    paper_2 = uuid4()

    try:
        s_key = generate_entity_key(project_id, EntityType.METHOD, "Method X")
        o_key = generate_entity_key(project_id, EntityType.DATASET, "ImageNet")

        real_repo.upsert_nodes(
            project_id,
            [
                {"key": s_key, "name": "Method X", "type": "Method"},
                {"key": o_key, "name": "ImageNet", "type": "Dataset"},
            ],
        )

        fact_1_id = f"fact_num_1_{uuid4().hex[:8]}"
        fact_2_id = f"fact_num_2_{uuid4().hex[:8]}"

        # Paper 1: Method X achieves 2.1% ECE on ImageNet
        real_repo.upsert_facts(
            project_id,
            paper_1,
            "gen_001",
            [
                {
                    "fact_id": fact_1_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key,
                    "object_name": "ImageNet",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {
                        "metric": "ECE",
                        "dataset": "ImageNet",
                        "result_value": 2.1,
                        "polarity": "POSITIVE",
                        "task": "classification",
                        "split": "test",
                        "unit": "%",
                        "comparison_condition": "standard evaluation",
                    },
                    "exact_quote": (
                        "Method X achieves 2.1% ECE on ImageNet for classification test "
                        "split under standard evaluation."
                    ),
                    "page_number": 2,
                }
            ],
        )

        # Paper 2: Method X achieves 8.5% ECE on ImageNet
        real_repo.upsert_facts(
            project_id,
            paper_2,
            "gen_002",
            [
                {
                    "fact_id": fact_2_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key,
                    "object_name": "ImageNet",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {
                        "metric": "ECE",
                        "dataset": "ImageNet",
                        "result_value": 8.5,
                        "polarity": "POSITIVE",
                        "task": "classification",
                        "split": "test",
                        "unit": "%",
                        "comparison_condition": "standard evaluation",
                    },
                    "exact_quote": (
                        "Method X reports 8.5% ECE on ImageNet for classification test "
                        "split under standard evaluation."
                    ),
                    "page_number": 5,
                }
            ],
        )

        contradictions = build_contradiction_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_id,
        )

        assert len(contradictions) == 1
        c = contradictions[0]
        assert c["conflict_type"] == "NUMERIC_VALUE"
        assert c["comparison_basis"]["dataset"].lower() == "imagenet"
        assert c["comparison_basis"]["metric"].lower() == "ece"

        f_ids = {c["fact_a"]["fact_id"], c["fact_b"]["fact_id"]}
        assert f_ids == {fact_1_id, fact_2_id}

    finally:
        real_repo.delete_project_graph(project_id)


# =============================================================================
# Test 4: Polarity contradiction: Positive claim vs "fails to converge"
# detected as POLARITY contradiction.
# =============================================================================


@skip_if_no_neo4j
def test_contradiction_candidate_polarity_conflict(real_repo: Neo4jRepository, db_session) -> None:
    project_id = uuid4()
    paper_1 = uuid4()
    paper_2 = uuid4()

    try:
        s_key = generate_entity_key(project_id, EntityType.METHOD, "Method X")
        o_key = generate_entity_key(project_id, EntityType.DATASET, "ImageNet")

        real_repo.upsert_nodes(
            project_id,
            [
                {"key": s_key, "name": "Method X", "type": "Method"},
                {"key": o_key, "name": "ImageNet", "type": "Dataset"},
            ],
        )

        fact_1_id = f"fact_pol_1_{uuid4().hex[:8]}"
        fact_2_id = f"fact_pol_2_{uuid4().hex[:8]}"

        # Paper 1: Positive claim achieves 2.1% ECE on ImageNet
        real_repo.upsert_facts(
            project_id,
            paper_1,
            "gen_001",
            [
                {
                    "fact_id": fact_1_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key,
                    "object_name": "ImageNet",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {
                        "metric": "ECE",
                        "dataset": "ImageNet",
                        "result_value": 2.1,
                        "polarity": "POSITIVE",
                        "task": "classification",
                        "split": "test",
                        "unit": "%",
                        "comparison_condition": "standard evaluation",
                    },
                    "exact_quote": (
                        "Method X achieves 2.1% ECE on ImageNet for classification test "
                        "split under standard evaluation."
                    ),
                    "page_number": 2,
                }
            ],
        )

        # Paper 2: Fails to converge on ImageNet with ECE
        real_repo.upsert_facts(
            project_id,
            paper_2,
            "gen_002",
            [
                {
                    "fact_id": fact_2_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key,
                    "object_name": "ImageNet",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {
                        "metric": "ECE",
                        "dataset": "ImageNet",
                        "polarity": "NEGATIVE",
                        "task": "classification",
                        "split": "test",
                        "unit": "%",
                        "comparison_condition": "standard evaluation",
                    },
                    "exact_quote": (
                        "Method X fails to converge when reporting ECE percentages on ImageNet "
                        "for classification test split under standard evaluation."
                    ),
                    "page_number": 6,
                }
            ],
        )

        contradictions = build_contradiction_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_id,
        )

        assert len(contradictions) == 1
        c = contradictions[0]
        assert c["conflict_type"] == "POLARITY"
        assert c["comparison_basis"]["dataset"].lower() == "imagenet"
        assert c["comparison_basis"]["metric"].lower() == "ece"

        f_ids = {c["fact_a"]["fact_id"], c["fact_b"]["fact_id"]}
        assert f_ids == {fact_1_id, fact_2_id}

    finally:
        real_repo.delete_project_graph(project_id)


# =============================================================================
# Test 5: Non-comparable pair: Different datasets (ImageNet 2.1% vs CIFAR-100 5.4%)
# are NOT flagged as contradictions.
# =============================================================================


@skip_if_no_neo4j
def test_contradiction_candidate_different_datasets_not_flagged(
    real_repo: Neo4jRepository, db_session
) -> None:
    project_id = uuid4()
    paper_1 = uuid4()
    paper_2 = uuid4()

    try:
        s_key = generate_entity_key(project_id, EntityType.METHOD, "Method X")
        o_key1 = generate_entity_key(project_id, EntityType.DATASET, "ImageNet")
        o_key2 = generate_entity_key(project_id, EntityType.DATASET, "CIFAR-100")

        real_repo.upsert_nodes(
            project_id,
            [
                {"key": s_key, "name": "Method X", "type": "Method"},
                {"key": o_key1, "name": "ImageNet", "type": "Dataset"},
                {"key": o_key2, "name": "CIFAR-100", "type": "Dataset"},
            ],
        )

        fact_1_id = f"fact_diff_1_{uuid4().hex[:8]}"
        fact_2_id = f"fact_diff_2_{uuid4().hex[:8]}"

        # Paper 1: Method X on ImageNet with 2.1% ECE
        real_repo.upsert_facts(
            project_id,
            paper_1,
            "gen_001",
            [
                {
                    "fact_id": fact_1_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key1,
                    "object_name": "ImageNet",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {
                        "metric": "ECE",
                        "dataset": "ImageNet",
                        "result_value": 2.1,
                        "polarity": "POSITIVE",
                    },
                    "exact_quote": "Method X achieves 2.1% ECE on ImageNet.",
                    "page_number": 2,
                }
            ],
        )

        # Paper 2: Method X on CIFAR-100 with 5.4% ECE (different dataset)
        real_repo.upsert_facts(
            project_id,
            paper_2,
            "gen_002",
            [
                {
                    "fact_id": fact_2_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key2,
                    "object_name": "CIFAR-100",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {
                        "metric": "ECE",
                        "dataset": "CIFAR-100",
                        "result_value": 5.4,
                        "polarity": "POSITIVE",
                    },
                    "exact_quote": "Method X achieves 5.4% ECE on CIFAR-100.",
                    "page_number": 3,
                }
            ],
        )

        contradictions = build_contradiction_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_id,
        )

        # Non-comparable pair must return empty list (not flagged as contradiction)
        assert contradictions == []

    finally:
        real_repo.delete_project_graph(project_id)


# =============================================================================
# Test 6: Corpus themes: Recurring method across papers grouped with contributing
# paper_ids and fact_ids.
# =============================================================================


@skip_if_no_neo4j
def test_corpus_themes_recurring_method_across_papers(
    real_repo: Neo4jRepository, db_session
) -> None:
    project_id = uuid4()
    paper_1 = uuid4()
    paper_2 = uuid4()
    paper_3 = uuid4()

    try:
        s_key = generate_entity_key(project_id, EntityType.METHOD, "Method X")
        o_key1 = generate_entity_key(project_id, EntityType.DATASET, "ImageNet")
        o_key2 = generate_entity_key(project_id, EntityType.DATASET, "CIFAR-10")

        real_repo.upsert_nodes(
            project_id,
            [
                {"key": s_key, "name": "Method X", "type": "Method"},
                {"key": o_key1, "name": "ImageNet", "type": "Dataset"},
                {"key": o_key2, "name": "CIFAR-10", "type": "Dataset"},
            ],
        )

        f1_id = f"fact_theme_1_{uuid4().hex[:8]}"
        f2_id = f"fact_theme_2_{uuid4().hex[:8]}"
        f3_id = f"fact_theme_3_{uuid4().hex[:8]}"

        # Paper 1: Method X on ImageNet
        real_repo.upsert_facts(
            project_id,
            paper_1,
            "gen_001",
            [
                {
                    "fact_id": f1_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key1,
                    "object_name": "ImageNet",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {"metric": "Accuracy", "dataset": "ImageNet"},
                    "exact_quote": "Method X evaluated on ImageNet in Paper 1.",
                    "page_number": 1,
                }
            ],
        )

        # Paper 2: Method X on CIFAR-10
        real_repo.upsert_facts(
            project_id,
            paper_2,
            "gen_002",
            [
                {
                    "fact_id": f2_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key2,
                    "object_name": "CIFAR-10",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {"metric": "Accuracy", "dataset": "CIFAR-10"},
                    "exact_quote": "Method X evaluated on CIFAR-10 in Paper 2.",
                    "page_number": 2,
                }
            ],
        )

        # Paper 3: Method X on ImageNet
        real_repo.upsert_facts(
            project_id,
            paper_3,
            "gen_003",
            [
                {
                    "fact_id": f3_id,
                    "subject_key": s_key,
                    "subject_name": "Method X",
                    "subject_type": "Method",
                    "object_key": o_key1,
                    "object_name": "ImageNet",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {"metric": "Accuracy", "dataset": "ImageNet"},
                    "exact_quote": "Method X evaluated on ImageNet in Paper 3.",
                    "page_number": 3,
                }
            ],
        )

        themes = build_corpus_themes(
            db=db_session,
            repo=real_repo,
            project_id=project_id,
            min_papers=2,
        )

        assert len(themes) >= 1

        # Look for Method X entity theme
        method_themes = [
            t for t in themes if t["theme_type"] == "ENTITY" and t["name"] == "Method X"
        ]
        assert len(method_themes) == 1
        mt = method_themes[0]
        assert mt["paper_count"] == 3
        assert set(mt["paper_ids"]) == {str(paper_1), str(paper_2), str(paper_3)}
        assert set(mt["fact_ids"]) == {f1_id, f2_id, f3_id}

        # Look for ImageNet entity theme (in Paper 1 and Paper 3)
        imagenet_themes = [
            t for t in themes if t["theme_type"] == "ENTITY" and t["name"] == "ImageNet"
        ]
        assert len(imagenet_themes) == 1
        it = imagenet_themes[0]
        assert it["paper_count"] == 2
        assert set(it["paper_ids"]) == {str(paper_1), str(paper_3)}

        # Look for Method X EVALUATED_ON ImageNet relation theme
        rel_themes = [
            t
            for t in themes
            if t["theme_type"] == "RELATION"
            and t["predicate"] == "EVALUATED_ON"
            and t["object_name"] == "ImageNet"
        ]
        assert len(rel_themes) == 1
        rt = rel_themes[0]
        assert rt["paper_count"] == 2
        assert set(rt["paper_ids"]) == {str(paper_1), str(paper_3)}

    finally:
        real_repo.delete_project_graph(project_id)


# =============================================================================
# Test 7: Single-paper or no-theme case: Entity appearing in only 1 paper is
# excluded when min_papers=2.
# =============================================================================


@skip_if_no_neo4j
def test_corpus_themes_single_paper_excluded(real_repo: Neo4jRepository, db_session) -> None:
    project_id = uuid4()
    paper_1 = uuid4()

    try:
        s_key = generate_entity_key(project_id, EntityType.METHOD, "Method Solo")
        o_key = generate_entity_key(project_id, EntityType.DATASET, "Dataset Solo")

        real_repo.upsert_nodes(
            project_id,
            [
                {"key": s_key, "name": "Method Solo", "type": "Method"},
                {"key": o_key, "name": "Dataset Solo", "type": "Dataset"},
            ],
        )

        f1_id = f"fact_solo_{uuid4().hex[:8]}"
        real_repo.upsert_facts(
            project_id,
            paper_1,
            "gen_001",
            [
                {
                    "fact_id": f1_id,
                    "subject_key": s_key,
                    "subject_name": "Method Solo",
                    "subject_type": "Method",
                    "object_key": o_key,
                    "object_name": "Dataset Solo",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {"metric": "F1", "dataset": "Dataset Solo"},
                    "exact_quote": "Method Solo on Dataset Solo in Paper 1 only.",
                    "page_number": 1,
                }
            ],
        )

        # With min_papers=2, single paper entities/relations are excluded
        themes = build_corpus_themes(
            db=db_session,
            repo=real_repo,
            project_id=project_id,
            min_papers=2,
        )

        assert themes == []

    finally:
        real_repo.delete_project_graph(project_id)


# =============================================================================
# Test 8: Security & bounds: Limits are capped, project isolation strictly
# enforced (Project A query never sees Project B nodes/facts).
# =============================================================================


@skip_if_no_neo4j
def test_security_bounds_and_project_isolation(real_repo: Neo4jRepository, db_session) -> None:
    project_a = uuid4()
    project_b = uuid4()
    paper_a1 = uuid4()
    paper_a2 = uuid4()

    try:
        s_key = generate_entity_key(project_a, EntityType.METHOD, "Shared Method")
        o_key = generate_entity_key(project_a, EntityType.DATASET, "Shared Dataset")

        real_repo.upsert_nodes(
            project_a,
            [
                {"key": s_key, "name": "Shared Method", "type": "Method"},
                {"key": o_key, "name": "Shared Dataset", "type": "Dataset"},
            ],
        )

        # Populate Project A with facts across 2 papers
        facts_a = []
        for i in range(1, 10):
            facts_a.append(
                {
                    "fact_id": f"fact_a_p1_{i}_{uuid4().hex[:6]}",
                    "subject_key": s_key,
                    "subject_name": "Shared Method",
                    "subject_type": "Method",
                    "object_key": o_key,
                    "object_name": "Shared Dataset",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {
                        "metric": "BLEU",
                        "dataset": "Shared Dataset",
                        "result_value": 30.0 + i,
                        "polarity": "POSITIVE",
                    },
                    "exact_quote": f"Paper 1 BLEU score is {30.0 + i}",
                    "page_number": i,
                }
            )
        real_repo.upsert_facts(project_a, paper_a1, "gen_a1", facts_a)

        facts_a2 = []
        for i in range(1, 10):
            facts_a2.append(
                {
                    "fact_id": f"fact_a_p2_{i}_{uuid4().hex[:6]}",
                    "subject_key": s_key,
                    "subject_name": "Shared Method",
                    "subject_type": "Method",
                    "object_key": o_key,
                    "object_name": "Shared Dataset",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {
                        "metric": "BLEU",
                        "dataset": "Shared Dataset",
                        "result_value": 50.0 + i,
                        "polarity": "POSITIVE",
                    },
                    "exact_quote": f"Paper 2 BLEU score is {50.0 + i}",
                    "page_number": i,
                }
            )
        real_repo.upsert_facts(project_a, paper_a2, "gen_a2", facts_a2)

        # 1. Project Isolation: Project B must NOT see Project A facts/relations
        rels_b = build_relationship_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_b,
            entity_a_name="Shared Method",
            entity_b_name="Shared Dataset",
        )
        assert rels_b == []

        contras_b = build_contradiction_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_b,
        )
        assert contras_b == []

        themes_b = build_corpus_themes(
            db=db_session,
            repo=real_repo,
            project_id=project_b,
        )
        assert themes_b == []

        # 2. Limit capping verification:
        # Requesting limit=1000 for relationships is capped at MAX_RELATIONSHIP_LIMIT (50)
        rels_capped = build_relationship_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_a,
            entity_a_name="Shared Method",
            entity_b_name="Shared Dataset",
            limit=1000,
        )
        assert len(rels_capped) <= MAX_RELATIONSHIP_LIMIT

        # Requesting limit=1000 for contradictions is capped at MAX_CONTRADICTION_LIMIT (50)
        contras_capped = build_contradiction_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_a,
            limit=1000,
        )
        assert len(contras_capped) <= MAX_CONTRADICTION_LIMIT

        # Requesting limit=100 for themes is capped at MAX_THEMES_LIMIT (20)
        themes_capped = build_corpus_themes(
            db=db_session,
            repo=real_repo,
            project_id=project_a,
            limit=100,
        )
        assert len(themes_capped) <= MAX_THEMES_LIMIT

        # 3. Non-positive limits return empty list
        assert (
            build_relationship_candidates(
                db=db_session,
                repo=real_repo,
                project_id=project_a,
                entity_a_name="Shared Method",
                entity_b_name="Shared Dataset",
                limit=0,
            )
            == []
        )
        assert (
            build_contradiction_candidates(
                db=db_session, repo=real_repo, project_id=project_a, limit=-5
            )
            == []
        )
        assert (
            build_corpus_themes(db=db_session, repo=real_repo, project_id=project_a, limit=0) == []
        )

    finally:
        real_repo.delete_project_graph(project_a)
        real_repo.delete_project_graph(project_b)


# =============================================================================
# Additional Unit Tests: Helper functions and Edge Cases
# =============================================================================


def test_parse_qualifiers_variants() -> None:
    from app.services.graphrag.query_engine import _parse_qualifiers

    assert _parse_qualifiers(None) == {}
    assert _parse_qualifiers({"metric": "ECE"}) == {"metric": "ECE"}
    assert _parse_qualifiers('{"dataset": "ImageNet"}') == {"dataset": "ImageNet"}
    assert _parse_qualifiers("not valid json") == {}
    assert _parse_qualifiers(123) == {}


def test_find_best_node_logic() -> None:
    from app.services.graphrag.query_engine import _find_best_node

    assert _find_best_node([], "any") is None

    nodes = [
        {"key": "k1", "name": "Deep Residual Learning", "aliases": ["ResNet", "ResNet-50"]},
        {"key": "k2", "name": "ResNet", "aliases": []},
    ]

    # Exact name match
    assert _find_best_node(nodes, "ResNet")["key"] == "k2"
    # Alias match
    assert _find_best_node(nodes, "ResNet-50")["key"] == "k1"
    # Fallback to first
    assert _find_best_node(nodes, "Residual")["key"] == "k1"


def test_bidirectional_relationship_candidate_retrieval(
    real_repo: Neo4jRepository, db_session
) -> None:
    """Tests that querying (Dataset Y, Method X) resolves Method X -> Dataset Y."""
    project_id = uuid4()
    paper_id = uuid4()
    try:
        s_key = generate_entity_key(project_id, EntityType.METHOD, "Method Reverse")
        o_key = generate_entity_key(project_id, EntityType.DATASET, "Dataset Reverse")

        real_repo.upsert_nodes(
            project_id,
            [
                {"key": s_key, "name": "Method Reverse", "type": "Method"},
                {"key": o_key, "name": "Dataset Reverse", "type": "Dataset"},
            ],
        )

        real_repo.upsert_facts(
            project_id,
            paper_id,
            "gen_001",
            [
                {
                    "fact_id": f"fact_rev_{uuid4().hex[:8]}",
                    "subject_key": s_key,
                    "subject_name": "Method Reverse",
                    "subject_type": "Method",
                    "object_key": o_key,
                    "object_name": "Dataset Reverse",
                    "object_type": "Dataset",
                    "predicate": "EVALUATED_ON",
                    "qualifiers": {"dataset": "Dataset Reverse"},
                    "exact_quote": "Reverse quote.",
                    "page_number": 1,
                }
            ],
        )

        # Call with reversed order: Dataset first, then Method
        cands = build_relationship_candidates(
            db=db_session,
            repo=real_repo,
            project_id=project_id,
            entity_a_name="Dataset Reverse",
            entity_b_name="Method Reverse",
        )
        assert len(cands) == 1
        assert cands[0]["subject_name"] == "Method Reverse"
        assert cands[0]["object_name"] == "Dataset Reverse"

        # Call with empty entity names
        assert (
            build_relationship_candidates(db_session, real_repo, project_id, "", "Method Reverse")
            == []
        )
        assert (
            build_relationship_candidates(
                db_session, real_repo, project_id, "Method Reverse", "   "
            )
            == []
        )

    finally:
        real_repo.delete_project_graph(project_id)


def test_contradiction_qualifier_method_and_quote_fallbacks() -> None:
    """Test contradiction matching when method is in qualifiers, and polarity comes from quote."""
    from unittest.mock import MagicMock

    mock_repo = MagicMock(spec=Neo4jRepository)
    p_id = uuid4()
    p1 = uuid4()
    p2 = uuid4()

    mock_repo.get_project_facts.return_value = [
        {
            "fact_id": "f1",
            "paper_id": p1,
            "subject_key": "k_res1",
            "subject_name": "Result A",
            "object_key": "k_ds",
            "object_name": "ImageNet",
            "object_type": "Dataset",
            "predicate": "ACHIEVES_RESULT",
            "qualifiers": {
                "method": "Transformer",
                "metric": "ECE",
                "result_value": 2.1,
                "task": "calibration",
                "split": "test",
                "unit": "%",
                "comparison_condition": "standard evaluation",
            },
            "exact_quote": (
                "Transformer attains 2.1 ECE percent on ImageNet for calibration test split "
                "under standard evaluation."
            ),
        },
        {
            "fact_id": "f2",
            "paper_id": p2,
            "subject_key": "k_res2",
            "subject_name": "Result B",
            "object_key": "k_ds",
            "object_name": "ImageNet",
            "object_type": "Dataset",
            "predicate": "ACHIEVES_RESULT",
            "qualifiers": {
                "method": "Transformer",
                "metric": "ECE",
                "task": "calibration",
                "split": "test",
                "unit": "%",
                "comparison_condition": "standard evaluation",
            },
            "exact_quote": (
                "Transformer fails to converge for ECE percent on ImageNet during calibration "
                "test split under standard evaluation."
            ),
        },
    ]

    contras = build_contradiction_candidates(db=None, repo=mock_repo, project_id=p_id)
    assert len(contras) == 1
    assert contras[0]["conflict_type"] == "POLARITY"
    assert contras[0]["comparison_basis"] == {
        "method": "Transformer",
        "dataset": "ImageNet",
        "metric": "ECE",
        "task": "calibration",
        "split": "test",
        "unit": "%",
        "comparison_condition": "standard evaluation",
    }


def _make_comparison_fact(
    fact_id: str,
    paper_id: str,
    value: float,
    *,
    split: str | None = "test",
    unit: str | None = "%",
    task: str | None = "classification",
    comparison_condition: str | None = "standard setup",
    metric: str = "ECE",
    polarity: str = "POSITIVE",
) -> dict[str, Any]:
    quote = (
        f"Method X reports {value}{unit or ''} {metric} on ImageNet for {task or 'unknown task'} "
        f"{split or 'unknown split'} split under {comparison_condition or 'unknown conditions'}."
    )
    return {
        "fact_id": fact_id,
        "paper_id": paper_id,
        "subject_key": "method-x",
        "subject_name": "Method X",
        "subject_type": "Method",
        "object_key": "imagenet",
        "object_name": "ImageNet",
        "object_type": "Dataset",
        "predicate": "EVALUATED_ON",
        "qualifiers": {
            "metric": metric,
            "dataset": "ImageNet",
            "result_value": value,
            "polarity": polarity,
            **({"task": task} if task else {}),
            **({"split": split} if split else {}),
            **({"unit": unit} if unit else {}),
            **({"comparison_condition": comparison_condition} if comparison_condition else {}),
        },
        "exact_quote": quote,
        "page_number": 3,
    }


def test_comparisons_abstain_for_different_or_unknown_conditions() -> None:
    from unittest.mock import MagicMock

    from app.services.graphrag.query_engine import build_contradiction_candidates

    project_id = uuid4()
    base = _make_comparison_fact("a", str(uuid4()), 90)
    train_test = _make_comparison_fact("a", str(uuid4()), 90, metric="accuracy")
    cases = [
        (
            train_test,
            _make_comparison_fact("b", str(uuid4()), 80, split="train", metric="accuracy"),
        ),
        (base, _make_comparison_fact("b", str(uuid4()), 80, unit="count")),
        (
            base,
            _make_comparison_fact("b", str(uuid4()), 80, comparison_condition="augmented setup"),
        ),
        (base, _make_comparison_fact("b", str(uuid4()), 80, split=None)),
        (base, _make_comparison_fact("b", str(uuid4()), 80, unit=None)),
        (base, _make_comparison_fact("b", str(uuid4()), 80, task=None)),
        (base, _make_comparison_fact("b", str(uuid4()), 80, comparison_condition=None)),
    ]

    for first, other in cases:
        mock_repo = MagicMock(spec=Neo4jRepository)
        mock_repo.get_project_facts.return_value = [first, other]
        assert build_contradiction_candidates(None, mock_repo, project_id) == []


def test_matching_supported_comparison_preserves_both_sources_and_basis() -> None:
    from unittest.mock import MagicMock

    from app.services.graphrag.query_engine import build_contradiction_candidates

    paper_a, paper_b = str(uuid4()), str(uuid4())
    fact_a = _make_comparison_fact("fact-a", paper_a, 90)
    fact_b = _make_comparison_fact("fact-b", paper_b, 80)
    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.get_project_facts.return_value = [fact_a, fact_b]

    [candidate] = build_contradiction_candidates(None, mock_repo, uuid4())

    assert candidate["conflict_type"] == "NUMERIC_VALUE"
    assert candidate["fact_a"]["paper_id"] != candidate["fact_b"]["paper_id"]
    assert candidate["fact_a"]["exact_quote"] == fact_a["exact_quote"]
    assert candidate["fact_b"]["exact_quote"] == fact_b["exact_quote"]
    assert candidate["fact_a"]["page_number"] == candidate["fact_b"]["page_number"] == 3
    assert candidate["comparison_basis"] == {
        "method": "Method X",
        "dataset": "ImageNet",
        "metric": "ECE",
        "task": "classification",
        "split": "test",
        "unit": "%",
        "comparison_condition": "standard setup",
    }

    positive = _make_comparison_fact("polarity-a", paper_a, 90, polarity="POSITIVE")
    negative = _make_comparison_fact("polarity-b", paper_b, 90, polarity="NEGATIVE")
    negative["exact_quote"] = (
        "Method X fails to achieve 90% ECE on ImageNet for classification test split "
        "under standard setup."
    )
    mock_repo.get_project_facts.return_value = [positive, negative]
    [polarity_candidate] = build_contradiction_candidates(None, mock_repo, uuid4())
    assert polarity_candidate["conflict_type"] == "POLARITY"
    assert polarity_candidate["fact_a"]["fact_id"] == "polarity-a"
    assert polarity_candidate["fact_b"]["fact_id"] == "polarity-b"


def test_empty_facts_scenarios() -> None:
    """Test query engine functions when no facts exist in the project."""
    from unittest.mock import MagicMock

    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.get_project_facts.return_value = []
    p_id = uuid4()

    assert build_contradiction_candidates(db=None, repo=mock_repo, project_id=p_id) == []
    assert build_corpus_themes(db=None, repo=mock_repo, project_id=p_id) == []
