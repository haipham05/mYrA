"""Unit and integration tests for Neo4jRepository (Task 5.09).

Covers:
1. Unit Tests (Mocked driver):
   - from_settings factory behavior
   - Explicit database parameter and session management
   - Parameterization verification (Cypher queries are static and use parameters)
   - Pagination capping (limit capped to 100, min 1)
   - Direction validation
   - Rollback on exception in managed transaction

2. Integration Tests (explicit disposable test Neo4j target only):
   - ensure_schema constraint and index idempotency
   - Duplicate MERGE idempotency (nodes and facts)
   - Cross-project isolation (same name in Project A vs B creates distinct nodes)
   - One-hop neighbor traversal (OUTGOING, INCOMING, BOTH, predicate filtering)
   - get_fact_by_id and find_relationships_between
   - Generation retirement (older generations removed, active preserved)
   - Paper-level deletion and full project graph deletion
   - Isolated teardown
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import ClientError, ServiceUnavailable

from app.config import Settings
from app.schemas.graph import (
    EntityType,
    GraphEntitySchema,
    GraphFactCandidate,
    GraphProvenanceSchema,
    GraphQualifierSchema,
    RelationshipPredicate,
)
from app.services.graphrag.identity import generate_entity_key
from app.services.graphrag.neo4j_repository import (
    MAX_PAGE_LIMIT,
    Neo4jRepository,
)

# =============================================================================
# Unit Tests (Mocked Driver)
# =============================================================================


def test_from_settings_disabled() -> None:
    """Repository factory returns None when graphrag is disabled or URI is unset."""
    settings = Settings(graphrag_enabled=False, neo4j_uri="bolt://localhost:7687")
    assert Neo4jRepository.from_settings(settings) is None

    settings_no_uri = Settings(graphrag_enabled=True, neo4j_uri=None)
    assert Neo4jRepository.from_settings(settings_no_uri) is None


def test_from_settings_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repository factory creates instance with configured database and credentials."""
    mock_driver = MagicMock(spec=Driver)
    mock_graph_database = MagicMock()
    mock_graph_database.driver.return_value = mock_driver

    monkeypatch.setattr(
        "app.services.graphrag.neo4j_repository.GraphDatabase",
        mock_graph_database,
    )

    settings = Settings(
        graphrag_enabled=True,
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        neo4j_database="custom_db",
        neo4j_timeout_seconds=15.0,
    )
    repo = Neo4jRepository.from_settings(settings)
    assert repo is not None
    assert repo.database == "custom_db"
    assert repo.driver == mock_driver

    mock_graph_database.driver.assert_called_once_with(
        "bolt://localhost:7687",
        auth=("neo4j", "password"),
        connection_timeout=15.0,
    )


def test_session_pinned_to_explicit_database() -> None:
    """Verifies that sessions are explicitly pinned to the configured database."""
    mock_driver = MagicMock(spec=Driver)
    repo = Neo4jRepository(driver=mock_driver, database="analytics_graph")

    _ = repo._get_session()
    mock_driver.session.assert_called_once_with(database="analytics_graph")


def test_parameterization_queries_are_static() -> None:
    """Verify that all Cypher statements executed by the repository are static

    parameterized queries and never dynamically interpolate project_id or user input.
    """
    mock_driver = MagicMock(spec=Driver)
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    # Capture the transaction work callable passed to execute_read / execute_write
    def fake_execute(work_callable):
        return work_callable(mock_tx)

    mock_session.execute_read.side_effect = fake_execute
    mock_session.execute_write.side_effect = fake_execute

    repo = Neo4jRepository(driver=mock_driver, database="neo4j")
    project_id = uuid4()
    paper_id = uuid4()

    # 1. search_nodes
    mock_res = MagicMock()
    mock_res.__iter__.return_value = []
    mock_res.single.return_value = None
    mock_tx.run.return_value = mock_res

    repo.search_nodes(project_id=project_id, query="transformer", entity_type="Model", limit=20)
    search_call = mock_tx.run.call_args
    cypher_query, params = search_call[0][0], search_call[0][1]
    assert "$project_id" in cypher_query
    assert "$query" in cypher_query
    assert "$entity_type" in cypher_query
    assert str(project_id) not in cypher_query  # Must NOT be in the Cypher string itself
    assert params["project_id"] == str(project_id)
    assert params["query"] == "transformer"
    assert params["entity_type"] == "Model"

    scoped_paper_id = uuid4()
    repo.search_nodes(project_id, limit=10, paper_ids={scoped_paper_id})
    scoped_search_query, scoped_search_params = mock_tx.run.call_args[0]
    assert "f.paper_id IN $paper_ids" in scoped_search_query
    assert scoped_search_query.index("f.paper_id IN $paper_ids") < scoped_search_query.index(
        "SKIP $skip"
    )
    assert scoped_search_params["paper_ids"] == [str(scoped_paper_id)]

    # The pagination total uses the exact same project/query/type predicates.
    mock_res.single.return_value = {"total": 11}
    assert (
        repo.count_matching_nodes(
            project_id=project_id,
            query="transformer",
            entity_type="Model",
        )
        == 11
    )
    count_call = mock_tx.run.call_args
    count_query, count_params = count_call[0][0], count_call[0][1]
    assert "$project_id" in count_query
    assert "$query" in count_query
    assert "$entity_type" in count_query
    assert count_params == {
        "project_id": params["project_id"],
        "query": params["query"],
        "entity_type": params["entity_type"],
    }

    # 2. get_node_by_key
    mock_res.single.return_value = None
    repo.get_node_by_key(project_id=project_id, key="model_bert")
    key_call = mock_tx.run.call_args
    assert "$key" in key_call[0][0]
    assert "model_bert" not in key_call[0][0]
    assert key_call[0][1]["key"] == "model_bert"

    # 3. get_node_neighbors
    mock_res.__iter__.return_value = []
    repo.get_node_neighbors(project_id=project_id, key="model_bert", direction="OUTGOING")
    neighbor_call = mock_tx.run.call_args
    assert "$key" in neighbor_call[0][0]
    assert "$project_id" in neighbor_call[0][0]
    assert neighbor_call[0][1]["key"] == "model_bert"

    repo.get_node_neighbors(project_id=project_id, key="model_bert", paper_ids={scoped_paper_id})
    scoped_neighbor_query, scoped_neighbor_params = mock_tx.run.call_args[0]
    assert "f.paper_id IN $paper_ids" in scoped_neighbor_query
    assert scoped_neighbor_query.index("f.paper_id IN $paper_ids") < scoped_neighbor_query.index(
        "LIMIT $limit"
    )
    assert scoped_neighbor_params["paper_ids"] == [str(scoped_paper_id)]

    repo.find_relationships_between(project_id, "subject", "object", paper_ids={scoped_paper_id})
    scoped_relationship_query, scoped_relationship_params = mock_tx.run.call_args[0]
    assert "f.paper_id IN $paper_ids" in scoped_relationship_query
    assert scoped_relationship_params["paper_ids"] == [str(scoped_paper_id)]

    repo.get_project_facts(project_id, limit=10, paper_ids={scoped_paper_id})
    scoped_facts_query, scoped_facts_params = mock_tx.run.call_args[0]
    assert "f.paper_id IN $paper_ids" in scoped_facts_query
    assert (
        scoped_facts_query.index("f.paper_id IN $paper_ids")
        < scoped_facts_query.index("SKIP $skip")
        < scoped_facts_query.index("LIMIT $limit")
    )
    assert scoped_facts_params["paper_ids"] == [str(scoped_paper_id)]

    # 4. retire_older_generations
    mock_res.single.return_value = {"cnt": 0}
    repo.retire_older_generations(
        project_id=project_id,
        paper_id=paper_id,
        active_generation_id="gen_v2",
    )
    retire_call = mock_tx.run.call_args
    assert "$active_generation_id" in retire_call[0][0]
    assert "gen_v2" not in retire_call[0][0]
    assert retire_call[0][1]["active_generation_id"] == "gen_v2"


def test_pagination_capping() -> None:
    """Verify pagination limit is strictly capped to MAX_PAGE_LIMIT (100) and at least 1."""
    mock_driver = MagicMock(spec=Driver)
    mock_session = MagicMock()
    mock_tx = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    def fake_execute(work_callable):
        return work_callable(mock_tx)

    mock_session.execute_read.side_effect = fake_execute
    mock_tx.run.return_value = []

    repo = Neo4jRepository(driver=mock_driver, database="neo4j")
    project_id = uuid4()

    # Requesting 1000 items is capped to 100
    repo.search_nodes(project_id=project_id, limit=1000)
    params = mock_tx.run.call_args[0][1]
    assert params["limit"] == MAX_PAGE_LIMIT

    # Requesting negative or 0 limit is clamped to 1
    repo.search_nodes(project_id=project_id, limit=-10)
    params = mock_tx.run.call_args[0][1]
    assert params["limit"] == 1

    # Neighbors limit capped as well
    repo.get_node_neighbors(project_id=project_id, key="test_key", limit=5000)
    params = mock_tx.run.call_args[0][1]
    assert params["limit"] == MAX_PAGE_LIMIT


def test_invalid_direction_raises() -> None:
    """Invalid direction in get_node_neighbors raises ValueError."""
    mock_driver = MagicMock(spec=Driver)
    repo = Neo4jRepository(driver=mock_driver, database="neo4j")
    with pytest.raises(ValueError, match="Invalid direction"):
        repo.get_node_neighbors(project_id=uuid4(), key="test", direction="SIDEWAYS")


def test_rollback_on_write_failure() -> None:
    """Simulates a write failure inside a managed transaction and verifies error propagation."""
    mock_driver = MagicMock(spec=Driver)
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session

    def simulate_failure(work_callable):
        mock_tx = MagicMock()
        mock_tx.run.side_effect = ClientError(
            "Neo.ClientError.Statement.SyntaxError", "Syntax error"
        )
        return work_callable(mock_tx)

    mock_session.execute_write.side_effect = simulate_failure

    repo = Neo4jRepository(driver=mock_driver, database="neo4j")
    with pytest.raises(ClientError):
        repo.upsert_nodes(project_id=uuid4(), nodes=[{"name": "test", "type": "Method"}])


def test_connectivity_and_context_manager() -> None:
    """Verify connectivity check, close, and context manager semantics."""
    mock_driver = MagicMock(spec=Driver)
    mock_driver.verify_connectivity.return_value = None

    repo = Neo4jRepository(driver=mock_driver, database="test_db")
    assert repo.verify_connectivity() is True
    mock_driver.verify_connectivity.assert_called_once()

    # When driver raises, verify_connectivity returns False
    mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Cannot connect")
    assert repo.verify_connectivity() is False

    # Context manager calls close
    with repo:
        pass
    mock_driver.close.assert_called_once()


def test_empty_inputs_short_circuit() -> None:
    """Verify empty or whitespace-only inputs safely short circuit without db calls."""
    mock_driver = MagicMock(spec=Driver)
    repo = Neo4jRepository(driver=mock_driver, database="neo4j")
    project_id = uuid4()
    paper_id = uuid4()

    assert repo.upsert_nodes(project_id, []) == 0
    assert repo.upsert_nodes(project_id, [{"name": "   "}]) == 0
    assert repo.upsert_facts(project_id, paper_id, "gen_1", []) == 0
    assert repo.upsert_facts(project_id, paper_id, "gen_1", [{}]) == 0
    assert repo.get_node_by_key(project_id, "   ") is None
    assert repo.get_node_neighbors(project_id, "   ") == []
    assert repo.get_fact_by_id(project_id, "   ") is None
    assert repo.find_relationships_between(project_id, "", "key") == []
    assert repo.find_relationships_between(project_id, "key", "") == []
    mock_driver.session.assert_not_called()


# =============================================================================
# Integration Tests (Local Neo4j)
# =============================================================================


@pytest.fixture
def real_repo(disposable_neo4j_uri: str) -> Generator[Neo4jRepository, None, None]:
    """Connect only to the explicitly acknowledged disposable test target."""
    driver = GraphDatabase.driver(disposable_neo4j_uri, auth=None)
    driver.verify_connectivity()
    repo = Neo4jRepository(driver=driver, database="neo4j")
    repo.ensure_schema()
    yield repo
    driver.close()


def test_ensure_schema_idempotency(real_repo: Neo4jRepository) -> None:
    """ensure_schema creates constraints and indexes and is safely repeatable."""
    real_repo.ensure_schema()
    real_repo.ensure_schema()

    with real_repo._get_session() as session:
        # Check constraints exist
        constraints = session.run("SHOW CONSTRAINTS").data()
        constraint_names = {c.get("name") for c in constraints}
        assert "node_key_unique" in constraint_names
        assert "fact_id_unique" in constraint_names

        # Check indexes exist
        indexes = session.run("SHOW INDEXES").data()
        index_names = {idx.get("name") for idx in indexes}
        assert "node_project_id" in index_names
        assert "fact_project_id" in index_names


def test_duplicate_merge_idempotency(real_repo: Neo4jRepository) -> None:
    """Repeatedly upserting identical nodes and facts yields the exact same counts."""
    project_id = uuid4()
    paper_id = uuid4()
    gen_id = "gen_001"

    node_1 = GraphEntitySchema(
        id=generate_entity_key(project_id, EntityType.METHOD, "Attention Mechanism"),
        name="Attention Mechanism",
        type=EntityType.METHOD,
        description="Core seq2seq component",
    )
    node_2 = GraphEntitySchema(
        id=generate_entity_key(project_id, EntityType.DATASET, "WMT14 En-De"),
        name="WMT14 En-De",
        type=EntityType.DATASET,
        description="Bilingual benchmark",
    )

    prov = GraphProvenanceSchema(
        paper_id=paper_id,
        chunk_id=uuid4(),
        page_number=3,
        element_id=uuid4(),
        exact_quote="We evaluate on WMT14 En-De benchmark.",
        char_start=0,
        char_end=37,
        document_sha256="a" * 64,
        parser_version="v1",
    )
    fact_1 = GraphFactCandidate(
        subject=node_1,
        predicate=RelationshipPredicate.EVALUATED_ON,
        object=node_2,
        qualifiers=GraphQualifierSchema(metric="BLEU", result_value=28.4),
        provenance=prov,
    )

    try:
        # First write
        cnt_nodes_1 = real_repo.upsert_nodes(project_id, [node_1, node_2])
        assert cnt_nodes_1 == 2
        cnt_facts_1 = real_repo.upsert_facts(project_id, paper_id, gen_id, [fact_1])
        assert cnt_facts_1 == 1

        counts_1 = real_repo.count_project_elements(project_id)
        assert counts_1["nodes"] == 2
        assert counts_1["facts"] == 1

        # Second identical write
        cnt_nodes_2 = real_repo.upsert_nodes(project_id, [node_1, node_2])
        assert cnt_nodes_2 == 2
        cnt_facts_2 = real_repo.upsert_facts(project_id, paper_id, gen_id, [fact_1])
        assert cnt_facts_2 == 1

        # Final counts must remain unchanged (zero duplicates created)
        counts_2 = real_repo.count_project_elements(project_id)
        assert counts_2["nodes"] == 2
        assert counts_2["facts"] == 1

    finally:
        real_repo.delete_project_graph(project_id)


def test_cross_project_isolation(real_repo: Neo4jRepository) -> None:
    """Two projects with identical entity name produce separate nodes;

    searching in Project A never leaks Project B's node.
    """
    proj_a = uuid4()
    proj_b = uuid4()

    key_a = generate_entity_key(proj_a, EntityType.TASK, "ASR")
    key_b = generate_entity_key(proj_b, EntityType.TASK, "ASR")
    assert key_a != key_b, "Project-scoped keys for identical name must differ"

    node_a = GraphEntitySchema(id=key_a, name="ASR", type=EntityType.TASK)
    node_b = GraphEntitySchema(id=key_b, name="ASR", type=EntityType.TASK)

    try:
        real_repo.upsert_nodes(proj_a, [node_a])
        real_repo.upsert_nodes(proj_b, [node_b])

        # Search Project A
        res_a = real_repo.search_nodes(proj_a, query="ASR")
        assert len(res_a) == 1
        assert res_a[0]["key"] == key_a
        assert res_a[0]["project_id"] == str(proj_a)

        # Search Project B
        res_b = real_repo.search_nodes(proj_b, query="ASR")
        assert len(res_b) == 1
        assert res_b[0]["key"] == key_b
        assert res_b[0]["project_id"] == str(proj_b)

        # Cross lookup by key must return None
        assert real_repo.get_node_by_key(proj_a, key_b) is None
        assert real_repo.get_node_by_key(proj_b, key_a) is None

    finally:
        real_repo.delete_project_graph(proj_a)
        real_repo.delete_project_graph(proj_b)


def test_neighbor_traversal_and_filtering(real_repo: Neo4jRepository) -> None:
    """Verify one-hop neighbor traversal with OUTGOING, INCOMING, BOTH and predicate filter."""
    project_id = uuid4()
    paper_id = uuid4()
    gen_id = "gen_001"

    node_s = GraphEntitySchema(
        id=generate_entity_key(project_id, EntityType.MODEL, "Alpaca"),
        name="Alpaca",
        type=EntityType.MODEL,
    )
    node_o = GraphEntitySchema(
        id=generate_entity_key(project_id, EntityType.MODEL, "LLaMA"),
        name="LLaMA",
        type=EntityType.MODEL,
    )
    prov = GraphProvenanceSchema(
        paper_id=paper_id,
        chunk_id=uuid4(),
        page_number=1,
        element_id=uuid4(),
        exact_quote="We fine-tune Alpaca from LLaMA.",
        char_start=0,
        char_end=31,
        document_sha256="b" * 64,
        parser_version="v1",
    )
    fact = GraphFactCandidate(
        subject=node_s,
        predicate=RelationshipPredicate.EXTENDS,
        object=node_o,
        qualifiers=GraphQualifierSchema(task="Instruction-tuning"),
        provenance=prov,
    )

    try:
        real_repo.upsert_nodes(project_id, [node_s, node_o])
        real_repo.upsert_facts(project_id, paper_id, gen_id, [fact])

        # OUTGOING from Alpaca -> LLaMA
        out_neighbors = real_repo.get_node_neighbors(
            project_id, key=node_s.id, direction="OUTGOING"
        )
        assert len(out_neighbors) == 1
        assert out_neighbors[0]["neighbor_key"] == node_o.id
        assert out_neighbors[0]["neighbor_name"] == "LLaMA"
        assert out_neighbors[0]["direction"] == "OUTGOING"
        assert out_neighbors[0]["predicate"] == "EXTENDS"
        assert out_neighbors[0]["qualifiers"] == {
            "task": "Instruction-tuning",
            "polarity": "POSITIVE",
        }

        # INCOMING from Alpaca -> none
        inc_s = real_repo.get_node_neighbors(project_id, key=node_s.id, direction="INCOMING")
        assert len(inc_s) == 0

        # INCOMING to LLaMA -> Alpaca
        inc_o = real_repo.get_node_neighbors(project_id, key=node_o.id, direction="INCOMING")
        assert len(inc_o) == 1
        assert inc_o[0]["neighbor_key"] == node_s.id
        assert inc_o[0]["direction"] == "INCOMING"

        # BOTH from Alpaca
        both_s = real_repo.get_node_neighbors(project_id, key=node_s.id, direction="BOTH")
        assert len(both_s) == 1
        assert both_s[0]["neighbor_key"] == node_o.id

        # Predicate filtering
        pred_match = real_repo.get_node_neighbors(
            project_id, key=node_s.id, predicate=RelationshipPredicate.EXTENDS
        )
        assert len(pred_match) == 1

        pred_mismatch = real_repo.get_node_neighbors(
            project_id, key=node_s.id, predicate=RelationshipPredicate.EVALUATED_ON
        )
        assert len(pred_mismatch) == 0

        # find_relationships_between
        rels = real_repo.find_relationships_between(
            project_id, subject_key=node_s.id, object_key=node_o.id
        )
        assert len(rels) == 1
        assert rels[0]["predicate"] == "EXTENDS"
        assert rels[0]["exact_quote"] == "We fine-tune Alpaca from LLaMA."

        # get_fact_by_id
        fact_id = rels[0]["fact_id"]
        fact_data = real_repo.get_fact_by_id(project_id, fact_id)
        assert fact_data is not None
        assert fact_data["id"] == fact_id
        assert fact_data["subject_key"] == node_s.id
        assert fact_data["object_key"] == node_o.id

    finally:
        real_repo.delete_project_graph(project_id)


def test_generation_retirement_and_paper_deletion(real_repo: Neo4jRepository) -> None:
    """Verify retiring older generations detaches old facts while keeping active ones."""
    project_id = uuid4()
    paper_id = uuid4()

    node_1 = GraphEntitySchema(
        id=generate_entity_key(project_id, EntityType.METHOD, "Method A"),
        name="Method A",
        type=EntityType.METHOD,
    )
    node_2 = GraphEntitySchema(
        id=generate_entity_key(project_id, EntityType.DATASET, "Dataset D"),
        name="Dataset D",
        type=EntityType.DATASET,
    )
    real_repo.upsert_nodes(project_id, [node_1, node_2])

    fact_old = {
        "fact_id": f"fact_old_{uuid4().hex[:12]}",
        "subject_key": node_1.id,
        "object_key": node_2.id,
        "predicate": "EVALUATED_ON",
        "char_start": 0,
        "char_end": 10,
        "exact_quote": "old quote",
    }
    fact_new = {
        "fact_id": f"fact_new_{uuid4().hex[:12]}",
        "subject_key": node_1.id,
        "object_key": node_2.id,
        "predicate": "EVALUATED_ON",
        "char_start": 0,
        "char_end": 10,
        "exact_quote": "new quote",
    }

    try:
        # Upsert generation 1 and generation 2
        real_repo.upsert_facts(project_id, paper_id, "gen_1", [fact_old])
        real_repo.upsert_facts(project_id, paper_id, "gen_2", [fact_new])

        counts_before = real_repo.count_project_elements(project_id)
        assert counts_before["facts"] == 2

        # Retire older generation, keeping gen_2
        deleted = real_repo.retire_older_generations(
            project_id=project_id,
            paper_id=paper_id,
            active_generation_id="gen_2",
        )
        assert deleted == 1

        counts_after = real_repo.count_project_elements(project_id)
        assert counts_after["facts"] == 1

        # Old fact should be gone, new fact still exists
        assert real_repo.get_fact_by_id(project_id, fact_old["fact_id"]) is None
        assert real_repo.get_fact_by_id(project_id, fact_new["fact_id"]) is not None

        # Delete paper facts
        deleted_paper = real_repo.delete_paper_facts(project_id, paper_id)
        assert deleted_paper == 1
        assert real_repo.count_project_elements(project_id)["facts"] == 0

    finally:
        real_repo.delete_project_graph(project_id)


def test_delete_project_graph_cleanup(real_repo: Neo4jRepository) -> None:
    """delete_project_graph removes all facts and nodes belonging to that project."""
    project_id = uuid4()
    node = GraphEntitySchema(
        id=generate_entity_key(project_id, EntityType.CONCEPT, "Concept X"),
        name="Concept X",
        type=EntityType.CONCEPT,
    )
    real_repo.upsert_nodes(project_id, [node])
    counts_before = real_repo.count_project_elements(project_id)
    assert counts_before["nodes"] == 1

    res = real_repo.delete_project_graph(project_id)
    assert res["nodes_deleted"] == 1
    assert res["facts_deleted"] == 0

    counts_after = real_repo.count_project_elements(project_id)
    assert counts_after["nodes"] == 0
    assert counts_after["facts"] == 0


def test_upsert_from_dict_and_qualifiers_parsing(real_repo: Neo4jRepository) -> None:
    """Verify upserting nodes and facts using dict representations with nested and flat forms."""
    project_id = uuid4()
    paper_id = uuid4()
    gen_id = "gen_dict_01"

    # Upsert nodes via dict with external_id and no key
    nodes = [
        {
            "name": "ResNet",
            "type": "Model",
            "description": "Residual network",
            "external_id": "arxiv:1512.03385",
        },
        {
            "name": "ImageNet",
            "type": "Dataset",
            "description": "Visual recognition benchmark",
        },
    ]
    upserted = real_repo.upsert_nodes(project_id, nodes)
    assert upserted == 2

    # Upsert facts via dict with nested subject/object
    facts = [
        {
            "subject": {
                "name": "ResNet",
                "type": "Model",
                "external_id": "arxiv:1512.03385",
            },
            "object": {"name": "ImageNet", "type": "Dataset"},
            "predicate": "EVALUATED_ON",
            "qualifiers": {"metric": "Top-1 Accuracy", "result_value": 75.3},
            "char_start": 5,
            "char_end": 45,
            "page_number": 2,
            "exact_quote": "ResNet achieves 75.3% top-1 accuracy on ImageNet.",
        }
    ]
    fact_upserted = real_repo.upsert_facts(project_id, paper_id, gen_id, facts)
    assert fact_upserted == 1

    try:
        resnet = real_repo.search_nodes(project_id, query="ResNet")
        assert len(resnet) == 1
        assert "arxiv:1512.03385" in resnet[0]["key"]

        neighbors = real_repo.get_node_neighbors(project_id, key=resnet[0]["key"])
        assert len(neighbors) == 1
        assert neighbors[0]["neighbor_name"] == "ImageNet"
        assert neighbors[0]["qualifiers"]["metric"] == "Top-1 Accuracy"

    finally:
        real_repo.delete_project_graph(project_id)
