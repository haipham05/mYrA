"""Tests for project-scoped GraphRAG API endpoints (Task 5.29).

Verifies:
1. Status endpoint: Returns valid counts, booleans, zero credentials, zero raw Cypher.
2. Node search & pagination: Respects limit and skip, caps limit at 100, filters by type.
3. Node detail & neighbors: Returns 1-hop neighbors; 404 for nonexistent or foreign project node.
4. Fact detail with live citation: Returns fact with verified CitationAnchor and DOM range offsets;
   404 for foreign project fact.
5. Project isolation (Project A vs Project B): Project A cannot access Project B nodes or facts.
6. Relationships between endpoints: Finds relationships and constructs Citation-compatible response.
7. Index endpoint: dry_run defaults to True; explicit opt-in enqueues papers.
8. Graph unavailable: When Neo4j is offline, node/neighbor routes return 503 cleanly;
   status route returns 200 with neo4j_available: false.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool
from tests.fixtures.graphrag.corpus_fixtures import (
    PAPER_A1_ID,
    PROJECT_A_ID,
    PROJECT_B_ID,
    ManifestRelationship,
    load_manifest,
    seed_synthetic_corpus,
    validate_manifest,
)

from app.api.v1.graph import get_graph_repo
from app.db.base import Base
from app.db.models import GraphEvent, GraphFactSnapshot, Paper
from app.db.session import get_db
from app.main import app
from app.schemas.evidence import AnchorStatus
from app.services.graphrag.neo4j_repository import Neo4jRepository


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Provide an isolated, in-memory SQLite database populated with the synthetic corpus."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = session_factory()
    try:
        seed_synthetic_corpus(session)
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def mock_repo() -> MagicMock:
    """Mock Neo4jRepository with default connected state."""
    repo = MagicMock(spec=Neo4jRepository)
    repo.verify_connectivity.return_value = True
    repo.count_project_elements.return_value = {
        "nodes": 10,
        "facts": 25,
        "node_count": 10,
        "fact_count": 25,
    }
    repo.count_matching_nodes.return_value = 0
    repo.get_node_by_key.return_value = None
    repo.get_fact_by_id.return_value = None
    repo.search_nodes.return_value = []
    repo.get_node_neighbors.return_value = []
    repo.find_relationships_between.return_value = []
    return repo


@pytest.fixture
def client(db_session: Session, mock_repo: MagicMock) -> Generator[TestClient, None, None]:
    """Test client with overridden DB session and Neo4jRepository."""

    def override_get_db() -> Generator[Session, None, None]:
        yield db_session

    def override_get_repo() -> Neo4jRepository | None:
        return mock_repo

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_graph_repo] = override_get_repo

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def _create_snapshot(
    db: Session,
    project_id: UUID,
    rel: ManifestRelationship,
    document_sha256: str,
    fact_id: str | None = None,
    generation_id: str = "gen-test-01",
) -> GraphFactSnapshot:
    """Helper to persist a GraphFactSnapshot from a manifest relationship."""
    prov = rel.provenance
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
        chunk_id=prov.chunk_id,
        page_number=prov.page_number,
        element_id=prov.element_id,
        char_start=prov.char_start,
        char_end=prov.char_end,
        exact_quote=prov.exact_quote,
        document_sha256=document_sha256,
        validation_version="1.0.0",
    )
    db.add(snapshot)
    db.commit()
    db.refresh(snapshot)
    return snapshot


# ============================================================================
# Test 1: Project status endpoint returns valid counts, booleans, 0 credentials
# ============================================================================


def test_status_endpoint_returns_valid_metrics_and_zero_credentials(
    client: TestClient,
    db_session: Session,
    mock_repo: MagicMock,
) -> None:
    """Test 1: Status endpoint returns correct event and element counts with no leaked secrets."""
    # Seed events for PROJECT_A_ID
    db_session.add_all(
        [
            GraphEvent(
                project_id=PROJECT_A_ID,
                paper_id=PAPER_A1_ID,
                generation_id="gen-1",
                action="UPSERT",
                status="PENDING",
            ),
            GraphEvent(
                project_id=PROJECT_A_ID,
                paper_id=PAPER_A1_ID,
                generation_id="gen-2",
                action="UPSERT",
                status="PROCESSING",
            ),
            GraphEvent(
                project_id=PROJECT_A_ID,
                paper_id=PAPER_A1_ID,
                generation_id="gen-3",
                action="UPSERT",
                status="COMPLETED",
            ),
            GraphEvent(
                project_id=PROJECT_A_ID,
                paper_id=PAPER_A1_ID,
                generation_id="gen-4",
                action="UPSERT",
                status="COMPLETED",
            ),
            GraphEvent(
                project_id=PROJECT_A_ID,
                paper_id=PAPER_A1_ID,
                generation_id="gen-5",
                action="UPSERT",
                status="FAILED",
            ),
        ]
    )
    db_session.commit()

    mock_repo.count_project_elements.return_value = {
        "nodes": 12,
        "facts": 34,
        "node_count": 12,
        "fact_count": 34,
    }

    resp = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/status")
    assert resp.status_code == 200
    data = resp.json()

    assert data["project_id"] == str(PROJECT_A_ID)
    assert data["neo4j_available"] is True
    assert data["node_count"] == 12
    assert data["fact_count"] == 34
    assert data["pending_events_count"] == 2  # PENDING + PROCESSING
    assert data["completed_events_count"] == 2
    assert data["failed_events_count"] == 1

    # Security check: Zero credentials or raw Cypher in payload
    raw_text = resp.text.lower()
    for sensitive_token in ("password", "neo4j_password", "bearer", "secret", "bolt://", "match ("):
        assert sensitive_token not in raw_text


# ============================================================================
# Test 2: Node search and pagination
# ============================================================================


def test_node_search_and_pagination(
    client: TestClient,
    mock_repo: MagicMock,
) -> None:
    """Test 2: Node search respects query/type filters, pagination, and caps limit at 100."""
    fake_nodes = [
        {
            "key": "model:bert",
            "name": "BERT",
            "type": "Model",
            "description": "Pretrained language model",
            "aliases": ["BERT-Base"],
            "project_id": str(PROJECT_A_ID),
            "updated_at": "2026-09-01T12:00:00Z",
        },
        {
            "key": "model:roberta",
            "name": "RoBERTa",
            "type": "Model",
            "description": "Robustly optimized BERT",
            "aliases": [],
            "project_id": str(PROJECT_A_ID),
            "updated_at": "2026-09-01T12:00:00Z",
        },
    ]
    mock_repo.search_nodes.return_value = fake_nodes
    mock_repo.count_matching_nodes.return_value = 2

    # Valid search with query, entity_type, limit, skip
    resp = client.get(
        f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes?query=bert&entity_type=Model&limit=10&skip=2"
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["total"] == 2
    assert data["limit"] == 10
    assert data["skip"] == 2
    assert len(data["items"]) == 2
    assert data["items"][0]["key"] == "model:bert"
    assert data["items"][0]["project_id"] == str(PROJECT_A_ID)

    mock_repo.search_nodes.assert_called_once_with(
        project_id=PROJECT_A_ID,
        query="bert",
        entity_type="Model",
        limit=10,
        skip=2,
    )
    mock_repo.count_matching_nodes.assert_called_once_with(
        project_id=PROJECT_A_ID,
        query="bert",
        entity_type="Model",
    )

    # Limit capped at 100 via schema validation (Query(le=100))
    resp_over_limit = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes?limit=150")
    assert resp_over_limit.status_code == 422


def test_filtered_node_pagination_uses_matching_total(client: TestClient, mock_repo: MagicMock):
    matching_nodes = [
        {
            "key": f"model:bert-{i}",
            "name": f"BERT {i}",
            "type": "Model",
            "project_id": str(PROJECT_A_ID),
        }
        for i in range(10)
    ]
    last_node = {
        "key": "model:bert-last",
        "name": "BERT Last",
        "type": "Model",
        "project_id": str(PROJECT_A_ID),
    }
    mock_repo.count_matching_nodes.return_value = 11
    mock_repo.search_nodes.side_effect = [matching_nodes, [last_node]]

    first = client.get(
        f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes?query=bert&entity_type=Model&limit=10"
    ).json()
    second = client.get(
        f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes?query=bert&entity_type=Model&limit=10&skip=10"
    ).json()

    assert len(first["items"]) == 10
    assert first["total"] == 11
    assert len(second["items"]) == 1
    assert second["total"] == 11
    assert mock_repo.search_nodes.call_args_list[0].kwargs["project_id"] == PROJECT_A_ID
    assert mock_repo.search_nodes.call_args_list[1].kwargs["project_id"] == PROJECT_A_ID


# ============================================================================
# Test 3: Node detail and neighbors
# ============================================================================


def test_node_detail_and_neighbors(
    client: TestClient,
    mock_repo: MagicMock,
) -> None:
    """Test 3: Returns 1-hop neighbors; 404 for nonexistent node; 400 for invalid direction."""
    node_data = {
        "key": "model:conformer",
        "name": "Conformer",
        "type": "Model",
        "description": "Convolution-augmented Transformer",
        "aliases": ["Conf"],
        "project_id": str(PROJECT_A_ID),
        "updated_at": "2026-09-01T12:00:00Z",
    }

    def mock_get_node(pid: UUID, key: str):
        if pid == PROJECT_A_ID and key == "model:conformer":
            return node_data
        return None

    mock_repo.get_node_by_key.side_effect = mock_get_node

    # 1. Existing node detail
    resp = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes/model:conformer")
    assert resp.status_code == 200
    assert resp.json()["key"] == "model:conformer"
    assert resp.json()["name"] == "Conformer"

    # 2. Nonexistent node detail returns 404
    resp_missing = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes/nonexistent:node")
    assert resp_missing.status_code == 404
    assert resp_missing.json()["detail"] == "Node not found in this project"

    # 3. Neighbors of existing node
    mock_repo.get_node_neighbors.return_value = [
        {
            "neighbor_key": "dataset:libri",
            "neighbor_name": "LibriSpeech",
            "neighbor_type": "Dataset",
            "direction": "OUTGOING",
            "predicate": "EVALUATED_ON",
            "fact_id": "fact-libri-01",
            "qualifiers": {"split": "test-clean"},
        }
    ]

    resp_nbr = client.get(
        f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes/model:conformer/neighbors?direction=OUTGOING"
    )
    assert resp_nbr.status_code == 200
    nbr_data = resp_nbr.json()
    assert nbr_data["node_key"] == "model:conformer"
    assert nbr_data["total"] == 1
    assert nbr_data["neighbors"][0]["neighbor_key"] == "dataset:libri"
    assert nbr_data["neighbors"][0]["predicate"] == "EVALUATED_ON"

    # 4. Neighbors of nonexistent node returns 404
    resp_nbr_missing = client.get(
        f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes/nonexistent:node/neighbors"
    )
    assert resp_nbr_missing.status_code == 404

    # 5. Invalid neighbor direction returns 400
    resp_nbr_bad_dir = client.get(
        f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes/model:conformer/neighbors?direction=UPWARD"
    )
    assert resp_nbr_bad_dir.status_code == 400


# ============================================================================
# Test 4: Fact detail with live citation anchor
# ============================================================================


def test_fact_detail_with_live_citation(
    client: TestClient,
    db_session: Session,
) -> None:
    """Test 4: Returns fact with verified CitationAnchor and DOM range-compatible offsets;
    404 for foreign project.
    """
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]

    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).first()
    assert paper is not None

    snapshot = _create_snapshot(
        db=db_session,
        project_id=PROJECT_A_ID,
        rel=rel,
        document_sha256=paper.document_sha256,
        fact_id="fact-grounded-001",
    )

    # 1. Fetch fact detail in Project A -> verified citation anchor
    resp = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/facts/{snapshot.fact_id}")
    assert resp.status_code == 200
    fact_resp = resp.json()

    assert fact_resp["id"] == snapshot.fact_id
    assert fact_resp["project_id"] == str(PROJECT_A_ID)
    assert fact_resp["paper_id"] == str(paper.id)
    assert fact_resp["anchor_status"] == AnchorStatus.VERIFIED.value
    assert fact_resp["citation"] is not None

    citation = fact_resp["citation"]
    assert citation["citation_index"] == 1
    assert citation["evidence_id"] == "G1"
    assert citation["quote"] == snapshot.exact_quote
    assert citation["anchor_status"] == AnchorStatus.VERIFIED.value
    assert len(citation["anchors"]) == 1

    anchor = citation["anchors"][0]
    assert anchor["page_number"] == snapshot.page_number
    assert anchor["exact_quote"] == snapshot.exact_quote
    assert anchor["source_char_start"] == snapshot.char_start
    assert anchor["source_char_end"] == snapshot.char_end
    assert anchor["anchor_status"] == AnchorStatus.VERIFIED.value

    # 2. Accessing Project A's fact from Project B -> 404
    resp_foreign = client.get(f"/api/v1/projects/{PROJECT_B_ID}/graph/facts/{snapshot.fact_id}")
    assert resp_foreign.status_code == 404
    assert resp_foreign.json()["detail"] == "Fact not found in this project"


# ============================================================================
# Test 5: Project isolation (Project A vs Project B)
# ============================================================================


def test_project_isolation_nodes_and_facts(
    client: TestClient,
    db_session: Session,
    mock_repo: MagicMock,
) -> None:
    """Test 5: Cross-project access is strictly blocked for both nodes and facts."""
    manifest = validate_manifest(load_manifest())

    # Find relationship in Project B
    b_rel = manifest.projects["project_b"].papers[0].relationships[0]
    paper_b = db_session.query(Paper).filter(Paper.id == b_rel.provenance.paper_id).first()
    assert paper_b is not None

    snapshot_b = _create_snapshot(
        db=db_session,
        project_id=PROJECT_B_ID,
        rel=b_rel,
        document_sha256=paper_b.document_sha256,
        fact_id="fact-proj-b-exclusive",
    )

    # 1. Fact isolation: Project A cannot read Project B's snapshot
    resp_a_reads_b = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/facts/{snapshot_b.fact_id}")
    assert resp_a_reads_b.status_code == 404

    resp_b_reads_b = client.get(f"/api/v1/projects/{PROJECT_B_ID}/graph/facts/{snapshot_b.fact_id}")
    assert resp_b_reads_b.status_code == 200

    # 2. Node isolation in Neo4j repo
    def mock_get_node(pid: UUID, key: str):
        if pid == PROJECT_B_ID and key == "b_only_node":
            return {
                "key": key,
                "name": "B Only Node",
                "type": "Concept",
                "description": None,
                "aliases": [],
                "project_id": str(PROJECT_B_ID),
                "updated_at": "2026-09-01T00:00:00Z",
            }
        return None

    mock_repo.get_node_by_key.side_effect = mock_get_node

    resp_node_a = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes/b_only_node")
    assert resp_node_a.status_code == 404

    resp_node_b = client.get(f"/api/v1/projects/{PROJECT_B_ID}/graph/nodes/b_only_node")
    assert resp_node_b.status_code == 200

    # 3. Nonexistent project ID -> 404 for all endpoints
    random_project_id = uuid4()
    assert client.get(f"/api/v1/projects/{random_project_id}/graph/status").status_code == 404
    assert client.get(f"/api/v1/projects/{random_project_id}/graph/nodes").status_code == 404


# ============================================================================
# Test 6: Relationships between endpoints
# ============================================================================


def test_relationships_between_endpoints(
    client: TestClient,
    db_session: Session,
    mock_repo: MagicMock,
) -> None:
    """Test 6: Finds relationships between entities and builds Citation-compatible responses."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]

    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).first()
    assert paper is not None

    snapshot = _create_snapshot(
        db=db_session,
        project_id=PROJECT_A_ID,
        rel=rel,
        document_sha256=paper.document_sha256,
        fact_id="fact-rel-001",
    )

    mock_repo.find_relationships_between.return_value = [
        {
            "fact_id": snapshot.fact_id,
            "project_id": str(PROJECT_A_ID),
            "paper_id": str(snapshot.paper_id),
            "generation_id": snapshot.generation_id,
            "predicate": snapshot.predicate,
            "subject_key": snapshot.subject_key,
            "subject_name": snapshot.subject_name,
            "subject_type": snapshot.subject_type,
            "object_key": snapshot.object_key,
            "object_name": snapshot.object_name,
            "object_type": snapshot.object_type,
            "qualifiers": snapshot.qualifiers,
            "char_start": snapshot.char_start,
            "char_end": snapshot.char_end,
            "page_number": snapshot.page_number,
            "exact_quote": snapshot.exact_quote,
            "updated_at": "2026-09-01T12:00:00Z",
        }
    ]

    resp = client.get(
        f"/api/v1/projects/{PROJECT_A_ID}/graph/relationships"
        f"?subject_key={snapshot.subject_key}&object_key={snapshot.object_key}&predicate={snapshot.predicate}"
    )
    assert resp.status_code == 200
    data = resp.json()

    assert data["total"] == 1
    assert len(data["items"]) == 1
    item = data["items"][0]

    assert item["id"] == snapshot.fact_id
    assert item["predicate"] == snapshot.predicate
    assert item["anchor_status"] == AnchorStatus.VERIFIED.value
    assert item["citation"] is not None
    assert item["citation"]["citation_index"] == 1
    assert item["citation"]["evidence_id"] == "G1"
    assert item["citation"]["quote"] == snapshot.exact_quote


# ============================================================================
# Test 7: Index endpoint
# ============================================================================


def test_index_endpoint_previews_but_requires_persisted_assistant_approval(
    client: TestClient,
    db_session: Session,
) -> None:
    """The legacy graph route previews targets but cannot bypass assistant approval."""
    # 1. Default request body: dry_run=True
    resp_dry = client.post(f"/api/v1/projects/{PROJECT_A_ID}/graph/index", json={})
    assert resp_dry.status_code == 200
    dry_data = resp_dry.json()

    assert dry_data["dry_run"] is True
    assert dry_data["target_project_id"] == str(PROJECT_A_ID)
    assert len(dry_data["eligible_paper_ids"]) > 0
    assert dry_data["enqueued_count"] == 0

    # No events committed to database
    initial_events_count = (
        db_session.query(GraphEvent).filter(GraphEvent.project_id == PROJECT_A_ID).count()
    )
    assert initial_events_count == 0

    # Persistent index writes are available only through an approved assistant action.
    resp_live = client.post(
        f"/api/v1/projects/{PROJECT_A_ID}/graph/index",
        json={"dry_run": False, "limit": 5},
    )
    assert resp_live.status_code == 409
    assert "approved through the assistant" in resp_live.json()["detail"]
    assert db_session.query(GraphEvent).filter(GraphEvent.project_id == PROJECT_A_ID).count() == 0


def test_index_endpoint_dry_run_does_not_require_neo4j(client: TestClient) -> None:
    response = client.post(
        f"/api/v1/projects/{PROJECT_A_ID}/graph/index",
        json={"dry_run": True, "limit": 1},
    )
    assert response.status_code == 200
    assert response.json()["dry_run"] is True


# ============================================================================
# Test 8: Graph unavailable outage handling
# ============================================================================


def test_graph_unavailable_behavior(
    client: TestClient,
    mock_repo: MagicMock,
) -> None:
    """Test 8: Node/neighbor routes return 503 cleanly when Neo4j is offline;
    status returns 200 with neo4j_available: false.
    """
    mock_repo.verify_connectivity.return_value = False

    # 1. Status route succeeds with 200 and neo4j_available = False
    resp_status = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/status")
    assert resp_status.status_code == 200
    status_data = resp_status.json()
    assert status_data["neo4j_available"] is False
    assert status_data["node_count"] == 0
    assert status_data["fact_count"] == 0

    # 2. Node search returns 503
    resp_nodes = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes")
    assert resp_nodes.status_code == 503
    assert resp_nodes.json()["detail"] == "Graph service unavailable"

    # 3. Node detail returns 503
    resp_node = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes/some:node")
    assert resp_node.status_code == 503
    assert resp_node.json()["detail"] == "Graph service unavailable"

    # 4. Node neighbors returns 503
    resp_nbr = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/nodes/some:node/neighbors")
    assert resp_nbr.status_code == 503
    assert resp_nbr.json()["detail"] == "Graph service unavailable"

    # 5. Relationships returns 503
    resp_rel = client.get(
        f"/api/v1/projects/{PROJECT_A_ID}/graph/relationships?subject_key=s&object_key=o"
    )
    assert resp_rel.status_code == 503
    assert resp_rel.json()["detail"] == "Graph service unavailable"

    # 6. Fact detail when snapshot is missing and Neo4j is offline -> 404 (Fact not found)
    resp_fact = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/facts/missing-fact-id")
    assert resp_fact.status_code == 404
    assert resp_fact.json()["detail"] == "Fact not found in this project"


def test_fact_found_in_neo4j_fallback(
    client: TestClient,
    db_session: Session,
    mock_repo: MagicMock,
) -> None:
    """Fact not in PostgreSQL snapshot is fetched from Neo4j and resolved."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).first()
    assert paper is not None

    neo4j_fact = {
        "id": "fact-neo4j-only-01",
        "project_id": str(PROJECT_A_ID),
        "paper_id": str(paper.id),
        "generation_id": "gen-neo4j-01",
        "predicate": rel.predicate,
        "subject_key": "method:aurc",
        "subject_name": "AURC",
        "subject_type": "Method",
        "object_key": "dataset:imagenet",
        "object_name": "ImageNet",
        "object_type": "Dataset",
        "qualifiers": rel.qualifiers,
        "exact_quote": rel.provenance.exact_quote,
        "page_number": rel.provenance.page_number,
        "char_start": rel.provenance.char_start,
        "char_end": rel.provenance.char_end,
        "document_sha256": paper.document_sha256,
        "chunk_id": str(rel.provenance.chunk_id),
        "updated_at": "2026-09-01T12:00:00Z",
    }
    mock_repo.get_fact_by_id.return_value = neo4j_fact

    resp = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/facts/fact-neo4j-only-01")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "fact-neo4j-only-01"
    assert data["anchor_status"] == AnchorStatus.VERIFIED.value
    assert data["citation"] is not None


def test_unverified_fact_status_unresolved(
    client: TestClient,
    db_session: Session,
) -> None:
    """Snapshot with non-matching quote resolves to AnchorStatus.UNRESOLVED with citation=None."""
    manifest = validate_manifest(load_manifest())
    rel = manifest.projects["project_a"].papers[0].relationships[0]
    paper = db_session.query(Paper).filter(Paper.id == rel.provenance.paper_id).first()
    assert paper is not None

    snapshot = _create_snapshot(
        db=db_session,
        project_id=PROJECT_A_ID,
        rel=rel,
        document_sha256=paper.document_sha256,
        fact_id="fact-unverified-001",
    )
    # Alter the quote so verification fails
    snapshot.exact_quote = "This quote definitely does not exist in the source paper text."
    db_session.commit()

    resp = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/facts/{snapshot.fact_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == snapshot.fact_id
    assert data["anchor_status"] == AnchorStatus.UNRESOLVED.value
    assert data["citation"] is None


def test_get_graph_repo_and_connectivity_helpers(monkeypatch) -> None:
    """Directly test get_graph_repo and _check_repo_available exception paths."""
    from app.api.v1.graph import _check_repo_available, _format_updated_at, get_graph_repo

    assert _format_updated_at(None) is None
    assert _check_repo_available(None) is False

    exploding_repo = MagicMock(spec=Neo4jRepository)
    exploding_repo.verify_connectivity.side_effect = RuntimeError("Neo4j exploded")
    assert _check_repo_available(exploding_repo) is False

    # Default get_graph_repo returns None when unconfigured
    repo = get_graph_repo()
    assert repo is None or isinstance(repo, Neo4jRepository)


def test_status_endpoint_handles_count_exception(
    client: TestClient,
    mock_repo: MagicMock,
) -> None:
    """Status endpoint recovers cleanly when count_project_elements raises an error."""
    mock_repo.count_project_elements.side_effect = RuntimeError("Count failed")
    resp = client.get(f"/api/v1/projects/{PROJECT_A_ID}/graph/status")
    assert resp.status_code == 200
    assert resp.json()["neo4j_available"] is False
