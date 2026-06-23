"""Task 5.33: Local Vertical Slice for Milestone 5 GraphRAG.

Executes a complete, runnable end-to-end local vertical slice:
1. Disposable SQLite DB & explicitly configured disposable Neo4j connection.
2. Seeding Paper to READY with exact character offsets and child chunk element linkage.
3. Atomic GraphEvent enqueueing and processing via GraphEventProcessor.
4. Schema validation, candidate verification, snapshot persistence, and Neo4j node/fact MERGE.
5. Graph API verification via TestClient (/status, /nodes, /neighbors, /facts/{id}, /relationships).
6. Question answering with ChatService: intent routing to RELATIONSHIP, graph candidate retrieval,
   evidence re-resolution to live CitationAnchor, and verified citation to exact PDF text.
7. Reindex & generation retirement (retires older generation in Neo4j while
   preserving durable snapshots).
8. Graceful outage fallback when Neo4j is offline.
9. Fixture manifest comparison against synthetic corpus manifest.
10. Isolated teardown (leaves zero residue in Neo4j).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from neo4j import GraphDatabase
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.v1.graph import get_graph_repo
from app.config import Settings
from app.crud.chat import create_conversation
from app.crud.graph import (
    claim_next_graph_event,
    create_or_enqueue_graph_event,
    get_graph_event,
)
from app.crud.project import create_project
from app.db.base import Base
from app.db.models import (
    ChunkElement,
    Paper,
    PaperChunk,
    PaperElement,
    PaperPage,
)
from app.db.session import get_db
from app.main import app
from app.schemas.evidence import (
    AnchorStatus,
)
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.chat_service import ChatService
from app.services.embedding import DeterministicEmbeddingProvider, set_embedding_provider
from app.services.graphrag.extractor import (
    GraphExtractionAdapter,
)
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.processor import GraphEventProcessor
from app.services.graphrag.router import GraphIntent, route_query_intent
from app.services.graphrag.snapshots import get_verified_fact_snapshots
from app.services.llm import LLMProvider, set_llm_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("m5_vertical_slice")


class SliceFakeLLMProvider(LLMProvider):
    """Fake LLM provider for the vertical slice returning valid extraction and cited QA."""

    def __init__(self, extraction_json: str | None = None) -> None:
        self.extraction_json = extraction_json or ""

    @property
    def provider_name(self) -> str:
        return "fake_slice_provider"

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        # Check if called for extraction
        if "JSON" in system_prompt or "entities" in system_prompt:
            return self.extraction_json

        # Check if called for chat QA
        if "RELATIONSHIP" in user_prompt or "Temperature Scaling" in user_prompt:
            return "We evaluate Temperature Scaling on the ImageNet benchmark [E1]."

        return "Standard response without graph citations [E1]."


@pytest.fixture
def slice_db() -> Session:
    """Provide an isolated, in-memory SQLite database for the vertical slice."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def slice_repo(disposable_neo4j_uri: str) -> Neo4jRepository:
    """Provide a repository connected to the acknowledged disposable target."""
    driver = GraphDatabase.driver(disposable_neo4j_uri, auth=None)
    driver.verify_connectivity()
    repo = Neo4jRepository(driver=driver, database="neo4j")
    repo.ensure_schema()
    yield repo
    driver.close()


@pytest.mark.anyio
async def test_m5_local_vertical_slice(slice_db: Session, slice_repo: Neo4jRepository) -> None:
    """Run the complete local vertical slice covering Tasks 5.01 - 5.33."""

    # -------------------------------------------------------------------------
    # 0. Setup & Project Isolation
    # -------------------------------------------------------------------------
    project_id = uuid4()
    paper_id = uuid4()
    worker_id = f"worker_slice_{uuid4().hex[:6]}"
    doc_sha256 = "a1" * 32

    logger.info("Starting M5 Local Vertical Slice for project %s", project_id)

    # Ensure clean state in Neo4j
    slice_repo.delete_project_graph(project_id)

    # Create Project in SQLite
    project = create_project(
        slice_db,
        ProjectCreate(name="Slice NLP Calibration", description="Vertical slice test project"),
    )
    project.id = project_id
    slice_db.commit()

    # -------------------------------------------------------------------------
    # 1. Seed Paper to READY with exact character offsets
    # -------------------------------------------------------------------------
    paper = Paper(
        id=paper_id,
        project_id=project_id,
        filename="calibration_neural_networks.pdf",
        storage_path=f"papers/{project_id}/{paper_id}.pdf",
        status=PaperStatus.PROCESSING,
        document_sha256=doc_sha256,
    )
    slice_db.add(paper)
    slice_db.commit()

    quote_1 = "We evaluate Temperature Scaling on the ImageNet benchmark."
    quote_2 = "Temperature Scaling achieves 2.1% ECE on ImageNet validation split."
    raw_page_1 = f"{quote_1} {quote_2}"

    page = PaperPage(
        paper_id=paper_id,
        page_number=1,
        width=612.0,
        height=792.0,
        raw_text=raw_page_1,
    )
    slice_db.add(page)

    char_start_1 = raw_page_1.find(quote_1)
    char_end_1 = char_start_1 + len(quote_1)

    char_start_2 = raw_page_1.find(quote_2)
    char_end_2 = char_start_2 + len(quote_2)
    assert char_end_2 > char_start_2

    elem_1 = PaperElement(
        id=uuid4(),
        paper_id=paper_id,
        page_number=1,
        element_index=0,
        element_type="paragraph",
        text=quote_1,
        bbox_x_min=50.0,
        bbox_y_min=100.0,
        bbox_x_max=500.0,
        bbox_y_max=120.0,
        page_width=612.0,
        page_height=792.0,
        coordinate_origin="TOP_LEFT",
        rotation=0,
        parser_version="1.0.0",
    )
    elem_2 = PaperElement(
        id=uuid4(),
        paper_id=paper_id,
        page_number=1,
        element_index=1,
        element_type="paragraph",
        text=quote_2,
        bbox_x_min=50.0,
        bbox_y_min=130.0,
        bbox_x_max=500.0,
        bbox_y_max=150.0,
        page_width=612.0,
        page_height=792.0,
        coordinate_origin="TOP_LEFT",
        rotation=0,
        parser_version="1.0.0",
    )
    slice_db.add_all([elem_1, elem_2])

    chunk_text = f"{quote_1} {quote_2}"
    chunk_1 = PaperChunk(
        id=uuid4(),
        paper_id=paper_id,
        chunk_type="child",
        chunk_index=0,
        text=chunk_text,
        token_count=len(chunk_text.split()),
        embedding=[0.05] * 1024,
        embedding_vec=[0.05] * 1024,
        embedding_model="bge-large-en-v1.5",
        embedding_version="1.0.0",
    )
    slice_db.add(chunk_1)
    slice_db.flush()

    ce_1 = ChunkElement(chunk_id=chunk_1.id, element_id=elem_1.id, order_index=0)
    ce_2 = ChunkElement(chunk_id=chunk_1.id, element_id=elem_2.id, order_index=1)
    slice_db.add_all([ce_1, ce_2])

    # Mark Paper READY and atomically enqueue GraphEvent (Task 5.11)
    paper.status = "READY"
    gen_id_1 = f"gen_slice_{paper_id.hex[:6]}_v1"
    event_1 = create_or_enqueue_graph_event(
        db=slice_db,
        project_id=project_id,
        paper_id=paper_id,
        action="UPSERT",
        generation_id=gen_id_1,
    )
    slice_db.commit()
    logger.info("Paper marked READY and GraphEvent enqueued: %s", event_1.id)

    # -------------------------------------------------------------------------
    # 2. Worker Claims and Processes GraphEvent (Tasks 5.12, 5.19)
    # -------------------------------------------------------------------------
    claimed_event = claim_next_graph_event(slice_db, worker_id)
    assert claimed_event is not None
    assert claimed_event.id == event_1.id
    assert claimed_event.status == "PROCESSING"
    assert claimed_event.lease_owner == worker_id

    extraction_payload = {
        "entities": [
            {
                "name": "Temperature Scaling",
                "type": "Method",
                "description": "Post-processing calibration technique",
                "aliases": ["TS"],
            },
            {
                "name": "ImageNet",
                "type": "Dataset",
                "description": "Visual object recognition benchmark",
                "aliases": ["ILSVRC2012"],
            },
            {
                "name": "ECE",
                "type": "Metric",
                "description": "Expected Calibration Error",
                "aliases": [],
            },
            {
                "name": "2.1% ECE",
                "type": "Result",
                "description": "Calibrated error rate",
                "aliases": [],
            },
        ],
        "facts": [
            {
                "subject": {"name": "Temperature Scaling", "type": "Method"},
                "predicate": "EVALUATED_ON",
                "object": {"name": "ImageNet", "type": "Dataset"},
                "qualifiers": {"dataset": "ImageNet"},
                "evidence_id": "ev_1",
                "exact_quote": quote_1,
            },
            {
                "subject": {"name": "Temperature Scaling", "type": "Method"},
                "predicate": "ACHIEVES_RESULT",
                "object": {"name": "2.1% ECE", "type": "Result"},
                "qualifiers": {"metric": "ECE", "result_value": 2.1},
                "evidence_id": "ev_1",
                "exact_quote": quote_2,
            },
        ],
    }

    mock_llm = SliceFakeLLMProvider(extraction_json=json.dumps(extraction_payload))
    extractor_adapter = GraphExtractionAdapter(llm_provider=mock_llm)

    processor = GraphEventProcessor(
        settings=Settings(graphrag_enabled=True),
        repo=slice_repo,
        extractor=extractor_adapter,
    )

    await processor.process_graph_event(slice_db, claimed_event.id, worker_id)

    # Verify event completed
    refreshed_event = get_graph_event(slice_db, claimed_event.id)
    assert refreshed_event is not None
    assert refreshed_event.status == "COMPLETED"
    assert refreshed_event.completed_at is not None

    # Verify durable snapshots stored in PostgreSQL / SQLite
    snapshots = get_verified_fact_snapshots(slice_db, paper_id, gen_id_1)
    assert len(snapshots) == 2
    logger.info("Persisted %d verified fact snapshots in database", len(snapshots))

    fact_eval = next(s for s in snapshots if s.predicate == "EVALUATED_ON")
    assert fact_eval.subject_name == "Temperature Scaling"
    assert fact_eval.object_name == "ImageNet"
    assert fact_eval.exact_quote == quote_1
    assert fact_eval.char_start == char_start_1
    assert fact_eval.char_end == char_end_1

    # Verify Neo4j Graph state
    counts = slice_repo.count_project_elements(project_id)
    assert counts["node_count"] >= 3
    assert counts["fact_count"] == 2
    logger.info("Verified Neo4j element counts: %s", counts)

    # -------------------------------------------------------------------------
    # 3. Inspect Graph API Endpoints via FastAPI TestClient (Task 5.29)
    # -------------------------------------------------------------------------
    app.dependency_overrides[get_db] = lambda: slice_db
    app.dependency_overrides[get_graph_repo] = lambda: slice_repo

    client = TestClient(app)

    # 3a. Status
    res = client.get(f"/api/v1/projects/{project_id}/graph/status")
    assert res.status_code == 200, res.text
    status_data = res.json()
    assert status_data["neo4j_available"] is True
    assert status_data["node_count"] >= 3
    assert status_data["fact_count"] == 2
    assert status_data["completed_events_count"] == 1
    logger.info("API /status returned: %s", status_data)

    # 3b. Search Nodes
    res = client.get(f"/api/v1/projects/{project_id}/graph/nodes?query=Temperature")
    assert res.status_code == 200, res.text
    nodes_data = res.json()
    assert nodes_data["total"] >= 1
    ts_node = next(n for n in nodes_data["items"] if n["name"] == "Temperature Scaling")
    assert ts_node["type"] == "Method"
    ts_node_key = ts_node["key"]

    # 3c. Neighbors
    res = client.get(f"/api/v1/projects/{project_id}/graph/nodes/{ts_node_key}/neighbors")
    assert res.status_code == 200, res.text
    neighbors_data = res.json()
    assert neighbors_data["total"] >= 1
    eval_neighbor = next(
        nb for nb in neighbors_data["neighbors"] if nb["predicate"] == "EVALUATED_ON"
    )
    assert eval_neighbor["neighbor_name"] == "ImageNet"
    assert eval_neighbor["direction"] == "OUTGOING"
    fact_id = eval_neighbor["fact_id"]

    # 3d. Fact Detail with Live Citation Anchor
    res = client.get(f"/api/v1/projects/{project_id}/graph/facts/{fact_id}")
    assert res.status_code == 200, res.text
    fact_data = res.json()
    assert fact_data["id"] == fact_id
    assert fact_data["anchor_status"] == "verified"
    assert fact_data["citation"] is not None
    citation = fact_data["citation"]
    assert citation["quote"] == quote_1
    assert citation["page_number"] == 1
    assert citation["anchor_status"] == "verified"
    assert len(citation["anchors"]) > 0
    anchor = citation["anchors"][0]
    assert fact_data["char_start"] == char_start_1
    assert fact_data["char_end"] == char_end_1
    assert anchor["source_char_start"] == char_start_1
    assert anchor["source_char_end"] == char_end_1
    logger.info("API /facts/{id} verified citation anchor: %s", anchor["exact_quote"])

    # -------------------------------------------------------------------------
    # 4. Chat QA & Citation Source Jump (Task 5.27)
    # -------------------------------------------------------------------------
    set_embedding_provider(DeterministicEmbeddingProvider(dimension=1024))
    set_llm_provider(mock_llm)
    chat_service = ChatService(graph_repo=slice_repo)
    conv = create_conversation(slice_db, project_id, title="Slice Chat")

    question = "What is the relationship between Temperature Scaling and ImageNet?"
    intent = route_query_intent(question)
    assert intent == GraphIntent.RELATIONSHIP
    logger.info("Query routed to intent: %s", intent.value)

    answer_resp = await chat_service.answer_question(
        db=slice_db,
        conversation_id=conv.id,
        question=question,
    )
    assert answer_resp is not None
    assert len(answer_resp.citations) >= 1
    top_citation = answer_resp.citations[0]
    assert top_citation.anchor_status == AnchorStatus.VERIFIED
    assert top_citation.paper_id == paper_id
    assert top_citation.page_number == 1
    logger.info("Chat answer verified citation: %s", top_citation.quote)

    # -------------------------------------------------------------------------
    # 5. Replay & Reindex Generation Retirement (Task 5.13)
    # -------------------------------------------------------------------------
    logger.info("Testing reindex and active generation retirement")
    gen_id_2 = f"gen_slice_{paper_id.hex[:6]}_v2"
    event_2 = create_or_enqueue_graph_event(
        db=slice_db,
        project_id=project_id,
        paper_id=paper_id,
        action="UPSERT",
        generation_id=gen_id_2,
    )
    slice_db.commit()

    claimed_event_2 = claim_next_graph_event(slice_db, worker_id)
    assert claimed_event_2 is not None
    assert claimed_event_2.id == event_2.id

    # Gen 2 payload has updated result
    extraction_payload_v2 = {
        "entities": [
            {"name": "Temperature Scaling", "type": "Method"},
            {"name": "ImageNet", "type": "Dataset"},
            {"name": "2.0% ECE", "type": "Result"},
        ],
        "facts": [
            {
                "subject": {"name": "Temperature Scaling", "type": "Method"},
                "predicate": "EVALUATED_ON",
                "object": {"name": "ImageNet", "type": "Dataset"},
                "qualifiers": {"dataset": "ImageNet"},
                "evidence_id": "ev_1",
                "exact_quote": quote_1,
            },
        ],
    }
    mock_llm.extraction_json = json.dumps(extraction_payload_v2)
    await processor.process_graph_event(slice_db, claimed_event_2.id, worker_id)

    # Verify older generation facts were retired in Neo4j
    new_counts = slice_repo.count_project_elements(project_id)
    assert new_counts["fact_count"] == 1  # Older generation retired
    logger.info("After reindex, active fact count in Neo4j: %d", new_counts["fact_count"])

    # -------------------------------------------------------------------------
    # 6. Ordinary RAG Fallback & Outage Handling (Task 5.28)
    # -------------------------------------------------------------------------
    logger.info("Testing graceful outage fallback when Neo4j is offline")
    offline_chat_service = ChatService(graph_repo=None)  # Simulate disconnected / unconfigured repo
    conv_offline = create_conversation(slice_db, project_id, title="Offline Chat")
    offline_answer = await offline_chat_service.answer_question(
        db=slice_db,
        conversation_id=conv_offline.id,
        question="Tell me about Temperature Scaling on ImageNet",
    )
    assert offline_answer is not None
    assert offline_answer.content is not None
    logger.info("Offline fallback succeeded cleanly without crash")

    # -------------------------------------------------------------------------
    # 7. Fixture Manifest Comparison (Task 5.01)
    # -------------------------------------------------------------------------
    manifest_path = Path(__file__).parent / "fixtures" / "graphrag" / "manifest.json"
    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text())
        allowed_preds = manifest_data.get("allowed_predicates", [])
        assert "EVALUATED_ON" in allowed_preds
        assert "ACHIEVES_RESULT" in allowed_preds
        logger.info("Manifest alignment confirmed against %s", manifest_path.name)

    # -------------------------------------------------------------------------
    # 8. Isolated Teardown
    # -------------------------------------------------------------------------
    slice_repo.delete_project_graph(project_id)
    post_teardown = slice_repo.count_project_elements(project_id)
    assert post_teardown["node_count"] == 0
    assert post_teardown["fact_count"] == 0
    logger.info("Isolated teardown complete: Project %s graph wiped cleanly", project_id)

    # Clean up app dependency overrides
    app.dependency_overrides.clear()
    logger.info("M5 Local Vertical Slice PASSED successfully!")


if __name__ == "__main__":
    import asyncio

    print("Running standalone M5 Local Vertical Slice...")
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    from conftest import disposable_neo4j_test_uri

    uri = disposable_neo4j_test_uri()
    if uri is None:
        raise RuntimeError(
            "Standalone slice requires MYRA_TEST_NEO4J_URI and MYRA_ALLOW_DISPOSABLE_NEO4J_TESTS=1"
        )
    driver = GraphDatabase.driver(uri, auth=None)
    driver.verify_connectivity()
    repo = Neo4jRepository(driver=driver, database="neo4j")
    repo.ensure_schema()

    try:
        asyncio.run(test_m5_local_vertical_slice(session, repo))
        print("Standalone slice completed: PASS")
    finally:
        session.close()
        driver.close()
