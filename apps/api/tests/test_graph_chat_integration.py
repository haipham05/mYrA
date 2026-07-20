"""Integration tests for GraphRAG Chat, Intent Routing, and Outage Handling (Tasks 5.27 & 5.28).

Verifies:
1. Intent router classification: Labeled queries map to FACTUAL, RELATIONSHIP,
   CONTRADICTION, and CORPUS_THEMES.
2. Two-sided cited contradiction answer: Contradiction pair retrieved from graph,
   LLM produces two-sided answer citing both sides ([E1] and [E2]), validated citations
   retained with verified CitationAnchors.
3. Ordinary factual QA unchanged: Standard hybrid chunks retrieved; answers generated
   and validated without disruption or regression.
4. Unsupported graph fact abstention: Stale/unsupported facts dropped; hallucinations
   rejected by sentence validator with graceful abstention.
5. Graph outage handling (Task 5.28): When Neo4j is None or disconnected, router returns
   outage notice, ChatService continues with text chunks and includes outage notice.
6. Ingestion and Paper status remain decoupled: Graph outage does not prevent paper
   ingestion or affect paper READY status.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.crud.chat import create_conversation
from app.crud.job import create_job
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.base import Base
from app.db.models import (
    ChunkElement,
    GraphEvent,
    GraphFactSnapshot,
    Paper,
    PaperChunk,
    PaperElement,
    PaperPage,
    Project,
)
from app.ingestion.parser import ParsedElement, ParsedPage, ParseResult
from app.schemas.evidence import (
    AnchorStatus,
    BoundingBox,
    CitationAnchor,
    ClaimSupportKind,
    CoordinateOrigin,
    EvidenceItem,
)
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.chat_service import ChatService
from app.services.embedding import DeterministicEmbeddingProvider, set_embedding_provider
from app.services.graphrag.neo4j_repository import Neo4jRepository
from app.services.graphrag.router import (
    GraphIntent,
    retrieve_graph_candidates_for_query,
    retrieve_graph_evidence,
    route_query_intent,
)
from app.services.ingestion import IngestionPipeline
from app.storage.factory import set_storage
from app.storage.local import MemoryStorage


@pytest.fixture
def db() -> Session:
    """Create an isolated, in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


def _setup_ground_truth_paper(
    db: Session,
    project_id: UUID,
    title: str,
    quote: str,
    document_sha256: str,
    page_number: int = 1,
) -> tuple[Paper, PaperChunk, PaperElement]:
    """Helper to persist a READY Paper, PaperPage, PaperElement, PaperChunk, and ChunkElement."""
    paper = Paper(
        id=uuid4(),
        project_id=project_id,
        filename=f"{title.lower().replace(' ', '_')}.pdf",
        storage_path=f"projects/{project_id}/papers/{uuid4()}.pdf",
        document_sha256=document_sha256,
        status=PaperStatus.READY.value,
        page_count=1,
    )
    db.add(paper)
    db.flush()

    page = PaperPage(
        id=uuid4(),
        paper_id=paper.id,
        page_number=page_number,
        width=612.0,
        height=792.0,
        rotation=0,
        raw_text=quote,
    )
    db.add(page)
    db.flush()

    element = PaperElement(
        id=uuid4(),
        paper_id=paper.id,
        page_number=page_number,
        element_index=0,
        element_type="paragraph",
        text=quote,
        bbox_x_min=50.0,
        bbox_y_min=100.0,
        bbox_x_max=500.0,
        bbox_y_max=150.0,
        page_width=612.0,
        page_height=792.0,
        parser_version="v1",
    )
    db.add(element)
    db.flush()

    chunk = PaperChunk(
        id=uuid4(),
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text=quote,
    )
    db.add(chunk)
    db.flush()

    chunk_elem = ChunkElement(
        chunk_id=chunk.id,
        element_id=element.id,
        order_index=0,
    )
    db.add(chunk_elem)
    db.commit()

    return paper, chunk, element


# =============================================================================
# Test 1: Intent router classification
# =============================================================================


def test_intent_router_classification():
    """Verify that labeled query sets accurately classify into FACTUAL,

    RELATIONSHIP, CONTRADICTION, and CORPUS_THEMES.
    """
    contradiction_queries = [
        "Are there contradictory results regarding calibration?",
        "Do papers conflict on learning rate convergence?",
        "Is there any disagreement between the reported findings?",
        "What are opposing claims regarding model size?",
        "Are there differing results reported for temperature scaling?",
        "Any contrasting findings observed across the evaluations?",
        "Explain the discrepancy in reported ECE values.",
    ]
    for q in contradiction_queries:
        assert route_query_intent(q) == GraphIntent.CONTRADICTION, (
            f"Expected CONTRADICTION for '{q}'"
        )

    corpus_themes_queries = [
        "What are the recurring themes across papers in this project?",
        "What are common themes in the literature?",
        "What trends across all papers can be identified?",
        "Which architectures are used across papers?",
        "Are attention mechanisms applied across all papers?",
        "Please provide a corpus overview.",
        "What are common methods across the corpus?",
        "What patterns across datasets are observed?",
        "Provide a summary of the corpus.",
    ]
    for q in corpus_themes_queries:
        assert route_query_intent(q) == GraphIntent.CORPUS_THEMES, (
            f"Expected CORPUS_THEMES for '{q}'"
        )

    relationship_queries = [
        "How is Adam related to SGD?",
        "What is the relationship between temperature scaling and ECE?",
        "What is the connection between self-attention and convolution?",
        "How does ResNet compare to Vision Transformer?",
        "How do CNNs and Transformers relate in speech recognition?",
        "Does Branchformer use depthwise convolution?",
        "What is the association between model depth and calibration?",
        "Is there a link between dropout and robustness?",
    ]
    for q in relationship_queries:
        assert route_query_intent(q) == GraphIntent.RELATIONSHIP, f"Expected RELATIONSHIP for '{q}'"

    factual_queries = [
        "What was the accuracy achieved on ImageNet?",
        "Who wrote the paper on attention?",
        "What learning rate was used for training?",
        "Explain the decoder architecture.",
        "",
        "   ",
    ]
    for q in factual_queries:
        assert route_query_intent(q) == GraphIntent.FACTUAL, f"Expected FACTUAL for '{q}'"


# =============================================================================
# Test 2: Two-sided cited contradiction answer
# =============================================================================


@pytest.mark.anyio
async def test_two_sided_cited_contradiction_answer(db: Session):
    """Query about conflicting results -> graph retrieves contradiction pair

    (e.g. 2.1% ECE on ImageNet [E1] vs 8.5% ECE on ImageNet [E2]) -> LLM produces
    two-sided answer citing [E1] and [E2] -> both citations are validated,
    retained in final answer, and return verified CitationAnchors.
    """
    project = Project(name="Calibration Contradiction Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Contradiction Query")

    quote_a = (
        "Temperature scaling achieves 2.1% ECE on ImageNet for classification on the test split "
        "under standard evaluation."
    )
    quote_b = (
        "Temperature scaling fails to converge, yielding an ECE of 8.5% on ImageNet for "
        "classification on the test split under standard evaluation."
    )
    hash_a = "a" * 64
    hash_b = "b" * 64

    paper_a, chunk_a, elem_a = _setup_ground_truth_paper(db, project.id, "Paper A", quote_a, hash_a)
    paper_b, chunk_b, elem_b = _setup_ground_truth_paper(db, project.id, "Paper B", quote_b, hash_b)

    # Persist snapshots in PostgreSQL
    snap_a = GraphFactSnapshot(
        fact_id="fact-contradiction-ece-a",
        project_id=project.id,
        paper_id=paper_a.id,
        generation_id="gen-1",
        subject_key="method:temperature_scaling",
        subject_name="Temperature Scaling",
        subject_type="Method",
        predicate="ACHIEVES_RESULT",
        object_key="dataset:imagenet",
        object_name="ImageNet",
        object_type="Dataset",
        qualifiers={
            "dataset": "ImageNet",
            "metric": "ECE",
            "result_value": 2.1,
            "unit": "%",
            "task": "classification",
            "split": "test",
            "comparison_condition": "standard evaluation",
            "polarity": "POSITIVE",
        },
        chunk_id=chunk_a.id,
        page_number=1,
        element_id=elem_a.id,
        char_start=0,
        char_end=len(quote_a),
        exact_quote=quote_a,
        document_sha256=hash_a,
        validation_version="1.0.0",
    )
    snap_b = GraphFactSnapshot(
        fact_id="fact-contradiction-ece-b",
        project_id=project.id,
        paper_id=paper_b.id,
        generation_id="gen-1",
        subject_key="method:temperature_scaling",
        subject_name="Temperature Scaling",
        subject_type="Method",
        predicate="ACHIEVES_RESULT",
        object_key="dataset:imagenet",
        object_name="ImageNet",
        object_type="Dataset",
        qualifiers={
            "dataset": "ImageNet",
            "metric": "ECE",
            "result_value": 8.5,
            "unit": "%",
            "task": "classification",
            "split": "test",
            "comparison_condition": "standard evaluation",
            "polarity": "NEGATIVE",
        },
        chunk_id=chunk_b.id,
        page_number=1,
        element_id=elem_b.id,
        char_start=0,
        char_end=len(quote_b),
        exact_quote=quote_b,
        document_sha256=hash_b,
        validation_version="1.0.0",
    )
    db.add_all([snap_a, snap_b])
    db.commit()

    # Mock Neo4j repository returning facts that build_contradiction_candidates detects
    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.verify_connectivity.return_value = True
    mock_repo.get_project_facts.return_value = [
        {
            "id": snap_a.fact_id,
            "paper_id": str(paper_a.id),
            "predicate": snap_a.predicate,
            "subject_key": snap_a.subject_key,
            "subject_name": snap_a.subject_name,
            "subject_type": snap_a.subject_type,
            "object_key": snap_a.object_key,
            "object_name": snap_a.object_name,
            "object_type": snap_a.object_type,
            "exact_quote": snap_a.exact_quote,
            "page_number": 1,
            "qualifiers": snap_a.qualifiers,
        },
        {
            "id": snap_b.fact_id,
            "paper_id": str(paper_b.id),
            "predicate": snap_b.predicate,
            "subject_key": snap_b.subject_key,
            "subject_name": snap_b.subject_name,
            "subject_type": snap_b.subject_type,
            "object_key": snap_b.object_key,
            "object_name": snap_b.object_name,
            "object_type": snap_b.object_type,
            "exact_quote": snap_b.exact_quote,
            "page_number": 1,
            "qualifiers": snap_b.qualifiers,
        },
    ]

    mock_llm = AsyncMock()
    # LLM produces two-sided answer citing E1 and E2 individually
    mock_llm.generate.return_value = f"{quote_a} [E1], whereas {quote_b} [E2]."
    mock_llm.model_name = "test-deepseek"

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService(retriever=mock_retriever, graph_repo=mock_repo)
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="Do papers conflict on temperature scaling ECE on ImageNet?",
        )

    # Both citations must be validated, retained, and renumbered to [1] and [2]
    assert "[1]" in resp.content
    assert "[2]" in resp.content
    assert len(resp.citations) == 2

    cit_a = next(c for c in resp.citations if c.paper_id == paper_a.id)
    cit_b = next(c for c in resp.citations if c.paper_id == paper_b.id)

    assert cit_a.anchor_status == AnchorStatus.VERIFIED
    assert cit_a.quote == quote_a
    assert cit_a.document_sha256 == hash_a

    assert cit_b.anchor_status == AnchorStatus.VERIFIED
    assert cit_b.quote == quote_b
    assert cit_b.document_sha256 == hash_b

    source_claims = [
        support
        for support in resp.claim_supports
        if support.support_kind == ClaimSupportKind.SOURCE_BACKED
    ]
    derived_claims = [
        support
        for support in resp.claim_supports
        if support.support_kind == ClaimSupportKind.DERIVED
    ]
    assert {support.evidence_ids[0] for support in source_claims} == {"E1", "E2"}
    assert len(derived_claims) == 1
    assert derived_claims[0].evidence_ids == ["E1", "E2"]
    assert "whereas" in derived_claims[0].claim_text

    # Verify contradiction instructions were added to system prompt
    _, kwargs = mock_llm.generate.call_args
    assert "CONTRADICTION ANALYSIS:" in kwargs["system_prompt"]
    assert "Present each side as its own source-supported statement" in kwargs["system_prompt"]

    mock_llm.generate.return_value = (
        f"Paper A reports “{quote_a.removesuffix('.')}” [E1] proving it is universally best, "
        f"whereas Paper B reports “{quote_b.removesuffix('.')}” [E2]."
    )
    wrapped_response = await chat_service.answer_question(
        db=db,
        conversation_id=conv.id,
        question="Compare those findings again.",
    )
    assert "universally best" not in wrapped_response.content
    assert "Insufficient evidence" in wrapped_response.content
    assert wrapped_response.citations == []


# =============================================================================
# Test 3: Ordinary factual QA unchanged
# =============================================================================


@pytest.mark.anyio
async def test_ordinary_factual_qa_unchanged(db: Session):
    """Factual question retrieves standard hybrid chunks; answers are generated

    and validated without disruption or regression.
    """
    project = Project(name="Factual QA Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Factual Query")

    quote = "Our architecture achieves 92.4% accuracy on ImageNet benchmarks."
    doc_hash = "c" * 64
    paper, chunk, elem = _setup_ground_truth_paper(
        db, project.id, "Vision Transformer", quote, doc_hash
    )

    anchor = CitationAnchor(
        paper_id=paper.id,
        page_number=1,
        source_element_id=elem.id,
        source_char_start=0,
        source_char_end=len(quote),
        exact_quote=quote,
        bounding_boxes=[
            BoundingBox(
                origin=CoordinateOrigin.TOP_LEFT,
                x_min=50.0,
                y_min=100.0,
                x_max=500.0,
                y_max=150.0,
                page_width=612.0,
                page_height=792.0,
            )
        ],
        anchor_status=AnchorStatus.VERIFIED,
        document_sha256=doc_hash,
        parser_version="v1",
    )
    evidence_item = EvidenceItem(
        id="E1",
        paper_id=paper.id,
        paper_title=paper.filename,
        chunk_id=chunk.id,
        quote=quote,
        parent_context=quote,
        page_number=1,
        bounding_boxes=anchor.bounding_boxes,
        source_element_ids=[elem.id],
        document_sha256=doc_hash,
        parser_version="v1",
        anchors=[anchor],
    )

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [evidence_item]

    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.verify_connectivity.return_value = True
    mock_repo.search_nodes.return_value = []
    mock_repo.get_node_neighbors.return_value = []

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (
        "Our architecture achieves 92.4% accuracy on ImageNet benchmarks [E1]."
    )
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService(retriever=mock_retriever, graph_repo=mock_repo)
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What was the accuracy achieved on ImageNet benchmarks?",
        )

    assert "[1]" in resp.content
    assert len(resp.citations) == 1
    assert resp.citations[0].anchor_status == AnchorStatus.VERIFIED
    assert resp.citations[0].paper_id == paper.id

    # Verify no contradiction instructions or outage notice in prompt
    _, kwargs = mock_llm.generate.call_args
    assert "CONTRADICTION ANALYSIS:" not in kwargs["system_prompt"]
    assert "[NOTE:" not in kwargs["system_prompt"]
    assert "[NOTE:" not in kwargs["user_prompt"]


# =============================================================================
# Test 4: Unsupported graph fact abstention
# =============================================================================


@pytest.mark.anyio
async def test_unsupported_graph_fact_abstention(db: Session):
    """If candidate fact points to a chunk whose text doesn't contain the quote

    or has stale hash, it is dropped from evidence; if model hallucinates an
    uncited claim, sentence validator drops it.
    """
    project = Project(name="Stale Fact Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Stale Fact Query")

    real_quote = "Residual connections enable training of deep feedforward networks."
    doc_hash = "d" * 64
    paper, chunk, elem = _setup_ground_truth_paper(db, project.id, "ResNet", real_quote, doc_hash)

    # Persist a snapshot with mismatched quote not in the document
    stale_quote = "Residual connections completely fail on deeper networks."
    stale_snap = GraphFactSnapshot(
        fact_id="fact-stale-001",
        project_id=project.id,
        paper_id=paper.id,
        generation_id="gen-1",
        subject_key="concept:residual_connections",
        subject_name="Residual Connections",
        subject_type="Concept",
        predicate="ACHIEVES_RESULT",
        object_key="metric:failure",
        object_name="Failure",
        object_type="Metric",
        chunk_id=chunk.id,
        page_number=1,
        element_id=elem.id,
        char_start=0,
        char_end=len(stale_quote),
        exact_quote=stale_quote,
        document_sha256=doc_hash,
        validation_version="1.0.0",
    )
    db.add(stale_snap)
    db.commit()

    # Verify that resolve_graph_facts_to_evidence drops this stale fact
    evidence_items, _ = retrieve_graph_evidence(
        db=db,
        repo=None,  # Not used when facts are directly resolved
        project_id=project.id,
        query="Tell me about residual connections failure",
        intent=GraphIntent.FACTUAL,
    )
    # Since repo is None, router returns []
    assert len(evidence_items) == 0

    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.verify_connectivity.return_value = True
    mock_repo.search_nodes.return_value = []
    mock_repo.get_node_neighbors.return_value = [{"fact_id": stale_snap.fact_id}]

    # Retrieve graph evidence with the stale fact
    graph_items, _ = retrieve_graph_evidence(
        db=db,
        repo=mock_repo,
        project_id=project.id,
        query="Tell me about residual connections",
        intent=GraphIntent.FACTUAL,
    )
    # Stale fact quote not matching ground truth chunk text is dropped
    assert len(graph_items) == 0

    # Test hallucination rejection in ChatService: model attempts to cite nonexistent E1
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "Residual connections completely fail on deeper networks [E1]."
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService(retriever=mock_retriever, graph_repo=mock_repo)
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="Tell me about residual connections",
        )

    # Hallucinated citation dropped, response abstains
    assert "fail on deeper networks" not in resp.content
    assert "Insufficient evidence" in resp.content
    assert len(resp.citations) == 0


# =============================================================================
# Test 5: Graph outage handling (Task 5.28)
# =============================================================================


@pytest.mark.anyio
async def test_graph_outage_handling(db: Session):
    """When Neo4j is None or disconnected:

    - Router returns outage notice.
    - ChatService continues without error, uses HybridRetriever text chunks, and
      includes outage notice.
    - Answer with valid text citations is returned.
    """
    project = Project(name="Outage Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Outage Query")

    quote = "Adam optimizer outperforms SGD across language modeling tasks."
    doc_hash = "e" * 64
    paper, chunk, elem = _setup_ground_truth_paper(
        db, project.id, "Optimizer Paper", quote, doc_hash
    )

    anchor = CitationAnchor(
        paper_id=paper.id,
        page_number=1,
        source_element_id=elem.id,
        source_char_start=0,
        source_char_end=len(quote),
        exact_quote=quote,
        bounding_boxes=[
            BoundingBox(
                origin=CoordinateOrigin.TOP_LEFT,
                x_min=50.0,
                y_min=100.0,
                x_max=500.0,
                y_max=150.0,
                page_width=612.0,
                page_height=792.0,
            )
        ],
        anchor_status=AnchorStatus.VERIFIED,
        document_sha256=doc_hash,
        parser_version="v1",
    )
    evidence_item = EvidenceItem(
        id="E1",
        paper_id=paper.id,
        paper_title=paper.filename,
        chunk_id=chunk.id,
        quote=quote,
        parent_context=quote,
        page_number=1,
        bounding_boxes=anchor.bounding_boxes,
        source_element_ids=[elem.id],
        document_sha256=doc_hash,
        parser_version="v1",
        anchors=[anchor],
    )

    # 1. Test when repo is None
    candidates, notice_none = retrieve_graph_candidates_for_query(
        db=db,
        repo=None,
        project_id=project.id,
        query="What is the relationship between Adam and SGD?",
        intent=GraphIntent.RELATIONSHIP,
    )
    assert candidates == []
    assert notice_none == "Graph service is not configured; using text retrieval."

    # 2. Test when repo verify_connectivity returns False
    mock_disconnected_repo = MagicMock(spec=Neo4jRepository)
    mock_disconnected_repo.verify_connectivity.return_value = False

    candidates_disc, notice_disc = retrieve_graph_candidates_for_query(
        db=db,
        repo=mock_disconnected_repo,
        project_id=project.id,
        query="What is the relationship between Adam and SGD?",
        intent=GraphIntent.RELATIONSHIP,
    )
    assert candidates_disc == []
    assert notice_disc == "Graph service is currently offline; using text retrieval."

    # 3. Test when verify_connectivity raises an exception
    mock_error_repo = MagicMock(spec=Neo4jRepository)
    mock_error_repo.verify_connectivity.side_effect = ConnectionError("Neo4j connection refused")

    candidates_err, notice_err = retrieve_graph_candidates_for_query(
        db=db,
        repo=mock_error_repo,
        project_id=project.id,
        query="What is the relationship between Adam and SGD?",
        intent=GraphIntent.RELATIONSHIP,
    )
    assert candidates_err == []
    assert notice_err == "Graph service is currently offline; using text retrieval."

    # 4. ChatService continues without error, uses HybridRetriever text chunks, and includes notice
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [evidence_item]

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = (
        "Adam optimizer outperforms SGD across language modeling tasks [E1]."
    )
    mock_llm.model_name = "test-deepseek"

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService(retriever=mock_retriever, graph_repo=mock_disconnected_repo)
        resp = await chat_service.answer_question(
            db=db,
            conversation_id=conv.id,
            question="What is the relationship between Adam and SGD?",
        )

    # Valid text citation is returned
    assert "[1]" in resp.content
    assert len(resp.citations) == 1
    assert resp.citations[0].paper_id == paper.id
    assert resp.citations[0].anchor_status == AnchorStatus.VERIFIED

    # Outage notice was included in the user prompt evidence block and system prompt
    _, kwargs = mock_llm.generate.call_args
    assert (
        "[NOTE: Graph service is currently offline; using text retrieval.]" in kwargs["user_prompt"]
    )
    assert (
        "[NOTE: Graph service is currently offline; using text retrieval.]"
        in kwargs["system_prompt"]
    )


# =============================================================================
# Test 6: Ingestion and Paper status remain decoupled
# =============================================================================


@pytest.mark.anyio
async def test_ingestion_and_paper_status_remain_decoupled(db: Session):
    """Verifies that graph outage or missing graph service does not prevent paper

    ingestion or affect paper READY status.
    """
    storage = MemoryStorage()
    set_storage(storage)
    set_embedding_provider(DeterministicEmbeddingProvider())

    proj = create_project(db, ProjectCreate(name="Decoupled Ingestion Project"))
    pdf_bytes = b"%PDF-1.4 dummy content for test"
    pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()
    storage_key = f"papers/{proj.id}/attention.pdf"
    await storage.put(storage_key, pdf_bytes)

    paper = create_paper(db, proj.id, "attention.pdf", storage_key, document_sha256=pdf_sha)
    job = create_job(db, paper.id)

    # Even if graphrag is enabled with a broken/nonexistent Neo4j URI
    settings = Settings(
        graphrag_enabled=True,
        neo4j_uri="bolt://127.0.0.1:9999",  # Nonexistent port
        neo4j_timeout_seconds=0.1,
    )

    class MockParser:
        def parse(self, data: bytes) -> ParseResult:
            elem = ParsedElement(
                text=(
                    "The dominant sequence transduction models are based on "
                    "complex recurrent networks."
                ),
                element_type="paragraph",
                page_number=1,
                bbox_x_min=10.0,
                bbox_y_min=10.0,
                bbox_x_max=200.0,
                bbox_y_max=30.0,
                page_width=612.0,
                page_height=792.0,
                element_index=0,
            )
            page = ParsedPage(
                page_number=1,
                width=612.0,
                height=792.0,
                raw_text=elem.text,
            )
            return ParseResult(pages=[page], elements=[elem])

    pipeline = IngestionPipeline(parser=MockParser(), settings=settings)
    await pipeline.process_paper(db, paper.id, job.id)

    db.refresh(paper)
    db.refresh(job)
    assert job.error_message is None, f"Job failed with error: {job.error_message}"
    assert paper.status == PaperStatus.READY.value

    # Graph event was enqueued atomically in PostgreSQL
    event = (
        db.query(GraphEvent)
        .filter(GraphEvent.paper_id == paper.id, GraphEvent.project_id == proj.id)
        .first()
    )
    assert event is not None
    assert event.status == "PENDING"
    assert event.action == "UPSERT"


# =============================================================================
# Additional Router & ChatService Branch Tests
# =============================================================================


def test_router_extract_relationship_entities():
    """Verify entity pair extraction across multiple relationship query regex patterns."""
    from app.services.graphrag.router import _extract_relationship_entities

    queries = [
        ("What is the relationship between Adam and SGD?", ("Adam", "SGD")),
        ("How is LoRA related to PEFT?", ("LoRA", "PEFT")),
        ("How does ResNet-50 compare to ViT-B?", ("ResNet-50", "ViT-B")),
        ("How do CNN and Transformer relate?", ("CNN", "Transformer")),
        ("Does BERT use self-attention?", ("BERT", "self-attention")),
        ("What is the accuracy of ResNet?", (None, None)),
    ]
    for q, (exp_a, exp_b) in queries:
        ea, eb = _extract_relationship_entities(q)
        assert ea == exp_a, f"Expected {exp_a} for '{q}', got {ea}"
        assert eb == exp_b, f"Expected {exp_b} for '{q}', got {eb}"


def test_router_candidate_retrieval_corpus_themes(db: Session):
    """Verify CORPUS_THEMES candidate retrieval calls build_corpus_themes."""
    project_id = uuid4()
    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.verify_connectivity.return_value = True

    with patch(
        "app.services.graphrag.router.build_corpus_themes",
        return_value=[{"theme": "calibration", "contributing_fact_ids": ["f1"]}],
    ) as mock_themes:
        candidates, notice = retrieve_graph_candidates_for_query(
            db=db,
            repo=mock_repo,
            project_id=project_id,
            query="What are recurring themes across all papers?",
            intent=GraphIntent.CORPUS_THEMES,
        )
        assert notice is None
        assert len(candidates) == 1
        assert candidates[0]["theme"] == "calibration"
        mock_themes.assert_called_once_with(db, mock_repo, project_id, min_papers=2, limit=10)


def test_router_candidate_retrieval_relationship_node_matching_and_neighbors(db: Session):
    """Verify relationship retrieval fallback: node matching and 1-hop neighbor traversal."""
    project_id = uuid4()
    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.verify_connectivity.return_value = True

    # 1. Test when repo.search_nodes finds 2 matching nodes in the project
    mock_repo.search_nodes.return_value = [
        {"key": "node:adam", "name": "Adam"},
        {"key": "node:sgd", "name": "SGD"},
    ]
    with patch(
        "app.services.graphrag.router.build_relationship_candidates",
        return_value=[{"fact_id": "rel_fact_1"}],
    ) as mock_rel:
        candidates, notice = retrieve_graph_candidates_for_query(
            db=db,
            repo=mock_repo,
            project_id=project_id,
            query="Compare Adam and SGD",
            intent=GraphIntent.RELATIONSHIP,
        )
        assert notice is None
        assert len(candidates) == 1
        assert candidates[0]["fact_id"] == "rel_fact_1"
        mock_rel.assert_called_once()

    # 2. Test when no 2 nodes match: falls back to neighbor traversal
    mock_repo.search_nodes.side_effect = lambda pid, query=None, limit=100: (
        [{"key": "node:transformer", "name": "Transformer"}] if query == "Transformer" else []
    )
    mock_repo.get_node_neighbors.return_value = [
        {"neighbor_key": "node:attention", "fact_id": "fact_nb_1"}
    ]
    candidates_nb, notice_nb = retrieve_graph_candidates_for_query(
        db=db,
        repo=mock_repo,
        project_id=project_id,
        query="Tell me how Transformer is used",
        intent=GraphIntent.RELATIONSHIP,
    )
    assert notice_nb is None
    assert len(candidates_nb) == 1
    assert candidates_nb[0]["fact_id"] == "fact_nb_1"


def test_router_candidate_retrieval_factual_one_hop(db: Session):
    """Verify factual query candidate retrieval extracts 1-hop facts."""
    project_id = uuid4()
    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.verify_connectivity.return_value = True

    mock_repo.search_nodes.return_value = [{"key": "node:resnet", "name": "ResNet"}]
    mock_repo.get_node_neighbors.return_value = [{"fact_id": "fact_factual_1"}]

    candidates, notice = retrieve_graph_candidates_for_query(
        db=db,
        repo=mock_repo,
        project_id=project_id,
        query="What is the architecture of ResNet?",
        intent=GraphIntent.FACTUAL,
    )
    assert notice is None
    assert len(candidates) == 1
    assert candidates[0]["fact_id"] == "fact_factual_1"


def test_router_candidate_retrieval_exception_handling(db: Session):
    """Verify that an unexpected error during graph traversal returns outage notice."""
    project_id = uuid4()
    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.verify_connectivity.return_value = True
    mock_repo.search_nodes.side_effect = RuntimeError("Deadlock in Neo4j")

    candidates, notice = retrieve_graph_candidates_for_query(
        db=db,
        repo=mock_repo,
        project_id=project_id,
        query="What is the relationship between X and Y?",
        intent=GraphIntent.RELATIONSHIP,
    )
    assert candidates == []
    assert notice == "Graph service is currently offline; using text retrieval."


def test_retrieve_graph_evidence_no_fact_ids(db: Session):
    """Verify retrieve_graph_evidence returns empty list when candidates yield no fact IDs."""
    project_id = uuid4()
    mock_repo = MagicMock(spec=Neo4jRepository)
    mock_repo.verify_connectivity.return_value = True

    with patch(
        "app.services.graphrag.router.retrieve_graph_candidates_for_query",
        return_value=([{"invalid": "candidate"}], None),
    ):
        items, notice = retrieve_graph_evidence(
            db=db,
            repo=mock_repo,
            project_id=project_id,
            query="Some query",
            intent=GraphIntent.FACTUAL,
        )
        assert items == []
        assert notice is None


def test_graph_telemetry_distinguishes_neo4j_outage_and_unresolved_evidence(db, monkeypatch):
    class Span:
        def __init__(self):
            self.updates = []

        def update(self, **kwargs):
            self.updates.append(kwargs)

    class RecordingTelemetry:
        def __init__(self):
            self.spans = []

        @contextmanager
        def stage(self, name, **kwargs):
            span = Span()
            self.spans.append((name, kwargs, span))
            yield span

    recorder = RecordingTelemetry()
    monkeypatch.setattr("app.services.graphrag.router.get_telemetry", lambda: recorder)

    items, outage = retrieve_graph_evidence(
        db=db,
        repo=None,
        project_id=uuid4(),
        query="does graph work",
        intent=GraphIntent.FACTUAL,
    )
    assert items == []
    assert outage == "Graph service is not configured; using text retrieval."
    assert recorder.spans[0][2].updates[-1]["metadata"]["outcome"] == "unavailable"

    monkeypatch.setattr(
        "app.services.graphrag.router.retrieve_graph_candidates_for_query",
        lambda **kwargs: ([{"fact_id": "fact-1"}], None),
    )
    monkeypatch.setattr(
        "app.services.graphrag.router.resolve_graph_facts_to_evidence",
        lambda **kwargs: [],
    )
    items, outage = retrieve_graph_evidence(
        db=db,
        repo=MagicMock(spec=Neo4jRepository),
        project_id=uuid4(),
        query="does graph work",
        intent=GraphIntent.FACTUAL,
    )
    assert items == []
    assert outage is None
    assert recorder.spans[1][2].updates[-1]["metadata"]["outcome"] == "success"
    resolution = recorder.spans[2][2].updates[-1]["metadata"]
    assert resolution["candidate_fact_count"] == 1
    assert resolution["verified_evidence_count"] == 0
    assert resolution["outcome"] == "unresolved"


@pytest.mark.anyio
async def test_chat_service_answer_alias(db: Session):
    """Verify that chat_service.answer is an alias for chat_service.answer_question."""
    project = Project(name="Alias Test Project")
    db.add(project)
    db.commit()

    conv = create_conversation(db, project_id=project.id, title="Alias Conversation")

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = "This is a response."
    mock_llm.model_name = "test-model"

    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with patch("app.services.chat_service.get_llm_provider", return_value=mock_llm):
        chat_service = ChatService(retriever=mock_retriever)
        resp = await chat_service.answer(
            db=db,
            conversation_id=conv.id,
            question="Hello world?",
        )
        assert resp.content == (
            "Insufficient evidence available in the uploaded papers to answer this question."
        )
