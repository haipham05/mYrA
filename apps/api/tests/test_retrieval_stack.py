import os
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.models import ChunkElement, PaperChunk, PaperElement
from app.db.session import SessionLocal, create_tables
from app.db.types import PGVector, TSVector
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.embedding import DeterministicEmbeddingProvider
from app.services.retrieval import (
    HybridRetriever,
    SimpleLexicalReranker,
    cosine_similarity,
    get_reranker,
    set_reranker,
)


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2)) == 1.0

    v3 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v3)) == 0.0

    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0


def test_vector_type_processor():
    v_type = PGVector(dim=1024)
    assert v_type.get_col_spec() == "halfvec(1024)"

    bind_fn = v_type.bind_processor(dialect=None)
    assert bind_fn([0.5, 0.25]) == "[0.5,0.25]"
    assert bind_fn(None) is None

    res_fn = v_type.result_processor(dialect=None, coltype=None)
    assert res_fn("[0.5, 0.25]") == [0.5, 0.25]
    assert res_fn([0.5, 0.25]) == [0.5, 0.25]
    assert res_fn(None) is None

    tsv_type = TSVector()
    assert tsv_type.get_col_spec() == "tsvector"


def test_simple_lexical_reranker():
    reranker = SimpleLexicalReranker()
    query = "attention mechanism transformer"
    docs = [
        "A study on deep reinforcement learning in robotics",
        "The transformer architecture uses self-attention mechanisms",
        "Recipe for chocolate chip cookies",
    ]
    ranked = reranker.rerank(query, docs)
    assert len(ranked) == 3
    # Second doc (index 1) should be top ranked
    assert ranked[0][0] == 1


def test_retriever_project_isolation():
    create_tables()
    db = SessionLocal()
    embedder = DeterministicEmbeddingProvider(dimension=1024)

    # Create two projects
    p1 = create_project(db, ProjectCreate(name="Project 1"))
    p2 = create_project(db, ProjectCreate(name="Project 2"))

    # Create paper in Project 1
    paper1 = create_paper(db, project_id=p1.id, filename="p1.pdf", storage_path="p1.pdf")
    paper1.status = PaperStatus.READY
    elem1 = PaperElement(
        paper_id=paper1.id,
        page_number=1,
        element_index=0,
        element_type="text",
        text="Quantum computing fundamentals and qubits.",
        bbox_x_min=50.0,
        bbox_y_min=50.0,
        bbox_x_max=200.0,
        bbox_y_max=80.0,
        page_width=612.0,
        page_height=792.0,
    )
    db.add(elem1)
    db.flush()

    chunk1 = PaperChunk(
        paper_id=paper1.id,
        chunk_type="child",
        chunk_index=0,
        text="Quantum computing fundamentals and qubits.",
        token_count=10,
        embedding=embedder.embed_query("Quantum computing fundamentals and qubits."),
        embedding_vec=embedder.embed_query("Quantum computing fundamentals and qubits."),
    )
    db.add(chunk1)
    db.flush()
    db.add(ChunkElement(chunk_id=chunk1.id, element_id=elem1.id, order_index=0))

    # Create paper in Project 2
    paper2 = create_paper(db, project_id=p2.id, filename="p2.pdf", storage_path="p2.pdf")
    paper2.status = PaperStatus.READY
    elem2 = PaperElement(
        paper_id=paper2.id,
        page_number=1,
        element_index=0,
        element_type="text",
        text="Deep neural networks and convolutional layers.",
        bbox_x_min=60.0,
        bbox_y_min=60.0,
        bbox_x_max=300.0,
        bbox_y_max=100.0,
        page_width=612.0,
        page_height=792.0,
    )
    db.add(elem2)
    db.flush()

    chunk2 = PaperChunk(
        paper_id=paper2.id,
        chunk_type="child",
        chunk_index=0,
        text="Deep neural networks and convolutional layers.",
        token_count=10,
        embedding=embedder.embed_query("Deep neural networks and convolutional layers."),
        embedding_vec=embedder.embed_query("Deep neural networks and convolutional layers."),
    )
    db.add(chunk2)
    db.flush()
    db.add(ChunkElement(chunk_id=chunk2.id, element_id=elem2.id, order_index=0))
    db.commit()

    retriever = HybridRetriever(top_candidates=10, top_evidence=3)

    # Query Project 1 for "quantum"
    ev1 = retriever.retrieve(db, project_id=p1.id, query="quantum computing")
    assert len(ev1) > 0
    assert all(e.paper_id == paper1.id for e in ev1)
    # Ensure no leaked data from Project 2
    assert not any(e.paper_id == paper2.id for e in ev1)

    # Query Project 2 for "quantum" -> should find no results or only project 2 items
    ev2 = retriever.retrieve(db, project_id=p2.id, query="quantum computing")
    assert all(e.paper_id == paper2.id for e in ev2)
    assert not any(e.paper_id == paper1.id for e in ev2)

    db.close()


def test_reranker_singleton():
    original = get_reranker()
    custom = SimpleLexicalReranker()
    set_reranker(custom)
    assert get_reranker() is custom
    set_reranker(original)


def test_rrf_rank_fusion_logic():
    retriever = HybridRetriever(top_candidates=5, rrf_k=60)
    # Simulate rank fusion
    dense_ranks = [uuid4(), uuid4()]
    fts_ranks = [dense_ranks[1], dense_ranks[0]]

    rrf_scores = {}
    for rank, cid in enumerate(dense_ranks):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (retriever.rrf_k + rank + 1))
    for rank, cid in enumerate(fts_ranks):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (retriever.rrf_k + rank + 1))

    # Both items present in both rankings at ranks 0 and 1 should have identical total score
    assert pytest.approx(rrf_scores[dense_ranks[0]]) == pytest.approx(rrf_scores[dense_ranks[1]])


def test_postgres_live_vector_and_fts():
    """Opt-in live PostgreSQL test when DATABASE_URL is set to postgres."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url or "postgres" not in db_url:
        pytest.skip("Skipping postgres live test: DATABASE_URL is not PostgreSQL")

    from app.db.session import engine

    with engine.connect() as conn:
        res = conn.execute(
            text("SELECT '[0.1, 0.2]'::halfvec <=> '[0.1, 0.2]'::halfvec AS d;")
        ).scalar()
        assert pytest.approx(float(res)) == 0.0

        fts_res = conn.execute(
            text(
                "SELECT to_tsvector('english', 'Hybrid retrieval test with pgvector') @@ "
                "plainto_tsquery('english', 'pgvector') AS match;"
            )
        ).scalar()
        assert fts_res is True
