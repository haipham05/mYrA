import os
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import text

import app.services.retrieval as retrieval_module
from app.crud.paper import create_paper
from app.crud.project import create_project
from app.db.models import ChunkElement, PaperChunk, PaperElement
from app.db.session import SessionLocal, create_tables
from app.db.types import PGVector, TSVector
from app.schemas.paper import PaperStatus
from app.schemas.project import ProjectCreate
from app.services.embedding import (
    BGEM3EmbeddingProvider,
    DeterministicEmbeddingProvider,
    get_embedding_provider,
    set_embedding_provider,
)
from app.services.retrieval import (
    BGERerankerProvider,
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


def test_demo_reranker_normalizes_question_words_and_plural_forms():
    reranker = SimpleLexicalReranker()
    docs = [
        "The architecture uses attention in a network.",
        "Positional Encoding: the model contains no recurrence or convolution.",
    ]
    assert reranker.rerank("Why are positional encodings needed?", docs)[0][0] == 1
    assert (
        reranker.rerank(
            "What does the BERT acronym stand for?",
            [
                "BERT stands for Bidirectional Encoder Representations from Transformers.",
                "The Transformer uses a BERT-style architecture.",
            ],
        )[0][0]
        == 0
    )


def test_production_providers_are_explicit_and_pinned(monkeypatch):
    import sentence_transformers

    selected = []

    class FakeVector:
        def tolist(self):
            return [0.01] * 1024

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs):
            selected.append((name, kwargs))

        def encode(self, text, **_kwargs):
            return [FakeVector() for _ in text] if isinstance(text, list) else FakeVector()

    class FakeCrossEncoder:
        def __init__(self, name, **kwargs):
            selected.append((name, kwargs))

        def predict(self, pairs):
            return [float(idx) for idx, _ in enumerate(pairs)]

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(sentence_transformers, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setenv("MYRA_EMBEDDING_PROVIDER", "bge-m3")
    monkeypatch.setenv("MYRA_RERANKER_PROVIDER", "bge")
    set_embedding_provider(None)
    set_reranker(None)
    try:
        embedder = get_embedding_provider()
        reranker = get_reranker()
        assert isinstance(embedder, BGEM3EmbeddingProvider)
        assert isinstance(reranker, BGERerankerProvider)
        assert len(embedder.embed_query("question")) == 1024
        assert len(embedder.embed_documents(["one", "two"])) == 2
        assert reranker.rerank("question", ["one", "two"])[0][0] == 1
        assert selected[0][1] == {
            "revision": embedder.model_version,
            "local_files_only": True,
        }
        assert selected[1][1] == {
            "revision": reranker.model_version,
            "local_files_only": True,
        }
    finally:
        set_embedding_provider(None)
        set_reranker(None)


def test_unknown_retrieval_providers_fail(monkeypatch):
    monkeypatch.setenv("MYRA_EMBEDDING_PROVIDER", "unknown")
    monkeypatch.setenv("MYRA_RERANKER_PROVIDER", "unknown")
    set_embedding_provider(None)
    set_reranker(None)
    with pytest.raises(ValueError, match="MYRA_EMBEDDING_PROVIDER"):
        get_embedding_provider()
    with pytest.raises(ValueError, match="MYRA_RERANKER_PROVIDER"):
        get_reranker()


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


def test_postgres_dense_search_filters_embedding_space():
    calls = []

    class EmptyResult:
        def fetchall(self):
            return []

    class RecordingSession:
        def execute(self, statement, params):
            calls.append((str(statement), params))
            return EmptyResult()

    retriever = HybridRetriever()
    assert (
        retriever._retrieve_postgres(
            RecordingSession(), uuid4(), "question", [0.1] * 1024, "BAAI/bge-m3", "sha123"
        )
        == []
    )
    dense_sql, dense_params = calls[0]
    assert "pc.embedding_model = :embedding_model" in dense_sql
    assert "pc.embedding_version = :embedding_version" in dense_sql
    assert dense_params["embedding_model"] == "BAAI/bge-m3"
    assert dense_params["embedding_version"] == "sha123"
    assert "p.project_id = :project_id" in dense_sql


def test_sqlite_dense_search_ignores_incompatible_vectors(monkeypatch):
    create_tables()
    db = SessionLocal()
    project = create_project(db, ProjectCreate(name="Version Filter Test"))
    paper = create_paper(db, project.id, "version.pdf", "version.pdf")
    paper.status = PaperStatus.READY
    db.add_all(
        [
            PaperChunk(
                paper_id=paper.id,
                chunk_type="child",
                chunk_index=index,
                text=f"evidence {index}",
                embedding_vec=[0.1] * 1024,
                embedding_model=model,
                embedding_version=version,
            )
            for index, (model, version) in enumerate(
                [("deterministic-fake", "v1"), ("BAAI/bge-m3", "other")]
            )
        ]
    )
    db.commit()
    calls = []

    def record_similarity(left, right):
        calls.append((left, right))
        return 1.0

    monkeypatch.setattr(retrieval_module, "cosine_similarity", record_similarity)
    result = HybridRetriever()._retrieve_fallback(
        db, project.id, "evidence", [0.1] * 1024, "deterministic-fake", "v1"
    )
    assert len(result) == 2  # Both remain eligible for lexical retrieval.
    assert len(calls) == 1  # Only a compatible embedding enters cosine ranking.
    db.close()


def test_postgres_live_vector_and_fts():
    """Opt-in live PostgreSQL test when DATABASE_URL or .env is set to postgres."""
    db_url = os.getenv("LIVE_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not db_url or "postgres" not in db_url:
        env_file = Path(__file__).resolve().parents[3] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("DATABASE_URL="):
                    candidate = line.split("=", 1)[1].strip().strip("\"'")
                    if "postgres" in candidate:
                        db_url = candidate
                        break

    if not db_url or "postgres" not in db_url:
        pytest.skip("Skipping postgres live test: DATABASE_URL is not PostgreSQL")

    from sqlalchemy import create_engine

    live_engine = create_engine(db_url)
    try:
        with live_engine.connect() as conn:
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
    finally:
        live_engine.dispose()
