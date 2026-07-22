import os
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

import app.services.retrieval as retrieval_module
from app.crud.corpus import bump_corpus_revision, has_pending_corpus_revision
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


def test_retrieve_forwards_shared_query_embedding(monkeypatch):
    retriever = HybridRetriever()
    observed = {}

    def fake_impl(
        db, project_id, query, telemetry, strategy, *, query_embedding=None, selected_paper_ids=None
    ):
        observed.update(
            db=db,
            project_id=project_id,
            query=query,
            strategy=strategy,
            query_embedding=query_embedding,
            selected_paper_ids=selected_paper_ids,
        )
        return []

    monkeypatch.setattr(retrieval_module, "get_telemetry", lambda: None)
    monkeypatch.setattr(retriever, "_retrieve_impl", fake_impl)
    vector = [0.1, 0.2, 0.3]

    assert retriever.retrieve(None, uuid4(), "shared query", query_embedding=vector) == []
    assert observed["query_embedding"] == vector
    assert observed["query"] == "shared query"


def test_retrieval_observation_records_metadata_without_content():
    recorded = []

    class Observation:
        def update(self, **kwargs):
            recorded.append(kwargs)

    class Telemetry:
        @contextmanager
        def stage(self, name, *, metadata):
            assert name == "retrieval.dense_search"
            assert "query" not in metadata
            yield Observation()

    with retrieval_module._retrieval_observation(
        Telemetry(), "retrieval.dense_search", {"backend": "test", "candidate_limit": 5}
    ) as observation:
        observation.update(metadata={"candidate_count": 2})

    result = recorded[0]["metadata"]
    assert result["candidate_count"] == 2
    assert result["candidate_limit"] == 5
    assert result["outcome"] == "success"
    assert result["duration_ms"] >= 0
    assert not any("text" in key or "query" in key for key in result)


def test_candidate_trace_item_sanitizes_before_truncating():
    text_value = "contact test@example.com; " + ("evidence " * 80)
    item = retrieval_module._candidate_trace_item(
        chunk_id=uuid4(),
        paper_id=uuid4(),
        text_value=text_value,
        rank=1,
        score_name="cosine_similarity",
        score=0.75,
    )
    assert len(item["text_prefix"]) <= retrieval_module._TRACE_PREFIX_CHARS
    assert "test@example.com" not in item["text_prefix"]
    assert "[EMAIL REDACTED]" in item["text_prefix"]
    assert item["text_truncated"] is True
    assert item["score"] == {"metric": "cosine_similarity", "value": 0.75}


def test_retrieval_observation_failure_does_not_change_product_exception():
    class BrokenTelemetry:
        def stage(self, *_args, **_kwargs):
            raise RuntimeError("telemetry unavailable")

    expected = ValueError("retrieval failed")
    with pytest.raises(ValueError) as raised:
        with retrieval_module._retrieval_observation(BrokenTelemetry(), "retrieval", {}):
            raise expected
    assert raised.value is expected


def test_reranking_cache_uses_model_and_ordered_content_hashes(monkeypatch):
    class Cache:
        values = {}
        keys = []

        def get(self, key, validator):
            self.keys.append(key)
            value = self.values.get(key)
            return validator(value) if value is not None else None

        def set(self, key, value, *, ttl_seconds):
            self.keys.append(key)
            assert ttl_seconds == 3600
            self.values[key] = value
            return True

    class Reranker:
        model_name = "test-reranker"
        model_version = "rev-1"
        calls = 0

        def rerank(self, query, documents):
            self.calls += 1
            return [(1, 0.9), (0, 0.2)]

    cache = Cache()
    reranker = Reranker()
    monkeypatch.setattr(retrieval_module, "get_cache", lambda: cache)

    class Observation:
        def __init__(self):
            self.metadata = {}

        def update(self, *, metadata=None, **_kwargs):
            if metadata:
                self.metadata.update(metadata)

    expected = [(1, 0.9), (0, 0.2)]
    docs = ["secret-document-A", "secret-document-B"]
    miss_observation = Observation()
    assert (
        retrieval_module._rerank_with_cache(
            "private query", docs, reranker, observation=miss_observation
        )
        == expected
    )
    hit_observation = Observation()
    assert (
        retrieval_module._rerank_with_cache(
            "private query", docs, reranker, observation=hit_observation
        )
        == expected
    )
    assert miss_observation.metadata["cache_status"] == "miss"
    assert hit_observation.metadata["cache_status"] == "hit"
    assert reranker.calls == 1

    retrieval_module._rerank_with_cache("private query", list(reversed(docs)), reranker)
    reranker.model_version = "rev-2"
    retrieval_module._rerank_with_cache("private query", docs, reranker)
    assert reranker.calls == 3
    assert all("private query" not in key and "secret-document" not in key for key in cache.keys)


def test_reranking_cache_rejects_invalid_cached_indexes_and_recomputes(monkeypatch):
    class Cache:
        def get(self, _key, validator):
            try:
                return validator([[3, 0.5]])
            except ValueError:
                return None

        def set(self, *_args, **_kwargs):
            return True

    class Reranker:
        model_name = "test-reranker"
        model_version = "rev-1"

        def rerank(self, _query, _documents):
            return [(0, 0.5)]

    monkeypatch.setattr(retrieval_module, "get_cache", lambda: Cache())
    assert retrieval_module._rerank_with_cache("question", ["only"], Reranker()) == [(0, 0.5)]


def test_candidate_cache_key_scopes_query_project_revision_model_and_policy():
    common = {
        "project_id": uuid4(),
        "corpus_revision": 4,
        "query": "exact query",
        "backend": "fallback",
        "embedding_model": "embedder",
        "embedding_version": "rev-1",
        "strategy": "hybrid-reranked",
        "candidate_limit": 40,
        "rrf_k": 60,
    }
    key = retrieval_module._candidate_cache_key(**common)
    assert "exact query" not in key
    for field, changed in (
        ("project_id", uuid4()),
        ("corpus_revision", 5),
        ("query", "different query"),
        ("embedding_version", "rev-2"),
        ("embedding_model", "other-model"),
        ("selected_paper_ids", (str(uuid4()),)),
    ):
        variant = {**common, field: changed}
        assert retrieval_module._candidate_cache_key(**variant) != key


def test_candidate_cache_id_validation_rejects_duplicates_and_overflow():
    item = str(uuid4())
    assert retrieval_module._validate_candidate_ids([item], limit=2) == [UUID(item)]
    with pytest.raises(ValueError):
        retrieval_module._validate_candidate_ids([item, item], limit=2)
    with pytest.raises(ValueError):
        retrieval_module._validate_candidate_ids([item, str(uuid4())], limit=1)


def test_candidate_cache_hit_hydrates_current_rows_and_misses_ineligible_ids():
    create_tables()
    db = SessionLocal()
    project = create_project(db, ProjectCreate(name="Candidate Cache"))
    paper = create_paper(db, project.id, "candidate.pdf", "candidate.pdf")
    paper.status = PaperStatus.READY
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text="cached candidate evidence",
        embedding_model="embedder",
        embedding_version="rev-1",
    )
    db.add(chunk)
    db.commit()

    hydrated = HybridRetriever._hydrate_candidate_ids(db, project.id, [chunk.id])
    assert hydrated is not None and [item.id for item in hydrated] == [chunk.id]
    assert (
        HybridRetriever._hydrate_candidate_ids(
            db, project.id, [chunk.id], selected_paper_ids=(uuid4(),)
        )
        is None
    )

    paper.status = "PROCESSING"
    db.commit()
    assert HybridRetriever._hydrate_candidate_ids(db, project.id, [chunk.id]) is None
    assert HybridRetriever._hydrate_candidate_ids(db, project.id, [uuid4()]) is None
    db.close()


def test_corpus_revision_bump_marks_session_until_commit_or_rollback():
    create_tables()
    db = SessionLocal()
    project = create_project(db, ProjectCreate(name="Corpus Revision"))
    db.commit()
    bump_corpus_revision(db, project.id)
    assert has_pending_corpus_revision(db)
    db.commit()
    assert not has_pending_corpus_revision(db)

    bump_corpus_revision(db, project.id)
    db.rollback()
    assert not has_pending_corpus_revision(db)
    db.close()


def test_retrieval_reuses_candidate_ids_without_caching_text(monkeypatch):
    create_tables()
    db = SessionLocal()
    project = create_project(db, ProjectCreate(name="Candidate Hit"))
    paper = create_paper(db, project.id, "candidate-hit.pdf", "candidate-hit.pdf")
    paper.status = PaperStatus.READY
    embedder = DeterministicEmbeddingProvider(dimension=1024)
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text="attention evidence text",
        embedding_vec=embedder.embed_query("attention question"),
        embedding_model=embedder.model_name,
        embedding_version=embedder.model_version,
    )
    db.add(chunk)
    db.commit()

    class Cache:
        enabled = True

        def __init__(self):
            self.values = {}
            self.read_keys = []
            self.written = []

        def get(self, key, validator):
            self.read_keys.append(key)
            value = self.values.get(key)
            return validator(value) if value is not None else None

        def set(self, key, value, *, ttl_seconds):
            self.written.append((key, value, ttl_seconds))
            self.values[key] = value
            return True

    cache = Cache()
    monkeypatch.setattr(retrieval_module, "get_cache", lambda: cache)
    trace_events = []

    class TraceObservation:
        def __init__(self, name):
            self.name = name

        def update(self, **kwargs):
            trace_events.append({"name": self.name, **kwargs})

    class TraceTelemetry:
        @contextmanager
        def stage(self, name, *, metadata=None, **_kwargs):
            trace_events.append({"name": name, "metadata": metadata or {}})
            yield TraceObservation(name)

        def event(self, name, *, metadata=None, **_kwargs):
            trace_events.append({"name": name, "metadata": metadata or {}})

    monkeypatch.setattr(retrieval_module, "get_telemetry", lambda: TraceTelemetry())
    set_embedding_provider(embedder)
    set_reranker(SimpleLexicalReranker())
    retriever = HybridRetriever(top_candidates=10)
    retrieval_calls = 0
    original_retrieve = retriever._retrieve_fallback

    def count_retrieval(*args, **kwargs):
        nonlocal retrieval_calls
        retrieval_calls += 1
        return original_retrieve(*args, **kwargs)

    monkeypatch.setattr(retriever, "_retrieve_fallback", count_retrieval)
    first = retriever.retrieve(db, project.id, "attention question", strategy="hybrid-unreranked")
    trace_events.clear()
    second = retriever.retrieve(db, project.id, "attention question", strategy="hybrid-unreranked")

    assert first and second
    assert retrieval_calls == 1
    assert len(cache.written) == 1
    _, value, ttl = cache.written[0]
    assert value == [str(chunk.id)]
    assert ttl == 300
    assert "attention question" not in cache.written[0][0]
    assert "attention evidence text" not in repr(value)
    cache_observation = next(
        event for event in reversed(trace_events) if event["name"] == "retrieval.candidate_cache"
    )
    candidate_output = cache_observation["output"]["candidates"][0]
    assert candidate_output["chunk_id"] == str(chunk.id)
    assert candidate_output["text_prefix"] == "attention evidence text"
    assert candidate_output["score"]["value"] is None
    assert cache_observation["metadata"]["cache_status"] == "hit"
    assert cache_observation["metadata"]["skipped_stages"] == [
        "retrieval.dense_search",
        "retrieval.fts_search",
        "retrieval.fusion",
    ]
    second_stage_names = {event["name"] for event in trace_events if "name" in event}
    assert "retrieval.dense_search" not in second_stage_names
    assert "retrieval.fts_search" not in second_stage_names
    assert "retrieval.fusion" not in second_stage_names
    db.close()
    set_embedding_provider(None)
    set_reranker(None)


@pytest.mark.parametrize(
    ("revisions", "expected_calls", "expected_evidence"),
    [([5, 6, 6, 6], 2, True), ([5, 6, 6, 7], 2, False)],
)
def test_retrieval_retries_corpus_revision_race_once(
    monkeypatch, revisions, expected_calls, expected_evidence
):
    create_tables()
    db = SessionLocal()
    project = create_project(db, ProjectCreate(name="Revision Race"))
    paper = create_paper(db, project.id, "revision-race.pdf", "revision-race.pdf")
    paper.status = PaperStatus.READY
    embedder = DeterministicEmbeddingProvider(dimension=1024)
    chunk = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text="revision race evidence",
        embedding_vec=embedder.embed_query("question"),
        embedding_model=embedder.model_name,
        embedding_version=embedder.model_version,
    )
    db.add(chunk)
    db.commit()

    class Cache:
        enabled = True

        def __init__(self):
            self.values = {}
            self.writes = []

        def get(self, key, validator):
            return None

        def set(self, key, value, *, ttl_seconds):
            self.writes.append((key, value, ttl_seconds))

    cache = Cache()
    revision_iter = iter(revisions)
    monkeypatch.setattr(retrieval_module, "get_cache", lambda: cache)
    monkeypatch.setattr(retrieval_module, "read_corpus_revision", lambda *_: next(revision_iter))
    set_embedding_provider(embedder)
    retriever = HybridRetriever(top_candidates=10)
    calls = 0

    def candidates(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return [chunk]

    monkeypatch.setattr(retriever, "_retrieve_fallback", candidates)
    result = retriever.retrieve(db, project.id, "question", strategy="hybrid-unreranked")

    assert calls == expected_calls
    assert bool(result) is expected_evidence
    assert len(cache.writes) == int(expected_evidence)
    if expected_evidence:
        assert '"corpus_revision":6' in cache.writes[0][0]
    db.close()
    set_embedding_provider(None)


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


def test_retriever_project_isolation(monkeypatch):
    observed = []

    class RecordingObservation:
        def __init__(self, name):
            self.name = name

        def update(self, **kwargs):
            observed.append({"name": self.name, **kwargs})

    class RecordingTelemetry:
        @contextmanager
        def stage(self, name, *, metadata=None, **_kwargs):
            observed.append({"name": name, **(metadata or {})})
            yield RecordingObservation(name)

        def event(self, name, *, metadata=None, **_kwargs):
            observed.append({"name": name, **(metadata or {})})

    monkeypatch.setattr(retrieval_module, "get_telemetry", lambda: RecordingTelemetry())
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
        embedding_model=embedder.model_name,
        embedding_version=embedder.model_version,
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
    ev1_explicit_default = retriever.retrieve(
        db, project_id=p1.id, query="quantum computing", strategy="hybrid-reranked"
    )
    assert len(ev1) > 0
    assert [(item.paper_id, item.chunk_id, item.page_number, item.quote) for item in ev1] == [
        (item.paper_id, item.chunk_id, item.page_number, item.quote)
        for item in ev1_explicit_default
    ]
    assert all(e.paper_id == paper1.id for e in ev1)
    # Ensure no leaked data from Project 2
    assert not any(e.paper_id == paper2.id for e in ev1)
    stage_names = {entry.get("name") for entry in observed if isinstance(entry, dict)}
    assert {
        "retrieval",
        "retrieval.query_embedding",
        "retrieval.dense_search",
        "retrieval.fts_search",
        "retrieval.fusion",
        "retrieval.reranking",
        "retrieval.evidence_built",
    } <= stage_names
    query_embedding = next(
        entry for entry in reversed(observed) if entry.get("name") == "retrieval.query_embedding"
    )
    assert query_embedding["input"] == {"question": "quantum computing"}
    dense = next(
        entry for entry in reversed(observed) if entry.get("name") == "retrieval.dense_search"
    )
    dense_candidates = dense["output"]["candidates"]
    assert dense_candidates[0]["chunk_id"] == str(chunk1.id)
    assert dense_candidates[0]["paper_id"] == str(paper1.id)
    assert dense_candidates[0]["score"]["metric"] == "cosine_similarity"
    assert dense_candidates[0]["text_prefix"].startswith("Quantum computing")
    fts = next(entry for entry in reversed(observed) if entry.get("name") == "retrieval.fts_search")
    assert fts["output"]["candidates"]
    assert fts["output"]["candidates"][0]["score"]["metric"] == "term_match_count"
    fusion = next(entry for entry in reversed(observed) if entry.get("name") == "retrieval.fusion")
    assert fusion["input"]["dense_candidates"]
    assert fusion["input"]["fts_candidates"]
    assert fusion["output"]["candidates"][0]["score"]["metric"] == ("reciprocal_rank_fusion")
    reranking = next(
        entry for entry in reversed(observed) if entry.get("name") == "retrieval.reranking"
    )
    assert reranking["input"]["question"] == "quantum computing"
    assert reranking["output"]["candidates"]
    assert reranking["output"]["candidates"][0]["score"]["metric"] == "reranker_score"

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
    stages = {}

    class EmptyResult:
        def fetchall(self):
            return []

    class Observation:
        def __init__(self, name):
            self.name = name

        def update(self, **kwargs):
            stages[self.name] = kwargs

    class Telemetry:
        @contextmanager
        def stage(self, name, **_kwargs):
            yield Observation(name)

    class RecordingSession:
        def execute(self, statement, params):
            calls.append((str(statement), params))
            return EmptyResult()

    retriever = HybridRetriever()
    assert (
        retriever._retrieve_postgres(
            RecordingSession(),
            uuid4(),
            "question",
            [0.1] * 1024,
            "BAAI/bge-m3",
            "sha123",
            telemetry=Telemetry(),
        )
        == []
    )
    dense_sql, dense_params = calls[0]
    assert "pc.embedding_model = :embedding_model" in dense_sql
    assert "pc.embedding_version = :embedding_version" in dense_sql
    assert dense_params["embedding_model"] == "BAAI/bge-m3"
    assert dense_params["embedding_version"] == "sha123"
    assert "p.project_id = :project_id" in dense_sql
    assert stages["retrieval.dense_search"]["metadata"]["candidate_count"] == 0
    assert stages["retrieval.dense_search"]["output"] == {"candidates": []}
    assert stages["retrieval.fts_search"]["metadata"]["candidate_count"] == 0
    assert stages["retrieval.fts_search"]["output"] == {"candidates": []}


def test_postgres_scope_is_applied_to_dense_and_fts_before_limits():
    selected_id = uuid4()
    calls = []

    class EmptyResult:
        def fetchall(self):
            return []

    class RecordingSession:
        def execute(self, statement, params):
            calls.append((str(statement), params))
            return EmptyResult()

    result = HybridRetriever()._retrieve_postgres(
        RecordingSession(),
        uuid4(),
        "question",
        [0.1] * 1024,
        "BAAI/bge-m3",
        "revision-1",
        selected_paper_ids=(selected_id,),
    )
    assert result == []
    assert len(calls) == 2
    for sql, params in calls:
        assert "p.id IN" in sql
        assert params["selected_paper_ids"] == (selected_id,)


def test_postgres_trace_reports_native_scores_in_search_order():
    dense_id, fts_id = uuid4(), uuid4()
    paper_id = uuid4()
    stages = {}

    class Result:
        def __init__(self, rows):
            self.rows = rows

        def fetchall(self):
            return self.rows

    class Observation:
        def __init__(self, name):
            self.name = name

        def update(self, **kwargs):
            stages[self.name] = kwargs

    class Telemetry:
        @contextmanager
        def stage(self, name, **_kwargs):
            yield Observation(name)

    class Query:
        def filter(self, *_args):
            return self

        def all(self):
            return [type("Chunk", (), {"id": fts_id})(), type("Chunk", (), {"id": dense_id})()]

    class RecordingSession:
        def __init__(self):
            self.calls = []

        def execute(self, statement, _params):
            sql = str(statement)
            self.calls.append(sql)
            if len(self.calls) == 1:
                return Result(
                    [
                        (dense_id, paper_id, "dense first", 0.91),
                        (fts_id, paper_id, "dense second", 0.72),
                    ]
                )
            return Result(
                [
                    (fts_id, paper_id, "fts first", 0.44),
                    (dense_id, paper_id, "fts second", 0.13),
                ]
            )

        def query(self, _model):
            return Query()

    session = RecordingSession()
    result = HybridRetriever(top_candidates=5)._retrieve_postgres(
        session,
        uuid4(),
        "attention",
        [0.1] * 1024,
        "BAAI/bge-m3",
        "revision-1",
        telemetry=Telemetry(),
    )

    assert [chunk.id for chunk in result] == [dense_id, fts_id]
    dense = stages["retrieval.dense_search"]["output"]["candidates"]
    fts = stages["retrieval.fts_search"]["output"]["candidates"]
    assert [candidate["chunk_id"] for candidate in dense] == [str(dense_id), str(fts_id)]
    assert [candidate["score"]["value"] for candidate in dense] == [0.91, 0.72]
    assert [candidate["chunk_id"] for candidate in fts] == [str(fts_id), str(dense_id)]
    assert [candidate["score"] for candidate in fts] == [
        {"metric": "postgres_fts_rank", "value": 0.44},
        {"metric": "postgres_fts_rank", "value": 0.13},
    ]
    assert "1.0 - (pc.embedding_vec <=> :query_vec) AS cosine_similarity" in session.calls[0]
    assert "ORDER BY pc.embedding_vec <=> :query_vec ASC" in session.calls[0]
    assert "ts_rank(pc.tsv_content" in session.calls[1]
    assert "ORDER BY ts_rank(pc.tsv_content" in session.calls[1]


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


def test_sqlite_scope_filters_before_candidate_ranking():
    create_tables()
    db = SessionLocal()
    project = create_project(db, ProjectCreate(name="Scoped retrieval"))
    included = create_paper(db, project.id, "included.pdf", "included")
    excluded = create_paper(db, project.id, "excluded.pdf", "excluded")
    included.status = excluded.status = PaperStatus.READY
    embedder = DeterministicEmbeddingProvider(dimension=1024)
    included_chunk = PaperChunk(
        paper_id=included.id,
        chunk_type="child",
        chunk_index=0,
        text="selected paper contains answer",
        embedding_vec=embedder.embed_query("selected paper answer"),
        embedding_model="deterministic-fake",
        embedding_version="v1",
    )
    excluded_chunk = PaperChunk(
        paper_id=excluded.id,
        chunk_type="child",
        chunk_index=0,
        text="excluded paper contains answer",
        embedding_vec=embedder.embed_query("selected paper answer"),
        embedding_model="deterministic-fake",
        embedding_version="v1",
    )
    db.add_all([included_chunk, excluded_chunk])
    db.commit()

    result = HybridRetriever(top_candidates=10)._retrieve_fallback(
        db,
        project.id,
        "answer",
        embedder.embed_query("answer"),
        "deterministic-fake",
        "v1",
        selected_paper_ids=(included.id,),
    )
    assert [chunk.id for chunk in result] == [included_chunk.id]
    db.close()


def test_dense_only_ablation_skips_lexical_candidates_and_reranking(monkeypatch):
    create_tables()
    db = SessionLocal()
    project = create_project(db, ProjectCreate(name="Retrieval Ablation"))
    paper = create_paper(db, project.id, "ablation.pdf", "ablation.pdf")
    paper.status = PaperStatus.READY
    embedder = DeterministicEmbeddingProvider(dimension=1024)
    matching = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text="attention vector source",
        embedding_vec=embedder.embed_query("attention vector source"),
        embedding_model="deterministic-fake",
        embedding_version="v1",
    )
    lexical_only = PaperChunk(
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=1,
        text="attention lexical-only source",
        embedding_vec=[0.0] * 1024,
        embedding_model="other-model",
        embedding_version="v2",
    )
    db.add_all([matching, lexical_only])
    db.commit()

    retriever = HybridRetriever(top_candidates=10)
    dense = retriever._retrieve_fallback(
        db,
        project.id,
        "attention vector",
        embedder.embed_query("attention vector"),
        "deterministic-fake",
        "v1",
        strategy="dense-only",
    )
    hybrid = retriever._retrieve_fallback(
        db,
        project.id,
        "attention vector",
        embedder.embed_query("attention vector"),
        "deterministic-fake",
        "v1",
        strategy="hybrid-unreranked",
    )
    assert [chunk.id for chunk in dense] == [matching.id]
    assert lexical_only.id not in {chunk.id for chunk in dense}
    assert lexical_only.id in {chunk.id for chunk in hybrid}
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


def test_project_12_attention_scoped_postgres_retrieval():
    """Opt-in, read-only native retrieval smoke scoped to the approved paper."""
    if os.getenv("MYRA_RUN_PROJECT12_POSTGRES_SCOPE_SMOKE") != "1":
        pytest.skip("Set MYRA_RUN_PROJECT12_POSTGRES_SCOPE_SMOKE=1 for the bounded live read")

    project_id = UUID("0f726c17-4f23-4914-ab0a-aefb31f05745")
    attention_paper_id = UUID("0a9a9b4f-b0cd-4326-ac8e-b28d19577f18")
    database_url = os.getenv("LIVE_DATABASE_URL")
    if not database_url:
        env_file = Path(__file__).resolve().parents[3] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("DATABASE_URL="):
                    database_url = line.split("=", 1)[1].strip().strip("\"'")
                    break
    if not database_url or "postgres" not in database_url:
        pytest.skip("A PostgreSQL LIVE_DATABASE_URL is required for this read-only smoke")

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    live_engine = create_engine(database_url)
    try:
        with sessionmaker(bind=live_engine)() as live_db:
            results = HybridRetriever(top_candidates=20)._retrieve_postgres(
                db=live_db,
                project_id=project_id,
                query="attention self-attention transformer",
                query_vec=[0.001] * 1024,
                embedding_model="phase-c-smoke-no-dense-match",
                embedding_version="phase-c-smoke-only",
                strategy="hybrid-unreranked",
                selected_paper_ids=(attention_paper_id,),
            )
        assert results, "The approved Attention paper should match the bounded FTS query"
        assert {chunk.paper_id for chunk in results} == {attention_paper_id}
    finally:
        live_engine.dispose()
