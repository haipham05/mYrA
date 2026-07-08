import json

import pytest

from app.services import embedding
from app.services.cache import JsonCache


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.ttls = {}
        self.error = None

    def get(self, key):
        if self.error:
            raise self.error
        return self.values.get(key)

    def set(self, key, value, *, ex):
        if self.error:
            raise self.error
        self.values[key] = value
        self.ttls[key] = ex
        return True


class CountingProvider(embedding.EmbeddingProvider):
    def __init__(self, model="test-model", revision="r1", dimension=3):
        self._model = model
        self._revision = revision
        self.dimension = dimension
        self.query_calls = 0
        self.document_batches = []

    @property
    def model_name(self):
        return self._model

    @property
    def model_version(self):
        return self._revision

    def embed_query(self, text):
        self.query_calls += 1
        seed = float(len(text))
        return [seed, seed + 1, seed + 2][: self.dimension]

    def embed_documents(self, texts):
        self.document_batches.append(list(texts))
        return [[float(len(text)), 2.0, 3.0][: self.dimension] for text in texts]


def install_cache(monkeypatch):
    redis = FakeRedis()
    cache = JsonCache(redis)
    monkeypatch.setattr(embedding, "get_cache", lambda: cache)
    return redis, cache


def test_query_embedding_hit_and_exact_model_identity(monkeypatch):
    redis, _ = install_cache(monkeypatch)
    provider = CountingProvider()
    first = embedding._cached_query(
        provider, "query text", lambda: provider.embed_query("query text")
    )
    again = embedding._cached_query(
        provider, "query text", lambda: provider.embed_query("query text")
    )

    assert first == again
    assert provider.query_calls == 1
    assert next(iter(redis.ttls.values())) == 24 * 60 * 60
    assert "query text" not in " ".join(redis.values)

    changed_revision = CountingProvider(revision="r2")
    embedding._cached_query(
        changed_revision,
        "query text",
        lambda: changed_revision.embed_query("query text"),
    )
    assert changed_revision.query_calls == 1


def test_document_batch_cache_preserves_positions_for_partial_hits(monkeypatch):
    redis, _ = install_cache(monkeypatch)
    provider = CountingProvider()
    first = embedding._cached_documents(provider, ["alpha", "beta"], provider.embed_documents)
    again = embedding._cached_documents(
        provider, ["beta", "gamma", "alpha"], provider.embed_documents
    )

    assert again == [first[1], [5.0, 2.0, 3.0], first[0]]
    assert provider.document_batches == [["alpha", "beta"], ["gamma"]]
    assert all(ttl == 24 * 60 * 60 for ttl in redis.ttls.values())


def test_document_input_hash_and_operation_are_distinct(monkeypatch):
    _, cache = install_cache(monkeypatch)
    provider = CountingProvider()
    embedding._cached_documents(provider, ["same"], provider.embed_documents)
    embedding._cached_query(provider, "same", lambda: provider.embed_query("same"))
    assert cache.stats.misses == 2
    assert len(cache._client.values) == 2


def test_invalid_cached_shape_is_recomputed(monkeypatch):
    redis, _ = install_cache(monkeypatch)
    provider = CountingProvider()
    key = embedding._cache_key(provider, "query", "query")
    redis.values[JsonCache._redis_key(key)] = json.dumps({"version": 1, "value": [1.0]})

    vector = embedding._cached_query(provider, "query", lambda: provider.embed_query("query"))

    assert len(vector) == 3
    assert provider.query_calls == 1


def test_non_finite_provider_output_is_rejected_and_not_cached(monkeypatch):
    redis, _ = install_cache(monkeypatch)
    provider = CountingProvider()
    with pytest.raises(ValueError, match="non-finite"):
        embedding._cached_query(provider, "bad", lambda: [float("nan"), 1.0, 2.0])
    assert redis.values == {}


def test_cache_outage_recomputes_and_returns_provider_output(monkeypatch):
    redis, _ = install_cache(monkeypatch)
    redis.error = TimeoutError("redis unavailable")
    provider = CountingProvider()

    assert embedding._cached_query(
        provider, "available", lambda: provider.embed_query("available")
    ) == [
        9.0,
        10.0,
        11.0,
    ]
    assert provider.query_calls == 1


def test_fresh_document_batch_must_match_requested_count(monkeypatch):
    install_cache(monkeypatch)
    with pytest.raises(RuntimeError, match="batch size"):
        embedding._cached_documents(
            CountingProvider(), ["one", "two"], lambda _texts: [[1.0, 2.0, 3.0]]
        )
