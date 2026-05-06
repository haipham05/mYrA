from pathlib import Path

import pytest

from app.services.embedding import (
    DeterministicEmbeddingProvider,
)
from app.services.llm import (
    FakeLLMProvider,
)
from app.services.retrieval import SimpleLexicalReranker, cosine_similarity
from app.storage.local import LocalStorage, MemoryStorage


@pytest.mark.anyio
async def test_local_storage(tmp_path):
    storage = LocalStorage(base_dir=str(tmp_path / "files"))
    key = "docs/test.pdf"
    content = b"%PDF-test"

    path = await storage.put(key, content)
    assert Path(path).exists()

    retrieved = await storage.get(key)
    assert retrieved == content

    await storage.delete(key)
    with pytest.raises(FileNotFoundError):
        await storage.get(key)


@pytest.mark.anyio
async def test_memory_storage():
    storage = MemoryStorage()
    key = "mem/test.pdf"
    content = b"pdf-bytes"

    await storage.put(key, content)
    retrieved = await storage.get(key)
    assert retrieved == content

    await storage.delete(key)
    with pytest.raises(FileNotFoundError):
        await storage.get(key)


def test_embedding_provider():
    provider = DeterministicEmbeddingProvider(dimension=1024)
    vec = provider.embed_query("machine learning")
    assert len(vec) == 1024
    docs = provider.embed_documents(["doc one", "doc two"])
    assert len(docs) == 2
    assert len(docs[0]) == 1024

    # Test empty string graceful
    empty_vec = provider.embed_query("")
    assert len(empty_vec) == 1024

    # Cosine similarity edge cases
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([1.0], [0.0]) == 0.0
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-6


def test_reranker_and_llm():
    reranker = SimpleLexicalReranker()
    scores = reranker.rerank(
        "transformer attention",
        ["attention model", "completely unrelated text"],
    )
    assert scores[0][0] == 0
    assert scores[0][1] > scores[1][1]


@pytest.mark.anyio
async def test_fake_llm():
    llm = FakeLLMProvider()
    resp = await llm.generate("system", "Here is [E1] evidence")
    assert "[E1]" in resp

    no_ev_resp = await llm.generate("system", "No evidence here")
    assert "Insufficient" in no_ev_resp

    custom_llm = FakeLLMProvider(fixed_response="Custom Answer")
    assert await custom_llm.generate("s", "u") == "Custom Answer"
