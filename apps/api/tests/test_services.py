import httpx
import pytest

from app.config import Settings
from app.services.embedding import (
    DeterministicEmbeddingProvider,
)
from app.services.llm import (
    DeepSeekLLMProvider,
    FakeLLMProvider,
    get_llm_provider,
    set_llm_provider,
)
from app.services.retrieval import SimpleLexicalReranker, cosine_similarity
from app.storage.local import LocalStorage, MemoryStorage


@pytest.mark.anyio
async def test_local_storage(tmp_path):
    storage = LocalStorage(base_dir=str(tmp_path / "files"))
    key = "docs/test.pdf"
    content = b"%PDF-test"

    path = await storage.put(key, content)
    assert path == key
    assert await storage.exists(path)
    assert await storage.get(str(tmp_path / "files" / key)) == content

    with pytest.raises(ValueError, match="outside the local storage"):
        await storage.get("../escape.pdf")

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
    resp = await llm.generate(
        "system", '[E1] (From: Paper, Page 1):\nEvidence quote: "The model improved recall."'
    )
    assert "[E1]" in resp
    assert resp.startswith("The model improved recall")

    no_ev_resp = await llm.generate("system", "No evidence here")
    assert "Insufficient" in no_ev_resp

    custom_llm = FakeLLMProvider(fixed_response="Custom Answer")
    assert await custom_llm.generate("s", "u") == "Custom Answer"


def test_generation_mode_requires_explicit_fake_or_deepseek_key(monkeypatch):
    set_llm_provider(None)
    try:
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
            get_llm_provider(settings=Settings(deepseek_api_key=None), mode="deepseek")
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
            get_llm_provider(settings=Settings(deepseek_api_key=None), mode="production")

        assert isinstance(
            get_llm_provider(settings=Settings(deepseek_api_key="unused"), mode="test"),
            FakeLLMProvider,
        )
        set_llm_provider(None)
        assert isinstance(
            get_llm_provider(settings=Settings(deepseek_api_key="unused"), mode="demo"),
            FakeLLMProvider,
        )
        set_llm_provider(None)
        assert isinstance(
            get_llm_provider(settings=Settings(deepseek_api_key="unused"), mode="deepseek"),
            DeepSeekLLMProvider,
        )
        set_llm_provider(None)
        monkeypatch.setenv("MYRA_LLM_MODE", "test")
        assert isinstance(get_llm_provider(), FakeLLMProvider)
        set_llm_provider(None)
        with pytest.raises(ValueError, match="Unknown MYRA_LLM_MODE"):
            get_llm_provider(settings=Settings(), mode="unknown")
    finally:
        set_llm_provider(None)


@pytest.mark.anyio
async def test_deepseek_request_uses_configured_endpoint_without_network(monkeypatch):
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://example.invalid/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert b"Only source evidence" in request.content
        return httpx.Response(200, json={"choices": [{"message": {"content": "Cited answer"}}]})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    provider = DeepSeekLLMProvider("test-key", "https://example.invalid/")
    assert provider.provider_name == "deepseek"
    assert await provider.generate("Only source evidence", "Question") == "Cited answer"
