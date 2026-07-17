import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest

from app.config import Settings
from app.observability.context import OperationContext, use_operation_context
from app.services.embedding import (
    DeterministicEmbeddingProvider,
)
from app.services.llm import (
    DeepSeekLLMProvider,
    FakeLLMProvider,
    GenerationOptions,
    GenerationResult,
    GenerationUsage,
    LLMProvider,
    generate_with_metadata,
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


@pytest.mark.anyio
async def test_generation_result_default_supports_legacy_provider():
    class LegacyProvider(LLMProvider):
        @property
        def provider_name(self) -> str:
            return "legacy"

        async def generate(self, system_prompt: str, user_prompt: str) -> str:
            return "legacy answer"

    result = await generate_with_metadata(LegacyProvider(), "system", "user")
    assert result == GenerationResult(content="legacy answer")
    assert result.usage is None


@pytest.mark.anyio
async def test_generation_helper_preserves_unspecced_async_mock_legacy_api():
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = "mock answer"

    result = await generate_with_metadata(mock_provider, "system", "user")

    assert result.content == "mock answer"
    mock_provider.generate.assert_awaited_once_with(system_prompt="system", user_prompt="user")
    mock_provider.generate_result.assert_not_awaited()


@pytest.mark.anyio
async def test_deepseek_generation_result_parses_usage_and_cache_metadata(monkeypatch):
    original_client = httpx.AsyncClient
    request_count = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "id": "response-123",
                "model": "deepseek-chat-2026-09",
                "choices": [{"message": {"content": "answer"}}],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 7,
                    "total_tokens": 27,
                    "prompt_cache_hit_tokens": 12,
                    "prompt_cache_miss_tokens": 8,
                },
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    provider = DeepSeekLLMProvider("test-key", "https://example.invalid")

    result = await provider.generate_result("system", "user")

    assert result.content == "answer"
    assert result.requested_model == "deepseek-chat"
    assert result.reported_model == "deepseek-chat-2026-09"
    assert result.response_id == "response-123"
    assert result.usage == GenerationUsage(20, 7, 27, 12, 8)
    assert request_count == 1
    with pytest.raises((AttributeError, TypeError)):
        result.content = "changed"  # type: ignore[misc]


@pytest.mark.anyio
async def test_deepseek_generation_without_usage_keeps_usage_unknown(monkeypatch):
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "answer"}}]},
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    result = await DeepSeekLLMProvider("test-key").generate_result("system", "user")

    assert result.content == "answer"
    assert result.reported_model is None
    assert result.response_id is None
    assert result.usage is None


@pytest.mark.anyio
async def test_deepseek_generation_options_control_model_output_and_routing(monkeypatch):
    original_client = httpx.AsyncClient
    observed_payload: dict[str, object] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        observed_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash-2026-10",
                "choices": [{"message": {"content": '{"intent":"qa"}'}}],
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    provider = DeepSeekLLMProvider(
        "test-key", "https://example.invalid", model_name="deepseek-chat"
    )

    result = await generate_with_metadata(
        provider,
        "Return a JSON object",
        "What does the paper claim?",
        options=GenerationOptions(
            model_name="deepseek-flash",
            max_output_tokens=512,
            structured_json=True,
            disable_thinking=True,
        ),
    )

    assert observed_payload["model"] == "deepseek-flash"
    assert observed_payload["max_tokens"] == 512
    assert observed_payload["response_format"] == {"type": "json_object"}
    assert observed_payload["thinking"] == {"type": "disabled"}
    assert result.requested_model == "deepseek-flash"
    assert result.reported_model == "deepseek-flash-2026-10"
    assert result.content == '{"intent":"qa"}'


def test_generation_options_validate_limits_and_are_immutable():
    with pytest.raises(ValueError, match="model_name"):
        GenerationOptions(model_name=" ")
    with pytest.raises(ValueError, match="between 1 and 65536"):
        GenerationOptions(max_output_tokens=65_537)
    with pytest.raises(ValueError, match="between 1 and 65536"):
        GenerationOptions(max_output_tokens=0)

    options = GenerationOptions(model_name="deepseek-flash", max_output_tokens=128)
    with pytest.raises((AttributeError, TypeError)):
        options.model_name = "deepseek-chat"  # type: ignore[misc]


@pytest.mark.anyio
async def test_deepseek_provider_defaults_apply_to_each_call_without_shared_mutation(monkeypatch):
    original_client = httpx.AsyncClient
    observed: list[dict[str, object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        observed.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    provider = DeepSeekLLMProvider(
        "test-key",
        "https://example.invalid",
        model_name="deepseek-flash",
        max_output_tokens=256,
        structured_json=True,
        disable_thinking=True,
    )

    results = await asyncio.gather(
        provider.generate_result("system", "one"),
        provider.generate_result(
            "system", "two", options=GenerationOptions(model_name="deepseek-chat")
        ),
    )

    assert [request["model"] for request in observed] == ["deepseek-flash", "deepseek-chat"]
    assert all(request["max_tokens"] == 256 for request in observed)
    assert all(request["response_format"] == {"type": "json_object"} for request in observed)
    assert all(request["thinking"] == {"type": "disabled"} for request in observed)
    assert [result.requested_model for result in results] == ["deepseek-flash", "deepseek-chat"]


@pytest.mark.anyio
async def test_deepseek_concurrent_calls_keep_metadata_per_response(monkeypatch):
    original_client = httpx.AsyncClient

    def respond(request: httpx.Request) -> httpx.Response:
        user_prompt = json.loads(request.content)["messages"][1]["content"]
        response_id = "response-one" if user_prompt == "one" else "response-two"
        token_count = 1 if response_id == "response-one" else 2
        return httpx.Response(
            200,
            json={
                "id": response_id,
                "model": "deepseek-chat",
                "choices": [{"message": {"content": response_id}}],
                "usage": {"prompt_tokens": token_count},
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    provider = DeepSeekLLMProvider("test-key", "https://example.invalid")

    first, second = await asyncio.gather(
        provider.generate_result("system", "one"),
        provider.generate_result("system", "two"),
    )

    assert first.response_id == "response-one"
    assert first.usage == GenerationUsage(prompt_tokens=1)
    assert second.response_id == "response-two"
    assert second.usage == GenerationUsage(prompt_tokens=2)


@pytest.mark.anyio
async def test_paid_generation_reserves_before_dispatch_and_settles_reported_usage(monkeypatch):
    original_client = httpx.AsyncClient
    reservation_id = uuid4()
    manager = Mock()
    manager.reserve.return_value = SimpleNamespace(reservation_id=reservation_id)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash",
                "choices": [{"message": {"content": "answer"}}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 4},
            },
        )

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    provider = DeepSeekLLMProvider(
        "test-key",
        model_name="deepseek-flash",
        max_output_tokens=200,
        budget_manager=manager,
    )
    context = OperationContext.validated(correlation_id="research-run-1")

    with use_operation_context(context):
        result = await provider.generate_result("system", "user")

    assert result.content == "answer"
    manager.reserve.assert_called_once_with(
        run_id="research-run-1",
        requested_model="deepseek-flash",
        input_bytes=10,
        max_output_tokens=200,
    )
    manager.settle.assert_called_once_with(
        reservation_id,
        prompt_tokens=15,
        completion_tokens=4,
        cache_hit_tokens=None,
        cache_miss_tokens=None,
    )
    manager.mark_unknown.assert_not_called()


@pytest.mark.anyio
async def test_paid_generation_refusal_happens_before_provider_dispatch(monkeypatch):
    from app.services.budget import BudgetDeniedError

    original_client = httpx.AsyncClient
    request_count = 0
    manager = Mock()
    manager.reserve.side_effect = BudgetDeniedError("run cap reached")

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "answer"}}]})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    provider = DeepSeekLLMProvider("test-key", budget_manager=manager)

    with pytest.raises(BudgetDeniedError, match="run cap"):
        await provider.generate_result("system", "user")

    assert request_count == 0
    manager.mark_unknown.assert_not_called()


@pytest.mark.anyio
async def test_provider_failure_keeps_the_attempt_reservation_unknown(monkeypatch):
    original_client = httpx.AsyncClient
    reservation_id = uuid4()
    manager = Mock()
    manager.reserve.return_value = SimpleNamespace(reservation_id=reservation_id)

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "temporary"})

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    provider = DeepSeekLLMProvider("test-key", budget_manager=manager)

    with pytest.raises(httpx.HTTPStatusError):
        await provider.generate_result("system", "user")

    manager.mark_unknown.assert_called_once_with(reservation_id)
    manager.settle.assert_not_called()
