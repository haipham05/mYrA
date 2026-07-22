import json
from uuid import uuid4

import pytest

from app.config import Settings
from app.schemas.assistant import (
    AssistantIntent,
    AssistantRunRequest,
    RouteOutcome,
    RoutePaperContext,
)
from app.services.assistant_router import AssistantRouter
from app.services.llm import DeepSeekLLMProvider, GenerationResult, GenerationUsage


def _request(**overrides: object) -> AssistantRunRequest:
    return AssistantRunRequest.model_validate(
        {
            "message": "What are its limitations?",
            "conversation_id": uuid4(),
            "project_id": uuid4(),
            "scope": "paper",
            "selected_paper_ids": [uuid4()],
            "idempotency_key": "route-request-123",
            **overrides,
        }
    )


class _StubProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.system_prompt = ""
        self.user_prompt = ""

    async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.response


class _CaptureDeepSeek(DeepSeekLLMProvider):
    def __init__(self, response: str) -> None:
        super().__init__("test-key", model_name="deepseek-flash")
        self.response = response
        self.options = None

    async def generate_result(self, system_prompt, user_prompt, *, options=None):
        self.options = options
        return GenerationResult(
            content=self.response,
            requested_model=options.model_name if options else self.model_name,
            reported_model="deepseek-flash-test",
            usage=GenerationUsage(prompt_tokens=23, completion_tokens=7, total_tokens=30),
        )


def _decision(intent: str = "qa", **extra: object) -> str:
    return json.dumps(
        {
            "intent": intent,
            "standalone_question": "What limitations does the paper report?",
            "resolved_paper_ids": [],
            "arguments": {},
            "missing_information": [],
            "clarification": None,
            "action_summary": "Answer using the selected paper",
            **extra,
        }
    )


@pytest.mark.anyio
async def test_router_resolves_followup_with_bounded_history_and_paper_context() -> None:
    request = _request()
    provider = _StubProvider(_decision(resolved_paper_ids=[str(request.selected_paper_ids[0])]))
    router = AssistantRouter(
        provider=provider, settings=Settings(deepseek_model_name="deepseek-flash")
    )

    result = await router.route(
        request,
        recent_history=[
            {"role": "USER", "content": "What does this paper propose?"},
            {"role": "ASSISTANT", "content": "It proposes attention."},
            {"role": "SYSTEM", "content": "must not be forwarded"},
        ],
        available_papers=[
            RoutePaperContext(id=request.selected_paper_ids[0], title="Attention Is All You Need")
        ],
    )

    assert result.outcome is RouteOutcome.ROUTED
    assert result.decision.intent is AssistantIntent.QA
    assert result.decision.resolved_paper_ids == request.selected_paper_ids
    assert "What limitations does the paper report?" == result.decision.standalone_question
    assert "must not be forwarded" not in provider.user_prompt
    assert "Attention Is All You Need" in provider.user_prompt
    assert "JSON Schema" in provider.system_prompt


@pytest.mark.anyio
async def test_router_forces_requested_scope_even_if_model_returns_other_paper() -> None:
    request = _request()
    provider = _StubProvider(_decision(resolved_paper_ids=[str(uuid4())]))
    result = await AssistantRouter(provider=provider).route(request)

    assert result.outcome is RouteOutcome.ROUTED
    assert result.decision.resolved_paper_ids == request.selected_paper_ids


@pytest.mark.anyio
async def test_project_scope_unknown_model_reference_becomes_clarification() -> None:
    request = _request(scope="project", selected_paper_ids=[])
    known_paper = RoutePaperContext(id=uuid4(), title="Known Paper")
    provider = _StubProvider(_decision("compare", resolved_paper_ids=[str(uuid4())]))
    result = await AssistantRouter(provider=provider).route(request, available_papers=[known_paper])

    assert result.outcome is RouteOutcome.NEEDS_CLARIFICATION
    assert result.decision.intent is AssistantIntent.CLARIFY
    assert not result.decision.resolved_paper_ids


@pytest.mark.anyio
@pytest.mark.parametrize(
    "response",
    [
        "not-json",
        _decision("run_shell"),
        _decision(extra_field="dangerous-tool"),
    ],
)
async def test_malformed_or_unsupported_model_output_fails_closed(response: str) -> None:
    result = await AssistantRouter(provider=_StubProvider(response)).route(_request())

    assert result.outcome is RouteOutcome.UNAVAILABLE
    assert result.decision.intent is AssistantIntent.CLARIFY
    assert result.decision.clarification
    assert "Traceback" not in result.decision.clarification


@pytest.mark.anyio
async def test_provider_error_returns_safe_retryable_routing_result() -> None:
    class FailingProvider:
        async def generate(self, *, system_prompt: str, user_prompt: str) -> str:
            raise RuntimeError("secret-bearing provider stacktrace")

    result = await AssistantRouter(provider=FailingProvider()).route(_request())

    assert result.outcome is RouteOutcome.UNAVAILABLE
    assert "secret-bearing" not in result.decision.clarification


@pytest.mark.anyio
async def test_explicit_intent_override_skips_paid_routing_call() -> None:
    request = _request(intent_override="compare")

    class MustNotCall:
        async def generate(self, **kwargs: object) -> str:
            raise AssertionError("explicit action override should skip LLM routing")

    result = await AssistantRouter(provider=MustNotCall()).route(request)

    assert result.outcome is RouteOutcome.ROUTED
    assert result.decision.intent is AssistantIntent.COMPARE
    assert result.decision.resolved_paper_ids == request.selected_paper_ids


@pytest.mark.anyio
async def test_deepseek_route_uses_bounded_json_non_thinking_options() -> None:
    request = _request()
    provider = _CaptureDeepSeek(_decision())
    result = await AssistantRouter(
        provider=provider,
        settings=Settings(deepseek_model_name="deepseek-flash"),
    ).route(request)

    assert provider.options is not None
    assert provider.options.model_name == "deepseek-flash"
    assert provider.options.max_output_tokens == 512
    assert provider.options.structured_json is True
    assert provider.options.disable_thinking is True
    assert result.usage == {"prompt_tokens": 23, "completion_tokens": 7, "total_tokens": 30}
