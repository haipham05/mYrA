from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.assistant import AssistantIntent, AssistantRunRequest, RouteDecision
from app.services.assistant_tools import (
    AssistantToolResult,
    CompareArguments,
    ToolContext,
    ToolStatus,
    build_tool_registry,
    execute_tool,
    tool_requires_approval,
    validate_tool_input,
)


def _request(*, scope: str = "selection", selected_paper_ids=None) -> AssistantRunRequest:
    return AssistantRunRequest.model_validate(
        {
            "message": "compare these papers",
            "conversation_id": uuid4(),
            "project_id": uuid4(),
            "scope": scope,
            "selected_paper_ids": selected_paper_ids or [uuid4(), uuid4()],
            "idempotency_key": "tool-request-123",
        }
    )


def test_registry_contains_only_fixed_intents_and_typed_callable_contracts() -> None:
    registry = build_tool_registry()

    assert set(registry) == set(AssistantIntent)
    assert all(callable(item.handler) for item in registry.values())
    assert all(item.input_model and item.output_model for item in registry.values())
    with pytest.raises(KeyError):
        registry["run_shell"]  # type: ignore[index]


def test_tool_validation_rejects_arbitrary_fields_and_checks_paper_cardinality() -> None:
    registry = build_tool_registry()
    request = _request()
    decision = RouteDecision(
        intent=AssistantIntent.COMPARE,
        resolved_paper_ids=request.selected_paper_ids,
        action_summary="Compare the selected papers",
        arguments={"dimensions": ["method", "results"]},
    )
    validated = validate_tool_input(registry[AssistantIntent.COMPARE], request, decision)
    assert isinstance(validated.arguments, CompareArguments)
    assert validated.arguments.dimensions == ["method", "results"]

    bad_decision = RouteDecision(
        intent=AssistantIntent.COMPARE,
        action_summary="Compare papers",
        arguments={"dimensions": [], "shell": "rm -rf"},
    )
    with pytest.raises(ValidationError, match="Extra inputs"):
        validate_tool_input(registry[AssistantIntent.COMPARE], request, bad_decision)

    one_paper = _request(selected_paper_ids=[uuid4()])
    one_paper_decision = RouteDecision(
        intent=AssistantIntent.COMPARE,
        action_summary="Compare papers",
    )
    with pytest.raises(ValueError, match="select more papers"):
        validate_tool_input(registry[AssistantIntent.COMPARE], one_paper, one_paper_decision)


def test_note_reads_do_not_require_approval_but_mutations_do() -> None:
    definition = build_tool_registry()[AssistantIntent.NOTES]
    request = _request()
    read_decision = RouteDecision(
        intent=AssistantIntent.NOTES,
        action_summary="List notes",
        arguments={"action": "list"},
    )
    update_decision = RouteDecision(
        intent=AssistantIntent.NOTES,
        action_summary="Propose note update",
        arguments={"action": "propose_update", "content": "new preference"},
    )
    assert not tool_requires_approval(
        definition, validate_tool_input(definition, request, read_decision)
    )
    assert tool_requires_approval(
        definition, validate_tool_input(definition, request, update_decision)
    )
    assert tool_requires_approval(
        build_tool_registry()[AssistantIntent.TRANSLATE],
        validate_tool_input(
            build_tool_registry()[AssistantIntent.TRANSLATE],
            _request(scope="paper", selected_paper_ids=[uuid4()]),
            RouteDecision(
                intent=AssistantIntent.TRANSLATE,
                action_summary="Translate the selected paper",
                arguments={"target_language": "vi"},
            ),
        ),
    )


@pytest.mark.anyio
async def test_qa_tool_reuses_existing_chat_service_once() -> None:
    class FakeChatService:
        def __init__(self) -> None:
            self.calls = []

        async def answer_question(self, db, conversation_id, question, **kwargs):
            self.calls.append((db, conversation_id, question, kwargs))
            return type(
                "Response",
                (),
                {
                    "id": uuid4(),
                    "model_name": "deepseek-flash",
                    "content": "Grounded result [1].",
                    "citations": [],
                    "evidence": [],
                    "provider_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    },
                },
            )()

    service = FakeChatService()
    request = _request(scope="paper", selected_paper_ids=[uuid4()])
    decision = RouteDecision(
        intent=AssistantIntent.QA,
        standalone_question="What are its limitations?",
        resolved_paper_ids=request.selected_paper_ids,
        action_summary="Answer from the selected paper",
    )
    registry = build_tool_registry()
    definition = registry[AssistantIntent.QA]
    tool_input = validate_tool_input(definition, request, decision)
    result = await execute_tool(
        definition,
        ToolContext(db=None, chat_service=service),  # type: ignore[arg-type]
        tool_input,
    )

    assert isinstance(result, AssistantToolResult)
    assert result.status is ToolStatus.SUCCEEDED
    assert result.display_text == "Grounded result [1]."
    assert len(service.calls) == 1
    assert service.calls[0][2] == request.message
    assert service.calls[0][3]["retrieval_question"] == decision.standalone_question


@pytest.mark.anyio
async def test_unimplemented_feature_returns_explicit_unavailable_result() -> None:
    registry = build_tool_registry()
    request = _request(scope="paper", selected_paper_ids=[uuid4()])
    decision = RouteDecision(
        intent=AssistantIntent.VISION,
        action_summary="Analyze the selected figure",
        arguments={"question": "What does the plot show?"},
    )
    definition = registry[AssistantIntent.VISION]
    result = await execute_tool(
        definition,
        ToolContext(db=None, chat_service=None),  # type: ignore[arg-type]
        validate_tool_input(definition, request, decision),
    )

    assert result.status is ToolStatus.UNAVAILABLE
    assert "not connected" in result.display_text
