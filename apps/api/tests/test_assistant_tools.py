from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.assistant import AssistantIntent, AssistantRunRequest, RouteDecision
from app.schemas.evidence import EvidenceItem
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
    selected_ids = [uuid4(), uuid4()] if selected_paper_ids is None else selected_paper_ids
    return AssistantRunRequest.model_validate(
        {
            "message": "compare these papers",
            "conversation_id": uuid4(),
            "project_id": uuid4(),
            "scope": scope,
            "selected_paper_ids": selected_ids,
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
async def test_read_paper_tool_reuses_scoped_chat_and_returns_brief_sections() -> None:
    class FakeChatService:
        def __init__(self) -> None:
            self.calls = []

        async def answer_question(self, db, conversation_id, question, **kwargs):
            self.calls.append((conversation_id, question, kwargs))
            return type(
                "Response",
                (),
                {
                    "id": uuid4(),
                    "model_name": "test-model",
                    "content": (
                        "## Research question\nThe paper studies attention [1].\n## Limitations\n"
                    ),
                    "citations": [],
                    "evidence": [],
                    "provider_usage": None,
                },
            )()

    request = _request(scope="paper", selected_paper_ids=[uuid4()])
    decision = RouteDecision(
        intent=AssistantIntent.READ_PAPER,
        resolved_paper_ids=request.selected_paper_ids,
        action_summary="Read the selected paper",
    )
    registry = build_tool_registry()
    definition = registry[AssistantIntent.READ_PAPER]
    validated = validate_tool_input(definition, request, decision)
    service = FakeChatService()
    result = await execute_tool(
        definition,
        ToolContext(
            db=None,
            chat_service=service,  # type: ignore[arg-type]
            assistant_run_id=uuid4(),
            worker_id="worker-read",
            attempt_count=2,
        ),
        validated,
    )

    assert result.result_type == "reading_brief"
    sections = {section["key"]: section for section in result.structured_payload["sections"]}
    assert sections["research_question"]["citation_indexes"] == [1]
    assert sections["limitations"]["content"] == "Not found in retrieved evidence."
    assert len(service.calls) == 1
    assert service.calls[0][2]["paper_scope"] == "selection"
    assert service.calls[0][2]["selected_paper_ids"] == request.selected_paper_ids
    assert service.calls[0][2]["response_guidance"]
    assert service.calls[0][2]["run_worker_id"] == "worker-read"


def test_read_paper_tool_requires_one_paper():
    registry = build_tool_registry()
    project_request = _request(scope="project", selected_paper_ids=[])
    decision = RouteDecision(
        intent=AssistantIntent.READ_PAPER,
        action_summary="Read the selected paper",
    )

    with pytest.raises(ValueError, match="select more papers"):
        validate_tool_input(registry[AssistantIntent.READ_PAPER], project_request, decision)


@pytest.mark.anyio
async def test_qa_tool_verifies_selected_passage_before_using_existing_chat(monkeypatch) -> None:
    paper_id = uuid4()
    project_id = uuid4()
    request = AssistantRunRequest.model_validate(
        {
            "message": "Explain this passage",
            "conversation_id": uuid4(),
            "project_id": project_id,
            "scope": "paper",
            "selected_paper_ids": [paper_id],
            "source_selection": {
                "paper_id": paper_id,
                "page_number": 3,
                "quote": "Attention connects all positions.",
                "document_sha256": "a" * 64,
            },
            "idempotency_key": "passage-request-123",
        }
    )
    decision = RouteDecision(
        intent=AssistantIntent.QA,
        resolved_paper_ids=[uuid4()],
        action_summary="Explain the passage",
    )

    class FakeChatService:
        calls = []

        async def answer_question(self, db, conversation_id, question, **kwargs):
            self.calls.append((question, kwargs))
            return type(
                "Response",
                (),
                {
                    "id": uuid4(),
                    "model_name": "test-model",
                    "content": "The passage says this [1].",
                    "citations": [],
                    "evidence": [],
                    "provider_usage": None,
                },
            )()

    monkeypatch.setattr(
        "app.services.assistant_tools.resolve_exact_source_anchor",
        lambda *args, **kwargs: type(
            "Anchor", (), {"page_number": 3, "source_char_start": 5, "source_char_end": 36}
        )(),
    )
    selected_evidence = EvidenceItem(
        id="selected-passage",
        paper_id=paper_id,
        chunk_id=uuid4(),
        quote="Attention connects all positions.",
        page_number=3,
    )
    monkeypatch.setattr(
        "app.services.assistant_tools.build_selected_passage_evidence",
        lambda *args, **kwargs: selected_evidence,
    )
    service = FakeChatService()
    definition = build_tool_registry()[AssistantIntent.QA]
    result = await execute_tool(
        definition,
        ToolContext(
            db=None,
            chat_service=service,  # type: ignore[arg-type]
            assistant_run_id=uuid4(),
        ),
        validate_tool_input(definition, request, decision),
    )

    assert result.result_type == "answer"
    assert len(service.calls) == 1
    assert service.calls[0][1]["selected_paper_ids"] == [paper_id]
    assert service.calls[0][1]["paper_scope"] == "selection"
    assert "Attention connects all positions." in service.calls[0][1]["retrieval_question"]
    assert (
        "define symbols only when the surrounding source supports"
        in service.calls[0][1]["response_guidance"]
    )
    assert service.calls[0][1]["additional_evidence"] == [selected_evidence]


@pytest.mark.anyio
async def test_qa_tool_does_not_call_chat_when_selected_passage_is_unverified(monkeypatch) -> None:
    paper_id = uuid4()
    request = AssistantRunRequest.model_validate(
        {
            "message": "Explain this passage",
            "conversation_id": uuid4(),
            "project_id": uuid4(),
            "scope": "paper",
            "selected_paper_ids": [paper_id],
            "source_selection": {
                "paper_id": paper_id,
                "page_number": 3,
                "quote": "Repeated phrase",
                "document_sha256": "b" * 64,
            },
            "idempotency_key": "passage-request-456",
        }
    )
    decision = RouteDecision(intent=AssistantIntent.QA, action_summary="Explain the passage")

    class FakeChatService:
        async def answer_question(self, *args, **kwargs):
            raise AssertionError("unverified selections must not reach generation")

    monkeypatch.setattr(
        "app.services.assistant_tools.resolve_exact_source_anchor", lambda *args, **kwargs: None
    )
    definition = build_tool_registry()[AssistantIntent.QA]
    result = await execute_tool(
        definition,
        ToolContext(db=None, chat_service=FakeChatService()),  # type: ignore[arg-type]
        validate_tool_input(definition, request, decision),
    )

    assert result.status.value == "NEEDS_INPUT"
    assert result.result_type == "source_selection_unavailable"


@pytest.mark.anyio
async def test_qa_tool_does_not_generate_when_verified_passage_is_not_index_linked(
    monkeypatch,
) -> None:
    paper_id = uuid4()
    project_id = uuid4()
    request = AssistantRunRequest.model_validate(
        {
            "message": "Explain this passage",
            "conversation_id": uuid4(),
            "project_id": project_id,
            "scope": "paper",
            "selected_paper_ids": [paper_id],
            "source_selection": {
                "paper_id": paper_id,
                "page_number": 3,
                "quote": "Verified but not indexed.",
                "document_sha256": "c" * 64,
            },
            "idempotency_key": "passage-request-789",
        }
    )

    class FakeChatService:
        async def answer_question(self, *args, **kwargs):
            raise AssertionError("unlinked selection must not reach generation")

    monkeypatch.setattr(
        "app.services.assistant_tools.resolve_exact_source_anchor",
        lambda *args, **kwargs: type("Anchor", (), {})(),
    )
    monkeypatch.setattr(
        "app.services.assistant_tools.build_selected_passage_evidence",
        lambda *args, **kwargs: None,
    )
    definition = build_tool_registry()[AssistantIntent.QA]
    result = await execute_tool(
        definition,
        ToolContext(db=None, chat_service=FakeChatService()),  # type: ignore[arg-type]
        validate_tool_input(
            definition,
            request,
            RouteDecision(
                intent=AssistantIntent.QA,
                resolved_paper_ids=[paper_id],
                action_summary="Explain the passage",
            ),
        ),
    )

    assert result.status is ToolStatus.NEEDS_INPUT
    assert result.result_type == "source_selection_unavailable"


def test_source_selection_requires_exact_single_paper_scope() -> None:
    paper_id = uuid4()
    payload = {
        "message": "Explain this passage",
        "conversation_id": uuid4(),
        "project_id": uuid4(),
        "scope": "selection",
        "selected_paper_ids": [paper_id],
        "source_selection": {
            "paper_id": paper_id,
            "page_number": 1,
            "quote": "Some source text.",
            "document_sha256": "c" * 64,
        },
        "idempotency_key": "passage-scope-123",
    }

    with pytest.raises(ValueError, match="single paper as the run scope"):
        AssistantRunRequest.model_validate(payload)


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
