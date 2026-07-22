from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.schemas.assistant import (
    AssistantIntent,
    AssistantRunRequest,
    RouteDecision,
    RunStatus,
)


def _request(**overrides: object) -> dict[str, object]:
    return {
        "message": "Compare these papers",
        "conversation_id": uuid4(),
        "project_id": uuid4(),
        "scope": "selection",
        "selected_paper_ids": [uuid4(), uuid4()],
        "idempotency_key": "request-123456",
        **overrides,
    }


def test_assistant_run_request_accepts_bounded_valid_contract() -> None:
    request = AssistantRunRequest.model_validate(_request(intent_override="compare"))

    assert request.intent_override is AssistantIntent.COMPARE
    assert request.scope.value == "selection"
    assert len(request.selected_paper_ids) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"message": "  "},
        {"intent_override": "run_shell"},
        {"selected_paper_ids": [uuid4()] * 7},
        {"idempotency_key": "bad key"},
        {"conversation_id": "not-a-uuid"},
        {"scope": "paper", "selected_paper_ids": []},
        {"scope": "project", "selected_paper_ids": [uuid4()]},
        {"scope": "selection", "selected_paper_ids": []},
    ],
)
def test_assistant_run_request_rejects_invalid_or_oversized_values(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        AssistantRunRequest.model_validate(_request(**overrides))


def test_assistant_run_request_rejects_duplicate_paper_ids() -> None:
    duplicate_id = uuid4()

    with pytest.raises(ValidationError, match="must be unique"):
        AssistantRunRequest.model_validate(
            _request(selected_paper_ids=[duplicate_id, duplicate_id])
        )


def test_route_decision_rejects_unknown_intent_and_unbounded_arguments() -> None:
    base = {"intent": "qa", "action_summary": "Answer from selected papers"}

    with pytest.raises(ValidationError):
        RouteDecision.model_validate({**base, "intent": "arbitrary_tool"})
    with pytest.raises(ValidationError, match="field count"):
        RouteDecision.model_validate({**base, "arguments": {str(i): i for i in range(25)}})
    with pytest.raises(ValidationError, match="allowed size"):
        RouteDecision.model_validate({**base, "arguments": {"query": "x" * 5000}})


def test_run_status_contract_contains_expected_lifecycle() -> None:
    assert {state.value for state in RunStatus} == {
        "QUEUED",
        "RUNNING",
        "NEEDS_INPUT",
        "AWAITING_APPROVAL",
        "SUCCEEDED",
        "FAILED",
        "CANCELLED",
    }
