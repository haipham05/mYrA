from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.crud.assistant_run import claim_next_assistant_run, create_assistant_run
from app.db.base import Base
from app.db.models import AssistantRun
from app.schemas.assistant import (
    AssistantIntent,
    AssistantRouteResult,
    AssistantRunRequest,
    RouteDecision,
    RouteOutcome,
)
from app.services.assistant_run_processor import AssistantRunProcessor


@pytest.fixture
def session_factory():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    yield factory
    engine.dispose()


def _queue_run(factory, *, intent_override: str | None = None):
    with factory() as db:
        from app.db.models import Conversation, Paper, Project

        project = Project(name="Assistant runs")
        db.add(project)
        db.flush()
        conversation = Conversation(project_id=project.id)
        paper = Paper(project_id=project.id, filename="paper.pdf", storage_path="paper.pdf")
        db.add_all([conversation, paper])
        db.commit()
        payload = {
            "message": "What are its limitations?",
            "project_id": project.id,
            "conversation_id": conversation.id,
            "scope": "paper",
            "selected_paper_ids": [paper.id],
            "intent_override": intent_override,
            "idempotency_key": f"run-{uuid4().hex}",
        }
        request = AssistantRunRequest.model_validate(payload)
        run, created = create_assistant_run(db, request)
        assert created
        return run.id


class _RouteStub:
    def __init__(self, intent: AssistantIntent, *, outcome=RouteOutcome.ROUTED):
        self.intent = intent
        self.outcome = outcome
        self.calls = 0

    async def route(self, request, **kwargs):
        self.calls += 1
        decision = RouteDecision(
            intent=self.intent,
            standalone_question="What limitations does the selected paper report?",
            resolved_paper_ids=request.selected_paper_ids,
            action_summary="Answer using the selected paper",
            clarification=(
                "Please select papers" if outcome_is_clarification(self.outcome) else None
            ),
            missing_information=(
                ["select_papers"] if outcome_is_clarification(self.outcome) else []
            ),
        )
        return AssistantRouteResult(outcome=self.outcome, decision=decision)


def outcome_is_clarification(outcome: RouteOutcome) -> bool:
    return outcome is RouteOutcome.NEEDS_CLARIFICATION


class _ChatStub:
    def __init__(self) -> None:
        self.calls = []

    async def answer_question(self, db, conversation_id, question, **kwargs):
        self.calls.append((conversation_id, question, kwargs))
        return type(
            "Response",
            (),
            {
                "id": uuid4(),
                "content": "Supported result [1].",
                "model_name": "deepseek-flash",
                "citations": [],
                "evidence": [],
                "provider_usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )()


@pytest.mark.anyio
async def test_run_processor_executes_routed_qa_and_persists_each_stage(session_factory) -> None:
    run_id = _queue_run(session_factory)
    with session_factory() as db:
        claimed = claim_next_assistant_run(db, worker_id="worker-a")
        assert claimed is not None
        attempt = claimed.attempt_count
    route = _RouteStub(AssistantIntent.QA)
    chat = _ChatStub()
    processor = AssistantRunProcessor(
        session_factory=session_factory,
        router=route,  # type: ignore[arg-type]
        chat_service=chat,  # type: ignore[arg-type]
    )

    await processor.process(run_id, worker_id="worker-a", attempt_count=attempt)

    with session_factory() as db:
        run = db.get(AssistantRun, run_id)
        assert run is not None
        assert run.status == "SUCCEEDED"
        assert run.intent == "qa"
        assert run.result_payload["display_text"] == "Supported result [1]."
        assert {step.step_key for step in run.steps} == {"assistant.route", "assistant.tool.qa"}
        assert all(step.status == "COMPLETED" for step in run.steps)
    assert route.calls == 1
    assert len(chat.calls) == 1
    assert chat.calls[0][1] == "What are its limitations?"
    assert (
        chat.calls[0][2]["retrieval_question"] == "What limitations does the selected paper report?"
    )


@pytest.mark.anyio
async def test_run_processor_marks_clarification_without_dispatching_tool(session_factory) -> None:
    run_id = _queue_run(session_factory)
    with session_factory() as db:
        claimed = claim_next_assistant_run(db, worker_id="worker-a")
        assert claimed is not None
        attempt = claimed.attempt_count
    route = _RouteStub(AssistantIntent.CLARIFY, outcome=RouteOutcome.NEEDS_CLARIFICATION)
    chat = _ChatStub()
    await AssistantRunProcessor(
        session_factory=session_factory,
        router=route,  # type: ignore[arg-type]
        chat_service=chat,  # type: ignore[arg-type]
    ).process(run_id, worker_id="worker-a", attempt_count=attempt)

    with session_factory() as db:
        run = db.get(AssistantRun, run_id)
        assert run is not None and run.status == "NEEDS_INPUT"
        assert run.result_payload["result_type"] == "clarification"
    assert chat.calls == []


@pytest.mark.anyio
async def test_run_processor_honors_cancel_before_any_provider_dispatch(session_factory) -> None:
    run_id = _queue_run(session_factory)
    with session_factory() as db:
        claimed = claim_next_assistant_run(db, worker_id="worker-a")
        assert claimed is not None
        attempt = claimed.attempt_count
        db.query(AssistantRun).filter(AssistantRun.id == run_id).update(
            {AssistantRun.cancel_requested: True}
        )
        db.commit()
    route = _RouteStub(AssistantIntent.QA)
    await AssistantRunProcessor(session_factory=session_factory, router=route).process(
        run_id, worker_id="worker-a", attempt_count=attempt
    )

    with session_factory() as db:
        run = db.get(AssistantRun, run_id)
        assert run is not None and run.status == "CANCELLED"
    assert route.calls == 0
