import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.crud.assistant_run import (
    claim_next_assistant_run,
    create_assistant_run,
    decide_assistant_approval,
    request_assistant_run_cancel,
    resume_assistant_run,
)
from app.db.base import Base
from app.db.models import AssistantApprovalAction, AssistantRun, Memory
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


class _ClarifyThenQaRouteStub(_RouteStub):
    def __init__(self):
        super().__init__(AssistantIntent.CLARIFY)
        self.outcomes = [
            (AssistantIntent.CLARIFY, RouteOutcome.NEEDS_CLARIFICATION),
            (AssistantIntent.QA, RouteOutcome.ROUTED),
        ]

    async def route(self, request, **kwargs):
        self.calls += 1
        intent, outcome = self.outcomes.pop(0)
        decision = RouteDecision(
            intent=intent,
            standalone_question="What limitations does the selected paper report?",
            resolved_paper_ids=request.selected_paper_ids,
            action_summary="Answer using the selected paper",
            clarification=("Please select papers" if intent is AssistantIntent.CLARIFY else None),
            missing_information=(["select_papers"] if intent is AssistantIntent.CLARIFY else []),
        )
        return AssistantRouteResult(outcome=outcome, decision=decision)


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
    assert chat.calls[0][2]["paper_scope"] == "selection"
    assert len(chat.calls[0][2]["selected_paper_ids"]) == 1


@pytest.mark.anyio
async def test_queued_qa_runs_keep_their_own_scope_when_completed_out_of_order(session_factory):
    from app.db.models import Conversation, Paper, Project

    with session_factory() as db:
        project = Project(name="Scoped queued runs")
        db.add(project)
        db.flush()
        conversation = Conversation(project_id=project.id)
        paper_a = Paper(project_id=project.id, filename="a.pdf", storage_path="a.pdf")
        paper_b = Paper(project_id=project.id, filename="b.pdf", storage_path="b.pdf")
        db.add_all([conversation, paper_a, paper_b])
        db.flush()
        requests = [
            AssistantRunRequest(
                message="Question about this paper",
                project_id=project.id,
                conversation_id=conversation.id,
                scope="selection",
                selected_paper_ids=[paper.id],
                intent_override="qa",
                idempotency_key=f"scope-{paper.filename}",
            )
            for paper in (paper_a, paper_b)
        ]
        run_a, _ = create_assistant_run(db, requests[0])
        run_b, _ = create_assistant_run(db, requests[1])
        run_a_id, run_b_id = run_a.id, run_b.id
        paper_a_id, paper_b_id = paper_a.id, paper_b.id

    with session_factory() as db:
        lease_a = claim_next_assistant_run(db, worker_id="worker-a")
        lease_b = claim_next_assistant_run(db, worker_id="worker-b")
        assert lease_a is not None and lease_b is not None
        assert {lease_a.id, lease_b.id} == {run_a_id, run_b_id}

    route = _RouteStub(AssistantIntent.QA)
    chat = _ChatStub()
    processor = AssistantRunProcessor(
        session_factory=session_factory,
        router=route,  # type: ignore[arg-type]
        chat_service=chat,  # type: ignore[arg-type]
    )
    leases = {
        lease_a.id: ("worker-a", lease_a.attempt_count),
        lease_b.id: ("worker-b", lease_b.attempt_count),
    }
    for run_id in (run_b_id, run_a_id):
        worker_id, attempt = leases[run_id]
        await processor.process(run_id, worker_id=worker_id, attempt_count=attempt)

    scopes_by_id = {
        kwargs["selected_paper_ids"][0]: kwargs["selected_paper_ids"] for _, _, kwargs in chat.calls
    }
    assert set(scopes_by_id) == {paper_a_id, paper_b_id}
    assert scopes_by_id[paper_a_id] == [paper_a_id]
    assert scopes_by_id[paper_b_id] == [paper_b_id]


@pytest.mark.anyio
async def test_cancellation_during_qa_discards_result_and_finishes_cancelled(session_factory):
    run_id = _queue_run(session_factory)
    with session_factory() as db:
        claimed = claim_next_assistant_run(db, worker_id="worker-a")
        assert claimed is not None
        attempt = claimed.attempt_count

    entered_tool = asyncio.Event()
    release_tool = asyncio.Event()

    class BlockingChat(_ChatStub):
        async def answer_question(self, db, conversation_id, question, **kwargs):
            entered_tool.set()
            await release_tool.wait()
            return await super().answer_question(db, conversation_id, question, **kwargs)

    chat = BlockingChat()
    processor = AssistantRunProcessor(
        session_factory=session_factory,
        router=_RouteStub(AssistantIntent.QA),  # type: ignore[arg-type]
        chat_service=chat,  # type: ignore[arg-type]
    )
    task = asyncio.create_task(
        processor.process(run_id, worker_id="worker-a", attempt_count=attempt)
    )
    await entered_tool.wait()
    with session_factory() as db:
        request_assistant_run_cancel(db, run_id)
    release_tool.set()
    await task

    with session_factory() as db:
        run = db.get(AssistantRun, run_id)
        assert run is not None and run.status == "CANCELLED"
        from app.db.models import Message

        assert db.query(Message).filter(Message.assistant_run_id == run_id).count() == 0


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
async def test_note_mutation_waits_for_approval_then_runs_once(session_factory) -> None:
    run_id = _queue_run(session_factory)

    class SaveNoteRoute:
        async def route(self, request, **kwargs):
            return AssistantRouteResult(
                outcome=RouteOutcome.ROUTED,
                decision=RouteDecision(
                    intent=AssistantIntent.NOTES,
                    resolved_paper_ids=request.selected_paper_ids,
                    action_summary="Save a research note",
                    arguments={
                        "action": "propose_save",
                        "title": "Useful limitation",
                        "content": "The paper evaluates only a narrow set of tasks.",
                    },
                ),
            )

    processor = AssistantRunProcessor(
        session_factory=session_factory,
        router=SaveNoteRoute(),  # type: ignore[arg-type]
        chat_service=_ChatStub(),  # type: ignore[arg-type]
    )
    with session_factory() as db:
        claim = claim_next_assistant_run(db, worker_id="worker-a")
        assert claim is not None
        first_attempt = claim.attempt_count

    await processor.process(run_id, worker_id="worker-a", attempt_count=first_attempt)

    with session_factory() as db:
        run = db.get(AssistantRun, run_id)
        action = db.query(AssistantApprovalAction).filter_by(run_id=run_id).one()
        assert run is not None and run.status == "AWAITING_APPROVAL"
        assert action.status == "PENDING"
        assert db.query(Memory).count() == 0
        action_id = action.id
        approved, transitioned = decide_assistant_approval(db, action_id, approve=True)
        assert transitioned and approved.status == "APPROVED"
        next_claim = claim_next_assistant_run(db, worker_id="worker-a")
        assert next_claim is not None
        second_attempt = next_claim.attempt_count

    await processor.process(run_id, worker_id="worker-a", attempt_count=second_attempt)

    with session_factory() as db:
        run = db.get(AssistantRun, run_id)
        memories = db.query(Memory).all()
        assert run is not None and run.status == "SUCCEEDED"
        assert len(memories) == 1
        assert memories[0].title == "Useful limitation"
        assert memories[0].content == "The paper evaluates only a narrow set of tasks."
        assert db.query(AssistantApprovalAction).filter_by(run_id=run_id).one().status == "APPROVED"


@pytest.mark.anyio
async def test_clarification_resume_routes_again_without_replaying_old_step(
    session_factory,
) -> None:
    run_id = _queue_run(session_factory)
    route = _ClarifyThenQaRouteStub()
    chat = _ChatStub()
    processor = AssistantRunProcessor(
        session_factory=session_factory,
        router=route,  # type: ignore[arg-type]
        chat_service=chat,  # type: ignore[arg-type]
    )

    with session_factory() as db:
        first_claim = claim_next_assistant_run(db, worker_id="worker-a")
        assert first_claim is not None
        first_attempt = first_claim.attempt_count
    await processor.process(run_id, worker_id="worker-a", attempt_count=first_attempt)

    with session_factory() as db:
        waiting = db.get(AssistantRun, run_id)
        assert waiting is not None and waiting.status == "NEEDS_INPUT"
        resume_assistant_run(
            db, run_id, additional_input="Use the Attention Is All You Need paper."
        )
        second_claim = claim_next_assistant_run(db, worker_id="worker-a")
        assert second_claim is not None
        second_attempt = second_claim.attempt_count

    await processor.process(run_id, worker_id="worker-a", attempt_count=second_attempt)

    with session_factory() as db:
        run = db.get(AssistantRun, run_id)
        assert run is not None and run.status == "SUCCEEDED"
        assert run.resume_count == 1
        assert {step.step_key for step in run.steps} == {
            "assistant.route",
            "assistant.route.resume.1",
            "assistant.tool.qa",
        }
        assert all(step.status == "COMPLETED" for step in run.steps)
        assert "Use the Attention Is All You Need paper." in run.request_payload["message"]
    assert route.calls == 2
    assert len(chat.calls) == 1


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
