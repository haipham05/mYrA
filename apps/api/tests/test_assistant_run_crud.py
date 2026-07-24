from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.crud.assistant_run import (
    IdempotencyConflict,
    RunLeaseLost,
    claim_next_assistant_run,
    create_assistant_run,
    finish_assistant_run,
    release_assistant_run,
    renew_assistant_run_lease,
    save_assistant_step,
    start_assistant_step,
)
from app.crud.chat import add_message
from app.db.base import Base
from app.db.models import AssistantRun, Conversation, Paper, Project
from app.schemas.assistant import AssistantRunRequest
from app.services.chat_service import ChatService


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def _seed(db: Session):
    project = Project(name="Run tests")
    db.add(project)
    db.flush()
    conversation = Conversation(project_id=project.id)
    paper = Paper(project_id=project.id, filename="paper.pdf", storage_path="local/paper.pdf")
    db.add_all([conversation, paper])
    db.commit()
    return project, conversation, paper


def _request(project, conversation, paper, **overrides) -> AssistantRunRequest:
    return AssistantRunRequest.model_validate(
        {
            "message": "What does this paper say?",
            "project_id": project.id,
            "conversation_id": conversation.id,
            "scope": "paper",
            "selected_paper_ids": [paper.id],
            "idempotency_key": "run-request-key",
            **overrides,
        }
    )


def test_create_assistant_run_is_idempotent_and_rejects_key_reuse(session: Session) -> None:
    project, conversation, paper = _seed(session)
    request = _request(project, conversation, paper)

    first, created = create_assistant_run(session, request)
    again, created_again = create_assistant_run(session, request)

    assert created is True
    assert created_again is False
    assert again.id == first.id
    assert again.status == "QUEUED"
    assert again.request_payload["idempotency_key"] == request.idempotency_key
    session.refresh(conversation)
    assert conversation.paper_scope == "paper"
    assert conversation.selected_paper_ids == [str(paper.id)]

    with pytest.raises(IdempotencyConflict):
        create_assistant_run(session, request.model_copy(update={"message": "different question"}))


def test_create_run_rejects_mismatched_conversation_and_foreign_paper(session: Session) -> None:
    project, conversation, paper = _seed(session)
    with pytest.raises(ValueError, match="conversation_not_found"):
        create_assistant_run(
            session,
            _request(project, conversation, paper, project_id=uuid4()),
        )

    other_project = Project(name="Other")
    session.add(other_project)
    session.commit()
    foreign_paper = Paper(
        project_id=other_project.id, filename="foreign.pdf", storage_path="foreign.pdf"
    )
    session.add(foreign_paper)
    session.commit()
    with pytest.raises(ValueError, match="selected_paper_not_found"):
        create_assistant_run(
            session,
            _request(project, conversation, paper, selected_paper_ids=[foreign_paper.id]),
        )


def test_run_claim_step_checkpoint_and_finish_are_fenced(session: Session) -> None:
    project, conversation, paper = _seed(session)
    run, _ = create_assistant_run(session, _request(project, conversation, paper))

    claimed = claim_next_assistant_run(session, worker_id="worker-a", lease_seconds=60)
    assert claimed is not None and claimed.id == run.id
    attempt = claimed.attempt_count
    assert renew_assistant_run_lease(
        session, run.id, worker_id="worker-a", attempt_count=attempt, lease_seconds=60
    )
    step = save_assistant_step(
        session,
        run.id,
        worker_id="worker-a",
        attempt_count=attempt,
        step_key="route",
        ordinal=1,
        input_fingerprint="a" * 64,
        output_payload={"intent": "qa"},
    )
    same_step = save_assistant_step(
        session,
        run.id,
        worker_id="worker-a",
        attempt_count=attempt,
        step_key="route",
        ordinal=1,
        input_fingerprint="a" * 64,
        output_payload={"intent": "should-not-replace-completed"},
    )
    assert same_step.id == step.id
    assert same_step.output_payload == {"intent": "qa"}
    resumed_step, should_execute = start_assistant_step(
        session,
        run.id,
        worker_id="worker-a",
        attempt_count=attempt,
        step_key="route",
        ordinal=1,
        input_fingerprint="a" * 64,
    )
    assert resumed_step.id == step.id
    assert not should_execute
    with pytest.raises(ValueError, match="fingerprint_changed"):
        save_assistant_step(
            session,
            run.id,
            worker_id="worker-a",
            attempt_count=attempt,
            step_key="route",
            ordinal=1,
            input_fingerprint="b" * 64,
            output_payload={},
        )

    finished = finish_assistant_run(
        session,
        run.id,
        worker_id="worker-a",
        attempt_count=attempt,
        status="SUCCEEDED",
        result_payload={"result_type": "answer"},
        intent="qa",
    )
    assert finished.status == "SUCCEEDED"
    assert finished.lease_owner is None
    assert claim_next_assistant_run(session, worker_id="worker-b") is None


def test_expired_attempt_is_reclaimed_and_stale_worker_cannot_publish(session: Session) -> None:
    project, conversation, paper = _seed(session)
    run, _ = create_assistant_run(session, _request(project, conversation, paper))
    old_claim = claim_next_assistant_run(session, worker_id="worker-old", lease_seconds=60)
    assert old_claim is not None
    old_attempt = old_claim.attempt_count
    session.query(AssistantRun).filter(AssistantRun.id == run.id).update(
        {AssistantRun.lease_expires_at: datetime.now(UTC) - timedelta(seconds=1)}
    )
    session.commit()

    new_claim = claim_next_assistant_run(session, worker_id="worker-new", lease_seconds=60)
    assert new_claim is not None
    assert new_claim.attempt_count == old_attempt + 1
    with pytest.raises(RunLeaseLost):
        save_assistant_step(
            session,
            run.id,
            worker_id="worker-old",
            attempt_count=old_attempt,
            step_key="result",
            ordinal=1,
            input_fingerprint="a" * 64,
            output_payload={"unsafe": "stale"},
        )
    assert release_assistant_run(
        session, run.id, worker_id="worker-new", attempt_count=new_claim.attempt_count
    )
    assert claim_next_assistant_run(session, worker_id="worker-retry") is not None


def test_interrupted_external_step_is_marked_unknown_not_replayed(session: Session) -> None:
    project, conversation, paper = _seed(session)
    run, _ = create_assistant_run(session, _request(project, conversation, paper))
    first = claim_next_assistant_run(session, worker_id="worker-first", lease_seconds=60)
    assert first is not None
    _, should_execute = start_assistant_step(
        session,
        run.id,
        worker_id="worker-first",
        attempt_count=first.attempt_count,
        step_key="assistant.route",
        ordinal=1,
        input_fingerprint=run.request_hash,
    )
    assert should_execute
    session.query(AssistantRun).filter(AssistantRun.id == run.id).update(
        {AssistantRun.lease_expires_at: datetime.now(UTC) - timedelta(seconds=1)}
    )
    session.commit()

    second = claim_next_assistant_run(session, worker_id="worker-second", lease_seconds=60)
    assert second is not None
    step, should_execute_again = start_assistant_step(
        session,
        run.id,
        worker_id="worker-second",
        attempt_count=second.attempt_count,
        step_key="assistant.route",
        ordinal=1,
        input_fingerprint=run.request_hash,
    )
    assert step.status == "UNKNOWN"
    assert step.safe_error == "ATTEMPT_OUTCOME_UNKNOWN"
    assert not should_execute_again


def test_needs_input_releases_run_for_a_later_resume(session: Session) -> None:
    project, conversation, paper = _seed(session)
    run, _ = create_assistant_run(session, _request(project, conversation, paper))
    claim = claim_next_assistant_run(session, worker_id="worker-a")
    assert claim is not None
    waiting = finish_assistant_run(
        session,
        run.id,
        worker_id="worker-a",
        attempt_count=claim.attempt_count,
        status="NEEDS_INPUT",
        result_payload={"result_type": "clarification"},
    )
    assert waiting.status == "NEEDS_INPUT"
    assert waiting.finished_at is None
    assert waiting.lease_owner is None


@pytest.mark.anyio
async def test_qa_message_is_reused_after_a_run_retry(session: Session) -> None:
    project, conversation, paper = _seed(session)
    run, _ = create_assistant_run(session, _request(project, conversation, paper))
    message = add_message(
        session,
        conversation.id,
        role="ASSISTANT",
        content="Previously completed answer",
        citations=[],
        evidence=[],
        assistant_run_id=run.id,
        provider_usage={"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    )

    class MustNotRetrieve:
        def retrieve(self, **kwargs):
            raise AssertionError("a persisted assistant response must be reused")

    service = ChatService(retriever=MustNotRetrieve())  # type: ignore[arg-type]
    response = await service.answer_question(
        session,
        conversation.id,
        "What does this paper say?",
        assistant_run_id=run.id,
    )

    assert response.id == message.id
    assert response.content == "Previously completed answer"
    assert response.provider_usage == message.provider_usage
