from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.crud.assistant_run import (
    IdempotencyConflict,
    RunLeaseLost,
    begin_assistant_run_publication,
    claim_next_assistant_run,
    create_assistant_approval,
    create_assistant_run,
    decide_assistant_approval,
    finish_assistant_run,
    get_valid_approved_assistant_action,
    release_assistant_run,
    renew_assistant_run_lease,
    request_assistant_run_cancel,
    resume_assistant_run,
    save_assistant_step,
    start_assistant_step,
)
from app.crud.chat import add_message
from app.db.base import Base
from app.db.models import AssistantRun, Conversation, Message, Paper, Project
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


def test_approval_proposal_is_exact_idempotent_and_bound_to_source_revision(
    session: Session,
) -> None:
    project, conversation, paper = _seed(session)
    run, _ = create_assistant_run(session, _request(project, conversation, paper))
    claim = claim_next_assistant_run(session, worker_id="worker-a")
    assert claim is not None
    arguments = {
        "message": "Translate the selected paper",
        "scope": "paper",
        "paper_ids": [str(paper.id)],
        "tool_arguments": {"target_language": "vi"},
    }
    action = create_assistant_approval(
        session,
        run.id,
        worker_id="worker-a",
        attempt_count=claim.attempt_count,
        action_type="translate",
        arguments=arguments,
        paper_ids=[paper.id],
    )
    repeated = create_assistant_approval(
        session,
        run.id,
        worker_id="worker-a",
        attempt_count=claim.attempt_count,
        action_type="translate",
        arguments=arguments,
        paper_ids=[paper.id],
    )
    assert repeated.id == action.id
    assert action.arguments == arguments
    assert action.status == "PENDING"
    assert action.source_fingerprint != ""

    project.corpus_revision += 1
    session.commit()
    changed_source = create_assistant_approval(
        session,
        run.id,
        worker_id="worker-a",
        attempt_count=claim.attempt_count,
        action_type="translate",
        arguments=arguments,
        paper_ids=[paper.id],
    )
    assert changed_source.id != action.id
    assert changed_source.source_fingerprint != action.source_fingerprint


def _pending_approval(session: Session):
    project, conversation, paper = _seed(session)
    run, _ = create_assistant_run(session, _request(project, conversation, paper))
    claim = claim_next_assistant_run(session, worker_id="worker-a")
    assert claim is not None
    action = create_assistant_approval(
        session,
        run.id,
        worker_id="worker-a",
        attempt_count=claim.attempt_count,
        action_type="translate",
        arguments={"paper_ids": [str(paper.id)], "tool_arguments": {"language": "vi"}},
        paper_ids=[paper.id],
    )
    finish_assistant_run(
        session,
        run.id,
        worker_id="worker-a",
        attempt_count=claim.attempt_count,
        status="AWAITING_APPROVAL",
        result_payload={"result_type": "approval_required"},
    )
    return project, run, action


def test_approval_decision_is_idempotent_and_requeues_exact_proposal(session: Session) -> None:
    project, run, action = _pending_approval(session)
    approved, transitioned = decide_assistant_approval(session, action.id, approve=True)
    assert approved is not None and approved.status == "APPROVED"
    assert transitioned is True
    repeated, repeated_transition = decide_assistant_approval(session, action.id, approve=True)
    assert repeated.id == action.id
    assert repeated_transition is False
    session.refresh(run)
    assert run.status == "QUEUED"
    assert run.result_payload is None
    exact_action = get_valid_approved_assistant_action(
        session,
        run.id,
        action_type="translate",
        expected_arguments=action.arguments,
        paper_ids=[UUID(action.arguments["paper_ids"][0])],
    )
    assert exact_action is not None and exact_action.id == action.id

    project.corpus_revision += 1
    session.commit()
    with pytest.raises(ValueError, match="no_longer_matches"):
        get_valid_approved_assistant_action(
            session,
            run.id,
            action_type="translate",
            expected_arguments=action.arguments,
            paper_ids=[UUID(action.arguments["paper_ids"][0])],
        )
    session.refresh(action)
    assert action.status == "STALE"


def test_rejection_cancels_without_queueing_mutation(session: Session) -> None:
    _, run, action = _pending_approval(session)
    rejected, transitioned = decide_assistant_approval(session, action.id, approve=False)
    assert rejected is not None and rejected.status == "REJECTED"
    assert transitioned is True
    session.refresh(run)
    assert run.status == "CANCELLED"
    assert run.safe_error == "ACTION_REJECTED"
    with pytest.raises(ValueError, match="already_decided"):
        decide_assistant_approval(session, action.id, approve=True)


def test_approval_refuses_changed_source_and_expires_stale_proposal(session: Session) -> None:
    project, run, action = _pending_approval(session)
    project.corpus_revision += 1
    session.commit()
    stale, transitioned = decide_assistant_approval(session, action.id, approve=True)
    assert stale is not None and stale.status == "STALE"
    assert transitioned is False
    session.refresh(run)
    assert run.status == "FAILED"
    assert run.safe_error == "APPROVAL_SOURCE_CHANGED"

    _, expired_run, expired_action = _pending_approval(session)
    expired_action.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    session.commit()
    expired, transitioned = decide_assistant_approval(session, expired_action.id, approve=True)
    assert expired is not None and expired.status == "EXPIRED"
    assert transitioned is False
    session.refresh(expired_run)
    assert expired_run.status == "FAILED"
    assert expired_run.safe_error == "APPROVAL_EXPIRED"


def test_cancel_and_clarification_resume_use_safe_lifecycle_states(session: Session) -> None:
    project, conversation, paper = _seed(session)
    cancelled_run, _ = create_assistant_run(
        session,
        _request(project, conversation, paper, idempotency_key="cancel-request-123"),
    )
    cancelled = request_assistant_run_cancel(session, cancelled_run.id)
    assert cancelled is not None and cancelled.status == "CANCELLED"
    assert cancelled.cancel_requested is True

    resumed_run, _ = create_assistant_run(
        session,
        _request(project, conversation, paper, idempotency_key="resume-request-123"),
    )
    claim = claim_next_assistant_run(session, worker_id="worker-resume")
    assert claim is not None and claim.id == resumed_run.id
    waiting = finish_assistant_run(
        session,
        resumed_run.id,
        worker_id="worker-resume",
        attempt_count=claim.attempt_count,
        status="NEEDS_INPUT",
        result_payload={"result_type": "clarification"},
    )
    previous_hash = waiting.request_hash
    queued = resume_assistant_run(
        session,
        waiting.id,
        additional_input="The Attention Is All You Need paper.",
    )
    assert queued is not None
    assert queued.status == "QUEUED"
    assert queued.resume_count == 1
    assert queued.request_hash != previous_hash
    assert "Additional user input" in queued.request_payload["message"]
    with pytest.raises(ValueError, match="not_waiting_for_input"):
        resume_assistant_run(session, queued.id, additional_input="again")


def test_publication_and_cancellation_have_one_winner(session: Session) -> None:
    project, conversation, paper = _seed(session)
    run, _ = create_assistant_run(
        session,
        _request(project, conversation, paper, idempotency_key="publish-race-123"),
    )
    claim = claim_next_assistant_run(session, worker_id="worker-publish")
    assert claim is not None and claim.id == run.id

    assert begin_assistant_run_publication(
        session,
        run.id,
        worker_id="worker-publish",
        attempt_count=claim.attempt_count,
    )
    message = add_message(
        session,
        conversation.id,
        role="ASSISTANT",
        content="Published answer",
        citations=[],
        evidence=[],
        assistant_run_id=run.id,
    )
    cancellation = request_assistant_run_cancel(session, run.id)

    assert cancellation is not None
    assert cancellation.status == "RUNNING"
    assert cancellation.cancel_requested is False
    assert session.get(Message, message.id) is not None


def test_cancellation_winning_before_publication_prevents_assistant_message(
    session: Session,
) -> None:
    project, conversation, paper = _seed(session)
    run, _ = create_assistant_run(
        session,
        _request(project, conversation, paper, idempotency_key="cancel-wins-123"),
    )
    claim = claim_next_assistant_run(session, worker_id="worker-cancel")
    assert claim is not None and claim.id == run.id

    cancellation = request_assistant_run_cancel(session, run.id)
    assert cancellation is not None and cancellation.cancel_requested is True
    assert not begin_assistant_run_publication(
        session,
        run.id,
        worker_id="worker-cancel",
        attempt_count=claim.attempt_count,
    )
    assert session.query(Message).filter(Message.assistant_run_id == run.id).count() == 0


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
