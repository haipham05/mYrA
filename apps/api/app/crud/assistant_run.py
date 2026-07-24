"""Durable assistant-run creation, lease ownership, and step checkpoints."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.db.models import AssistantRun, AssistantRunStep, Conversation, Paper
from app.schemas.assistant import AssistantRunRequest


class IdempotencyConflict(ValueError):
    """A request key was reused with a different normalized request."""


class RunLeaseLost(RuntimeError):
    """The caller no longer owns the current run attempt."""


def _request_hash(request: AssistantRunRequest) -> tuple[str, dict[str, object]]:
    payload = request.model_dump(mode="json")
    fingerprint_payload = {key: value for key, value in payload.items() if key != "idempotency_key"}
    encoded = json.dumps(
        fingerprint_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), payload


def create_assistant_run(db: Session, request: AssistantRunRequest) -> tuple[AssistantRun, bool]:
    """Create a queued run or return the existing run for the same request key."""
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == request.conversation_id,
            Conversation.project_id == request.project_id,
        )
        .first()
    )
    if conversation is None:
        raise ValueError("conversation_not_found")
    if request.parent_run_id is not None:
        parent = (
            db.query(AssistantRun)
            .filter(
                AssistantRun.id == request.parent_run_id,
                AssistantRun.project_id == request.project_id,
                AssistantRun.conversation_id == request.conversation_id,
            )
            .first()
        )
        if parent is None:
            raise ValueError("parent_run_not_found")
    if request.selected_paper_ids:
        selected_count = (
            db.query(Paper.id)
            .filter(
                Paper.project_id == request.project_id, Paper.id.in_(request.selected_paper_ids)
            )
            .count()
        )
        if selected_count != len(request.selected_paper_ids):
            raise ValueError("selected_paper_not_found")
    request_hash, payload = _request_hash(request)
    existing = (
        db.query(AssistantRun)
        .filter(
            AssistantRun.conversation_id == request.conversation_id,
            AssistantRun.idempotency_key == request.idempotency_key,
        )
        .first()
    )
    if existing is not None:
        if existing.request_hash != request_hash:
            raise IdempotencyConflict("idempotency_key_reused_with_different_request")
        return existing, False

    run = AssistantRun(
        project_id=request.project_id,
        conversation_id=request.conversation_id,
        parent_run_id=request.parent_run_id,
        idempotency_key=request.idempotency_key,
        request_hash=request_hash,
        request_payload=payload,
        status="QUEUED",
    )
    conversation.paper_scope = request.scope.value
    conversation.selected_paper_ids = [str(paper_id) for paper_id in request.selected_paper_ids]
    conversation.updated_at = datetime.now(UTC)
    db.add(run)
    try:
        db.commit()
    except Exception:
        db.rollback()
        existing = (
            db.query(AssistantRun)
            .filter(
                AssistantRun.conversation_id == request.conversation_id,
                AssistantRun.idempotency_key == request.idempotency_key,
            )
            .first()
        )
        if existing is None:
            raise
        if existing.request_hash != request_hash:
            raise IdempotencyConflict("idempotency_key_reused_with_different_request")
        return existing, False
    db.refresh(run)
    return run, True


def get_assistant_run(db: Session, run_id: UUID) -> AssistantRun | None:
    return db.query(AssistantRun).filter(AssistantRun.id == run_id).first()


def claim_next_assistant_run(
    db: Session, *, worker_id: str, lease_seconds: int = 300
) -> AssistantRun | None:
    now = datetime.now(UTC)
    query = (
        db.query(AssistantRun)
        .filter(
            or_(
                AssistantRun.status == "QUEUED",
                ((AssistantRun.status == "RUNNING") & (AssistantRun.lease_expires_at < now)),
            ),
            AssistantRun.cancel_requested.is_(False),
        )
        .order_by(AssistantRun.created_at.asc(), AssistantRun.id.asc())
    )
    if db.bind is not None and db.bind.dialect.name != "sqlite":
        query = query.with_for_update(skip_locked=True)
    candidate = query.first()
    if candidate is None:
        return None

    # Lease timestamps are written in UTC and use a CAS update so two pollers cannot claim one run.
    from sqlalchemy import update

    conditions = [AssistantRun.status == "QUEUED"]
    if candidate.status == "RUNNING":
        conditions = [
            AssistantRun.status == "RUNNING",
            or_(AssistantRun.lease_expires_at.is_(None), AssistantRun.lease_expires_at < now),
        ]
    changed = db.execute(
        update(AssistantRun)
        .where(AssistantRun.id == candidate.id, *conditions)
        .values(
            status="RUNNING",
            lease_owner=worker_id,
            lease_expires_at=now + timedelta(seconds=lease_seconds),
            attempt_count=AssistantRun.attempt_count + 1,
            started_at=func.coalesce(AssistantRun.started_at, now),
            updated_at=now,
        )
        .execution_options(synchronize_session=False)
    ).rowcount
    if changed != 1:
        db.rollback()
        return None
    db.commit()
    db.refresh(candidate)
    return candidate


def renew_assistant_run_lease(
    db: Session, run_id: UUID, *, worker_id: str, attempt_count: int, lease_seconds: int = 300
) -> bool:
    now = datetime.now(UTC)
    changed = (
        db.query(AssistantRun)
        .filter(
            AssistantRun.id == run_id,
            AssistantRun.status == "RUNNING",
            AssistantRun.lease_owner == worker_id,
            AssistantRun.attempt_count == attempt_count,
        )
        .update(
            {
                AssistantRun.lease_expires_at: now + timedelta(seconds=lease_seconds),
                AssistantRun.updated_at: now,
            },
            synchronize_session=False,
        )
    )
    db.commit()
    return changed == 1


def assert_assistant_run_lease(
    db: Session, run_id: UUID, *, worker_id: str, attempt_count: int
) -> AssistantRun:
    run = (
        db.query(AssistantRun)
        .filter(
            AssistantRun.id == run_id,
            AssistantRun.status == "RUNNING",
            AssistantRun.lease_owner == worker_id,
            AssistantRun.attempt_count == attempt_count,
        )
        .first()
    )
    if run is None:
        raise RunLeaseLost("assistant_run_lease_lost")
    return run


def start_assistant_step(
    db: Session,
    run_id: UUID,
    *,
    worker_id: str,
    attempt_count: int,
    step_key: str,
    ordinal: int,
    input_fingerprint: str,
    tool_name: str | None = None,
) -> tuple[AssistantRunStep, bool]:
    """Durably mark a step before external work; uncertain old attempts are never replayed."""
    run = assert_assistant_run_lease(db, run_id, worker_id=worker_id, attempt_count=attempt_count)
    step = (
        db.query(AssistantRunStep)
        .filter(AssistantRunStep.run_id == run_id, AssistantRunStep.step_key == step_key)
        .first()
    )
    if step is not None:
        if step.input_fingerprint != input_fingerprint:
            raise ValueError("assistant_step_fingerprint_changed")
        if step.status == "COMPLETED":
            return step, False
        if step.status == "RUNNING":
            step.status = "UNKNOWN"
            step.safe_error = "ATTEMPT_OUTCOME_UNKNOWN"
            db.commit()
            return step, False
        if step.status == "UNKNOWN":
            return step, False

    now = datetime.now(UTC)
    if step is None:
        step = AssistantRunStep(
            run_id=run_id,
            step_key=step_key,
            ordinal=ordinal,
            tool_name=tool_name,
            status="RUNNING",
            input_fingerprint=input_fingerprint,
            attempt_count=attempt_count,
            started_at=now,
        )
        db.add(step)
    else:
        step.status = "RUNNING"
        step.safe_error = None
        step.attempt_count = attempt_count
        step.started_at = now
    run.current_stage = step_key
    run.updated_at = now
    db.commit()
    db.refresh(step)
    return step, True


def assistant_run_cancel_requested(
    db: Session, run_id: UUID, *, worker_id: str, attempt_count: int
) -> bool:
    assert_assistant_run_lease(db, run_id, worker_id=worker_id, attempt_count=attempt_count)
    value = (
        db.query(AssistantRun.cancel_requested)
        .filter(
            AssistantRun.id == run_id,
            AssistantRun.lease_owner == worker_id,
            AssistantRun.attempt_count == attempt_count,
        )
        .scalar()
    )
    return bool(value)


def save_assistant_step(
    db: Session,
    run_id: UUID,
    *,
    worker_id: str,
    attempt_count: int,
    step_key: str,
    ordinal: int,
    input_fingerprint: str,
    output_payload: dict[str, object],
    tool_name: str | None = None,
    external_effect_id: str | None = None,
) -> AssistantRunStep:
    assert_assistant_run_lease(db, run_id, worker_id=worker_id, attempt_count=attempt_count)
    step = (
        db.query(AssistantRunStep)
        .filter(AssistantRunStep.run_id == run_id, AssistantRunStep.step_key == step_key)
        .first()
    )
    if step is None:
        step = AssistantRunStep(
            run_id=run_id,
            step_key=step_key,
            ordinal=ordinal,
            tool_name=tool_name,
            status="COMPLETED",
            input_fingerprint=input_fingerprint,
            output_payload=output_payload,
            external_effect_id=external_effect_id,
            attempt_count=attempt_count,
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
        )
        db.add(step)
    elif step.input_fingerprint != input_fingerprint:
        raise ValueError("assistant_step_fingerprint_changed")
    elif step.status != "COMPLETED":
        step.status = "COMPLETED"
        step.output_payload = output_payload
        step.external_effect_id = external_effect_id
        step.finished_at = datetime.now(UTC)
        step.attempt_count = attempt_count
    db.commit()
    db.refresh(step)
    return step


def finish_assistant_run(
    db: Session,
    run_id: UUID,
    *,
    worker_id: str,
    attempt_count: int,
    status: str,
    result_payload: dict[str, object] | None = None,
    intent: str | None = None,
    route_decision: dict[str, object] | None = None,
    action_summary: str | None = None,
    safe_error: str | None = None,
    usage: dict[str, object] | None = None,
) -> AssistantRun:
    if status not in {"NEEDS_INPUT", "AWAITING_APPROVAL", "SUCCEEDED", "FAILED", "CANCELLED"}:
        raise ValueError("invalid_terminal_assistant_run_status")
    run = assert_assistant_run_lease(db, run_id, worker_id=worker_id, attempt_count=attempt_count)
    now = datetime.now(UTC)
    run.status = status
    run.result_payload = result_payload
    run.intent = intent
    run.route_decision = route_decision
    run.action_summary = action_summary
    run.safe_error = safe_error
    run.usage = usage
    run.lease_owner = None
    run.lease_expires_at = None
    run.finished_at = now if status in {"SUCCEEDED", "FAILED", "CANCELLED"} else None
    run.updated_at = now
    db.commit()
    db.refresh(run)
    return run


def release_assistant_run(db: Session, run_id: UUID, *, worker_id: str, attempt_count: int) -> bool:
    changed = (
        db.query(AssistantRun)
        .filter(
            AssistantRun.id == run_id,
            AssistantRun.status == "RUNNING",
            AssistantRun.lease_owner == worker_id,
            AssistantRun.attempt_count == attempt_count,
        )
        .update(
            {
                AssistantRun.status: "QUEUED",
                AssistantRun.lease_owner: None,
                AssistantRun.lease_expires_at: None,
                AssistantRun.updated_at: datetime.now(UTC),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    return changed == 1
