"""Durable assistant-run creation, lease ownership, and step checkpoints."""

import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func, or_, update
from sqlalchemy.orm import Session

from app.db.models import (
    AssistantApprovalAction,
    AssistantRun,
    AssistantRunStep,
    Conversation,
    Paper,
    Project,
)
from app.schemas.assistant import AssistantRunRequest
from app.schemas.discovery import CatalogCandidate


class IdempotencyConflict(ValueError):
    """A request key was reused with a different normalized request."""


class RunLeaseLost(RuntimeError):
    """The caller no longer owns the current run attempt."""


ASSISTANT_PUBLISH_STAGE = "assistant.publish"


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


def request_assistant_run_cancel(db: Session, run_id: UUID) -> AssistantRun | None:
    run = get_assistant_run(db, run_id)
    if run is None:
        return None
    if run.status in {"SUCCEEDED", "FAILED", "CANCELLED"}:
        return run
    now = datetime.now(UTC)
    if run.status == "RUNNING":
        db.execute(
            update(AssistantRun)
            .where(
                AssistantRun.id == run_id,
                AssistantRun.status == "RUNNING",
                AssistantRun.cancel_requested.is_(False),
                or_(
                    AssistantRun.current_stage.is_(None),
                    AssistantRun.current_stage != ASSISTANT_PUBLISH_STAGE,
                ),
            )
            .values(cancel_requested=True, updated_at=now)
        )
        db.commit()
        db.refresh(run)
        return run
    run.cancel_requested = True
    run.updated_at = now
    run.status = "CANCELLED"
    run.finished_at = now
    db.query(AssistantApprovalAction).filter(
        AssistantApprovalAction.run_id == run_id,
        AssistantApprovalAction.status == "PENDING",
    ).update(
        {AssistantApprovalAction.status: "CANCELLED", AssistantApprovalAction.decided_at: now},
        synchronize_session=False,
    )
    db.commit()
    db.refresh(run)
    return run


def begin_assistant_run_publication(
    db: Session, run_id: UUID, *, worker_id: str, attempt_count: int
) -> bool:
    """Atomically win the cancellation race before persisting a QA assistant message."""
    result = db.execute(
        update(AssistantRun)
        .where(
            AssistantRun.id == run_id,
            AssistantRun.status == "RUNNING",
            AssistantRun.lease_owner == worker_id,
            AssistantRun.attempt_count == attempt_count,
            AssistantRun.cancel_requested.is_(False),
        )
        .values(current_stage=ASSISTANT_PUBLISH_STAGE, updated_at=datetime.now(UTC))
    )
    return result.rowcount == 1


def resume_assistant_run(
    db: Session, run_id: UUID, *, additional_input: str
) -> AssistantRun | None:
    run = get_assistant_run(db, run_id)
    if run is None:
        return None
    if run.status != "NEEDS_INPUT":
        raise ValueError("assistant_run_not_waiting_for_input")
    clarification = additional_input.strip()
    if not clarification or len(clarification) > 4000:
        raise ValueError("assistant_resume_input_invalid")
    payload = dict(run.request_payload)
    previous_message = str(payload.get("message", ""))
    combined_message = f"{previous_message}\n\nAdditional user input: {clarification}"
    if len(combined_message) > 10_000:
        raise ValueError("assistant_resume_input_too_long")
    payload["message"] = combined_message
    request = AssistantRunRequest.model_validate(payload)
    request_hash, normalized_payload = _request_hash(request)
    run.request_payload = normalized_payload
    run.request_hash = request_hash
    run.resume_count += 1
    run.status = "QUEUED"
    run.current_stage = None
    run.result_payload = None
    run.safe_error = None
    run.cancel_requested = False
    run.finished_at = None
    run.updated_at = datetime.now(UTC)
    db.commit()
    db.refresh(run)
    return run


def create_assistant_approval(
    db: Session,
    run_id: UUID,
    *,
    worker_id: str,
    attempt_count: int,
    action_type: str,
    arguments: dict[str, object],
    paper_ids: list[UUID],
    expires_in_seconds: int = 1800,
) -> AssistantApprovalAction:
    """Persist an exact proposal against the current project/corpus/source identities."""
    if expires_in_seconds <= 0 or expires_in_seconds > 86_400:
        raise ValueError("assistant_approval_expiry_out_of_range")
    run = assert_assistant_run_lease(db, run_id, worker_id=worker_id, attempt_count=attempt_count)
    source_fingerprint = _approval_source_fingerprint(db, run.project_id, paper_ids)
    if source_fingerprint is None:
        raise ValueError("assistant_approval_source_missing")
    action_payload = {"action_type": action_type, "arguments": arguments}
    idempotency_key = hashlib.sha256(
        json.dumps(
            {"run_id": str(run_id), **action_payload, "source": source_fingerprint},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    existing = (
        db.query(AssistantApprovalAction)
        .filter(
            AssistantApprovalAction.run_id == run_id,
            AssistantApprovalAction.idempotency_key == idempotency_key,
        )
        .first()
    )
    if existing is not None:
        return existing
    action = AssistantApprovalAction(
        run_id=run_id,
        action_type=action_type,
        arguments=action_payload["arguments"],
        source_fingerprint=source_fingerprint,
        idempotency_key=idempotency_key,
        status="PENDING",
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def create_discovery_import_approval(
    db: Session, run_id: UUID, candidate: CatalogCandidate, *, expires_in_seconds: int = 1800
) -> AssistantApprovalAction:
    """Persist a user's exact catalog candidate proposal without downloading it."""
    if expires_in_seconds <= 0 or expires_in_seconds > 86_400:
        raise ValueError("assistant_approval_expiry_out_of_range")
    run = get_assistant_run(db, run_id)
    if run is None:
        raise ValueError("assistant_run_not_found")
    if run.status != "SUCCEEDED" or run.intent != "discover" or not run.result_payload:
        raise ValueError("discovery_run_not_available")

    result = run.result_payload
    structured = result.get("structured_payload") if isinstance(result, dict) else None
    stored_items = structured.get("items") if isinstance(structured, dict) else None
    candidate_payload = candidate.model_dump(mode="json")
    if not isinstance(stored_items, list) or candidate_payload not in stored_items:
        raise ValueError("discovery_candidate_not_in_run")

    source_fingerprint = _approval_source_fingerprint(db, run.project_id, [])
    if source_fingerprint is None:
        raise ValueError("assistant_approval_source_missing")
    arguments = {"paper_ids": [], "candidate": candidate_payload}
    idempotency_key = hashlib.sha256(
        json.dumps(
            {
                "run_id": str(run_id),
                "action_type": "discovery_import",
                "arguments": arguments,
                "source": source_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    existing = (
        db.query(AssistantApprovalAction)
        .filter(
            AssistantApprovalAction.run_id == run_id,
            AssistantApprovalAction.idempotency_key == idempotency_key,
        )
        .first()
    )
    if existing is not None:
        return existing

    action = AssistantApprovalAction(
        run_id=run_id,
        action_type="discovery_import",
        arguments=arguments,
        source_fingerprint=source_fingerprint,
        idempotency_key=idempotency_key,
        status="PENDING",
        expires_at=datetime.now(UTC) + timedelta(seconds=expires_in_seconds),
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def _approval_source_fingerprint(
    db: Session, project_id: UUID, paper_ids: list[UUID]
) -> str | None:
    project = db.query(Project).filter(Project.id == project_id).first()
    if project is None:
        return None
    rows = []
    if paper_ids:
        papers = (
            db.query(Paper)
            .filter(Paper.project_id == project_id, Paper.id.in_(paper_ids))
            .order_by(Paper.id.asc())
            .all()
        )
        if len(papers) != len(set(paper_ids)):
            return None
        rows = [
            {"paper_id": str(paper.id), "sha256": paper.document_sha256, "status": paper.status}
            for paper in papers
        ]
    identity = {
        "project_id": str(project.id),
        "corpus_revision": project.corpus_revision,
        "papers": rows,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def list_assistant_approvals(db: Session, run_id: UUID) -> list[AssistantApprovalAction]:
    return (
        db.query(AssistantApprovalAction)
        .filter(AssistantApprovalAction.run_id == run_id)
        .order_by(AssistantApprovalAction.created_at.asc(), AssistantApprovalAction.id.asc())
        .all()
    )


def get_valid_approved_assistant_action(
    db: Session,
    run_id: UUID,
    *,
    action_type: str,
    expected_arguments: dict[str, object],
    paper_ids: list[UUID],
) -> AssistantApprovalAction | None:
    """Return the exact approved proposal, invalidating it if scope or sources changed."""
    action = (
        db.query(AssistantApprovalAction)
        .filter(
            AssistantApprovalAction.run_id == run_id,
            AssistantApprovalAction.action_type == action_type,
            AssistantApprovalAction.status == "APPROVED",
        )
        .order_by(AssistantApprovalAction.decided_at.desc(), AssistantApprovalAction.id.desc())
        .first()
    )
    if action is None:
        return None
    run = db.query(AssistantRun).filter(AssistantRun.id == run_id).first()
    expected_fingerprint = (
        _approval_source_fingerprint(db, run.project_id, paper_ids) if run is not None else None
    )
    if (
        action.arguments != expected_arguments
        or expected_fingerprint is None
        or expected_fingerprint != action.source_fingerprint
    ):
        action.status = "STALE"
        action.decided_at = datetime.now(UTC)
        db.commit()
        raise ValueError("assistant_approval_no_longer_matches_current_request")
    return action


def decide_assistant_approval(
    db: Session, action_id: UUID, *, approve: bool
) -> tuple[AssistantApprovalAction, bool] | None:
    """Approve the exact pending proposal only while its source fingerprint is current."""
    action = (
        db.query(AssistantApprovalAction)
        .filter(AssistantApprovalAction.id == action_id)
        .with_for_update()
        .first()
    )
    if action is None:
        return None
    desired_status = "APPROVED" if approve else "REJECTED"
    if action.status == desired_status:
        return action, False
    if action.status != "PENDING":
        raise ValueError("assistant_approval_already_decided")
    run = db.query(AssistantRun).filter(AssistantRun.id == action.run_id).first()
    discovery_import = action.action_type == "discovery_import"
    required_run_status = "SUCCEEDED" if discovery_import else "AWAITING_APPROVAL"
    if run is None or run.status != required_run_status:
        raise ValueError("assistant_run_not_awaiting_approval")

    now = datetime.now(UTC)
    expiry = action.expires_at
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=UTC)
    if expiry <= now:
        action.status = "EXPIRED"
        action.decided_at = now
        if not discovery_import:
            run.status = "FAILED"
            run.safe_error = "APPROVAL_EXPIRED"
            run.finished_at = now
            run.updated_at = now
        db.commit()
        db.refresh(action)
        return action, False

    if approve:
        raw_paper_ids = action.arguments.get("paper_ids", [])
        try:
            paper_ids = [UUID(str(value)) for value in raw_paper_ids]
        except (TypeError, ValueError):
            paper_ids = []
            source_fingerprint = None
        else:
            source_fingerprint = _approval_source_fingerprint(db, run.project_id, paper_ids)
        if source_fingerprint != action.source_fingerprint:
            action.status = "STALE"
            action.decided_at = now
            if not discovery_import:
                run.status = "FAILED"
                run.safe_error = "APPROVAL_SOURCE_CHANGED"
                run.finished_at = now
                run.updated_at = now
            db.commit()
            db.refresh(action)
            return action, False
        if not discovery_import:
            run.status = "QUEUED"
            run.current_stage = None
            run.result_payload = None
            run.safe_error = None
            run.finished_at = None
            run.cancel_requested = False
            run.lease_owner = None
            run.lease_expires_at = None
    else:
        if not discovery_import:
            run.status = "CANCELLED"
            run.safe_error = "ACTION_REJECTED"
            run.finished_at = now
    action.status = desired_status
    action.decided_at = now
    if not discovery_import:
        run.updated_at = now
    db.commit()
    db.refresh(action)
    return action, True


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
