"""Channel-independent assistant run and approval endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.crud.assistant_run import (
    IdempotencyConflict,
    create_assistant_run,
    create_discovery_import_approval,
    decide_assistant_approval,
    get_assistant_run,
    list_assistant_approvals,
    request_assistant_run_cancel,
    resume_assistant_run,
)
from app.db.models import AssistantApprovalAction, AssistantRun
from app.db.session import get_db
from app.schemas.assistant import (
    AssistantApprovalResponse,
    AssistantRunRequest,
    AssistantRunResponse,
    AssistantRunResult,
    AssistantRunResumeRequest,
)
from app.schemas.discovery import CatalogCandidate
from app.services.discovery.download import ImportDownloadError
from app.services.discovery.importer import import_approved_candidate

router = APIRouter(tags=["assistant"])


def _run_response(run: AssistantRun) -> AssistantRunResponse:
    result = None
    if run.result_payload is not None:
        try:
            payload = dict(run.result_payload)
            # Tool payloads also persist their execution status; public run results
            # use a compact schema and represent unknown usage as an empty object.
            payload.pop("status", None)
            structured_payload = payload.get("structured_payload")
            if not isinstance(structured_payload, dict):
                structured_payload = {}
                payload["structured_payload"] = structured_payload
            structured_payload.setdefault("run_id", str(run.id))
            if payload.get("usage") is None:
                payload["usage"] = {}
            result = AssistantRunResult.model_validate(payload)
        except ValueError:
            # Persisted JSON is internal state, but malformed content must not break polling.
            result = None
    return AssistantRunResponse(
        id=run.id,
        project_id=run.project_id,
        conversation_id=run.conversation_id,
        status=run.status,
        intent=run.intent,
        action_summary=run.action_summary,
        stage=run.current_stage,
        result=result,
        safe_error=run.safe_error,
        usage=run.usage,
        created_at=run.created_at,
        updated_at=run.updated_at,
    )


def _approval_response(action: AssistantApprovalAction) -> AssistantApprovalResponse:
    return AssistantApprovalResponse.model_validate(action, from_attributes=True)


def _get_run_or_404(db: Session, run_id: UUID) -> AssistantRun:
    run = get_assistant_run(db, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assistant run not found")
    return run


@router.post(
    "/conversations/{conversation_id}/runs",
    response_model=AssistantRunResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_assistant_run(
    conversation_id: UUID,
    request: AssistantRunRequest,
    db: Session = Depends(get_db),
) -> AssistantRunResponse:
    if request.conversation_id != conversation_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Request conversation does not match the URL",
        )
    try:
        run, _ = create_assistant_run(db, request)
    except IdempotencyConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        code = str(exc)
        http_status = (
            status.HTTP_404_NOT_FOUND
            if code
            in {"conversation_not_found", "parent_run_not_found", "selected_paper_not_found"}
            else status.HTTP_422_UNPROCESSABLE_ENTITY
        )
        raise HTTPException(status_code=http_status, detail=code) from exc
    return _run_response(run)


@router.get("/runs/{run_id}", response_model=AssistantRunResponse)
def inspect_assistant_run(run_id: UUID, db: Session = Depends(get_db)) -> AssistantRunResponse:
    return _run_response(_get_run_or_404(db, run_id))


@router.post("/runs/{run_id}/cancel", response_model=AssistantRunResponse)
def cancel_assistant_run(run_id: UUID, db: Session = Depends(get_db)) -> AssistantRunResponse:
    run = request_assistant_run_cancel(db, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assistant run not found")
    return _run_response(run)


@router.post("/runs/{run_id}/resume", response_model=AssistantRunResponse)
def continue_assistant_run(
    run_id: UUID,
    request: AssistantRunResumeRequest,
    db: Session = Depends(get_db),
) -> AssistantRunResponse:
    try:
        run = resume_assistant_run(db, run_id, additional_input=request.additional_input)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assistant run not found")
    return _run_response(run)


@router.get("/runs/{run_id}/actions", response_model=list[AssistantApprovalResponse])
def inspect_assistant_actions(
    run_id: UUID, db: Session = Depends(get_db)
) -> list[AssistantApprovalResponse]:
    _get_run_or_404(db, run_id)
    return [_approval_response(action) for action in list_assistant_approvals(db, run_id)]


@router.post(
    "/runs/{run_id}/discovery-import-proposals",
    response_model=AssistantApprovalResponse,
    status_code=status.HTTP_201_CREATED,
)
def propose_discovery_import(
    run_id: UUID,
    candidate: CatalogCandidate,
    db: Session = Depends(get_db),
) -> AssistantApprovalResponse:
    try:
        action = create_discovery_import_approval(db, run_id, candidate)
    except ValueError as exc:
        code = str(exc)
        http_status = (
            status.HTTP_404_NOT_FOUND
            if code == "assistant_run_not_found"
            else status.HTTP_409_CONFLICT
        )
        raise HTTPException(status_code=http_status, detail=code) from exc
    return _approval_response(action)


async def _decide_action(
    action_id: UUID, *, approve: bool, db: Session
) -> AssistantApprovalResponse:
    try:
        action = decide_assistant_approval(db, action_id, approve=approve)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if action is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Assistant approval action not found"
        )
    response = _approval_response(action)
    if response.status in {"STALE", "EXPIRED"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=response.status)
    if approve and action.action_type == "discovery_import" and response.status == "APPROVED":
        try:
            response.import_result = await import_approved_candidate(db, action)
        except ImportDownloadError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"DISCOVERY_IMPORT_FAILED: {exc}",
            ) from exc
    return response


@router.post("/actions/{action_id}/approve", response_model=AssistantApprovalResponse)
async def approve_assistant_action(
    action_id: UUID, db: Session = Depends(get_db)
) -> AssistantApprovalResponse:
    return await _decide_action(action_id, approve=True, db=db)


@router.post("/actions/{action_id}/reject", response_model=AssistantApprovalResponse)
async def reject_assistant_action(
    action_id: UUID, db: Session = Depends(get_db)
) -> AssistantApprovalResponse:
    return await _decide_action(action_id, approve=False, db=db)
