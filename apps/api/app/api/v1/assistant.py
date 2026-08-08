"""Channel-independent assistant run and approval endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.v1.artifacts import _response as artifact_response
from app.crud.artifact import create_artifact, list_artifacts
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
from app.schemas.artifact import ArtifactCreate, ArtifactResponse, ArtifactType
from app.schemas.assistant import (
    AssistantApprovalResponse,
    AssistantRunArtifactSaveRequest,
    AssistantRunRequest,
    AssistantRunResponse,
    AssistantRunResult,
    AssistantRunResumeRequest,
)
from app.schemas.discovery import CatalogCandidate
from app.schemas.evidence import Citation, EvidenceItem
from app.services.discovery.download import ImportDownloadError
from app.services.discovery.importer import import_approved_candidate
from app.services.research_report import report_source_manifest

router = APIRouter(tags=["assistant"])

_RUN_ARTIFACT_TYPES = {
    "reading_brief": ArtifactType.READING_BRIEF,
    "comparison": ArtifactType.COMPARISON,
    "claim_verification": ArtifactType.CLAIM_CHECK,
    "research_report": ArtifactType.REPORT,
    "research_draft": ArtifactType.REPORT,
    "gap_analysis": ArtifactType.GAP_ANALYSIS,
    "experiment_proposal": ArtifactType.EXPERIMENT_PROPOSAL,
}


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


@router.post(
    "/runs/{run_id}/artifact",
    response_model=ArtifactResponse,
    status_code=status.HTTP_201_CREATED,
)
def save_assistant_run_artifact(
    run_id: UUID,
    request: AssistantRunArtifactSaveRequest,
    db: Session = Depends(get_db),
) -> ArtifactResponse:
    """Explicitly save a completed, cited assistant draft as a versioned project artifact."""
    run = _get_run_or_404(db, run_id)
    result = run.result_payload if isinstance(run.result_payload, dict) else {}
    result_type = result.get("result_type")
    artifact_type = _RUN_ARTIFACT_TYPES.get(result_type)
    structured = result.get("structured_payload")
    if run.status != "SUCCEEDED" or artifact_type is None or not isinstance(structured, dict):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only completed research drafts can be saved as artifacts",
        )

    existing = next(
        (
            artifact
            for artifact in list_artifacts(db, run.project_id)
            if any(
                revision.scope_snapshot.get("assistant_run_id") == str(run.id)
                for revision in artifact.revisions
            )
        ),
        None,
    )
    if existing is not None:
        response = artifact_response(db, existing)
        assert isinstance(response, ArtifactResponse)
        return response

    citations = [Citation.model_validate(item) for item in result.get("citations", [])]
    evidence = [EvidenceItem.model_validate(item) for item in result.get("evidence", [])]
    manifest = structured.get("source_manifest")
    if not isinstance(manifest, list):
        manifest = report_source_manifest(citations, evidence)
    if not manifest:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A verified source is required before saving this artifact",
        )

    request_payload = run.request_payload if isinstance(run.request_payload, dict) else {}
    route_decision = run.route_decision if isinstance(run.route_decision, dict) else {}
    paper_ids = route_decision.get("resolved_paper_ids") or request_payload.get(
        "selected_paper_ids", []
    )
    title = request.title or run.action_summary or result_type.replace("_", " ").title()
    data = ArtifactCreate(
        artifact_type=artifact_type,
        title=title,
        payload={
            "markdown": result.get("display_text", ""),
            "structured_payload": structured,
            "citations": result.get("citations", []),
            "run_id": str(run.id),
        },
        scope_snapshot={
            "assistant_run_id": str(run.id),
            "scope": request_payload.get("scope", "project"),
            "paper_ids": paper_ids,
        },
        source_manifest=manifest,
        config_snapshot={
            "intent": run.intent,
            "model_name": structured.get("model_name"),
        },
        usage=result.get("usage") or {},
    )
    artifact = create_artifact(db, project_id=run.project_id, data=data)
    response = artifact_response(db, artifact)
    assert isinstance(response, ArtifactResponse)
    return response


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
        decision = decide_assistant_approval(db, action_id, approve=approve)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    if decision is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Assistant approval action not found"
        )
    action, transitioned = decision
    response = _approval_response(action)
    if response.status in {"STALE", "EXPIRED"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=response.status)
    if (
        approve
        and transitioned
        and action.action_type == "discovery_import"
        and response.status == "APPROVED"
    ):
        try:
            response.import_result = await import_approved_candidate(db, action)
        except ImportDownloadError as exc:
            action.status = "PENDING"
            action.decided_at = None
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"DISCOVERY_IMPORT_FAILED: {exc}",
            ) from exc
        except Exception:
            action.status = "PENDING"
            action.decided_at = None
            db.commit()
            raise
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
