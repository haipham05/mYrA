from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.crud.assistant_run import (
    claim_next_assistant_run,
    create_assistant_approval,
    finish_assistant_run,
)
from app.db.base import Base
from app.db.models import AssistantApprovalAction, AssistantRun, Conversation, Paper, Project
from app.db.session import get_db
from app.main import app
from app.schemas.paper import PaperStatus, PaperUploadResponse


@pytest.fixture
def assistant_api_context(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'assistant-api.sqlite'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)
    project = Project(name="Assistant API")
    session.add(project)
    session.flush()
    conversation = Conversation(project_id=project.id)
    paper = Paper(project_id=project.id, filename="paper.pdf", storage_path="paper.pdf")
    session.add_all([conversation, paper])
    session.commit()

    def override_get_db():
        yield session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as client:
        yield client, session, project, conversation, paper
    app.dependency_overrides.clear()
    session.close()
    Base.metadata.drop_all(engine)
    engine.dispose()


def _request_body(project: Project, conversation: Conversation, paper: Paper) -> dict[str, object]:
    return {
        "message": "Explain this paper",
        "project_id": str(project.id),
        "conversation_id": str(conversation.id),
        "scope": "paper",
        "selected_paper_ids": [str(paper.id)],
        "intent_override": "qa",
        "idempotency_key": f"test-{uuid4().hex}",
    }


def test_submit_poll_cancel_and_resume_assistant_run(assistant_api_context):
    client, session, project, conversation, paper = assistant_api_context
    body = _request_body(project, conversation, paper)
    submitted = client.post(f"/api/v1/conversations/{conversation.id}/runs", json=body)
    assert submitted.status_code == 202
    run_id = submitted.json()["id"]
    assert submitted.json()["status"] == "QUEUED"

    polled = client.get(f"/api/v1/runs/{run_id}")
    assert polled.status_code == 200
    assert polled.json()["conversation_id"] == str(conversation.id)

    cancelled = client.post(f"/api/v1/runs/{run_id}/cancel")
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "CANCELLED"

    body["idempotency_key"] = f"resume-{uuid4().hex}"
    second = client.post(f"/api/v1/conversations/{conversation.id}/runs", json=body)
    run = session.get(AssistantRun, UUID(second.json()["id"]))
    assert run is not None
    run.status = "NEEDS_INPUT"
    run.result_payload = {"result_type": "clarification", "display_text": "Pick a paper"}
    session.commit()
    resumed = client.post(
        f"/api/v1/runs/{run.id}/resume", json={"additional_input": "Use this selected paper"}
    )
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "QUEUED"


def test_successful_run_poll_returns_tool_result_with_unknown_usage(assistant_api_context):
    client, session, project, conversation, paper = assistant_api_context
    submitted = client.post(
        f"/api/v1/conversations/{conversation.id}/runs",
        json=_request_body(project, conversation, paper),
    )
    run = session.get(AssistantRun, UUID(submitted.json()["id"]))
    assert run is not None
    run.status = "SUCCEEDED"
    run.result_payload = {
        "status": "SUCCEEDED",
        "result_type": "answer",
        "display_text": "Grounded answer.",
        "structured_payload": {"model_name": "test-model"},
        "citations": [],
        "evidence": [],
        "warnings": [],
        "usage": None,
    }
    session.commit()

    result = client.get(f"/api/v1/runs/{run.id}")

    assert result.status_code == 200
    assert result.json()["result"]["display_text"] == "Grounded answer."
    assert result.json()["result"]["usage"] == {}


def test_run_submit_rejects_path_mismatch_and_idempotency_conflict(assistant_api_context):
    client, _, project, conversation, paper = assistant_api_context
    body = _request_body(project, conversation, paper)
    mismatch = client.post(f"/api/v1/conversations/{uuid4()}/runs", json=body)
    assert mismatch.status_code == 422

    first = client.post(f"/api/v1/conversations/{conversation.id}/runs", json=body)
    assert first.status_code == 202
    changed = {**body, "message": "A different request"}
    conflict = client.post(f"/api/v1/conversations/{conversation.id}/runs", json=changed)
    assert conflict.status_code == 409


def test_approval_endpoints_expose_and_decide_persisted_proposal(assistant_api_context):
    client, session, project, conversation, paper = assistant_api_context
    response = client.post(
        f"/api/v1/conversations/{conversation.id}/runs",
        json=_request_body(project, conversation, paper),
    )
    run_id = response.json()["id"]
    run = session.get(AssistantRun, UUID(run_id))
    assert run is not None
    claim = claim_next_assistant_run(session, worker_id="worker-api")
    assert claim is not None and str(claim.id) == run_id
    action = create_assistant_approval(
        session,
        claim.id,
        worker_id="worker-api",
        attempt_count=claim.attempt_count,
        action_type="notes",
        arguments={"paper_ids": [str(paper.id)], "tool_arguments": {"text": "Save note"}},
        paper_ids=[paper.id],
    )
    finish_assistant_run(
        session,
        claim.id,
        worker_id="worker-api",
        attempt_count=claim.attempt_count,
        status="AWAITING_APPROVAL",
        result_payload={"result_type": "approval_required", "display_text": "Needs approval"},
    )

    actions = client.get(f"/api/v1/runs/{run_id}/actions")
    assert actions.status_code == 200
    assert actions.json()[0]["id"] == str(action.id)
    approved = client.post(f"/api/v1/actions/{action.id}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"
    assert client.get(f"/api/v1/runs/{run_id}").json()["status"] == "QUEUED"


def test_discovery_result_proposes_exact_candidate_without_importing(
    assistant_api_context, monkeypatch
):
    client, session, project, conversation, paper = assistant_api_context

    async def fake_import(_db, _action):
        return PaperUploadResponse(paper_id=paper.id, job_id=paper.id, status=PaperStatus.READY)

    monkeypatch.setattr("app.api.v1.assistant.import_approved_candidate", fake_import)
    candidate = {
        "catalog": "arxiv",
        "catalog_id": "2401.12345v1",
        "title": "A Catalog Candidate",
        "authors": [],
        "publication_year": None,
        "doi": None,
        "arxiv_id": "2401.12345v1",
        "abstract": None,
        "source_url": "https://arxiv.org/abs/2401.12345v1",
        "pdf_url": "https://arxiv.org/pdf/2401.12345v1",
        "open_access": True,
        "possible_duplicate": False,
    }
    run = AssistantRun(
        project_id=project.id,
        conversation_id=conversation.id,
        idempotency_key="discovery-result-run",
        request_hash="a" * 64,
        request_payload={},
        status="SUCCEEDED",
        intent="discover",
        result_payload={
            "result_type": "discovery_results",
            "display_text": "Metadata only",
            "structured_payload": {"items": [candidate]},
            "citations": [],
            "warnings": [],
            "usage": {},
        },
    )
    session.add(run)
    session.commit()

    result = client.get(f"/api/v1/runs/{run.id}")
    proposal = client.post(f"/api/v1/runs/{run.id}/discovery-import-proposals", json=candidate)

    assert result.status_code == 200
    assert result.json()["result"]["structured_payload"]["run_id"] == str(run.id)
    assert proposal.status_code == 201
    assert proposal.json()["action_type"] == "discovery_import"
    assert proposal.json()["arguments"]["candidate"] == candidate
    assert session.query(AssistantApprovalAction).count() == 1
    assert session.query(Paper).count() == 1

    approved = client.post(f"/api/v1/actions/{proposal.json()['id']}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"
    assert approved.json()["import_result"]["paper_id"] == str(paper.id)
    assert session.get(AssistantRun, run.id).status == "SUCCEEDED"
    assert session.query(Paper).count() == 1


@pytest.mark.parametrize("suffix", ["approve", "reject"])
def test_unknown_assistant_action_returns_404(assistant_api_context, suffix: str):
    client, *_ = assistant_api_context
    response = client.post(f"/api/v1/actions/{uuid4()}/{suffix}")
    assert response.status_code == 404
