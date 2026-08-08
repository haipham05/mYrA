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
from app.services.discovery.download import ImportDownloadError


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


def test_research_draft_is_saved_only_when_requested_and_once_per_run(assistant_api_context):
    client, session, project, conversation, paper = assistant_api_context
    run = AssistantRun(
        project_id=project.id,
        conversation_id=conversation.id,
        idempotency_key="research-artifact-run",
        request_hash="f" * 64,
        request_payload={"scope": "paper", "selected_paper_ids": [str(paper.id)]},
        status="SUCCEEDED",
        intent="research",
        action_summary="Review selected-paper evidence",
        route_decision={"resolved_paper_ids": [str(paper.id)]},
        result_payload={
            "status": "SUCCEEDED",
            "result_type": "research_draft",
            "display_text": "A grounded research draft [1].",
            "structured_payload": {
                "goal": "Review the method",
                "model_name": "deepseek-flash",
                "source_manifest": [
                    {
                        "evidence_id": "1",
                        "citation_index": 1,
                        "paper_id": str(paper.id),
                        "paper_title": "Selected paper",
                        "page_number": 1,
                        "chunk_id": str(uuid4()),
                        "document_sha256": "a" * 64,
                        "quote": "The selected paper supports this point.",
                    }
                ],
                "saved": False,
            },
            "citations": [],
            "evidence": [],
            "usage": {"total_tokens": 24},
        },
    )
    session.add(run)
    session.commit()

    from app.db.models import ResearchArtifact

    assert session.query(ResearchArtifact).count() == 0
    first = client.post(f"/api/v1/runs/{run.id}/artifact", json={"title": "Method review"})
    second = client.post(f"/api/v1/runs/{run.id}/artifact", json={})

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["id"] == second.json()["id"]
    assert first.json()["artifact_type"] == "report"
    assert first.json()["latest"]["payload"]["run_id"] == str(run.id)
    assert first.json()["latest"]["scope_snapshot"]["paper_ids"] == [str(paper.id)]
    assert first.json()["latest"]["config_snapshot"]["model_name"] == "deepseek-flash"
    assert first.json()["latest"]["usage"] == {"total_tokens": 24}
    assert session.query(ResearchArtifact).count() == 1


def test_run_artifact_requires_completed_result_and_verified_sources(assistant_api_context):
    client, session, project, conversation, _paper = assistant_api_context
    run = AssistantRun(
        project_id=project.id,
        conversation_id=conversation.id,
        idempotency_key="unfinished-artifact-run",
        request_hash="e" * 64,
        request_payload={},
        status="NEEDS_INPUT",
        intent="research",
        result_payload={
            "result_type": "research_draft",
            "structured_payload": {"source_manifest": []},
        },
    )
    session.add(run)
    session.commit()

    response = client.post(f"/api/v1/runs/{run.id}/artifact", json={})

    assert response.status_code == 409
    run.status = "SUCCEEDED"
    run.result_payload = {
        "result_type": "research_draft",
        "structured_payload": {"source_manifest": []},
    }
    session.commit()
    no_sources = client.post(f"/api/v1/runs/{run.id}/artifact", json={})

    assert no_sources.status_code == 409
    from app.db.models import ResearchArtifact

    assert session.query(ResearchArtifact).count() == 0


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
    import_calls = 0

    async def fake_import(_db, _action):
        nonlocal import_calls
        import_calls += 1
        if import_calls == 1:
            raise ImportDownloadError("temporary source failure")
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

    failed = client.post(f"/api/v1/actions/{proposal.json()['id']}/approve")
    assert failed.status_code == 502
    assert session.query(AssistantApprovalAction).one().status == "PENDING"

    approved = client.post(f"/api/v1/actions/{proposal.json()['id']}/approve")
    assert approved.status_code == 200
    assert approved.json()["status"] == "APPROVED"
    assert approved.json()["import_result"]["paper_id"] == str(paper.id)
    repeated = client.post(f"/api/v1/actions/{proposal.json()['id']}/approve")
    assert repeated.status_code == 200
    assert repeated.json()["status"] == "APPROVED"
    assert repeated.json()["import_result"] is None
    assert import_calls == 2
    assert session.get(AssistantRun, run.id).status == "SUCCEEDED"
    assert session.query(Paper).count() == 1


@pytest.mark.parametrize("suffix", ["approve", "reject"])
def test_unknown_assistant_action_returns_404(assistant_api_context, suffix: str):
    client, *_ = assistant_api_context
    response = client.post(f"/api/v1/actions/{uuid4()}/{suffix}")
    assert response.status_code == 404
