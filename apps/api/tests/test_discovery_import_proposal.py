import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.crud.assistant_run import create_discovery_import_approval, decide_assistant_approval
from app.db.base import Base
from app.db.models import AssistantApprovalAction, AssistantRun, Conversation, Paper, Project
from app.schemas.discovery import CatalogCandidate


@pytest.fixture
def proposal_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'discovery-proposal.db'}")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        project = Project(name="Discovery proposal project")
        db.add(project)
        db.flush()
        conversation = Conversation(project_id=project.id, title="Discovery")
        db.add(conversation)
        db.flush()
        candidate = CatalogCandidate(
            catalog="arxiv",
            catalog_id="2401.12345v1",
            title="A Catalog Candidate",
            arxiv_id="2401.12345v1",
            source_url="https://arxiv.org/abs/2401.12345v1",
            pdf_url="https://arxiv.org/pdf/2401.12345v1",
            open_access=True,
        )
        run = AssistantRun(
            project_id=project.id,
            conversation_id=conversation.id,
            idempotency_key="discovery-run-test",
            request_hash="a" * 64,
            request_payload={},
            status="SUCCEEDED",
            intent="discover",
            result_payload={
                "result_type": "discovery_results",
                "structured_payload": {
                    "items": [candidate.model_dump(mode="json")],
                },
            },
        )
        db.add(run)
        db.commit()
        yield db, project, run, candidate
    engine.dispose()


def test_proposal_is_persisted_idempotently_without_download_or_paper(proposal_db):
    db, _project, run, candidate = proposal_db

    action = create_discovery_import_approval(db, run.id, candidate)
    repeated = create_discovery_import_approval(db, run.id, candidate)

    assert action.id == repeated.id
    assert action.status == "PENDING"
    assert action.action_type == "discovery_import"
    assert action.arguments["candidate"] == candidate.model_dump(mode="json")
    assert db.query(Paper).count() == 0
    assert db.query(AssistantApprovalAction).count() == 1


def test_candidate_must_be_in_the_completed_discovery_result(proposal_db):
    db, _project, run, candidate = proposal_db
    candidate.title = "A different paper"

    with pytest.raises(ValueError, match="candidate_not_in_run"):
        create_discovery_import_approval(db, run.id, candidate)

    assert db.query(AssistantApprovalAction).count() == 0
    assert db.query(Paper).count() == 0


def test_import_approval_does_not_rerun_or_mutate_discovery_run(proposal_db):
    db, _project, run, candidate = proposal_db
    action = create_discovery_import_approval(db, run.id, candidate)

    decided, transitioned = decide_assistant_approval(db, action.id, approve=True)

    assert decided.status == "APPROVED"
    assert transitioned is True
    assert run.status == "SUCCEEDED"
    assert run.result_payload["result_type"] == "discovery_results"
    assert db.query(Paper).count() == 0


def test_changed_project_corpus_marks_pending_import_proposal_stale(proposal_db):
    db, project, run, candidate = proposal_db
    action = create_discovery_import_approval(db, run.id, candidate)
    project.corpus_revision += 1
    db.commit()

    decided, transitioned = decide_assistant_approval(db, action.id, approve=True)

    assert decided.status == "STALE"
    assert transitioned is False
    assert run.status == "SUCCEEDED"
    assert db.query(Paper).count() == 0


def test_rejection_leaves_discovery_run_complete(proposal_db):
    db, _project, run, candidate = proposal_db
    action = create_discovery_import_approval(db, run.id, candidate)

    decided, transitioned = decide_assistant_approval(db, action.id, approve=False)

    assert decided.status == "REJECTED"
    assert transitioned is True
    assert run.status == "SUCCEEDED"
    assert db.query(Paper).count() == 0
