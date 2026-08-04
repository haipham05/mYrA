import anyio
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.crud.assistant_run import create_discovery_import_approval, decide_assistant_approval
from app.crud.paper import create_paper_with_job
from app.db.base import Base
from app.db.models import AssistantRun, Conversation, Job, Paper, Project
from app.schemas.discovery import CatalogCandidate
from app.schemas.paper import PaperStatus, PaperUploadResponse
from app.services.discovery.download import DownloadedPdf, ImportDownloadError
from app.services.discovery.importer import import_approved_candidate


@pytest.fixture
def import_context(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'discovery-import.db'}")
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as db:
        project = Project(name="Discovery import project")
        db.add(project)
        db.flush()
        conversation = Conversation(project_id=project.id, title="Discovery")
        db.add(conversation)
        db.flush()
        candidate = CatalogCandidate(
            catalog="arxiv",
            catalog_id="2401.12345v1",
            title="Imported Paper",
            authors=["Author One"],
            publication_year=2024,
            arxiv_id="2401.12345v1",
            abstract="Catalog abstract",
            source_url="https://arxiv.org/abs/2401.12345v1",
            pdf_url="https://arxiv.org/pdf/2401.12345v1",
            open_access=True,
        )
        run = AssistantRun(
            project_id=project.id,
            conversation_id=conversation.id,
            idempotency_key="discovery-import-run",
            request_hash="b" * 64,
            request_payload={},
            status="SUCCEEDED",
            intent="discover",
            result_payload={
                "result_type": "discovery_results",
                "structured_payload": {"items": [candidate.model_dump(mode="json")]},
            },
        )
        db.add(run)
        db.commit()
        action = create_discovery_import_approval(db, run.id, candidate)
        action, transitioned = decide_assistant_approval(db, action.id, approve=True)
        assert transitioned is True
        yield db, project, candidate, action
    engine.dispose()


def test_approved_import_reuses_regular_upload_and_creates_job(import_context, monkeypatch):
    db, project, candidate, action = import_context
    pdf = DownloadedPdf(
        data=b"%PDF-1.7 fixture",
        final_url=candidate.pdf_url,
        sha256="a" * 64,
    )
    calls = []

    async def fake_upload_paper(**kwargs):
        calls.append(kwargs)
        paper, job = create_paper_with_job(
            kwargs["db"],
            kwargs["project_id"],
            kwargs["file"].filename,
            f"papers/{project.id}/{pdf.sha256}.pdf",
            document_sha256=pdf.sha256,
        )
        return PaperUploadResponse(paper_id=paper.id, job_id=job.id, status=PaperStatus.PROCESSING)

    monkeypatch.setattr("app.api.v1.projects._upload_paper", fake_upload_paper, raising=False)

    uploaded = anyio.run(lambda: import_approved_candidate(db, action, downloaded=pdf))

    paper = db.get(Paper, uploaded.paper_id)
    assert len(calls) == 1
    assert calls[0]["idempotency_key"] == f"discovery-{action.id.hex}"
    assert paper is not None
    assert paper.project_id == project.id
    assert paper.status == "PROCESSING"
    assert paper.title == candidate.title
    assert paper.authors == candidate.authors
    assert paper.arxiv_id == candidate.arxiv_id
    assert paper.metadata_provenance["title"] == "catalog:arxiv"
    assert db.query(Job).filter(Job.paper_id == paper.id).count() == 1


def test_existing_arxiv_identity_is_returned_without_download(import_context):
    db, project, candidate, action = import_context
    paper = Paper(
        project_id=project.id,
        filename="existing.pdf",
        storage_path="papers/existing.pdf",
        status="READY",
        arxiv_id="https://arxiv.org/abs/2401.12345v1",
    )
    db.add(paper)
    db.commit()

    uploaded = anyio.run(import_approved_candidate, db, action)

    assert uploaded.paper_id == paper.id
    assert uploaded.status is PaperStatus.READY
    assert db.query(Paper).count() == 1


def test_pending_action_cannot_download_or_create_paper(import_context):
    db, _project, candidate, action = import_context
    action.status = "PENDING"
    db.commit()

    with pytest.raises(ValueError, match="not_approved"):
        anyio.run(import_approved_candidate, db, action)
    assert db.query(Paper).count() == 0


def test_download_failure_leaves_no_paper(import_context, monkeypatch):
    db, _project, _candidate, action = import_context

    async def failed_download(_candidate):
        raise ImportDownloadError("source unavailable")

    monkeypatch.setattr("app.services.discovery.importer.download_open_access_pdf", failed_download)

    with pytest.raises(ImportDownloadError, match="source unavailable"):
        anyio.run(import_approved_candidate, db, action)
    assert db.query(Paper).count() == 0
