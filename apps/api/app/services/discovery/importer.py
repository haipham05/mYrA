"""Run an approved catalog import through the existing PDF upload/ingestion path."""

from __future__ import annotations

import re
from io import BytesIO
from uuid import UUID

from fastapi import Response, UploadFile
from sqlalchemy.orm import Session

from app.db.models import AssistantApprovalAction, AssistantRun, Job, Paper
from app.observability.telemetry import TelemetryAdapter, get_telemetry
from app.schemas.discovery import CatalogCandidate
from app.schemas.paper import PaperStatus, PaperUploadResponse
from app.services.discovery.download import (
    DownloadedPdf,
    download_open_access_pdf,
)

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)


def _normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    result = _DOI_PREFIX.sub("", value.strip()).strip().rstrip(".,;)").casefold()
    return result or None


def _normalize_arxiv(value: str | None) -> str | None:
    if not value:
        return None
    result = value.strip().casefold()
    for prefix in ("https://arxiv.org/abs/", "https://arxiv.org/pdf/", "arxiv:"):
        if result.startswith(prefix):
            return result.removeprefix(prefix)
    return result


def _matching_paper(db: Session, project_id: UUID, candidate: CatalogCandidate) -> Paper | None:
    doi = _normalize_doi(candidate.doi)
    arxiv_id = _normalize_arxiv(candidate.arxiv_id)
    if doi is None and arxiv_id is None:
        return None
    papers = (
        db.query(Paper)
        .filter(Paper.project_id == project_id)
        .filter((Paper.doi.is_not(None)) | (Paper.arxiv_id.is_not(None)))
        .all()
    )
    return next(
        (
            paper
            for paper in papers
            if (doi is not None and _normalize_doi(paper.doi) == doi)
            or (arxiv_id is not None and _normalize_arxiv(paper.arxiv_id) == arxiv_id)
        ),
        None,
    )


def _existing_upload_response(db: Session, paper: Paper) -> PaperUploadResponse:
    latest_job = (
        db.query(Job).filter(Job.paper_id == paper.id).order_by(Job.created_at.desc()).first()
    )
    return PaperUploadResponse(
        paper_id=paper.id,
        job_id=latest_job.id if latest_job else paper.id,
        status=PaperStatus(paper.status),
    )


async def import_approved_candidate(
    db: Session,
    action: AssistantApprovalAction,
    *,
    telemetry: TelemetryAdapter | None = None,
    downloaded: DownloadedPdf | None = None,
) -> PaperUploadResponse:
    """Download only an approved candidate, then reuse normal upload validation and queueing."""
    if action.action_type != "discovery_import" or action.status != "APPROVED":
        raise ValueError("discovery_import_not_approved")
    run = db.query(AssistantRun).filter(AssistantRun.id == action.run_id).first()
    if run is None:
        raise ValueError("discovery_run_not_found")
    candidate_payload = action.arguments.get("candidate")
    candidate = CatalogCandidate.model_validate(candidate_payload)

    existing = _matching_paper(db, run.project_id, candidate)
    if existing is not None:
        return _existing_upload_response(db, existing)

    telemetry = telemetry or get_telemetry()
    with telemetry.stage(
        "import.download",
        metadata={"catalog": candidate.catalog, "open_access": candidate.open_access},
    ) as observation:
        downloaded = downloaded or await download_open_access_pdf(candidate)
        if observation is not None:
            observation.update(
                metadata={
                    "outcome": "downloaded",
                    "sha256": downloaded.sha256,
                    "bytes": len(downloaded.data),
                }
            )

    # Use the ordinary upload validator/storage/idempotency/job creator so imported PDFs
    # follow exactly the same worker ingestion path as a user-selected file.
    from app.api.v1.projects import _upload_paper  # avoid router import cycle at module load

    filename = f"discovery-{downloaded.sha256[:16]}.pdf"
    file = UploadFile(filename=filename, file=BytesIO(downloaded.data))
    response = Response()
    upload = await _upload_paper(
        project_id=run.project_id,
        response=response,
        file=file,
        idempotency_key=f"discovery-{action.id.hex}",
        db=db,
        telemetry=telemetry,
    )
    paper = db.query(Paper).filter(Paper.id == upload.paper_id).first()
    if paper is not None:
        updates = {
            "title": candidate.title,
            "authors": candidate.authors or None,
            "publication_year": candidate.publication_year,
            "doi": candidate.doi,
            "arxiv_id": candidate.arxiv_id,
            "abstract": candidate.abstract,
            "source_url": candidate.source_url,
        }
        for field, value in updates.items():
            if getattr(paper, field) is None and value is not None:
                setattr(paper, field, value)
        provenance = dict(paper.metadata_provenance or {})
        for field, value in updates.items():
            if value is not None:
                provenance.setdefault(field, f"catalog:{candidate.catalog}")
        paper.metadata_provenance = provenance
        db.commit()
    return upload
