from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.orm import Session

from app.crud.artifact import (
    create_artifact,
    create_artifact_revision,
    get_artifact,
    list_artifacts,
)
from app.db.models import Project, ResearchArtifact
from app.db.session import get_db
from app.observability.telemetry import get_telemetry
from app.schemas.artifact import (
    ArtifactCreate,
    ArtifactDetailResponse,
    ArtifactExportFormat,
    ArtifactResponse,
    ArtifactRevisionCreate,
    ArtifactRevisionResponse,
)
from app.services.artifact_export import export_bibtex, export_comparison_csv, export_markdown
from app.services.artifact_sources import revalidate_source_manifest

router = APIRouter(prefix="/projects/{project_id}/artifacts", tags=["artifacts"])


def _response(
    db: Session, artifact: ResearchArtifact, *, include_history: bool = False
) -> ArtifactResponse | ArtifactDetailResponse:
    revisions = []
    for revision in artifact.revisions:
        response = ArtifactRevisionResponse.model_validate(revision, from_attributes=True)
        response.source_manifest = revalidate_source_manifest(
            db, project_id=artifact.project_id, source_manifest=response.source_manifest
        )
        revisions.append(response)
    latest = next(
        revision for revision in revisions if revision.revision_number == artifact.latest_revision
    )
    payload = {
        "id": artifact.id,
        "project_id": artifact.project_id,
        "artifact_type": artifact.artifact_type,
        "title": artifact.title,
        "latest_revision": artifact.latest_revision,
        "created_at": artifact.created_at,
        "updated_at": artifact.updated_at,
        "latest": latest,
    }
    if include_history:
        return ArtifactDetailResponse(**payload, revisions=revisions)
    return ArtifactResponse(**payload)


@router.get("", response_model=list[ArtifactResponse])
def get_project_artifacts(
    project_id: UUID, db: Session = Depends(get_db)
) -> list[ArtifactResponse]:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return [_response(db, item) for item in list_artifacts(db, project_id)]


@router.post("", response_model=ArtifactResponse, status_code=status.HTTP_201_CREATED)
def save_artifact(
    project_id: UUID,
    data: ArtifactCreate,
    db: Session = Depends(get_db),
) -> ArtifactResponse:
    if db.get(Project, project_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return _response(db, create_artifact(db, project_id=project_id, data=data))


@router.get("/{artifact_id}", response_model=ArtifactDetailResponse)
def get_project_artifact(
    project_id: UUID, artifact_id: UUID, db: Session = Depends(get_db)
) -> ArtifactDetailResponse:
    artifact = get_artifact(db, project_id=project_id, artifact_id=artifact_id)
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return _response(db, artifact, include_history=True)


@router.post("/{artifact_id}/revisions", response_model=ArtifactResponse)
def save_artifact_revision(
    project_id: UUID,
    artifact_id: UUID,
    data: ArtifactRevisionCreate,
    db: Session = Depends(get_db),
) -> ArtifactResponse:
    artifact = create_artifact_revision(
        db, project_id=project_id, artifact_id=artifact_id, data=data
    )
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    return _response(db, artifact)


@router.get("/{artifact_id}/export")
def export_project_artifact(
    project_id: UUID,
    artifact_id: UUID,
    export_format: ArtifactExportFormat = Query(alias="format"),
    revision_number: int | None = Query(default=None, ge=1, alias="revision"),
    db: Session = Depends(get_db),
) -> Response:
    artifact = get_artifact(db, project_id=project_id, artifact_id=artifact_id)
    if artifact is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
    number = revision_number or artifact.latest_revision
    revision = next((item for item in artifact.revisions if item.revision_number == number), None)
    if revision is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Revision not found")

    extension: str
    media_type: str
    with get_telemetry().stage(
        "artifact.export",
        metadata={
            "artifact_type": artifact.artifact_type,
            "format": export_format.value,
            "revision": number,
        },
    ) as observation:
        if export_format is ArtifactExportFormat.MARKDOWN:
            manifest = revalidate_source_manifest(
                db, project_id=project_id, source_manifest=revision.source_manifest
            )
            content = export_markdown(revision, artifact.artifact_type, source_manifest=manifest)
            extension, media_type = "md", "text/markdown; charset=utf-8"
        elif export_format is ArtifactExportFormat.CSV:
            if artifact.artifact_type != "comparison":
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="CSV export is available for comparison artifacts",
                )
            manifest = revalidate_source_manifest(
                db, project_id=project_id, source_manifest=revision.source_manifest
            )
            content = export_comparison_csv(revision, source_manifest=manifest)
            extension, media_type = "csv", "text/csv; charset=utf-8"
        else:
            manifest = revalidate_source_manifest(
                db, project_id=project_id, source_manifest=revision.source_manifest
            )
            content = export_bibtex(db, revision, project_id, source_manifest=manifest)
            extension, media_type = "bib", "application/x-bibtex; charset=utf-8"
        if observation is not None:
            observation.update(metadata={"outcome": "exported", "content_characters": len(content)})

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="artifact-{artifact.id}-r{number}.{extension}"'
            )
        },
    )
