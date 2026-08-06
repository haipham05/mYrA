from uuid import UUID

from sqlalchemy.orm import Session, selectinload

from app.db.models import ResearchArtifact, ResearchArtifactRevision
from app.schemas.artifact import ArtifactCreate, ArtifactRevisionCreate


def list_artifacts(db: Session, project_id: UUID) -> list[ResearchArtifact]:
    return (
        db.query(ResearchArtifact)
        .options(selectinload(ResearchArtifact.revisions))
        .filter(ResearchArtifact.project_id == project_id)
        .order_by(ResearchArtifact.updated_at.desc())
        .all()
    )


def get_artifact(db: Session, *, project_id: UUID, artifact_id: UUID) -> ResearchArtifact | None:
    return (
        db.query(ResearchArtifact)
        .options(selectinload(ResearchArtifact.revisions))
        .filter(
            ResearchArtifact.id == artifact_id,
            ResearchArtifact.project_id == project_id,
        )
        .first()
    )


def create_artifact(db: Session, *, project_id: UUID, data: ArtifactCreate) -> ResearchArtifact:
    artifact = ResearchArtifact(
        project_id=project_id,
        artifact_type=data.artifact_type.value,
        title=data.title,
        latest_revision=1,
    )
    db.add(artifact)
    db.flush()
    db.add(
        ResearchArtifactRevision(
            artifact_id=artifact.id,
            revision_number=1,
            title=data.title,
            payload=data.payload,
            scope_snapshot=data.scope_snapshot,
            source_manifest=data.source_manifest,
            config_snapshot=data.config_snapshot,
            usage=data.usage,
        )
    )
    db.commit()
    created = get_artifact(db, project_id=project_id, artifact_id=artifact.id)
    assert created is not None
    return created


def create_artifact_revision(
    db: Session,
    *,
    project_id: UUID,
    artifact_id: UUID,
    data: ArtifactRevisionCreate,
) -> ResearchArtifact | None:
    artifact = (
        db.query(ResearchArtifact)
        .filter(
            ResearchArtifact.id == artifact_id,
            ResearchArtifact.project_id == project_id,
        )
        .with_for_update()
        .first()
    )
    if artifact is None:
        return None
    revision_number = artifact.latest_revision + 1
    db.add(
        ResearchArtifactRevision(
            artifact_id=artifact.id,
            revision_number=revision_number,
            title=data.title,
            payload=data.payload,
            scope_snapshot=data.scope_snapshot,
            source_manifest=data.source_manifest,
            config_snapshot=data.config_snapshot,
            usage=data.usage,
        )
    )
    artifact.latest_revision = revision_number
    artifact.title = data.title
    db.commit()
    return get_artifact(db, project_id=project_id, artifact_id=artifact.id)
