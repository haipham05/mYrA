from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import (
    Paper,
    Project,
    ProjectTranslationGlossaryEntry,
    TranslationDocument,
    TranslationSegment,
)

_RETRYABLE_TRANSLATION_ERROR_CODES = {
    "ENGINE_DEADLINE_EXCEEDED",
    "ENGINE_INCOMPLETE",
    "ENGINE_PROTOCOL_ERROR",
    "PROVIDER_INVALID_JSON",
    "PROVIDER_MARKER_MISMATCH",
    "PROVIDER_RATE_LIMITED",
    "PROVIDER_SCIENTIFIC_TOKEN_MISMATCH",
    "PROVIDER_UNAVAILABLE",
}


class TranslationConflict(ValueError):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _translation_query():
    return select(TranslationDocument).options(selectinload(TranslationDocument.segments))


def create_translation(
    db: Session,
    *,
    project_id: UUID,
    paper_id: UUID,
    idempotency_key: str,
    acknowledge_external_processing: bool,
) -> tuple[TranslationDocument, bool]:
    if not acknowledge_external_processing:
        raise TranslationConflict("External text processing must be acknowledged")
    project = db.get(Project, project_id)
    paper = db.get(Paper, paper_id)
    if not project or not paper or paper.project_id != project_id:
        raise TranslationConflict("Paper was not found in this project")
    if paper.status != "READY":
        raise TranslationConflict("Only READY papers can be translated")
    if not paper.document_sha256:
        raise TranslationConflict("The original paper does not have a verified SHA-256")

    existing = db.scalar(
        _translation_query().where(
            TranslationDocument.project_id == project_id,
            TranslationDocument.paper_id == paper_id,
            TranslationDocument.idempotency_key == idempotency_key,
        )
    )
    if existing:
        if existing.source_sha256 != paper.document_sha256:
            raise TranslationConflict("The paper changed after this request key was used")
        return existing, False

    glossary = db.scalars(
        select(ProjectTranslationGlossaryEntry)
        .where(ProjectTranslationGlossaryEntry.project_id == project_id)
        .order_by(func.lower(ProjectTranslationGlossaryEntry.source_term))
    ).all()
    translation = TranslationDocument(
        project_id=project_id,
        paper_id=paper_id,
        status="PENDING",
        stage="QUEUED",
        idempotency_key=idempotency_key,
        acknowledge_external_processing=True,
        source_sha256=paper.document_sha256,
        source_storage_path=paper.storage_path,
        source_filename=paper.filename,
        source_page_count=paper.page_count,
        glossary_snapshot=[
            {"source_term": entry.source_term, "preferred_translation": entry.preferred_translation}
            for entry in glossary
        ],
    )
    db.add(translation)
    try:
        db.commit()
    except Exception:
        db.rollback()
        # The unique key is the concurrency guard for simultaneous duplicate requests.
        existing = db.scalar(
            _translation_query().where(
                TranslationDocument.project_id == project_id,
                TranslationDocument.paper_id == paper_id,
                TranslationDocument.idempotency_key == idempotency_key,
            )
        )
        if existing:
            return existing, False
        raise
    db.refresh(translation)
    return translation, True


def get_translation(
    db: Session, translation_id: UUID, *, project_id: UUID
) -> TranslationDocument | None:
    return db.scalar(
        _translation_query().where(
            TranslationDocument.id == translation_id,
            TranslationDocument.project_id == project_id,
        )
    )


def list_translations(
    db: Session, paper_id: UUID, *, project_id: UUID
) -> list[TranslationDocument]:
    return list(
        db.scalars(
            _translation_query()
            .where(
                TranslationDocument.paper_id == paper_id,
                TranslationDocument.project_id == project_id,
            )
            .order_by(TranslationDocument.created_at.desc())
        ).all()
    )


def cancel_translation(db: Session, translation: TranslationDocument) -> TranslationDocument:
    if translation.status in {"COMPLETED", "FAILED", "CANCELLED"}:
        return translation
    translation.status = "CANCELLED"
    translation.stage = "CANCELLED"
    translation.lease_owner = None
    translation.lease_expires_at = None
    translation.attempt_token = uuid4().hex
    translation.updated_at = _utcnow()
    db.commit()
    db.refresh(translation)
    return translation


def retry_translation(db: Session, translation: TranslationDocument) -> TranslationDocument:
    retryable = translation.is_retryable or translation.error_code in (
        _RETRYABLE_TRANSLATION_ERROR_CODES
    )
    if translation.status != "FAILED" or not retryable:
        raise TranslationConflict("This translation is not in a retryable failed state")
    paper = db.get(Paper, translation.paper_id)
    if not paper or paper.project_id != translation.project_id:
        raise TranslationConflict("The source paper is no longer available in this project")
    if paper.document_sha256 != translation.source_sha256:
        raise TranslationConflict("The source paper changed; create a new translation request")
    translation.status = "PENDING"
    translation.stage = "QUEUED"
    translation.error_code = None
    translation.error_message = None
    translation.is_retryable = False
    translation.lease_owner = None
    translation.lease_expires_at = None
    translation.attempt_token = None
    translation.updated_at = _utcnow()
    db.commit()
    db.refresh(translation)
    return translation


def replace_glossary(
    db: Session, *, project_id: UUID, entries: list[tuple[str, str]]
) -> list[ProjectTranslationGlossaryEntry]:
    if not db.get(Project, project_id):
        raise LookupError("Project not found")
    normalized: dict[str, tuple[str, str]] = {}
    for source, preferred in entries:
        source = source.strip()
        preferred = preferred.strip()
        if not source or not preferred:
            raise TranslationConflict("Glossary terms cannot be blank")
        key = source.casefold()
        if key in normalized:
            raise TranslationConflict("Glossary source terms must be unique")
        normalized[key] = (source, preferred)
    current = db.scalars(
        select(ProjectTranslationGlossaryEntry).where(
            ProjectTranslationGlossaryEntry.project_id == project_id
        )
    ).all()
    for entry in current:
        db.delete(entry)
    db.flush()
    result = [
        ProjectTranslationGlossaryEntry(
            project_id=project_id,
            source_term=source,
            preferred_translation=preferred,
        )
        for source, preferred in normalized.values()
    ]
    db.add_all(result)
    db.commit()
    return list(
        db.scalars(
            select(ProjectTranslationGlossaryEntry)
            .where(ProjectTranslationGlossaryEntry.project_id == project_id)
            .order_by(ProjectTranslationGlossaryEntry.source_term)
        ).all()
    )


def get_glossary(db: Session, *, project_id: UUID) -> list[ProjectTranslationGlossaryEntry]:
    return list(
        db.scalars(
            select(ProjectTranslationGlossaryEntry)
            .where(ProjectTranslationGlossaryEntry.project_id == project_id)
            .order_by(ProjectTranslationGlossaryEntry.source_term)
        ).all()
    )


def claim_next_translation(db: Session, *, worker_id: str) -> TranslationDocument | None:
    now = _utcnow()
    query = (
        select(TranslationDocument)
        .where(
            or_(
                TranslationDocument.status == "PENDING",
                (TranslationDocument.status == "PROCESSING")
                & (TranslationDocument.lease_expires_at < now),
            )
        )
        .order_by(TranslationDocument.created_at)
        .with_for_update(skip_locked=True)
        .limit(1)
    )
    translation = db.scalar(query)
    if translation is None:
        return None
    token = uuid4().hex
    translation.status = "PROCESSING"
    translation.stage = "SOURCE_DOWNLOAD"
    translation.attempt_count += 1
    translation.attempt_token = token
    translation.lease_owner = worker_id
    translation.lease_expires_at = now + timedelta(seconds=300)
    translation.updated_at = now
    db.commit()
    db.refresh(translation)
    return translation


def renew_translation_lease(
    db: Session, translation_id: UUID, *, worker_id: str, attempt_token: str
) -> bool:
    translation = db.get(TranslationDocument, translation_id)
    if (
        not translation
        or translation.status != "PROCESSING"
        or translation.lease_owner != worker_id
        or translation.attempt_token != attempt_token
    ):
        return False
    translation.lease_expires_at = _utcnow() + timedelta(seconds=300)
    translation.updated_at = _utcnow()
    db.commit()
    return True


def update_translation_stage(
    db: Session,
    translation_id: UUID,
    *,
    worker_id: str,
    attempt_token: str,
    stage: str,
    completed_units: int | None = None,
    total_units: int | None = None,
) -> bool:
    translation = db.get(TranslationDocument, translation_id)
    if (
        not translation
        or translation.status != "PROCESSING"
        or translation.lease_owner != worker_id
        or translation.attempt_token != attempt_token
    ):
        return False
    translation.stage = stage
    if completed_units is not None:
        translation.completed_units = max(0, completed_units)
    if total_units is not None:
        translation.total_units = max(0, total_units)
    translation.updated_at = _utcnow()
    db.commit()
    return True


def save_translation_segment(
    db: Session,
    translation_id: UUID,
    *,
    worker_id: str,
    attempt_token: str,
    engine_checkpoint_key: str,
    ordinal: int,
    source_page_number: int,
    source_text_hash: str,
    source_quote: str,
    translated_text: str,
    translated_text_hash: str,
    output_page_number: int | None = None,
    output_boxes: list[dict[str, float]] | None = None,
    source_element_id: UUID | None = None,
    status: str = "VALIDATED",
) -> bool:
    """Persist a validated unit only while the owning worker still holds the lease."""
    translation = db.scalar(
        select(TranslationDocument)
        .where(TranslationDocument.id == translation_id)
        .with_for_update()
    )
    if (
        not translation
        or translation.status != "PROCESSING"
        or translation.lease_owner != worker_id
        or translation.attempt_token != attempt_token
    ):
        db.rollback()
        return False
    if (
        ordinal < 0
        or source_page_number < 1
        or not source_quote
        or not source_text_hash
        or not translated_text
        or not translated_text_hash
        or len(engine_checkpoint_key) != 64
        or any(character not in "0123456789abcdef" for character in engine_checkpoint_key)
        or status not in {"VALIDATED", "PRESERVED"}
    ):
        raise ValueError("Translation checkpoint fields failed validation")

    segment = db.scalar(
        select(TranslationSegment)
        .where(
            TranslationSegment.translation_id == translation_id,
            TranslationSegment.ordinal == ordinal,
        )
        .with_for_update()
    )
    if segment and segment.source_text_hash != source_text_hash:
        raise TranslationConflict("Checkpoint source identity changed")
    if segment is None:
        segment = TranslationSegment(translation_id=translation_id, ordinal=ordinal)
        db.add(segment)
    segment.source_page_number = source_page_number
    segment.engine_checkpoint_key = engine_checkpoint_key
    segment.source_element_id = source_element_id
    segment.source_text_hash = source_text_hash
    segment.source_quote = source_quote
    segment.translated_text = translated_text
    segment.translated_text_hash = translated_text_hash
    segment.output_page_number = output_page_number
    segment.output_boxes = output_boxes
    segment.status = status
    db.flush()
    translation.completed_units = (
        db.scalar(
            select(func.count(TranslationSegment.id)).where(
                TranslationSegment.translation_id == translation_id,
                TranslationSegment.status.in_(["VALIDATED", "PRESERVED"]),
            )
        )
        or 0
    )
    translation.updated_at = _utcnow()
    db.commit()
    return True


def fail_translation(
    db: Session,
    translation_id: UUID,
    *,
    worker_id: str,
    attempt_token: str,
    error_code: str,
    error_message: str,
    retryable: bool,
) -> bool:
    translation = db.get(TranslationDocument, translation_id)
    if (
        not translation
        or translation.status != "PROCESSING"
        or translation.lease_owner != worker_id
        or translation.attempt_token != attempt_token
    ):
        return False
    translation.status = "FAILED"
    translation.stage = "FAILED"
    translation.error_code = error_code[:80]
    translation.error_message = error_message[:500]
    translation.is_retryable = retryable
    translation.lease_owner = None
    translation.lease_expires_at = None
    translation.updated_at = _utcnow()
    db.commit()
    return True


def complete_translation(
    db: Session,
    translation_id: UUID,
    *,
    worker_id: str,
    attempt_token: str,
    output_storage_path: str,
    output_sha256: str,
    source_map: list[dict],
) -> bool:
    translation = db.get(TranslationDocument, translation_id)
    if (
        not translation
        or translation.status != "PROCESSING"
        or translation.lease_owner != worker_id
        or translation.attempt_token != attempt_token
    ):
        return False
    translation.status = "COMPLETED"
    translation.stage = "COMPLETED"
    translation.output_storage_path = output_storage_path
    translation.output_sha256 = output_sha256
    translation.source_map = source_map
    translation.completed_at = _utcnow()
    translation.lease_owner = None
    translation.lease_expires_at = None
    translation.updated_at = _utcnow()
    db.commit()
    return True
