import logging
from hashlib import sha256
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.crud.translation import (
    TranslationConflict,
    cancel_translation,
    create_translation,
    get_glossary,
    get_translation,
    list_translations,
    replace_glossary,
    retry_translation,
)
from app.db.models import Paper, PaperPage
from app.db.session import get_db
from app.ingestion.parser import find_verbatim_span
from app.schemas.translation import (
    TranslationCreate,
    TranslationGlossaryEntry,
    TranslationGlossaryReplace,
    TranslationGlossaryResponse,
    TranslationListResponse,
    TranslationResponse,
)
from app.storage.factory import get_storage

logger = logging.getLogger("myra.api.translations")
router = APIRouter(tags=["translations"])


def _response(translation, *, include_segments: bool = True) -> TranslationResponse:
    return TranslationResponse(
        id=translation.id,
        project_id=translation.project_id,
        paper_id=translation.paper_id,
        status=translation.status,
        stage=translation.stage,
        source_sha256=translation.source_sha256,
        source_filename=translation.source_filename,
        source_page_count=translation.source_page_count,
        completed_units=translation.completed_units,
        total_units=translation.total_units,
        output_sha256=translation.output_sha256,
        output_available=bool(translation.output_storage_path),
        error_code=translation.error_code,
        error_message=translation.error_message,
        is_retryable=translation.is_retryable,
        created_at=translation.created_at,
        updated_at=translation.updated_at,
        completed_at=translation.completed_at,
        segments=[
            {
                "id": segment.id,
                "ordinal": segment.ordinal,
                "source_page_number": segment.source_page_number,
                "source_element_id": segment.source_element_id,
                "source_quote": segment.source_quote,
                "translated_text": segment.translated_text,
                "output_page_number": segment.output_page_number,
                "output_boxes": segment.output_boxes,
                "status": segment.status,
            }
            for segment in (translation.segments if include_segments else [])
        ],
    )


def _get_project_translation(db: Session, translation_id: UUID, project_id: UUID):
    translation = get_translation(db, translation_id, project_id=project_id)
    if not translation:
        raise HTTPException(status_code=404, detail="Translation not found")
    return translation


@router.post(
    "/papers/{paper_id}/translations",
    response_model=TranslationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def submit_translation(
    paper_id: UUID,
    payload: TranslationCreate,
    response: Response,
    idempotency_header: str | None = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
) -> TranslationResponse:
    key = payload.idempotency_key or idempotency_header or uuid4().hex
    try:
        translation, created = create_translation(
            db,
            project_id=payload.project_id,
            paper_id=paper_id,
            idempotency_key=key,
            acknowledge_external_processing=payload.acknowledge_external_processing,
        )
    except TranslationConflict as err:
        message = str(err)
        code = 404 if "not found" in message.lower() else 409
        raise HTTPException(status_code=code, detail=message) from err
    response.status_code = status.HTTP_202_ACCEPTED if created else status.HTTP_200_OK
    return _response(translation)


@router.get("/papers/{paper_id}/translations", response_model=TranslationListResponse)
def get_paper_translations(
    paper_id: UUID,
    project_id: UUID = Query(...),
    db: Session = Depends(get_db),
) -> TranslationListResponse:
    items = list_translations(db, paper_id, project_id=project_id)
    return TranslationListResponse(items=[_response(item) for item in items])


@router.get("/translations/{translation_id}", response_model=TranslationResponse)
def get_translation_status(
    translation_id: UUID,
    project_id: UUID = Query(...),
    db: Session = Depends(get_db),
) -> TranslationResponse:
    return _response(_get_project_translation(db, translation_id, project_id))


@router.post("/translations/{translation_id}/cancel", response_model=TranslationResponse)
def cancel_translation_job(
    translation_id: UUID,
    project_id: UUID = Query(...),
    db: Session = Depends(get_db),
) -> TranslationResponse:
    return _response(
        cancel_translation(db, _get_project_translation(db, translation_id, project_id))
    )


@router.post("/translations/{translation_id}/retry", response_model=TranslationResponse)
def retry_translation_job(
    translation_id: UUID,
    project_id: UUID = Query(...),
    db: Session = Depends(get_db),
) -> TranslationResponse:
    try:
        translation = retry_translation(
            db, _get_project_translation(db, translation_id, project_id)
        )
    except TranslationConflict as err:
        raise HTTPException(status_code=409, detail=str(err)) from err
    return _response(translation)


@router.get("/translations/{translation_id}/source-map")
def get_translation_source_map(
    translation_id: UUID,
    project_id: UUID = Query(...),
    db: Session = Depends(get_db),
) -> dict:
    translation = _get_project_translation(db, translation_id, project_id)
    paper = db.get(Paper, translation.paper_id)
    source_is_current = bool(paper and paper.document_sha256 == translation.source_sha256)
    pages = {
        page.page_number: page
        for page in db.query(PaperPage).filter(PaperPage.paper_id == translation.paper_id).all()
    }
    segments = []
    for item in translation.segments:
        page = pages.get(item.source_page_number)
        match = (
            find_verbatim_span(page.raw_text, item.source_quote)
            if source_is_current and page and page.raw_text
            else None
        )
        verified = match is not None
        segments.append(
            {
                "ordinal": item.ordinal,
                "source_page_number": item.source_page_number,
                "source_element_id": None,
                "source_text_hash": item.source_text_hash,
                "source_quote": item.source_quote,
                "output_page_number": item.output_page_number,
                "output_boxes": item.output_boxes,
                "anchor_status": "verified" if verified else "page_only",
                "highlight_label": None if verified else "Exact highlight unavailable",
                "source_char_start": match[0] if match else None,
                "source_char_end": match[1] if match else None,
                "source_page_text_sha256": (
                    sha256(page.raw_text.encode("utf-8")).hexdigest()
                    if page and page.raw_text
                    else None
                ),
                "document_sha256": translation.source_sha256,
                "parser_version": "translation-page-rawtext-v1" if verified else None,
            }
        )
    return {
        "translation_id": str(translation.id),
        "project_id": str(translation.project_id),
        "paper_id": str(translation.paper_id),
        "source_sha256": translation.source_sha256,
        "segments": segments,
    }


@router.get("/translations/{translation_id}/pdf")
async def download_translation_pdf(
    translation_id: UUID,
    project_id: UUID = Query(...),
    inline: bool = Query(False),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    translation = _get_project_translation(db, translation_id, project_id)
    if translation.status != "COMPLETED" or not translation.output_storage_path:
        raise HTTPException(status_code=409, detail="Translated PDF is not ready")
    key = translation.output_storage_path
    if key.startswith("gs://"):
        parts = key.split("/", 3)
        key = parts[3] if len(parts) > 3 else key
    elif key.startswith("memory://"):
        key = key.removeprefix("memory://")
    storage = get_storage()
    if not await storage.exists(key):
        raise HTTPException(status_code=404, detail="Translated PDF was not found in storage")
    filename = translation.source_filename.rsplit("/", 1)[-1].replace('"', "")
    disposition = "inline" if inline else "attachment"
    headers = {
        "Content-Disposition": f'{disposition}; filename="translated-{filename}"',
        "Cache-Control": "private, no-store",
    }
    if translation.output_sha256:
        headers["ETag"] = f'"{translation.output_sha256}"'
    return StreamingResponse(
        storage.open_stream(key), media_type="application/pdf", headers=headers
    )


@router.get(
    "/projects/{project_id}/translation-glossary", response_model=TranslationGlossaryResponse
)
def read_translation_glossary(
    project_id: UUID,
    db: Session = Depends(get_db),
) -> TranslationGlossaryResponse:
    entries = get_glossary(db, project_id=project_id)
    return TranslationGlossaryResponse(
        project_id=project_id,
        entries=[
            TranslationGlossaryEntry.model_validate(item, from_attributes=True) for item in entries
        ],
    )


@router.put(
    "/projects/{project_id}/translation-glossary", response_model=TranslationGlossaryResponse
)
def update_translation_glossary(
    project_id: UUID,
    payload: TranslationGlossaryReplace,
    db: Session = Depends(get_db),
) -> TranslationGlossaryResponse:
    try:
        entries = replace_glossary(
            db,
            project_id=project_id,
            entries=[(item.source_term, item.preferred_translation) for item in payload.entries],
        )
    except LookupError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except TranslationConflict as err:
        raise HTTPException(status_code=422, detail=str(err)) from err
    return TranslationGlossaryResponse(
        project_id=project_id,
        entries=[
            TranslationGlossaryEntry.model_validate(item, from_attributes=True) for item in entries
        ],
    )
