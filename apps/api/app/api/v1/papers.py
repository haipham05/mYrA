import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.crud.paper import get_paper, get_paper_elements
from app.db.session import get_db
from app.schemas.evidence import BoundingBox, CoordinateOrigin, SourceElement
from app.schemas.paper import PaperResponse
from app.storage.factory import get_storage

logger = logging.getLogger("myra.api.papers")
router = APIRouter(prefix="/papers", tags=["papers"])


@router.get("/{paper_id}", response_model=PaperResponse)
def get_single_paper(
    paper_id: UUID,
    db: Session = Depends(get_db),
) -> PaperResponse:
    paper = get_paper(db, paper_id)
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found",
        )
    return PaperResponse.model_validate(paper, from_attributes=True)


@router.get("/{paper_id}/document")
async def get_paper_document(
    paper_id: UUID,
    project_id: UUID | None = None,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    paper = get_paper(db, paper_id)
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found",
        )
    if project_id and paper.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found in project {project_id}",
        )

    storage = get_storage()
    # Resolve storage key from storage_path or format
    key = paper.storage_path
    if key.startswith("gs://"):
        parts = key.split("/", 3)
        key = parts[3] if len(parts) > 3 else key
    elif key.startswith("memory://"):
        key = key.replace("memory://", "")

    resolved_key = None
    if await storage.exists(key):
        resolved_key = key
    elif await storage.exists(paper.storage_path):
        resolved_key = paper.storage_path

    if resolved_key is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document file not found in storage",
        )

    stream = storage.open_stream(resolved_key)
    headers = {
        "Content-Disposition": f'inline; filename="{paper.filename}"',
        "Content-Type": "application/pdf",
        "Cache-Control": "public, max-age=3600",
    }
    if paper.document_sha256:
        headers["ETag"] = f'"{paper.document_sha256}"'

    return StreamingResponse(stream, media_type="application/pdf", headers=headers)


@router.get("/{paper_id}/elements", response_model=list[SourceElement])
def get_paper_debug_elements(
    paper_id: UUID,
    db: Session = Depends(get_db),
) -> list[SourceElement]:
    """Debug endpoint for inspecting parsed elements and bounding box coordinates."""
    paper = get_paper(db, paper_id)
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found",
        )

    db_elements = get_paper_elements(db, paper_id)
    results = []
    for elem in db_elements:
        bbox = None
        if elem.bbox_x_min is not None and elem.page_width and elem.page_height:
            bbox = BoundingBox(
                x_min=elem.bbox_x_min,
                y_min=elem.bbox_y_min or 0.0,
                x_max=elem.bbox_x_max or 0.0,
                y_max=elem.bbox_y_max or 0.0,
                page_width=elem.page_width,
                page_height=elem.page_height,
                origin=CoordinateOrigin(elem.coordinate_origin),
                rotation=elem.rotation,
            )
        results.append(
            SourceElement(
                id=elem.id,
                element_index=elem.element_index,
                element_type=elem.element_type,
                text=elem.text,
                page_number=elem.page_number,
                bounding_box=bbox,
                section_path=elem.section_path or [],
            )
        )
    return results
