import hashlib
import io
import logging
import tempfile
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.config import Settings
from app.crud.paper import create_paper_with_job, list_papers_by_project
from app.crud.project import create_project, get_project, list_projects
from app.db.models import Paper
from app.db.session import get_db
from app.schemas.paper import PaperListResponse, PaperResponse, PaperStatus, PaperUploadResponse
from app.schemas.project import ProjectCreate, ProjectListResponse, ProjectResponse
from app.storage.factory import get_storage

logger = logging.getLogger("myra.api.projects")
router = APIRouter(prefix="/projects", tags=["projects"])
settings = Settings.from_environment()


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_new_project(
    project_in: ProjectCreate,
    db: Session = Depends(get_db),
) -> ProjectResponse:
    project = create_project(db, project_in)
    return ProjectResponse.model_validate(project, from_attributes=True)


@router.get("", response_model=ProjectListResponse)
def get_projects(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> ProjectListResponse:
    items, total = list_projects(db, limit=limit, offset=offset)
    return ProjectListResponse(
        items=[ProjectResponse.model_validate(p, from_attributes=True) for p in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{project_id}", response_model=ProjectResponse)
def get_single_project(
    project_id: UUID,
    db: Session = Depends(get_db),
) -> ProjectResponse:
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )
    return ProjectResponse.model_validate(project, from_attributes=True)


@router.post(
    "/{project_id}/papers",
    response_model=PaperUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_paper(
    project_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> PaperUploadResponse:
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    filename = file.filename or "document.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    # 1. Bounded streamed upload reading via spooled temporary file
    chunk_size = 1024 * 1024  # 1MB chunks
    total_bytes = 0
    hasher = hashlib.sha256()
    spooled = tempfile.SpooledTemporaryFile(max_size=5 * 1024 * 1024, mode="w+b")
    first_chunk = True

    try:
        while chunk := await file.read(chunk_size):
            total_bytes += len(chunk)
            if total_bytes > settings.max_upload_size_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File exceeds maximum size of {settings.max_upload_size_bytes} bytes",
                )
            if first_chunk:
                if not chunk.startswith(b"%PDF"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid PDF file signature",
                    )
                first_chunk = False
            hasher.update(chunk)
            spooled.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded",
            )

        document_sha256 = hasher.hexdigest()
        spooled.seek(0)
        content = spooled.read()
    finally:
        spooled.close()

    # 2. Check for duplicate upload within project
    existing_paper = (
        db.query(Paper)
        .filter(Paper.project_id == project_id, Paper.document_sha256 == document_sha256)
        .first()
    )
    if existing_paper:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Paper with identical content already exists in this project",
        )

    # 3. Validate page count using pypdf
    try:
        reader = PdfReader(io.BytesIO(content))
        page_count = len(reader.pages)
        if page_count > settings.max_pdf_pages:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"PDF page count ({page_count}) exceeds limit of {settings.max_pdf_pages}",
            )
    except HTTPException:
        raise
    except Exception as err:
        logger.warning("Unreadable or corrupt PDF upload", exc_info=err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupt or unreadable PDF document",
        ) from err

    paper_id = uuid4()
    storage_key = f"papers/{project_id}/{paper_id}.pdf"
    storage = get_storage(settings)

    # 4. Store file in storage backend
    try:
        storage_path = await storage.put(storage_key, content, content_type="application/pdf")
    except Exception as err:
        logger.error("Failed to store PDF", exc_info=err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store PDF document",
        ) from err

    # 5. Create paper & job transactionally in a single DB commit; compensate storage on DB failure
    try:
        paper, job = create_paper_with_job(
            db,
            project_id=project_id,
            filename=filename,
            storage_path=storage_path,
            document_sha256=document_sha256,
            status=PaperStatus.PROCESSING,
        )
    except Exception as err:
        db.rollback()
        try:
            await storage.delete(storage_key)
        except Exception:
            pass
        logger.error("Failed to initialize paper record", exc_info=err)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to initialize paper record",
        ) from err

    return PaperUploadResponse(
        paper_id=paper.id,
        job_id=job.id,
        status=PaperStatus.PROCESSING,
    )


@router.get("/{project_id}/papers", response_model=PaperListResponse)
def get_project_papers(
    project_id: UUID,
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PaperListResponse:
    project = get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project {project_id} not found",
        )

    items, total = list_papers_by_project(db, project_id, limit=limit, offset=offset)
    return PaperListResponse(
        items=[PaperResponse.model_validate(p, from_attributes=True) for p in items],
        total=total,
        limit=limit,
        offset=offset,
    )
