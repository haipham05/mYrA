import hashlib
import logging
import tempfile
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse
from pypdf import PdfReader
from sqlalchemy.orm import Session

from app.config import Settings
from app.crud.job import create_job
from app.crud.paper import create_paper_with_job, list_papers_by_project
from app.crud.project import create_project, get_project, list_projects
from app.db.models import Job, Paper
from app.db.session import get_db
from app.observability.context import get_operation_context
from app.observability.telemetry import TelemetryAdapter, get_telemetry
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
    response: Response,
    file: UploadFile = File(...),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
) -> PaperUploadResponse:
    telemetry = get_telemetry()
    with telemetry.operation(
        "api.projects.upload_paper",
        metadata={
            "http_method": "POST",
            "route_template": "/api/v1/projects/{project_id}/papers",
        },
    ) as operation:
        result = await _upload_paper(
            project_id=project_id,
            response=response,
            file=file,
            idempotency_key=idempotency_key,
            db=db,
            telemetry=telemetry,
        )
        if operation is not None:
            operation.update(
                metadata={
                    "outcome": "accepted" if result.status == PaperStatus.PROCESSING else "ready",
                    "http_status": response.status_code or status.HTTP_202_ACCEPTED,
                    "paper_id": str(result.paper_id),
                    "job_id": str(result.job_id),
                }
            )
        return result


async def _upload_paper(
    *,
    project_id: UUID,
    response: Response,
    file: UploadFile,
    idempotency_key: str | None,
    db: Session,
    telemetry: TelemetryAdapter,
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

        # 2. Check for duplicate upload within project before writing storage.
        # Idempotency policy (3.B3): return existing record if content exists
        existing_paper = (
            db.query(Paper)
            .filter(Paper.project_id == project_id, Paper.document_sha256 == document_sha256)
            .first()
        )
        if existing_paper:
            if idempotency_key is not None:
                latest_job = (
                    db.query(Job)
                    .filter(Job.paper_id == existing_paper.id)
                    .order_by(Job.created_at.desc())
                    .first()
                )
                if existing_paper.status == PaperStatus.READY:
                    response.status_code = status.HTTP_200_OK
                    return PaperUploadResponse(
                        paper_id=existing_paper.id,
                        job_id=latest_job.id if latest_job else existing_paper.id,
                        status=PaperStatus.READY,
                    )
                if existing_paper.status == PaperStatus.PROCESSING:
                    response.status_code = status.HTTP_202_ACCEPTED
                    return PaperUploadResponse(
                        paper_id=existing_paper.id,
                        job_id=latest_job.id if latest_job else existing_paper.id,
                        status=PaperStatus.PROCESSING,
                    )
                if existing_paper.status == PaperStatus.FAILED:
                    # Recover failed paper by creating a new job and re-queuing
                    queue_started = perf_counter()
                    with telemetry.stage(
                        "ingestion.queue.create_job",
                        metadata={"reason": "retry_failed_paper"},
                    ) as queue_observation:
                        try:
                            new_job = create_job(
                                db, existing_paper.id, trace_context=get_operation_context()
                            )
                            existing_paper.status = PaperStatus.PROCESSING
                            existing_paper.error_message = None
                            db.commit()
                        except Exception as err:
                            if queue_observation is not None:
                                queue_observation.update(
                                    metadata={
                                        "outcome": "failed",
                                        "duration_ms": round(
                                            (perf_counter() - queue_started) * 1000, 2
                                        ),
                                        "error_code": type(err).__name__,
                                    }
                                )
                            raise
                        if queue_observation is not None:
                            queue_observation.update(
                                metadata={
                                    "outcome": "queued",
                                    "duration_ms": round(
                                        (perf_counter() - queue_started) * 1000, 2
                                    ),
                                    "paper_id": str(existing_paper.id),
                                    "job_id": str(new_job.id),
                                    "retry_count": new_job.retry_count,
                                }
                            )
                    response.status_code = status.HTTP_202_ACCEPTED
                    return PaperUploadResponse(
                        paper_id=existing_paper.id,
                        job_id=new_job.id,
                        status=PaperStatus.PROCESSING,
                    )

            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Paper with identical content already exists in this project",
            )

        # 3. Validate page count on the seekable spool without materializing bytes.
        try:
            spooled.seek(0)
            reader = PdfReader(spooled)
            page_count = len(reader.pages)
            if page_count > settings.max_pdf_pages:
                detail = f"PDF page count ({page_count}) exceeds limit of {settings.max_pdf_pages}"
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=detail,
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

        # 4. Stream from the spool into the selected storage backend.
        storage_started = perf_counter()
        try:
            with telemetry.stage(
                "ingestion.storage.upload",
                metadata={"backend": type(storage).__name__},
            ) as storage_observation:
                try:
                    spooled.seek(0)
                    storage_path = await storage.put_stream(
                        storage_key, spooled, content_type="application/pdf"
                    )
                except Exception as err:
                    if storage_observation is not None:
                        storage_observation.update(
                            metadata={
                                "outcome": "failed",
                                "duration_ms": round((perf_counter() - storage_started) * 1000, 2),
                                "error_code": type(err).__name__,
                            }
                        )
                    raise
                if storage_observation is not None:
                    storage_observation.update(
                        metadata={
                            "outcome": "stored",
                            "duration_ms": round((perf_counter() - storage_started) * 1000, 2),
                            "bytes": total_bytes,
                        }
                    )
        except Exception as err:
            telemetry.event(
                "ingestion.storage.upload_failed",
                metadata={
                    "outcome": "failed",
                    "duration_ms": round((perf_counter() - storage_started) * 1000, 2),
                    "error_code": type(err).__name__,
                },
            )
            logger.error("Failed to store PDF", exc_info=err)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store PDF document",
            ) from err
    finally:
        spooled.close()

    # 5. Create paper & job transactionally in a single DB commit; compensate storage on DB failure
    queue_started = perf_counter()
    try:
        with telemetry.stage(
            "ingestion.queue.create_job",
            metadata={"reason": "new_paper"},
        ) as queue_observation:
            try:
                paper, job = create_paper_with_job(
                    db,
                    project_id=project_id,
                    filename=filename,
                    storage_path=storage_path,
                    document_sha256=document_sha256,
                    status=PaperStatus.PROCESSING,
                    trace_context=get_operation_context(),
                )
            except Exception as err:
                if queue_observation is not None:
                    queue_observation.update(
                        metadata={
                            "outcome": "failed",
                            "duration_ms": round((perf_counter() - queue_started) * 1000, 2),
                            "error_code": type(err).__name__,
                        }
                    )
                raise
            if queue_observation is not None:
                queue_observation.update(
                    metadata={
                        "outcome": "queued",
                        "duration_ms": round((perf_counter() - queue_started) * 1000, 2),
                        "paper_id": str(paper.id),
                        "job_id": str(job.id),
                        "retry_count": job.retry_count,
                    }
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


@router.get("/{project_id}/papers/{paper_id}/document")
async def get_project_paper_document(
    project_id: UUID,
    paper_id: UUID,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    from app.api.v1.papers import get_paper_document

    return await get_paper_document(paper_id=paper_id, project_id=project_id, db=db)
