from uuid import UUID

from sqlalchemy import Text, cast, or_
from sqlalchemy.orm import Session

from app.crud.corpus import bump_corpus_revision
from app.db.models import Job, Paper, PaperElement, PaperPage
from app.observability.context import OperationContext


def create_paper(
    db: Session,
    project_id: UUID,
    filename: str,
    storage_path: str,
    document_sha256: str | None = None,
    status: str = "PROCESSING",
) -> Paper:
    paper = Paper(
        project_id=project_id,
        filename=filename,
        storage_path=storage_path,
        document_sha256=document_sha256,
        status=status,
    )
    db.add(paper)
    if status == "READY":
        bump_corpus_revision(db, project_id)
    db.commit()
    db.refresh(paper)
    return paper


def create_paper_with_job(
    db: Session,
    project_id: UUID,
    filename: str,
    storage_path: str,
    document_sha256: str | None = None,
    status: str = "PROCESSING",
    trace_context: OperationContext | None = None,
) -> tuple[Paper, Job]:
    """Atomically create paper and job within a single database transaction."""
    paper = Paper(
        project_id=project_id,
        filename=filename,
        storage_path=storage_path,
        document_sha256=document_sha256,
        status=status,
    )
    db.add(paper)
    db.flush()
    if status == "READY":
        bump_corpus_revision(db, project_id)
    job = Job(
        paper_id=paper.id,
        status="PENDING",
        stage="QUEUED",
        progress=0.0,
        correlation_id=trace_context.correlation_id if trace_context else None,
        trace_id=trace_context.trace_id if trace_context else None,
        parent_span_id=trace_context.span_id if trace_context else None,
        trace_sampled=trace_context.sampled if trace_context else False,
    )
    db.add(job)
    db.commit()
    db.refresh(paper)
    db.refresh(job)
    return paper, job


def get_paper(db: Session, paper_id: UUID | str) -> Paper | None:
    if isinstance(paper_id, str):
        paper_id = UUID(paper_id)
    return db.query(Paper).filter(Paper.id == paper_id).first()


def list_papers_by_project(
    db: Session,
    project_id: UUID,
    limit: int = 50,
    offset: int = 0,
    query_text: str | None = None,
    status: str | None = None,
    publication_year: int | None = None,
) -> tuple[list[Paper], int]:
    query = db.query(Paper).filter(Paper.project_id == project_id)
    if query_text:
        pattern = f"%{query_text}%"
        query = query.filter(
            or_(
                Paper.filename.ilike(pattern),
                Paper.title.ilike(pattern),
                cast(Paper.authors, Text).ilike(pattern),
            )
        )
    if status:
        query = query.filter(Paper.status == status)
    if publication_year is not None:
        query = query.filter(Paper.publication_year == publication_year)
    query = query.order_by(Paper.created_at.desc(), Paper.id.asc())
    total = query.count()
    items = query.offset(offset).limit(limit).all()
    return items, total


def update_paper_status(
    db: Session,
    paper_id: UUID | str,
    status: str,
    page_count: int | None = None,
    error_message: str | None = None,
) -> Paper | None:
    if isinstance(paper_id, str):
        paper_id = UUID(paper_id)
    paper = get_paper(db, paper_id)
    if not paper:
        return None
    was_ready = paper.status == "READY"
    paper.status = status
    if was_ready != (status == "READY"):
        bump_corpus_revision(db, paper.project_id)
    if page_count is not None:
        paper.page_count = page_count
    if error_message is not None:
        paper.error_message = error_message
    db.commit()
    db.refresh(paper)
    return paper


def get_paper_elements(db: Session, paper_id: UUID | str) -> list[PaperElement]:
    if isinstance(paper_id, str):
        paper_id = UUID(paper_id)
    return (
        db.query(PaperElement)
        .filter(PaperElement.paper_id == paper_id)
        .order_by(PaperElement.page_number.asc(), PaperElement.element_index.asc())
        .all()
    )


def get_paper_pages(db: Session, paper_id: UUID | str) -> list[PaperPage]:
    if isinstance(paper_id, str):
        paper_id = UUID(paper_id)
    return (
        db.query(PaperPage)
        .filter(PaperPage.paper_id == paper_id)
        .order_by(PaperPage.page_number.asc())
        .all()
    )
