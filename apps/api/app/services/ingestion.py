import hashlib
from uuid import UUID

from sqlalchemy.orm import Session

from app.crud.job import get_job, update_job_progress
from app.crud.paper import get_paper, update_paper_status
from app.db.models import ChunkElement, PaperChunk, PaperElement, PaperPage
from app.ingestion.chunker import DocumentChunker
from app.ingestion.parser import DocumentParser
from app.schemas.job import JobStage, JobStatus
from app.schemas.paper import PaperStatus
from app.services.embedding import get_embedding_provider
from app.storage.factory import get_storage


def is_transient_error(err: Exception) -> bool:
    """Classify whether an exception is transient (retryable) or permanent (fatal)."""
    if isinstance(err, (ValueError, TypeError, KeyError, AttributeError, IndexError)):
        return False
    err_str = str(err).lower()
    if any(
        k in err_str
        for k in ("no extractable text", "unsupported", "invalid pdf", "corrupt", "not a pdf")
    ):
        return False
    if isinstance(err, (TimeoutError, ConnectionError, OSError)):
        return True
    err_cls_name = err.__class__.__name__
    if any(k in err_cls_name for k in ("OperationalError", "Timeout", "Connection", "Transient")):
        return True
    return False


class IngestionPipeline:
    """End-to-end ingestion pipeline: storage -> parser -> chunker -> embedding -> db."""

    def __init__(
        self,
        parser: DocumentParser | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.parser = parser or DocumentParser()
        self.chunker = chunker or DocumentChunker()

    async def process_paper(self, db: Session, paper_id: UUID, job_id: UUID) -> None:
        paper = get_paper(db, paper_id)
        job = get_job(db, job_id)
        if not paper or not job:
            return

        try:
            # 1. Fetch file from storage
            update_job_progress(db, job_id, stage=JobStage.PARSING, progress=0.1)
            storage = get_storage()
            key = paper.storage_path
            if key.startswith("gs://"):
                parts = key.split("/", 3)
                key = parts[3] if len(parts) > 3 else key
            elif key.startswith("memory://"):
                key = key.replace("memory://", "")

            try:
                pdf_bytes = await storage.get(key)
            except FileNotFoundError:
                pdf_bytes = await storage.get(paper.storage_path)

            # 2. Parse PDF
            update_job_progress(db, job_id, stage=JobStage.PARSING, progress=0.25)
            if not paper.document_sha256:
                paper.document_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
                db.flush()
            parse_result = self.parser.parse(pdf_bytes)

            # Clear any existing pages/elements if re-running (idempotent)
            db.query(PaperPage).filter(PaperPage.paper_id == paper_id).delete()
            db.query(PaperElement).filter(PaperElement.paper_id == paper_id).delete()
            db.flush()

            # Insert pages
            for p in parse_result.pages:
                db_page = PaperPage(
                    paper_id=paper_id,
                    page_number=p.page_number,
                    width=p.width,
                    height=p.height,
                    rotation=p.rotation,
                    crop_box=p.crop_box,
                    raw_text=p.raw_text,
                )
                db.add(db_page)

            # Insert elements and map element_index to DB element
            elem_map: dict[int, PaperElement] = {}
            for e in parse_result.elements:
                db_elem = PaperElement(
                    paper_id=paper_id,
                    page_number=e.page_number,
                    element_index=e.element_index,
                    element_type=e.element_type,
                    text=e.text,
                    bbox_x_min=e.bbox_x_min,
                    bbox_y_min=e.bbox_y_min,
                    bbox_x_max=e.bbox_x_max,
                    bbox_y_max=e.bbox_y_max,
                    page_width=e.page_width,
                    page_height=e.page_height,
                    coordinate_origin=e.coordinate_origin,
                    rotation=e.rotation,
                    section_path=e.section_path,
                    parser_version=e.parser_version,
                )
                db.add(db_elem)
                elem_map[e.element_index] = db_elem

            update_paper_status(
                db,
                paper_id,
                status=PaperStatus.PROCESSING,
                page_count=len(parse_result.pages),
            )
            db.flush()

            # 3. Chunk elements
            update_job_progress(db, job_id, stage=JobStage.CHUNKING, progress=0.5)
            chunk_specs = self.chunker.chunk(parse_result.elements)

            # Clear existing chunks
            db.query(PaperChunk).filter(PaperChunk.paper_id == paper_id).delete()
            db.flush()

            # 4. Embeddings
            update_job_progress(db, job_id, stage=JobStage.EMBEDDING, progress=0.7)
            embed_provider = get_embedding_provider()
            child_chunks = [c for c in chunk_specs if c.chunk_type == "child"]
            child_texts = [c.text for c in child_chunks]
            child_embeddings = embed_provider.embed_documents(child_texts) if child_texts else []

            embedding_map = {
                c.chunk_index: emb for c, emb in zip(child_chunks, child_embeddings, strict=False)
            }

            # 5. Insert Chunks and ChunkElements
            update_job_progress(db, job_id, stage=JobStage.INDEXING, progress=0.85)
            for c_spec in chunk_specs:
                emb = embedding_map.get(c_spec.chunk_index)
                db_chunk = PaperChunk(
                    paper_id=paper_id,
                    chunk_type=c_spec.chunk_type,
                    chunk_index=c_spec.chunk_index,
                    text=c_spec.text,
                    token_count=c_spec.token_count,
                    embedding=emb,
                    embedding_vec=emb,
                    embedding_model=embed_provider.model_name,
                    embedding_version=embed_provider.model_version,
                )
                db.add(db_chunk)
                db.flush()

                for order, elem_idx in enumerate(c_spec.element_indices):
                    if elem_idx in elem_map:
                        chunk_link = ChunkElement(
                            chunk_id=db_chunk.id,
                            element_id=elem_map[elem_idx].id,
                            order_index=order,
                        )
                        db.add(chunk_link)

            db.commit()

            # 6. Mark done
            update_paper_status(db, paper_id, status=PaperStatus.READY)
            update_job_progress(
                db,
                job_id,
                stage=JobStage.COMPLETED,
                progress=1.0,
                status=JobStatus.COMPLETED,
            )

        except Exception as err:
            db.rollback()
            err_msg = str(err)
            current_job = get_job(db, job_id)
            transient = is_transient_error(err)
            if current_job and transient and current_job.retry_count < current_job.max_retries:
                current_job.retry_count += 1
                current_job.status = JobStatus.PENDING
                current_job.stage = JobStage.QUEUED
                retries = f"{current_job.retry_count}/{current_job.max_retries}"
                msg = f"Transient failure ({retries}): {err_msg}"
                current_job.error_message = msg[:500]
                current_job.is_retryable = True
                db.commit()

            else:
                update_paper_status(db, paper_id, status=PaperStatus.FAILED, error_message=err_msg)
                update_job_progress(
                    db,
                    job_id,
                    stage=JobStage.FAILED,
                    progress=1.0,
                    status=JobStatus.FAILED,
                    error_message=err_msg,
                    is_retryable=False,
                )
