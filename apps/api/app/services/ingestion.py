import hashlib
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.crud.job import LostJobLeaseError, fence_job_for_publish, get_job, update_job_progress
from app.crud.paper import get_paper
from app.db.models import ChunkElement, PaperChunk, PaperElement, PaperPage
from app.ingestion.chunker import DocumentChunker
from app.ingestion.parser import DocumentParser
from app.schemas.job import JobStage, JobStatus
from app.schemas.paper import PaperStatus
from app.services.embedding import get_embedding_provider
from app.services.error_sanitizer import classify_and_sanitize_error
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

    async def process_paper(
        self, db: Session, paper_id: UUID, job_id: UUID, worker_id: str | None = None
    ) -> None:
        paper = get_paper(db, paper_id)
        job = get_job(db, job_id)
        if not paper or not job:
            return

        try:
            # 1. Fetch file from storage
            update_job_progress(
                db, job_id, stage=JobStage.PARSING, progress=0.1, worker_id=worker_id
            )
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
            update_job_progress(
                db, job_id, stage=JobStage.PARSING, progress=0.25, worker_id=worker_id
            )
            document_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
            if paper.document_sha256 and paper.document_sha256 != document_sha256:
                raise ValueError("Stored PDF checksum does not match the uploaded document")
            parse_result = self.parser.parse(pdf_bytes)

            # 3. Chunk elements
            update_job_progress(
                db, job_id, stage=JobStage.CHUNKING, progress=0.5, worker_id=worker_id
            )
            chunk_specs = self.chunker.chunk(parse_result.elements)

            # 4. Embeddings
            update_job_progress(
                db, job_id, stage=JobStage.EMBEDDING, progress=0.7, worker_id=worker_id
            )
            embed_provider = get_embedding_provider()
            child_chunks = [c for c in chunk_specs if c.chunk_type == "child"]
            child_texts = [c.text for c in child_chunks]
            child_embeddings = embed_provider.embed_documents(child_texts) if child_texts else []

            embedding_map = {
                c.chunk_index: emb for c, emb in zip(child_chunks, child_embeddings, strict=False)
            }

            # 5. The conditional UPDATE locks this job row through publication.
            update_job_progress(
                db, job_id, stage=JobStage.INDEXING, progress=0.85, worker_id=worker_id
            )
            fence_job_for_publish(db, job_id, worker_id)

            # Replace the entire index in one transaction. A failed/stale worker
            # cannot leave a half-updated paper marked READY.
            chunk_ids = select(PaperChunk.id).where(PaperChunk.paper_id == paper_id)
            db.query(ChunkElement).filter(ChunkElement.chunk_id.in_(chunk_ids)).delete(
                synchronize_session=False
            )
            db.query(PaperChunk).filter(PaperChunk.paper_id == paper_id).delete()
            db.query(PaperElement).filter(PaperElement.paper_id == paper_id).delete()
            db.query(PaperPage).filter(PaperPage.paper_id == paper_id).delete()
            db.flush()

            for page in parse_result.pages:
                db.add(
                    PaperPage(
                        paper_id=paper_id,
                        page_number=page.page_number,
                        width=page.width,
                        height=page.height,
                        rotation=page.rotation,
                        crop_box=page.crop_box,
                        raw_text=page.raw_text,
                    )
                )

            elem_map: dict[int, PaperElement] = {}
            for element in parse_result.elements:
                db_element = PaperElement(
                    paper_id=paper_id,
                    page_number=element.page_number,
                    element_index=element.element_index,
                    element_type=element.element_type,
                    text=element.text,
                    bbox_x_min=element.bbox_x_min,
                    bbox_y_min=element.bbox_y_min,
                    bbox_x_max=element.bbox_x_max,
                    bbox_y_max=element.bbox_y_max,
                    page_width=element.page_width,
                    page_height=element.page_height,
                    coordinate_origin=element.coordinate_origin,
                    rotation=element.rotation,
                    section_path=element.section_path,
                    parser_version=element.parser_version,
                )
                db.add(db_element)
                elem_map[element.element_index] = db_element
            db.flush()

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

            paper.document_sha256 = document_sha256
            paper.page_count = len(parse_result.pages)
            paper.status = PaperStatus.READY
            paper.error_message = None
            job.status = JobStatus.COMPLETED
            job.stage = JobStage.COMPLETED
            job.progress = 1.0
            job.error_message = None
            job.is_retryable = False
            db.commit()

        except LostJobLeaseError:
            db.rollback()
            return
        except Exception as err:
            db.rollback()
            classified = classify_and_sanitize_error(err)
            try:
                fence_job_for_publish(db, job_id, worker_id)
            except LostJobLeaseError:
                return
            current_job = get_job(db, job_id)
            db.refresh(current_job)
            transient = classified.is_transient or is_transient_error(err)
            if current_job and transient and current_job.retry_count < current_job.max_retries:
                current_job.retry_count += 1
                current_job.status = JobStatus.PENDING
                current_job.stage = JobStage.QUEUED
                retries = f"{current_job.retry_count}/{current_job.max_retries}"
                code_str = classified.code.value
                msg = f"[{code_str}] Transient failure ({retries}): {classified.sanitized_message}"
                current_job.error_message = msg[:500]
                current_job.is_retryable = True
                db.commit()

            else:
                msg = f"[{classified.code.value}] {classified.sanitized_message}"
                paper.status = PaperStatus.FAILED
                paper.error_message = msg
                current_job.stage = JobStage.FAILED
                current_job.progress = 1.0
                current_job.status = JobStatus.FAILED
                current_job.error_message = msg[:500]
                current_job.is_retryable = False
                db.commit()
