"""Bounded metadata-only restoration of missing authoritative PDF page text."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from urllib.parse import urlsplit
from uuid import UUID

from sqlalchemy.orm import Session

from app.crud.corpus import bump_corpus_revision
from app.db.models import Paper, PaperPage
from app.ingestion.parser import DocumentParser
from app.storage.base import ObjectStorage


@dataclass(frozen=True)
class PageTextBackfillResult:
    """Summary of one explicitly scoped page-text backfill attempt."""

    paper_id: UUID
    page_count: int
    missing_page_numbers: tuple[int, ...]
    updated_page_numbers: tuple[int, ...]
    dry_run: bool


def _storage_object_key(storage_path: str, storage: ObjectStorage) -> str:
    """Normalize supported stored URI forms without crossing storage buckets."""
    parsed = urlsplit(storage_path)
    if parsed.scheme == "gs":
        configured_bucket = getattr(storage, "bucket_name", None)
        if not configured_bucket or parsed.netloc != configured_bucket:
            raise ValueError("Stored GCS URI does not match the configured storage bucket")
        key = parsed.path.lstrip("/")
        if not key:
            raise ValueError("Stored GCS URI has no object key")
        return key
    if parsed.scheme == "memory":
        key = f"{parsed.netloc}{parsed.path}".lstrip("/")
        if not key:
            raise ValueError("Stored memory URI has no object key")
        return key
    if parsed.scheme:
        raise ValueError(f"Unsupported stored document URI scheme: {parsed.scheme}")
    return storage_path


async def backfill_missing_page_text(
    db: Session,
    *,
    project_id: UUID,
    paper_id: UUID,
    expected_document_sha256: str,
    storage: ObjectStorage,
    parser: DocumentParser,
    dry_run: bool = True,
) -> PageTextBackfillResult:
    """Restore only NULL page text for one hash-pinned, READY paper.

    This helper owns its transaction and therefore requires a clean Session.
    It never modifies paper/page identity, elements, chunks, embeddings or
    citations. Callers must provide both project and paper IDs plus the
    expected document hash; dry-run is the default.
    """
    if db.new or db.dirty or db.deleted:
        raise ValueError("Page-text backfill requires a clean database session")
    if len(expected_document_sha256) != 64:
        raise ValueError("Expected document SHA-256 must be a 64-character hex digest")
    try:
        int(expected_document_sha256, 16)
    except ValueError as err:
        raise ValueError("Expected document SHA-256 must be hexadecimal") from err

    try:
        paper = (
            db.query(Paper)
            .filter(Paper.id == paper_id, Paper.project_id == project_id)
            .with_for_update()
            .one_or_none()
        )
        if paper is None:
            raise ValueError("Paper does not exist in the explicitly selected project")
        if paper.status != "READY":
            raise ValueError("Only READY papers can have page text restored")
        if (
            not paper.document_sha256
            or paper.document_sha256.lower() != expected_document_sha256.lower()
        ):
            raise ValueError("Expected document hash does not match the stored paper version")
        if not paper.page_count or paper.page_count < 1:
            raise ValueError("Paper has no valid stored page count")

        object_key = _storage_object_key(paper.storage_path, storage)
        pdf_bytes = await storage.get(object_key)
        actual_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        if actual_sha256 != paper.document_sha256.lower():
            raise ValueError("Stored PDF hash does not match the selected paper version")

        parsed = parser.parse(pdf_bytes)
        expected_page_numbers = list(range(1, paper.page_count + 1))
        parsed_page_numbers = [page.page_number for page in parsed.pages]
        if parsed_page_numbers != expected_page_numbers:
            raise ValueError("Parsed page identities do not match the stored paper page count")

        pages = (
            db.query(PaperPage)
            .filter(PaperPage.paper_id == paper.id)
            .order_by(PaperPage.page_number)
            .with_for_update()
            .all()
        )
        if [page.page_number for page in pages] != expected_page_numbers:
            raise ValueError("Stored page identities do not match the selected paper version")

        raw_text_by_page = {page.page_number: page.raw_text for page in parsed.pages}
        missing = tuple(page.page_number for page in pages if page.raw_text is None)
        for page in pages:
            if page.raw_text is None:
                page.raw_text = raw_text_by_page[page.page_number]

        if dry_run:
            db.rollback()
            updated: tuple[int, ...] = ()
        else:
            if missing:
                bump_corpus_revision(db, paper.project_id)
            db.commit()
            updated = missing

        return PageTextBackfillResult(
            paper_id=paper.id,
            page_count=paper.page_count,
            missing_page_numbers=missing,
            updated_page_numbers=updated,
            dry_run=dry_run,
        )
    except Exception:
        db.rollback()
        raise
