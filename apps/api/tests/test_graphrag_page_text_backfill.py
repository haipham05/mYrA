"""Safe, idempotent page-text restoration tests."""

from __future__ import annotations

import hashlib
from collections.abc import Generator
from uuid import uuid4

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.crud.corpus import read_corpus_revision
from app.db.base import Base
from app.db.models import Paper, PaperChunk, PaperElement, PaperPage, Project
from app.ingestion.parser import ParsedPage, ParseResult
from app.services.graphrag.page_text_backfill import backfill_missing_page_text
from app.storage.local import MemoryStorage

PDF_BYTES = b"unchanged source pdf bytes"
DOCUMENT_SHA256 = hashlib.sha256(PDF_BYTES).hexdigest()


class StubParser:
    def __init__(self, pages: list[ParsedPage] | None = None) -> None:
        self.pages = pages or [
            ParsedPage(page_number=1, width=612, height=792, raw_text="Restored page one."),
            ParsedPage(page_number=2, width=612, height=792, raw_text="Restored page two."),
        ]

    def parse(self, _: bytes) -> ParseResult:
        return ParseResult(pages=self.pages, elements=[])


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def seed_legacy_paper(db: Session) -> tuple[Project, Paper, list[PaperPage]]:
    project = Project(id=uuid4(), name="Backfill test")
    db.add(project)
    db.flush()
    paper = Paper(
        id=uuid4(),
        project_id=project.id,
        filename="legacy.pdf",
        storage_path="papers/legacy.pdf",
        document_sha256=DOCUMENT_SHA256,
        page_count=2,
        status="READY",
    )
    pages = [
        PaperPage(
            id=uuid4(),
            paper_id=paper.id,
            page_number=1,
            width=612,
            height=792,
            raw_text=None,
        ),
        PaperPage(
            id=uuid4(),
            paper_id=paper.id,
            page_number=2,
            width=612,
            height=792,
            raw_text="Existing page text stays intact.",
        ),
    ]
    element = PaperElement(
        id=uuid4(),
        paper_id=paper.id,
        page_number=1,
        element_index=0,
        element_type="paragraph",
        text="Existing element identity",
    )
    chunk = PaperChunk(
        id=uuid4(),
        paper_id=paper.id,
        chunk_type="child",
        chunk_index=0,
        text="Existing chunk and embedding identity",
        embedding=[0.1, 0.2],
    )
    db.add_all([paper, *pages, element, chunk])
    db.commit()
    return project, paper, pages


@pytest.mark.anyio
async def test_dry_run_then_restore_only_missing_text_and_repeat_is_idempotent(
    db_session: Session,
) -> None:
    project, paper, pages = seed_legacy_paper(db_session)
    element_id = db_session.query(PaperElement.id).filter_by(paper_id=paper.id).scalar()
    chunk = db_session.query(PaperChunk).filter_by(paper_id=paper.id).one()
    chunk_id, chunk_embedding = chunk.id, chunk.embedding
    storage = MemoryStorage()
    paper.storage_path = "memory://papers/legacy.pdf"
    db_session.commit()
    await storage.put("papers/legacy.pdf", PDF_BYTES)
    parser = StubParser()

    preview = await backfill_missing_page_text(
        db_session,
        project_id=project.id,
        paper_id=paper.id,
        expected_document_sha256=DOCUMENT_SHA256,
        storage=storage,
        parser=parser,
    )
    assert preview.missing_page_numbers == (1,)
    assert preview.updated_page_numbers == ()
    assert read_corpus_revision(db_session, project.id) == 0
    assert db_session.get(PaperPage, pages[0].id).raw_text is None

    restored = await backfill_missing_page_text(
        db_session,
        project_id=project.id,
        paper_id=paper.id,
        expected_document_sha256=DOCUMENT_SHA256,
        storage=storage,
        parser=parser,
        dry_run=False,
    )
    assert restored.updated_page_numbers == (1,)
    assert read_corpus_revision(db_session, project.id) == 1
    assert db_session.get(PaperPage, pages[0].id).raw_text == "Restored page one."
    assert db_session.get(PaperPage, pages[1].id).raw_text == "Existing page text stays intact."
    assert db_session.query(PaperElement.id).filter_by(paper_id=paper.id).scalar() == element_id
    after_chunk = db_session.query(PaperChunk).filter_by(paper_id=paper.id).one()
    assert (after_chunk.id, after_chunk.embedding) == (chunk_id, chunk_embedding)

    repeated = await backfill_missing_page_text(
        db_session,
        project_id=project.id,
        paper_id=paper.id,
        expected_document_sha256=DOCUMENT_SHA256,
        storage=storage,
        parser=parser,
        dry_run=False,
    )
    assert repeated.missing_page_numbers == ()
    assert repeated.updated_page_numbers == ()
    assert read_corpus_revision(db_session, project.id) == 1


@pytest.mark.anyio
async def test_gcs_uri_backfill_uses_only_the_configured_bucket_key(db_session: Session) -> None:
    project, paper, pages = seed_legacy_paper(db_session)
    storage = MemoryStorage()
    storage.bucket_name = "approved-bucket"
    await storage.put("papers/legacy.pdf", PDF_BYTES)
    paper.storage_path = "gs://approved-bucket/papers/legacy.pdf"
    db_session.commit()

    result = await backfill_missing_page_text(
        db_session,
        project_id=project.id,
        paper_id=paper.id,
        expected_document_sha256=DOCUMENT_SHA256,
        storage=storage,
        parser=StubParser(),
        dry_run=False,
    )
    assert result.updated_page_numbers == (1,)

    paper.storage_path = "gs://different-bucket/papers/legacy.pdf"
    db_session.commit()
    db_session.get(PaperPage, pages[0].id).raw_text = None
    db_session.commit()
    with pytest.raises(ValueError, match="configured storage bucket"):
        await backfill_missing_page_text(
            db_session,
            project_id=project.id,
            paper_id=paper.id,
            expected_document_sha256=DOCUMENT_SHA256,
            storage=storage,
            parser=StubParser(),
            dry_run=False,
        )
    assert db_session.get(PaperPage, pages[0].id).raw_text is None


@pytest.mark.anyio
async def test_wrong_expected_hash_or_download_hash_aborts_without_writes(
    db_session: Session,
) -> None:
    project, paper, pages = seed_legacy_paper(db_session)
    storage = MemoryStorage()
    await storage.put(paper.storage_path, PDF_BYTES)

    with pytest.raises(ValueError, match="Expected document hash"):
        await backfill_missing_page_text(
            db_session,
            project_id=project.id,
            paper_id=paper.id,
            expected_document_sha256="b" * 64,
            storage=storage,
            parser=StubParser(),
            dry_run=False,
        )
    assert db_session.get(PaperPage, pages[0].id).raw_text is None

    await storage.put(paper.storage_path, b"different PDF bytes")
    with pytest.raises(ValueError, match="Stored PDF hash"):
        await backfill_missing_page_text(
            db_session,
            project_id=project.id,
            paper_id=paper.id,
            expected_document_sha256=DOCUMENT_SHA256,
            storage=storage,
            parser=StubParser(),
            dry_run=False,
        )
    assert db_session.get(PaperPage, pages[0].id).raw_text is None


@pytest.mark.anyio
async def test_scope_and_page_identity_mismatches_abort_without_writes(db_session: Session) -> None:
    project, paper, pages = seed_legacy_paper(db_session)
    storage = MemoryStorage()
    await storage.put(paper.storage_path, PDF_BYTES)

    with pytest.raises(ValueError, match="explicitly selected project"):
        await backfill_missing_page_text(
            db_session,
            project_id=uuid4(),
            paper_id=paper.id,
            expected_document_sha256=DOCUMENT_SHA256,
            storage=storage,
            parser=StubParser(),
            dry_run=False,
        )
    with pytest.raises(ValueError, match="Parsed page identities"):
        await backfill_missing_page_text(
            db_session,
            project_id=project.id,
            paper_id=paper.id,
            expected_document_sha256=DOCUMENT_SHA256,
            storage=storage,
            parser=StubParser(pages=[ParsedPage(1, 612, 792, raw_text="Only one page")]),
            dry_run=False,
        )
    assert db_session.get(PaperPage, pages[0].id).raw_text is None
