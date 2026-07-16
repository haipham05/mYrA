from __future__ import annotations

import asyncio
import hashlib
import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
from pypdf import PdfReader
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Paper, Project, TranslationDocument, TranslationSegment
from app.services.translation.processor import BabelDocTranslationProcessor
from app.storage.local import MemoryStorage
from app.translation_worker import TranslationJob, TranslationProcessingError

FIXTURE_PDF = Path(__file__).parents[3] / "tests/retrieval_eval/fixtures/vaswani2017_attention.pdf"


@pytest.fixture
def database() -> Generator[sessionmaker[Session], None, None]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


class FakeEngine:
    def __init__(self, output_pdf: Path, translated_text: str) -> None:
        self.output_pdf = output_pdf
        self.translated_text = translated_text
        self.request = None

    async def run(self, request, *, on_progress, on_checkpoint):
        self.request = request
        output_path = Path(request["output_dir"]) / "translated.pdf"
        shutil.copyfile(self.output_pdf, output_path)
        source = "Original source sentence for a translation checkpoint."
        await on_checkpoint(
            {
                "segment_key": hashlib.sha256(b"checkpoint").hexdigest(),
                "page_number": 1,
                "ordinal": 0,
                "source_quote": source,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "translated_text": self.translated_text,
                "translated_sha256": hashlib.sha256(self.translated_text.encode()).hexdigest(),
                "status": "validated",
            }
        )
        await on_progress({"stage": "translation", "current": 1, "total": 1})
        return {
            "output_pdf": str(output_path),
            "segment_counts": {"total": 1, "completed": 1, "skipped": 0, "failed": 0},
            "failure_code": None,
        }


def _create_processing_job(factory: sessionmaker[Session], source: bytes) -> TranslationJob:
    with factory() as db:
        project = Project(name="Processor test")
        db.add(project)
        db.flush()
        paper = Paper(
            project_id=project.id,
            filename="fixture.pdf",
            storage_path="papers/fixture.pdf",
            document_sha256=hashlib.sha256(source).hexdigest(),
            status="READY",
            page_count=1,
        )
        db.add(paper)
        db.flush()
        document = TranslationDocument(
            project_id=project.id,
            paper_id=paper.id,
            status="PROCESSING",
            stage="TRANSLATION",
            idempotency_key="processor-test",
            acknowledge_external_processing=True,
            source_sha256=paper.document_sha256,
            source_storage_path=paper.storage_path,
            source_filename=paper.filename,
            source_page_count=1,
            glossary_snapshot=[{"source_term": "attention", "preferred_translation": "chú ý"}],
            attempt_token="attempt-token",
            lease_owner="test-worker",
            attempt_count=1,
        )
        db.add(document)
        db.commit()
        return TranslationJob(
            id=document.id,
            project_id=project.id,
            paper_id=paper.id,
            source_sha256=paper.document_sha256,
            source_storage_path=paper.storage_path,
            source_filename=paper.filename,
            source_page_count=1,
            glossary_snapshot=({"source_term": "attention", "preferred_translation": "chú ý"},),
            worker_id="test-worker",
            attempt_token="attempt-token",
            attempt_count=1,
        )


def test_processor_checkpoints_validates_and_publishes_attempt_scoped_pdf(database, tmp_path):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    storage = MemoryStorage()
    asyncio.run(storage.put("papers/fixture.pdf", source))
    output_text = PdfReader(FIXTURE_PDF).pages[0].extract_text()
    fake_engine = FakeEngine(FIXTURE_PDF, output_text)
    processor = BabelDocTranslationProcessor(
        storage=storage,
        session_factory=database,
        engine_factory=lambda: fake_engine,
        layout_model=str(tmp_path / "unused-layout.onnx"),
    )
    stages = []

    async def process():
        return await processor.process(
            job,
            on_progress=lambda stage, **kwargs: stages.append((stage, kwargs)),
        )

    result = asyncio.run(process())
    with database() as db:
        saved = db.scalars(
            select(TranslationSegment).where(TranslationSegment.translation_id == job.id)
        ).one()
    assert saved.engine_checkpoint_key == hashlib.sha256(b"checkpoint").hexdigest()
    assert saved.status == "VALIDATED"
    assert result.output_storage_path.startswith("memory://translations/")
    assert result.output_sha256 == hashlib.sha256(FIXTURE_PDF.read_bytes()).hexdigest()
    assert result.source_map[0]["source_page_number"] == 1
    assert fake_engine.request["lang_out"] == "Vietnamese"
    assert fake_engine.request["glossary"] == [{"source": "attention", "target": "chú ý"}]
    assert any(stage == "layout_analysis" for stage, _ in stages)
    assert any(stage == "translation" for stage, _ in stages)


def test_processor_rejects_incomplete_engine_output(database, tmp_path):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    storage = MemoryStorage()
    asyncio.run(storage.put("papers/fixture.pdf", source))

    class IncompleteEngine:
        async def run(self, request, *, on_progress, on_checkpoint):
            return {
                "output_pdf": str(FIXTURE_PDF),
                "segment_counts": {"total": 2, "completed": 1, "skipped": 0, "failed": 1},
                "failure_code": None,
            }

    processor = BabelDocTranslationProcessor(
        storage=storage,
        session_factory=database,
        engine_factory=IncompleteEngine,
        layout_model=str(tmp_path / "unused-layout.onnx"),
    )

    async def process():
        await processor.process(job, on_progress=lambda *args, **kwargs: None)

    with pytest.raises(TranslationProcessingError, match="ENGINE_INCOMPLETE") as error:
        asyncio.run(process())
    assert error.value.retryable is True


def test_processor_rejects_unchanged_required_prose_checkpoint(database):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    processor = BabelDocTranslationProcessor(session_factory=database)
    unchanged = "This required scientific prose must not remain in English."
    with pytest.raises(TranslationProcessingError, match="ENGINE_INCOMPLETE") as error:
        processor._persist_checkpoint(
            job,
            {
                "segment_key": hashlib.sha256(b"same").hexdigest(),
                "page_number": 1,
                "ordinal": 0,
                "source_quote": unchanged,
                "source_sha256": hashlib.sha256(unchanged.encode()).hexdigest(),
                "translated_text": unchanged,
                "translated_sha256": hashlib.sha256(unchanged.encode()).hexdigest(),
                "status": "validated",
            },
        )
    assert error.value.retryable is True


def test_processor_persists_explicitly_preserved_official_title(database):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    processor = BabelDocTranslationProcessor(session_factory=database)
    title = "Attention Is All You Need for Sequence Modeling"
    title_hash = hashlib.sha256(title.encode()).hexdigest()

    processor._persist_checkpoint(
        job,
        {
            "segment_key": hashlib.sha256(b"preserved-title").hexdigest(),
            "page_number": 1,
            "ordinal": 0,
            "source_quote": title,
            "source_sha256": title_hash,
            "translated_text": title,
            "translated_sha256": title_hash,
            "status": "preserved",
            "layout_label": "title",
        },
    )

    with database() as db:
        saved = db.scalars(
            select(TranslationSegment).where(TranslationSegment.translation_id == job.id)
        ).one()
    assert saved.status == "PRESERVED"
    assert saved.translated_text == title


def test_processor_rejects_changed_checkpoint_mislabeled_as_preserved_title(database):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    processor = BabelDocTranslationProcessor(session_factory=database)
    title = "Attention Is All You Need for Sequence Modeling"
    translated = "Một tiêu đề đã được thay đổi"

    with pytest.raises(TranslationProcessingError, match="ENGINE_PROTOCOL_ERROR"):
        processor._persist_checkpoint(
            job,
            {
                "segment_key": hashlib.sha256(b"changed-preserved-title").hexdigest(),
                "page_number": 1,
                "ordinal": 0,
                "source_quote": title,
                "source_sha256": hashlib.sha256(title.encode()).hexdigest(),
                "translated_text": translated,
                "translated_sha256": hashlib.sha256(translated.encode()).hexdigest(),
                "status": "preserved",
                "layout_label": "title",
            },
        )


def test_processor_withholds_unchanged_checkpoint_until_document_finishes(database, tmp_path):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    storage = MemoryStorage()
    asyncio.run(storage.put("papers/fixture.pdf", source))
    unchanged = "This required scientific prose remains untranslated."

    class UnchangedEngine:
        async def run(self, request, *, on_progress, on_checkpoint):
            await on_checkpoint(
                {
                    "segment_key": hashlib.sha256(b"unchanged").hexdigest(),
                    "page_number": 1,
                    "ordinal": 0,
                    "source_quote": unchanged,
                    "source_sha256": hashlib.sha256(unchanged.encode()).hexdigest(),
                    "translated_text": unchanged,
                    "translated_sha256": hashlib.sha256(unchanged.encode()).hexdigest(),
                    "status": "validated",
                }
            )
            output = Path(request["output_dir"]) / "output.pdf"
            shutil.copyfile(FIXTURE_PDF, output)
            return {
                "output_pdf": str(output),
                "segment_counts": {"total": 1, "completed": 1, "skipped": 0, "failed": 0},
                "failure_code": None,
            }

    processor = BabelDocTranslationProcessor(
        storage=storage,
        session_factory=database,
        engine_factory=UnchangedEngine,
        layout_model=str(tmp_path / "unused-layout.onnx"),
    )

    async def process():
        await processor.process(job, on_progress=lambda *args, **kwargs: None)

    with pytest.raises(TranslationProcessingError, match="ENGINE_INCOMPLETE") as error:
        asyncio.run(process())
    assert error.value.retryable is True
    with database() as db:
        saved = db.scalars(
            select(TranslationSegment).where(TranslationSegment.translation_id == job.id)
        ).all()
    assert saved == []
