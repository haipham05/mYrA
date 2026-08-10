from __future__ import annotations

import asyncio
import hashlib
import shutil
from collections.abc import Generator
from contextlib import nullcontext
from pathlib import Path

import pytest
from pypdf import PdfReader
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.models import Paper, Project, TranslationDocument, TranslationSegment
from app.observability.telemetry import TelemetryAdapter, TelemetryConfig
from app.services.translation.engine import TranslationEngineError
from app.services.translation.processor import (
    BabelDocTranslationProcessor,
    _translation_text_is_present,
    _translation_tokens,
)
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
    def __init__(
        self, output_pdf: Path, translated_text: str, *, progress_stage: str = "translation"
    ) -> None:
        self.output_pdf = output_pdf
        self.translated_text = translated_text
        self.progress_stage = progress_stage
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
        await on_progress({"stage": self.progress_stage, "current": 1, "total": 1})
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


@pytest.mark.parametrize("telemetry_mode", ["disabled", "exporter_unavailable"])
def test_processor_publishes_when_telemetry_disabled_or_unavailable(
    database, tmp_path, monkeypatch, telemetry_mode
):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    storage = MemoryStorage()
    asyncio.run(storage.put("papers/fixture.pdf", source))
    output_text = PdfReader(FIXTURE_PDF).pages[0].extract_text()
    fake_engine = FakeEngine(FIXTURE_PDF, output_text)
    if telemetry_mode == "disabled":
        telemetry = TelemetryAdapter(config=TelemetryConfig(enabled=False))
    else:

        class UnavailableClient:
            def start_as_current_observation(self, **kwargs):
                raise RuntimeError("test exporter unavailable")

        telemetry = TelemetryAdapter(UnavailableClient(), config=TelemetryConfig(enabled=True))
    monkeypatch.setattr("app.services.translation.processor.get_telemetry", lambda: telemetry)
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


@pytest.mark.parametrize(
    ("progress_stage", "observed"),
    [("rendering", True), ("translation", False)],
)
def test_processor_reports_render_duration_only_when_engine_reports_render_progress(
    database, tmp_path, monkeypatch, progress_stage, observed
):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    storage = MemoryStorage()
    asyncio.run(storage.put("papers/fixture.pdf", source))
    output_text = PdfReader(FIXTURE_PDF).pages[0].extract_text()
    fake_engine = FakeEngine(FIXTURE_PDF, output_text, progress_stage=progress_stage)

    class RecordingTelemetry:
        def __init__(self):
            self.events = []

        def stage(self, *_args, **_kwargs):
            return nullcontext(None)

        def event(self, name, **kwargs):
            self.events.append((name, kwargs))

    telemetry = RecordingTelemetry()
    monkeypatch.setattr("app.services.translation.processor.get_telemetry", lambda: telemetry)
    processor = BabelDocTranslationProcessor(
        storage=storage,
        session_factory=database,
        engine_factory=lambda: fake_engine,
        layout_model=str(tmp_path / "unused-layout.onnx"),
    )

    result = asyncio.run(processor.process(job, on_progress=lambda *_args, **_kwargs: None))

    assert result.output_storage_path.startswith("memory://translations/")
    name, event = telemetry.events[-1]
    assert name == "translation.render"
    output = event["output"]
    assert output["outcome"] == "complete"
    assert output["render_progress_observed"] is observed
    assert (output["duration_ms"] is not None) is observed
    assert output["duration_ms"] is None or output["duration_ms"] >= 0
    assert output["segment_counts"] == {
        "total": 1,
        "completed": 1,
        "skipped": 0,
        "failed": 0,
    }
    assert output["output_bytes"] == len(source)
    assert set(output) == {
        "outcome",
        "render_progress_observed",
        "duration_ms",
        "segment_counts",
        "output_bytes",
    }
    assert all(str(tmp_path) not in str(value) for value in output.values())


@pytest.mark.parametrize(
    ("failure_code", "retryable"),
    [
        ("PROVIDER_UNAVAILABLE", True),
        ("PROVIDER_INVALID_JSON", True),
        ("PROVIDER_MARKER_MISMATCH", True),
        ("PROVIDER_SCIENTIFIC_TOKEN_MISMATCH", True),
        ("ENGINE_PROTOCOL_ERROR", False),
    ],
)
def test_render_telemetry_failure_does_not_mask_engine_failure(
    database, tmp_path, monkeypatch, failure_code, retryable
):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    storage = MemoryStorage()
    asyncio.run(storage.put("papers/fixture.pdf", source))

    class FailingTelemetry:
        def stage(self, *_args, **_kwargs):
            return nullcontext(None)

        def event(self, *_args, **_kwargs):
            raise RuntimeError("private exporter diagnostic")

    class FailingEngine:
        async def run(self, _request, *, on_progress, on_checkpoint):
            del on_checkpoint
            await on_progress({"stage": "rendering"})
            raise TranslationEngineError(failure_code)

    telemetry = FailingTelemetry()
    monkeypatch.setattr("app.services.translation.processor.get_telemetry", lambda: telemetry)
    processor = BabelDocTranslationProcessor(
        storage=storage,
        session_factory=database,
        engine_factory=FailingEngine,
        layout_model=str(tmp_path / "unused-layout.onnx"),
    )

    with pytest.raises(TranslationProcessingError) as error:
        asyncio.run(processor.process(job, on_progress=lambda *_args, **_kwargs: None))

    assert error.value.code == failure_code
    assert error.value.retryable is retryable


def test_restarted_processor_loads_only_integrity_checked_checkpoints(database):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    output_text = PdfReader(FIXTURE_PDF).pages[0].extract_text() or ""
    valid_text = output_text[:120]
    assert len(valid_text) >= 24
    valid_key = hashlib.sha256(b"valid persisted checkpoint identity").hexdigest()
    valid_hash = hashlib.sha256(valid_text.encode()).hexdigest()

    with database() as db:
        db.add_all(
            [
                TranslationSegment(
                    translation_id=job.id,
                    engine_checkpoint_key=valid_key,
                    ordinal=0,
                    source_page_number=1,
                    source_text_hash=hashlib.sha256(b"original source").hexdigest(),
                    source_quote="Original source",
                    translated_text=valid_text,
                    translated_text_hash=valid_hash,
                    status="VALIDATED",
                ),
                TranslationSegment(
                    translation_id=job.id,
                    engine_checkpoint_key=hashlib.sha256(b"corrupt checkpoint").hexdigest(),
                    ordinal=1,
                    source_page_number=1,
                    source_text_hash=hashlib.sha256(b"another source").hexdigest(),
                    source_quote="Another source",
                    translated_text="Corrupt cached translation",
                    translated_text_hash="0" * 64,
                    status="VALIDATED",
                ),
                TranslationSegment(
                    translation_id=job.id,
                    engine_checkpoint_key=hashlib.sha256(b"unfinished checkpoint").hexdigest(),
                    ordinal=2,
                    source_page_number=1,
                    source_text_hash=hashlib.sha256(b"unfinished source").hexdigest(),
                    source_quote="Unfinished source",
                    translated_text="Unfinished translation",
                    translated_text_hash=hashlib.sha256(b"Unfinished translation").hexdigest(),
                    status="PENDING",
                ),
            ]
        )
        db.commit()

    # A new processor models a worker restart and must ignore corrupt or partial rows.
    restarted_processor = BabelDocTranslationProcessor(
        session_factory=database,
    )
    assert restarted_processor._existing_checkpoints(job) == [
        {
            "segment_key": valid_key,
            "translated_text": valid_text,
            "translated_sha256": valid_hash,
        }
    ]


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


def test_pdf_validation_matches_translation_after_stripping_layout_markers(database, tmp_path):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    output_text = " ".join((PdfReader(FIXTURE_PDF).pages[0].extract_text() or "").split())
    visible_text = output_text[:100]
    translated_text = f"{visible_text[:35]}<b1>{visible_text[35:]}</b1>"
    translated_hash = hashlib.sha256(translated_text.encode()).hexdigest()
    with database() as db:
        db.add(
            TranslationSegment(
                translation_id=job.id,
                engine_checkpoint_key=hashlib.sha256(b"marker-checkpoint").hexdigest(),
                ordinal=0,
                source_page_number=1,
                source_text_hash=hashlib.sha256(b"source sentence").hexdigest(),
                source_quote="source sentence",
                translated_text=translated_text,
                translated_text_hash=translated_hash,
                status="VALIDATED",
            )
        )
        db.commit()
    output_pdf = tmp_path / "rendered.pdf"
    shutil.copyfile(FIXTURE_PDF, output_pdf)

    validated_pdf, source_map = asyncio.run(
        BabelDocTranslationProcessor(session_factory=database)._validate_pdf(output_pdf, job)
    )

    assert validated_pdf == source
    assert source_map[0]["translated_text_hash"] == translated_hash


def test_pdf_validation_still_rejects_absent_translation(database, tmp_path):
    source = FIXTURE_PDF.read_bytes()
    job = _create_processing_job(database, source)
    translated_text = "A completely unrelated translated paragraph that is not in this PDF."
    with database() as db:
        db.add(
            TranslationSegment(
                translation_id=job.id,
                engine_checkpoint_key=hashlib.sha256(b"absent-checkpoint").hexdigest(),
                ordinal=0,
                source_page_number=1,
                source_text_hash=hashlib.sha256(b"source sentence").hexdigest(),
                source_quote="source sentence",
                translated_text=translated_text,
                translated_text_hash=hashlib.sha256(translated_text.encode()).hexdigest(),
                status="VALIDATED",
            )
        )
        db.commit()
    output_pdf = tmp_path / "rendered.pdf"
    shutil.copyfile(FIXTURE_PDF, output_pdf)

    with pytest.raises(TranslationProcessingError, match="OUTPUT_TRANSLATION_NOT_FOUND"):
        asyncio.run(
            BabelDocTranslationProcessor(session_factory=database)._validate_pdf(output_pdf, job)
        )


def test_translation_presence_requires_a_bounded_window_on_one_page():
    expected = "alpha beta gamma delta epsilon zeta"
    scattered = ["alpha", *(["noise"] * 12), "beta", *(["noise"] * 12), "gamma"]
    scattered.extend(["delta", *(["noise"] * 12), "epsilon"])

    assert not _translation_text_is_present(expected, [scattered, ["zeta"]])
    assert _translation_text_is_present(
        expected,
        [["alpha", "noise", "beta", "gamma", "delta", "epsilon", "noise", "zeta"]],
    )


def test_translation_tokens_join_words_split_by_layout_markers():
    assert _translation_tokens("translated<b1>paragraph</b1>") == ["translatedparagraph"]


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
