"""Storage, checkpoint, and publication boundary for a translation attempt."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import re
import tempfile
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.crud.translation import save_translation_segment
from app.db.models import Paper, TranslationSegment
from app.db.session import SessionLocal
from app.observability.telemetry import get_telemetry
from app.services.translation.engine import TranslationEngineError, TranslationEngineProcess
from app.storage import ObjectStorage, get_storage
from app.translation_worker import (
    TranslationJob,
    TranslationProcessingError,
    TranslationResult,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENGINE_PYTHON = "/opt/myra-translation/.venv/bin/python"
_ENGINE_RUNNER = "/app/app/services/translation/engine_runner.py"
logger = logging.getLogger("myra.translation.processor")


def _storage_key(path: str) -> str:
    if path.startswith("gs://"):
        parts = path.split("/", 3)
        if len(parts) != 4:
            raise TranslationProcessingError(
                "SOURCE_NOT_FOUND", "The original PDF is unavailable.", retryable=False
            )
        return parts[3]
    if path.startswith("memory://"):
        return path.removeprefix("memory://")
    return path


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")[:100]
    return name or "paper.pdf"


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


class BabelDocTranslationProcessor:
    """Run the isolated PDFMathTranslate runtime and publish an immutable output."""

    def __init__(
        self,
        *,
        storage: ObjectStorage | None = None,
        session_factory: Callable[[], Session] = SessionLocal,
        engine_factory: Callable[[], TranslationEngineProcess] | None = None,
        engine_python: str | None = None,
        engine_runner: str | None = None,
        layout_model: str | None = None,
    ) -> None:
        self.storage = storage
        self.session_factory = session_factory
        self.engine_factory = engine_factory
        self.engine_python = engine_python or os.getenv(
            "MYRA_TRANSLATION_ENGINE_PYTHON", _ENGINE_PYTHON
        )
        self.engine_runner = engine_runner or os.getenv(
            "MYRA_TRANSLATION_ENGINE_RUNNER", _ENGINE_RUNNER
        )
        self.layout_model = layout_model or os.getenv(
            "MYRA_TRANSLATION_LAYOUT_MODEL",
            "/tmp/.cache/babeldoc/models/doclayout_yolo_docstructbench_imgsz1024.onnx",
        )

    def _engine(self, *, home_dir: Path) -> TranslationEngineProcess:
        if self.engine_factory:
            return self.engine_factory()
        return TranslationEngineProcess(
            python=self.engine_python,
            runner=self.engine_runner,
            env={
                "PATH": os.getenv("PATH", "/usr/local/bin:/usr/bin:/bin"),
                "HOME": str(home_dir),
                "MYRA_TRANSLATION_ASSET_DIR": "/tmp/.cache/babeldoc",
            },
        )

    @staticmethod
    def _trace_metadata(**fields: Any) -> dict[str, Any]:
        return {
            "test_run": os.getenv("MYRA_TRANSLATION_TEST_RUN", "false").lower() == "true",
            **fields,
        }

    def _existing_checkpoints(self, job: TranslationJob) -> list[dict[str, str]]:
        with self.session_factory() as db:
            rows = db.scalars(
                select(TranslationSegment).where(
                    TranslationSegment.translation_id == job.id,
                    TranslationSegment.status.in_(["VALIDATED", "PRESERVED"]),
                    TranslationSegment.engine_checkpoint_key.is_not(None),
                )
            ).all()
            return [
                {
                    "segment_key": row.engine_checkpoint_key,
                    "translated_text": row.translated_text,
                    "translated_sha256": row.translated_text_hash,
                }
                for row in rows
                if row.engine_checkpoint_key
                and _SHA256.fullmatch(row.engine_checkpoint_key)
                and _sha256(row.translated_text.encode("utf-8")) == row.translated_text_hash
            ]

    def _persist_checkpoint(self, job: TranslationJob, segment: dict[str, Any]) -> None:
        source_quote = segment["source_quote"]
        translated_text = segment["translated_text"]
        preserved_title = (
            segment.get("status") == "preserved"
            and segment.get("layout_label") == "title"
            and _normalize_text(source_quote) == _normalize_text(translated_text)
        )
        if (
            segment.get("status") not in {"validated", "preserved"}
            or (segment.get("status") == "preserved" and not preserved_title)
            or not _SHA256.fullmatch(segment["segment_key"])
            or not _SHA256.fullmatch(segment["source_sha256"])
            or not _SHA256.fullmatch(segment["translated_sha256"])
            or _sha256(source_quote.encode("utf-8")) != segment["source_sha256"]
            or _sha256(translated_text.encode("utf-8")) != segment["translated_sha256"]
        ):
            raise TranslationProcessingError(
                "ENGINE_PROTOCOL_ERROR", "A translation unit failed validation.", retryable=False
            )
        if (
            not preserved_title
            and len(_normalize_text(source_quote)) >= 24
            and _normalize_text(source_quote) == _normalize_text(translated_text)
        ):
            raise TranslationProcessingError(
                "ENGINE_INCOMPLETE",
                "A required prose segment was returned without translation.",
                retryable=True,
            )
        with self.session_factory() as db:
            saved = save_translation_segment(
                db,
                job.id,
                worker_id=job.worker_id,
                attempt_token=job.attempt_token,
                engine_checkpoint_key=segment["segment_key"],
                ordinal=segment["ordinal"],
                source_page_number=segment["page_number"],
                source_text_hash=segment["source_sha256"],
                source_quote=source_quote,
                translated_text=translated_text,
                translated_text_hash=segment["translated_sha256"],
                status="PRESERVED" if preserved_title else "VALIDATED",
            )
        if not saved:
            raise TranslationProcessingError(
                "TRANSLATION_ATTEMPT_REVOKED",
                "Translation attempt was cancelled or superseded.",
                retryable=False,
            )

    def _source_is_current(self, job: TranslationJob) -> bool:
        with self.session_factory() as db:
            paper = db.get(Paper, job.paper_id)
            return bool(
                paper
                and paper.project_id == job.project_id
                and paper.status == "READY"
                and paper.document_sha256 == job.source_sha256
            )

    async def _validate_pdf(
        self, output_path: Path, job: TranslationJob
    ) -> tuple[bytes, list[dict[str, Any]]]:
        try:
            data = await asyncio.to_thread(output_path.read_bytes)
            if not data.startswith(b"%PDF") or len(data) < 128:
                raise ValueError("invalid PDF signature")
            reader = PdfReader(io.BytesIO(data), strict=True)
            if not reader.pages:
                raise ValueError("PDF has no pages")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if not text.strip():
                raise ValueError("PDF has no selectable text")
        except Exception as exc:
            raise TranslationProcessingError(
                "OUTPUT_VALIDATION_FAILED",
                "The translated PDF could not be validated.",
                retryable=False,
            ) from exc

        with self.session_factory() as db:
            rows = list(
                db.scalars(
                    select(TranslationSegment)
                    .where(
                        TranslationSegment.translation_id == job.id,
                        TranslationSegment.status.in_(["VALIDATED", "PRESERVED"]),
                    )
                    .order_by(TranslationSegment.ordinal)
                ).all()
            )
        translated_rows = [row for row in rows if row.status == "VALIDATED"]
        normalized_pdf_text = _normalize_text(text)
        meaningful_rows = [
            row for row in translated_rows if len(_normalize_text(row.translated_text)) >= 24
        ]
        matched_rows = [
            row
            for row in meaningful_rows
            if _normalize_text(row.translated_text) in normalized_pdf_text
        ]
        minimum_matches = max(1, (len(meaningful_rows) * 3 + 4) // 5)
        if not meaningful_rows or len(matched_rows) < minimum_matches:
            raise TranslationProcessingError(
                "OUTPUT_TRANSLATION_NOT_FOUND",
                "The translated PDF did not contain a verified translated text segment.",
                retryable=False,
            )
        source_map = [
            {
                "ordinal": row.ordinal,
                "source_page_number": row.source_page_number,
                "source_text_hash": row.source_text_hash,
                "source_quote": row.source_quote,
                "translated_text_hash": row.translated_text_hash,
                "status": row.status.lower(),
            }
            for row in rows
        ]
        return data, source_map

    async def process(
        self,
        job: TranslationJob,
        *,
        on_progress: Callable[..., None],
    ) -> TranslationResult:
        storage = self.storage or get_storage()
        telemetry = get_telemetry()
        artifact_key: str | None = None
        with tempfile.TemporaryDirectory(prefix=f"myra-translation-{job.id}-") as tmp:
            root = Path(tmp)
            input_path = root / "source.pdf"
            output_dir = root / "output"
            working_dir = root / "working"
            output_dir.mkdir()
            working_dir.mkdir()
            engine_home = working_dir / "engine-home"
            engine_home.mkdir()

            with telemetry.stage(
                "translation.source_download",
                metadata=self._trace_metadata(
                    translation_id=str(job.id), attempt=job.attempt_count
                ),
            ) as observation:
                try:
                    source = await storage.get(_storage_key(job.source_storage_path))
                except FileNotFoundError as exc:
                    raise TranslationProcessingError(
                        "SOURCE_NOT_FOUND", "The original PDF is unavailable.", retryable=False
                    ) from exc
                if _sha256(source) != job.source_sha256:
                    raise TranslationProcessingError(
                        "SOURCE_HASH_MISMATCH",
                        "The original PDF changed after this translation was requested.",
                        retryable=False,
                    )
                if observation is not None:
                    observation.update(output={"bytes": len(source), "sha256": job.source_sha256})
            await asyncio.to_thread(input_path.write_bytes, source)

            checkpoints = self._existing_checkpoints(job)
            request = {
                "input_pdf": str(input_path),
                "output_dir": str(output_dir),
                "working_dir": str(working_dir),
                "layout_model": self.layout_model,
                "lang_in": "English",
                "lang_out": "Vietnamese",
                "source_pdf_sha256": job.source_sha256,
                "glossary": [
                    {"source": item["source_term"], "target": item["preferred_translation"]}
                    for item in job.glossary_snapshot
                ],
                "checkpoint_results": checkpoints,
            }

            async def on_engine_progress(event: dict[str, Any]) -> None:
                if event.get("type") == "segment_summary":
                    logger.info(
                        "translation_segment_validation",
                        extra={
                            "stage": "validation",
                            "total_units": event.get("total"),
                            "completed_units": event.get("completed"),
                            "skipped_units": event.get("skipped"),
                            "failure_count": event.get("failed"),
                            "failure_reasons": event.get("failure_reasons", {}),
                            "failure_causes": event.get("failure_causes", {}),
                            "skip_reasons": event.get("skip_reasons", {}),
                            "failure_units": event.get("failure_units", [])[:20],
                        },
                    )
                    get_telemetry().event(
                        "translation.validation",
                        metadata=self._trace_metadata(
                            translation_id=str(job.id),
                            failure_reasons=event.get("failure_reasons", {}),
                            failure_causes=event.get("failure_causes", {}),
                            skip_reasons=event.get("skip_reasons", {}),
                            failure_units=event.get("failure_units", []),
                        ),
                        output={
                            "outcome": "complete" if event.get("failed") == 0 else "partial",
                            "total": event.get("total"),
                            "completed": event.get("completed"),
                            "skipped": event.get("skipped"),
                            "failed": event.get("failed"),
                        },
                    )
                    on_progress(
                        "translation",
                        completed_units=event.get("completed"),
                        total_units=event.get("total"),
                    )
                    return
                on_progress(
                    event.get("stage", "translation"),
                )

            rejected_prose_units: list[dict[str, int | str]] = []

            async def on_checkpoint(segment: dict[str, Any]) -> None:
                source_quote = segment.get("source_quote")
                translated_text = segment.get("translated_text")
                if (
                    segment.get("status") != "preserved"
                    and isinstance(source_quote, str)
                    and isinstance(translated_text, str)
                    and len(_normalize_text(source_quote)) >= 24
                    and _normalize_text(source_quote) == _normalize_text(translated_text)
                ):
                    rejected_prose_units.append(
                        {
                            "page_number": segment.get("page_number", 0),
                            "ordinal": segment.get("ordinal", 0),
                            "status": "unchanged_prose",
                            "source_chars": len(source_quote),
                        }
                    )
                    return
                self._persist_checkpoint(job, segment)

            with telemetry.stage(
                "translation.layout_analysis",
                metadata=self._trace_metadata(
                    translation_id=str(job.id), source_pages=job.source_page_count
                ),
            ):
                on_progress("layout_analysis")
            with telemetry.stage(
                "translation.translate",
                input={"glossary_terms": len(job.glossary_snapshot)},
                metadata=self._trace_metadata(
                    translation_id=str(job.id),
                    engine_version="2.9.0",
                    babeldoc_version="0.6.2",
                    provider_policy="siliconflowfree-v1",
                    documented_model="THUDM/GLM-4-9B-0414",
                    provider_reported_model=None,
                    provider_usage=None,
                    checkpoint_hits=len(checkpoints),
                ),
            ):
                try:
                    completion = await self._engine(home_dir=engine_home).run(
                        request,
                        on_progress=on_engine_progress,
                        on_checkpoint=on_checkpoint,
                    )
                except TranslationEngineError as exc:
                    raise TranslationProcessingError(
                        exc.code,
                        "Translation provider or engine did not complete this request.",
                        retryable=exc.code
                        in {
                            "PROVIDER_RATE_LIMITED",
                            "PROVIDER_UNAVAILABLE",
                            "ENGINE_DEADLINE_EXCEEDED",
                            "ENGINE_INCOMPLETE",
                        },
                    ) from exc

            counts = completion.get("segment_counts", {})
            if (
                completion.get("failure_code")
                or counts.get("failed", 0) != 0
                or counts.get("completed", 0) + counts.get("skipped", 0) != counts.get("total")
                or counts.get("completed", 0) == 0
                or rejected_prose_units
            ):
                failure_count = max(counts.get("failed", 0), len(rejected_prose_units))
                failure_units = rejected_prose_units[:20]
                logger.warning(
                    "translation_segment_validation_failed",
                    extra={
                        "stage": "validation",
                        "total_units": counts.get("total"),
                        "completed_units": counts.get("completed"),
                        "skipped_units": counts.get("skipped"),
                        "failure_count": failure_count,
                        "failure_reasons": (
                            {"unchanged_prose": len(rejected_prose_units)}
                            if rejected_prose_units
                            else {}
                        ),
                        "failure_causes": {},
                        "failure_units": failure_units,
                    },
                )
                raise TranslationProcessingError(
                    completion.get("failure_code") or "ENGINE_INCOMPLETE",
                    "Some document text could not be translated safely.",
                    retryable=completion.get("failure_code") == "PROVIDER_RATE_LIMITED"
                    or counts.get("failed", 0) > 0
                    or counts.get("completed", 0) + counts.get("skipped", 0) != counts.get("total")
                    or bool(rejected_prose_units),
                )

            output_path = Path(completion["output_pdf"]).resolve()
            if not output_path.is_relative_to(output_dir.resolve()) or not output_path.is_file():
                raise TranslationProcessingError(
                    "ENGINE_PROTOCOL_ERROR", "Translation output was unavailable.", retryable=False
                )
            with telemetry.stage(
                "translation.validation",
                metadata=self._trace_metadata(translation_id=str(job.id), segment_counts=counts),
            ):
                output, source_map = await self._validate_pdf(output_path, job)
            if not self._source_is_current(job):
                raise TranslationProcessingError(
                    "SOURCE_CHANGED",
                    "The original paper changed before translation publication.",
                    retryable=False,
                )
            latest_source = await storage.get(_storage_key(job.source_storage_path))
            if _sha256(latest_source) != job.source_sha256:
                raise TranslationProcessingError(
                    "SOURCE_HASH_MISMATCH",
                    "The original PDF changed before translation publication.",
                    retryable=False,
                )

            output_sha = _sha256(output)
            artifact_key = (
                f"translations/{job.project_id}/{job.paper_id}/{job.id}/"
                f"attempt-{job.attempt_count}-{job.attempt_token}/{output_sha}.pdf"
            )
            with telemetry.stage(
                "translation.publication",
                metadata=self._trace_metadata(translation_id=str(job.id), output_sha256=output_sha),
            ):
                try:
                    stored_path = await storage.put_stream(
                        artifact_key, io.BytesIO(output), content_type="application/pdf"
                    )
                except BaseException:
                    try:
                        await asyncio.shield(storage.delete(artifact_key))
                    except Exception:
                        pass
                    raise
            return TranslationResult(
                output_storage_path=stored_path,
                output_sha256=output_sha,
                source_map=source_map,
            )

    async def cleanup(self, result: TranslationResult) -> None:
        storage = self.storage or get_storage()
        await storage.delete(_storage_key(result.output_storage_path))
