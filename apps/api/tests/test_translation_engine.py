from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.services.translation import engine_runner
from app.services.translation.engine import (
    TranslationEngineError,
    TranslationEngineProcess,
    _minimal_environment,
    _safe_event,
)


def _paragraph(text: str, debug_id: str = "p-1") -> SimpleNamespace:
    return SimpleNamespace(unicode=text, debug_id=debug_id, layout_label="text")


def test_engine_environment_does_not_inherit_application_secrets(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "do-not-forward")
    monkeypatch.setenv("DATABASE_URL", "postgresql://private")
    monkeypatch.setenv("HF_HOME", "/models/cache")

    env = _minimal_environment()

    assert "DEEPSEEK_API_KEY" not in env
    assert "DATABASE_URL" not in env
    assert env["HF_HOME"] == "/models/cache"


def test_protocol_sanitizes_progress_and_validates_checkpoints() -> None:
    progress = _safe_event(
        {
            "type": "progress",
            "stage": "untrusted raw stage",
            "current": -1,
            "total": 3,
            "progress": 125,
        }
    )
    assert progress == {
        "type": "progress",
        "stage": "translation",
        "current": None,
        "total": 3,
        "progress": 100.0,
    }

    checkpoint = {
        "type": "checkpoint",
        "segment": {
            "segment_key": "k" * 64,
            "page_number": 1,
            "ordinal": 0,
            "source_quote": "original quote",
            "source_start": 0,
            "source_end": 14,
            "source_sha256": "a" * 64,
            "translated_text": "translated quote",
            "translated_sha256": "b" * 64,
            "status": "validated",
        },
    }
    assert _safe_event(checkpoint)["segment"]["source_quote"] == "original quote"
    checkpoint["segment"]["translated_text"] = "x" * 65_537
    with pytest.raises(TranslationEngineError, match="ENGINE_PROTOCOL_ERROR"):
        _safe_event(checkpoint)


def test_protocol_rejects_completion_with_untranslated_segments() -> None:
    event = {
        "type": "complete",
        "output_pdf": "/private/path.pdf",
        "segment_counts": {"total": 3, "completed": 2, "skipped": 0, "failed": 1},
    }
    with pytest.raises(TranslationEngineError, match="ENGINE_INCOMPLETE"):
        _safe_event(event)


def test_checkpoint_recorder_emits_page_quote_offsets_and_hashes(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(engine_runner, "PROTOCOL_STDOUT", output)
    paragraph = _paragraph("Text with {v1} marker")
    page = SimpleNamespace(page_number=2, pdf_paragraph=[paragraph])
    docs = SimpleNamespace(page=[page])
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[{"source": "attention", "target": "chú ý"}],
        checkpoint_results=[],
    )

    recorder.begin(docs)
    record = recorder.get(paragraph)
    assert record is not None
    recorder.note_preprocessed(paragraph, "Text with {v1} marker")
    recorder.complete(record, "Văn bản có {v1} dấu")
    counts = recorder.finish()

    event = json.loads(output.getvalue().splitlines()[0])
    assert event["type"] == "checkpoint"
    assert event["segment"]["page_number"] == 2
    assert event["segment"]["source_start"] == 0
    assert event["segment"]["source_end"] == len("Text with {v1} marker")
    assert event["segment"]["offset_basis"] == "babeldoc_reading_order_paragraphs"
    assert counts == {"total": 1, "completed": 1, "skipped": 0, "failed": 0}


def test_checkpoint_recorder_rejects_dropped_protected_placeholders(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(engine_runner, "PROTOCOL_STDOUT", output)
    paragraph = _paragraph("Formula {v1}")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )
    record = recorder.get(paragraph)
    assert record is not None
    recorder.note_preprocessed(paragraph, "Formula {v1}")

    recorder.complete(record, "Công thức")

    assert output.getvalue() == ""
    assert recorder.finish()["failed"] == 1


def test_first_glossary_occurrence_includes_english_and_preferred_vietnamese() -> None:
    assert (
        engine_runner._apply_first_occurrence_terms(
            "Mô hình chú ý cải thiện chất lượng.",
            [{"english": "attention", "vietnamese": "chú ý"}],
        )
        == "Mô hình attention (chú ý) cải thiện chất lượng."
    )
    assert (
        engine_runner._apply_first_occurrence_terms(
            "Attention (chú ý) là một cơ chế.",
            [{"english": "attention", "vietnamese": "chú ý"}],
        )
        == "Attention (chú ý) là một cơ chế."
    )


def test_llm_adapter_reuses_checkpoint_without_provider_call() -> None:
    paragraph = _paragraph("Hello {v1}")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )
    record = recorder.get(paragraph)
    assert record is not None
    record["preprocessed_input"] = "Hello {v1}"
    with recorder.batch_scope():
        recorder.note_preprocessed(paragraph, "Hello {v1}")
        prefix = "instructions\n\n## Here is the input:\n\n"
        item = {"id": 0, "input": "Hello {v1}", "layout_label": "text"}
        key = recorder.key_for(record, engine_runner._context_hash(prefix, "", item["input"]))
        recorder.checkpoints[key] = "Xin chào {v1}"
        translator = engine_runner.SiliconFlowFreeTranslator(
            "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
        )
        translator.client.close()
        translator.client = SimpleNamespace(
            post=lambda *args, **kwargs: pytest.fail("provider called")
        )

        response = translator.llm_translate(prefix + json.dumps([item]))

    assert json.loads(response) == [{"id": 0, "output": "Xin chào {v1}"}]


def test_llm_adapter_sends_only_missed_units_when_resuming_partial_batch() -> None:
    first = _paragraph("First {v1}", "p-1")
    second = _paragraph("Second paragraph", "p-2")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[first, second])])
    )
    first_record = recorder.get(first)
    second_record = recorder.get(second)
    assert first_record is not None and second_record is not None
    prefix = "instructions\n\n## Here is the input:\n\n"
    items = [
        {"id": 0, "input": "First {v1}", "layout_label": "text"},
        {"id": 1, "input": "Second paragraph", "layout_label": "text"},
    ]
    first_key = recorder.key_for(
        first_record,
        engine_runner._context_hash(prefix, "", items[0]["input"]),
    )
    recorder.checkpoints[first_key] = "Thứ nhất {v1}"
    with recorder.batch_scope():
        recorder.note_preprocessed(first, items[0]["input"])
        recorder.note_preprocessed(second, items[1]["input"])
        translator = engine_runner.SiliconFlowFreeTranslator(
            "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
        )

        class FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, str]:
                return {"content": '[{"id":1,"output":"Đoạn thứ hai"}]'}

        class FakeClient:
            def __init__(self) -> None:
                self.payloads: list[dict] = []

            def post(self, _url: str, **kwargs) -> FakeResponse:
                self.payloads.append(kwargs["json"])
                return FakeResponse()

        client = FakeClient()
        translator.client.close()
        translator.client = client
        response = translator.llm_translate(prefix + json.dumps(items))

    assert json.loads(response) == [
        {"id": 0, "output": "Thứ nhất {v1}"},
        {"id": 1, "output": "Đoạn thứ hai"},
    ]
    assert len(client.payloads) == 1
    assert "Second paragraph" in client.payloads[0]["text"]
    assert "First {v1}" not in client.payloads[0]["text"]


def test_siliconflow_transport_retries_at_most_three_times(monkeypatch) -> None:
    class FakeResponse:
        def __init__(self, status_code: int, content: str | None = None) -> None:
            self.status_code = status_code
            self._content = content

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError("private response", request=None, response=None)

        def json(self) -> dict[str, str]:
            return {"content": self._content or ""}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.responses = [FakeResponse(429), FakeResponse(429), FakeResponse(200, "xin chào")]
            self.calls = 0

        def post(self, url: str, **kwargs) -> FakeResponse:
            self.calls += 1
            assert url == "https://api1.pdf2zh-next.com/chatproxy"
            assert set(kwargs["json"]) == {"text"}
            return self.responses.pop(0)

    fake = FakeClient()
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: fake)
    monkeypatch.setattr(engine_runner.time, "sleep", lambda delay: None)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )

    assert translator._request("private paper text") == "xin chào"
    assert fake.calls == 3


def test_siliconflow_failure_is_classified_after_three_attempts(monkeypatch) -> None:
    class FakeResponse:
        status_code = 429

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.calls = 0

        def post(self, *args, **kwargs) -> FakeResponse:
            self.calls += 1
            return FakeResponse()

    fake = FakeClient()
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: fake)
    monkeypatch.setattr(engine_runner.time, "sleep", lambda delay: None)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )

    with pytest.raises(engine_runner.SafeEngineError) as exc_info:
        translator._request("private paper text")

    assert exc_info.value.code == "PROVIDER_RATE_LIMITED"
    assert "private paper text" not in str(exc_info.value)
    assert fake.calls == 3


def test_subprocess_emits_progress_and_checkpoint_without_stderr(tmp_path: Path) -> None:
    runner = tmp_path / "fake_runner.py"
    runner.write_text(
        "import json, sys\n"
        "sys.stdin.readline()\n"
        "print(json.dumps({'type':'progress','stage':'translation','current':1,'total':2,'progress':50}))\n"
        "print(json.dumps({'type':'checkpoint','segment':{'segment_key':'k','page_number':1,'ordinal':0,'source_quote':'q','source_start':0,'source_end':1,'source_sha256':'a'*64,'translated_text':'t','translated_sha256':'b'*64,'status':'validated'}}))\n"
        "print(json.dumps({'type':'complete','output_pdf':'/tmp/result.pdf','source_mapping':'available','resume_supported':True,'segment_counts':{'total':1,'completed':1,'skipped':0,'failed':0},'failure_code':None}))\n"
        "print('sensitive diagnostic on stderr', file=sys.stderr)\n",
        encoding="utf-8",
    )
    seen_progress: list[dict] = []
    seen_checkpoints: list[dict] = []

    async def run() -> dict:
        process = TranslationEngineProcess(
            python=sys.executable,
            runner=runner,
            timeout_seconds=5,
        )
        return await process.run(
            {"input_pdf": "/tmp/input.pdf"},
            on_progress=lambda event: _append(seen_progress, event),
            on_checkpoint=lambda event: _append(seen_checkpoints, event),
        )

    completion = asyncio.run(run())
    assert seen_progress[0]["stage"] == "translation"
    assert seen_checkpoints[0]["source_quote"] == "q"
    assert completion["source_mapping"] == "available"
    assert completion["segment_counts"]["completed"] == 1


async def _append(target: list, value: dict) -> None:
    target.append(value)


def test_subprocess_deadline_kills_engine(tmp_path: Path) -> None:
    runner = tmp_path / "slow_runner.py"
    runner.write_text("import sys, time\nsys.stdin.readline()\ntime.sleep(10)\n", encoding="utf-8")

    async def run() -> None:
        process = TranslationEngineProcess(
            python=sys.executable,
            runner=runner,
            timeout_seconds=1,
        )
        await process.run({})

    with pytest.raises(TranslationEngineError, match="ENGINE_DEADLINE_EXCEEDED"):
        asyncio.run(run())
