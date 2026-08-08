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


def _translate_batch_with_responses(
    source: str,
    responses: list[str],
    *,
    layout_label: str = "text",
    glossary_terms: list[dict[str, str]] | None = None,
) -> tuple[str, list[str], dict, engine_runner.TranslationCheckpointRecorder]:
    paragraph = _paragraph(source)
    paragraph.layout_label = layout_label
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
    record["first_occurrence_terms"] = glossary_terms or []
    prefix = "instructions\n\n## Here is the input:\n\n"
    item = {"id": 0, "input": source, "layout_label": layout_label}
    request_texts: list[str] = []

    class FakeResponse:
        status_code = 200

        def __init__(self, output: str) -> None:
            self.output = output

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, str]:
            return {"content": json.dumps([{"id": 0, "output": self.output}], ensure_ascii=False)}

    class FakeClient:
        def post(self, _url: str, **kwargs) -> FakeResponse:
            request_texts.append(kwargs["json"]["text"])
            if not responses:
                pytest.fail("unexpected additional provider request")
            return FakeResponse(responses.pop(0))

    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )
    translator.client.close()
    translator.client = FakeClient()
    with recorder.batch_scope():
        recorder.note_preprocessed(paragraph, source)
        response = translator.llm_translate(prefix + json.dumps([item]))
    translated = json.loads(response)[0]["output"]
    recorder.complete(record, translated)
    return translated, request_texts, record, recorder


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


def test_protocol_keeps_failure_causes_bounded_and_sanitized() -> None:
    event = {
        "type": "segment_summary",
        "total": 1,
        "completed": 0,
        "skipped": 0,
        "failed": 1,
        "failure_reasons": {"untranslated": 1},
        "failure_causes": {"preprocessing_not_selected": 1},
        "skip_reasons": {},
        "failure_units": [
            {
                "page_number": 1,
                "ordinal": 2,
                "status": "untranslated",
                "failure_reason": "preprocessing_not_selected",
                "source_chars": 40,
                "layout_label": "plain text",
            }
        ],
    }

    assert _safe_event(event)["failure_causes"] == {"preprocessing_not_selected": 1}
    event["failure_causes"] = {"vertical_paragraph": 1, "unknown_decline": 1}
    assert _safe_event(event)["failure_causes"] == {
        "vertical_paragraph": 1,
        "unknown_decline": 1,
    }
    event["failure_units"][0]["failure_reason"] = "raw quote must not cross the protocol"
    with pytest.raises(TranslationEngineError, match="ENGINE_PROTOCOL_ERROR"):
        _safe_event(event)


def test_protocol_keeps_only_safe_failure_unit_metadata() -> None:
    event = _safe_event(
        {
            "type": "segment_summary",
            "total": 2,
            "completed": 0,
            "skipped": 0,
            "failed": 2,
            "failure_reasons": {"unchanged_prose": 1, "untranslated": 1},
            "skip_reasons": {"below_engine_minimum": 3, "numeric_or_symbol_only": 2},
            "failure_units": [
                {
                    "page_number": 2,
                    "ordinal": 4,
                    "status": "unchanged_prose",
                    "source_chars": 84,
                    "layout_label": "text",
                    "source_quote": "This source content must never enter the protocol.",
                }
            ],
        }
    )
    assert event["failure_reasons"] == {"unchanged_prose": 1, "untranslated": 1}
    assert event["failure_causes"] == {}
    assert event["skip_reasons"] == {"below_engine_minimum": 3, "numeric_or_symbol_only": 2}
    assert event["failure_units"] == [
        {
            "page_number": 2,
            "ordinal": 4,
            "status": "unchanged_prose",
            "failure_reason": None,
            "source_chars": 84,
            "layout_label": "text",
        }
    ]


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


def test_checkpoint_recorder_rejects_obvious_unchanged_english_prose(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(engine_runner, "PROTOCOL_STDOUT", output)
    text = "Attention Is All You Need for sequence modeling."
    paragraph = _paragraph(text)
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
    recorder.note_preprocessed(paragraph, text)

    recorder.complete(record, text)

    assert record["status"] == "unchanged_prose"
    assert output.getvalue() == ""
    assert recorder.finish() == {"total": 1, "completed": 0, "skipped": 0, "failed": 1}
    assert recorder.failure_reasons() == {"unchanged_prose": 1}


def test_unchanged_prose_guard_matches_processor_normalization() -> None:
    text = "Transformer design uses unusual technical terminology."
    assert engine_runner._is_unchanged_english_prose(text, text, layout_label="text")


def test_checkpoint_recorder_emits_safe_summary_once_on_engine_failure(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(engine_runner, "PROTOCOL_STDOUT", output)
    paragraph = _paragraph("A required scientific paragraph that is still English.")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )
    recorder.note_preprocessed(paragraph, paragraph.unicode)

    recorder.emit_summary()
    recorder.emit_summary()

    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert len(events) == 1
    assert events[0]["failure_reasons"] == {"untranslated": 1}
    assert events[0]["failure_causes"] == {"missing_validated_checkpoint": 1}
    assert events[0]["failure_units"][0]["source_chars"] == len(paragraph.unicode)
    assert events[0]["failure_units"][0]["failure_reason"] == "missing_validated_checkpoint"
    assert "source_quote" not in events[0]["failure_units"][0]


def test_checkpoint_recorder_fails_unvisited_required_prose() -> None:
    paragraph = _paragraph("A long required paragraph omitted by the engine selector.")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )

    counts = recorder.finish()

    assert counts == {"total": 1, "completed": 0, "skipped": 0, "failed": 1}
    assert recorder.failure_reasons() == {"untranslated": 1}
    assert recorder.failure_causes() == {"preprocessing_not_selected": 1}


def test_checkpoint_recorder_distinguishes_unselected_from_missing_checkpoint() -> None:
    selected = _paragraph("A prose unit selected for translation by the engine.")
    not_selected = _paragraph("A prose unit not selected for translation by the engine.")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(
            page=[SimpleNamespace(page_number=1, pdf_paragraph=[selected, not_selected])]
        )
    )
    recorder.note_preprocessed(selected, selected.unicode)

    recorder.finish()

    assert recorder.failure_causes() == {
        "missing_validated_checkpoint": 1,
        "preprocessing_not_selected": 1,
    }


@pytest.mark.parametrize(
    ("paragraph", "translate_input", "numeric", "placeholder_only", "expected"),
    [
        (_paragraph("vertical"), None, False, False, "vertical_paragraph"),
        (
            SimpleNamespace(unicode="", pdf_paragraph_composition=[]),
            None,
            False,
            False,
            "no_composition",
        ),
        (
            SimpleNamespace(
                unicode="42", pdf_paragraph_composition=[SimpleNamespace(pdf_line=object())]
            ),
            None,
            True,
            False,
            "pure_numeric",
        ),
        (
            SimpleNamespace(
                unicode="{formula}",
                pdf_paragraph_composition=[SimpleNamespace(pdf_line=object())],
            ),
            None,
            False,
            True,
            "placeholder_only",
        ),
        (
            SimpleNamespace(
                unicode="x",
                pdf_paragraph_composition=[SimpleNamespace(pdf_formula=object())],
            ),
            None,
            False,
            False,
            "formula_only",
        ),
        (
            SimpleNamespace(
                unicode="debug",
                pdf_paragraph_composition=[
                    SimpleNamespace(pdf_same_style_unicode_characters=object())
                ],
            ),
            None,
            False,
            False,
            "debug_unicode_composition",
        ),
        (
            SimpleNamespace(
                unicode="long enough raw text",
                pdf_paragraph_composition=[SimpleNamespace(pdf_line=object())],
            ),
            SimpleNamespace(unicode="tiny"),
            False,
            False,
            "below_minimum_length",
        ),
        (
            SimpleNamespace(
                unicode="unknown",
                pdf_paragraph_composition=[SimpleNamespace(unrecognized=object())],
            ),
            None,
            False,
            False,
            "unsupported_composition",
        ),
        (
            SimpleNamespace(
                unicode="unknown",
                pdf_paragraph_composition=[SimpleNamespace(pdf_line=object())],
            ),
            None,
            False,
            False,
            "unknown_decline",
        ),
    ],
)
def test_preprocess_decline_classifier_uses_known_structure_only(
    paragraph, translate_input, numeric: bool, placeholder_only: bool, expected: str
) -> None:
    if expected == "vertical_paragraph":
        paragraph.vertical = True

    reason = engine_runner._classify_preprocess_decline(
        paragraph,
        translate_input,
        minimum_text_length=5,
        is_pure_numeric=lambda _paragraph: numeric,
        is_placeholder_only=lambda _paragraph: placeholder_only,
    )

    assert reason == expected


def test_preprocess_decline_diagnostics_do_not_change_skip_or_completion_counts() -> None:
    skipped = _paragraph("{equation}")
    required = _paragraph("A required prose paragraph omitted by preprocessing.")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[skipped, required])])
    )
    recorder.get(skipped)["preprocess_decline_reason"] = "formula_only"
    recorder.get(required)["preprocess_decline_reason"] = "no_composition"

    counts = recorder.finish()

    assert counts == {"total": 2, "completed": 0, "skipped": 1, "failed": 1}
    assert recorder.failure_causes() == {
        "no_composition": 1,
    }


def test_checkpoint_recorder_skips_only_short_unselected_content() -> None:
    paragraph = _paragraph("abc")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )

    assert recorder.finish() == {"total": 1, "completed": 0, "skipped": 1, "failed": 0}
    assert recorder.skip_reasons() == {"below_engine_minimum": 1}


def test_checkpoint_recorder_preserves_unchanged_official_title(monkeypatch) -> None:
    output = io.StringIO()
    monkeypatch.setattr(engine_runner, "PROTOCOL_STDOUT", output)
    text = "Attention Is All You Need for sequence modeling."
    paragraph = _paragraph(text)
    paragraph.layout_label = "title"
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
    recorder.note_preprocessed(paragraph, text)
    recorder.complete(record, text)
    counts = recorder.finish()
    checkpoint = json.loads(output.getvalue().splitlines()[0])
    assert counts == {"total": 1, "completed": 1, "skipped": 0, "failed": 0}
    assert checkpoint["segment"]["status"] == "preserved"


@pytest.mark.parametrize("text", ["E=mc^2", "API and SDK"])
def test_checkpoint_recorder_allows_unchanged_equations_and_short_terms(
    monkeypatch, text: str
) -> None:
    output = io.StringIO()
    monkeypatch.setattr(engine_runner, "PROTOCOL_STDOUT", output)
    paragraph = _paragraph(text)
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
    recorder.note_preprocessed(paragraph, text)

    recorder.complete(record, text)

    assert record["status"] == "completed"
    assert recorder.finish()["failed"] == 0


def test_checkpoint_recorder_skips_short_abandoned_layout_fragment() -> None:
    paragraph = _paragraph("1")
    paragraph.layout_label = "abandon"
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )
    assert recorder.finish() == {"total": 1, "completed": 0, "skipped": 1, "failed": 0}
    assert recorder.skip_reasons() == {"below_engine_minimum": 1}


def test_checkpoint_recorder_preserves_babeldoc_intentionally_skipped_units() -> None:
    paragraph = _paragraph("Only an equation")
    paragraph.layout_label = "equation"
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )
    recorder.note_preprocessed(paragraph, None)

    assert recorder.finish() == {"total": 1, "completed": 0, "skipped": 1, "failed": 0}
    assert recorder.skip_reasons() == {"protected_scientific_content": 1}


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


def test_llm_adapter_retries_unchanged_prose_once_and_preserves_glossary_placeholders() -> None:
    translated, request_texts, record, recorder = _translate_batch_with_responses(
        "The attention {v1} model improves the result in practice.",
        [
            "The attention {v1} model improves the result in practice.",
            "Mô hình chú ý {v1} cải thiện kết quả trong thực tiễn.",
        ],
        glossary_terms=[{"english": "attention", "vietnamese": "chú ý"}],
    )

    assert len(request_texts) == 2
    assert "previous output left these prose items in English" in request_texts[1]
    assert "first_occurrence_glossary_terms" in request_texts[1]
    assert translated == "Mô hình attention (chú ý) {v1} cải thiện kết quả trong thực tiễn."
    assert record["status"] == "completed"
    assert recorder.finish()["completed"] == 1


def test_llm_adapter_bounds_unchanged_prose_retry_and_keeps_failure_untranslated() -> None:
    source = "The model improves the result in practice for this task."
    translated, request_texts, record, recorder = _translate_batch_with_responses(
        source, [source, source]
    )

    assert len(request_texts) == 2
    assert translated == source
    assert record["status"] == "unchanged_prose"
    assert recorder.finish()["failed"] == 1


def test_llm_adapter_does_not_retry_unchanged_scientific_protected_content() -> None:
    source = "E = mc^2 {v1}"
    translated, request_texts, record, recorder = _translate_batch_with_responses(
        source, [source], layout_label="formula"
    )

    assert len(request_texts) == 1
    assert translated == source
    assert record["status"] == "completed"
    assert recorder.finish()["completed"] == 1


def test_llm_adapter_keeps_normal_batch_response_on_single_request() -> None:
    translated, request_texts, record, recorder = _translate_batch_with_responses(
        "The model improves the result in practice for this task.",
        ["该模型在此任务中提高了实际效果。"],
    )

    assert len(request_texts) == 1
    assert translated == "该模型在此任务中提高了实际效果。"
    assert record["status"] == "completed"
    assert recorder.finish()["completed"] == 1


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
