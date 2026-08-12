from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import sys
import types
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.logging import configure_logging
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


@pytest.mark.parametrize(
    "code",
    [
        "PROVIDER_REJECTED",
        "PROVIDER_INVALID_RESPONSE",
        "PROVIDER_INVALID_SCHEMA",
        "PROVIDER_INVALID_JSON",
        "PROVIDER_INVALID_OUTPUT",
        "PROVIDER_MISSING_ITEM",
        "PROVIDER_MARKER_MISMATCH",
        "PROVIDER_SCIENTIFIC_TOKEN_MISMATCH",
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_UNAVAILABLE",
    ],
)
def test_protocol_preserves_safe_provider_error_classification(code: str) -> None:
    with pytest.raises(TranslationEngineError, match=code) as exc_info:
        _safe_event({"type": "error", "code": code})

    assert exc_info.value.code == code


def test_protocol_sanitizes_unknown_provider_error_classification() -> None:
    with pytest.raises(TranslationEngineError, match="ENGINE_FAILURE") as exc_info:
        _safe_event({"type": "error", "code": "private text from a provider response"})

    assert exc_info.value.code == "ENGINE_FAILURE"


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


def test_protocol_accepts_split_footnote_preservation_reason() -> None:
    event = {
        "type": "segment_summary",
        "total": 3,
        "completed": 0,
        "skipped": 3,
        "failed": 0,
        "skip_reasons": {"preserved_split_footnote_layout_content": 3},
    }

    assert _safe_event(event)["skip_reasons"] == {"preserved_split_footnote_layout_content": 3}


def test_protocol_accepts_arxiv_stamp_skip_reason() -> None:
    event = {
        "type": "segment_summary",
        "total": 1,
        "completed": 0,
        "skipped": 1,
        "failed": 0,
        "skip_reasons": {"arxiv_version_stamp": 1},
    }

    assert _safe_event(event)["skip_reasons"] == {"arxiv_version_stamp": 1}


def test_protocol_accepts_short_fallback_omission_reason() -> None:
    event = {
        "type": "segment_summary",
        "total": 1,
        "completed": 0,
        "skipped": 1,
        "failed": 0,
        "failure_reasons": {},
        "skip_reasons": {"untranslated_short_fallback": 1},
        "failure_units": [],
    }

    assert _safe_event(event)["skip_reasons"] == {"untranslated_short_fallback": 1}


def test_protocol_accepts_reviewed_embedded_figure_skip_reason() -> None:
    event = {
        "type": "segment_summary",
        "total": 1,
        "completed": 0,
        "skipped": 1,
        "failed": 0,
        "skip_reasons": {
            "preserved_embedded_figure_text": 1,
            "preserved_scientific_table_content": 1,
        },
    }

    assert _safe_event(event)["skip_reasons"] == {
        "preserved_embedded_figure_text": 1,
        "preserved_scientific_table_content": 1,
    }


def test_protocol_accepts_reviewed_figure_and_citation_metadata_skip_reason() -> None:
    event = {
        "type": "segment_summary",
        "total": 3,
        "completed": 0,
        "skipped": 3,
        "failed": 0,
        "skip_reasons": {"preserved_figure_or_citation_metadata": 3},
    }

    assert _safe_event(event)["skip_reasons"] == {"preserved_figure_or_citation_metadata": 3}


def test_checkpoint_recorder_preserves_only_contiguous_split_footnotes() -> None:
    fragments = [
        _paragraph("5We used values of 2.8, 3.", "footnote-1"),
        _paragraph("7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100", "footnote-2"),
        _paragraph(", respectively.", "footnote-3"),
    ]
    fragments[0].layout_label = "text"
    fragments[1].layout_label = "text"
    fragments[2].layout_label = "text"
    next_order = 100
    for index, paragraph in enumerate(fragments):
        characters = []
        for char in paragraph.unicode:
            if char.isspace():
                characters.append(SimpleNamespace(render_order=None, char_unicode=char))
            else:
                characters.append(SimpleNamespace(render_order=next_order, char_unicode=char))
                next_order += 1
        if index == 0:
            paragraph.pdf_paragraph_composition = [
                SimpleNamespace(pdf_formula=SimpleNamespace(pdf_character=characters[:1])),
                SimpleNamespace(
                    pdf_same_style_characters=SimpleNamespace(pdf_character=characters[1:])
                ),
            ]
        else:
            paragraph.pdf_paragraph_composition = [
                SimpleNamespace(pdf_line=SimpleNamespace(pdf_character=characters))
            ]

    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(SimpleNamespace(page=[SimpleNamespace(page_number=7, pdf_paragraph=fragments)]))

    for paragraph in fragments:
        assert recorder.get(paragraph)["preserve_reason"] == (
            "preserved_split_footnote_layout_content"
        )
        recorder.note_preprocessed(paragraph, None)

    assert recorder.finish() == {"total": 3, "completed": 0, "skipped": 3, "failed": 0}
    assert recorder.skip_reasons() == {"preserved_split_footnote_layout_content": 3}


@pytest.mark.parametrize(
    "break_order",
    ["gap", "unknown_composition", "different_page", "no_sentence_end"],
)
def test_checkpoint_recorder_does_not_group_ambiguous_footnote_fragments(
    break_order: str,
) -> None:
    first = _paragraph("1This is a footnote fragment ending", "footnote-1")
    second_text = " with more text" if break_order == "no_sentence_end" else " with more text."
    second = _paragraph(second_text, "footnote-2")
    first.layout_label = second.layout_label = "text"

    def set_characters(paragraph: SimpleNamespace, start: int) -> None:
        paragraph.pdf_paragraph_composition = [
            SimpleNamespace(
                pdf_line=SimpleNamespace(
                    pdf_character=[
                        SimpleNamespace(render_order=order)
                        for order in range(start, start + len(paragraph.unicode))
                    ]
                )
            )
        ]

    set_characters(first, 10)
    second_start = 10 + len(first.unicode) + (1 if break_order == "gap" else 0)
    set_characters(second, second_start)
    if break_order == "unknown_composition":
        second.pdf_paragraph_composition = [SimpleNamespace(unrecognized=True)]
    pages = [SimpleNamespace(page_number=1, pdf_paragraph=[first, second])]
    if break_order == "different_page":
        pages = [
            SimpleNamespace(page_number=1, pdf_paragraph=[first]),
            SimpleNamespace(page_number=2, pdf_paragraph=[second]),
        ]

    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(SimpleNamespace(page=pages))

    assert all("preserve_reason" not in recorder.get(p) for p in (first, second))


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


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("lang_in", "Vietnamese"),
        ("lang_out", "English"),
        ("translation_policy_version", "another-provider-v2"),
    ],
)
def test_checkpoint_identity_includes_language_pair_and_policy(field, changed_value) -> None:
    paragraph = _paragraph("A stable source paragraph.")
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

    identity = {
        "source_pdf_sha256": recorder.source_sha256,
        "page_number": record["page_number"],
        "ordinal": record["ordinal"],
        "page_ordinal": record["page_ordinal"],
        "source_sha256": record["source_sha256"],
        "glossary_sha256": recorder.glossary_sha256,
        "pdf2zh_version": engine_runner.PDF2ZH_VERSION,
        "babeldoc_version": engine_runner.BABELDOC_VERSION,
        "lang_in": engine_runner.TRANSLATION_LANG_IN,
        "lang_out": engine_runner.TRANSLATION_LANG_OUT,
        "translation_policy_version": engine_runner.TRANSLATION_POLICY_VERSION,
    }

    assert identity["lang_in"] == "English"
    assert identity["lang_out"] == "Vietnamese"
    assert record["base_key"] == engine_runner._checkpoint_base_key(identity)
    changed_identity = {**identity, field: changed_value}
    assert engine_runner._checkpoint_base_key(changed_identity) != record["base_key"]


def test_translation_engine_request_requires_fixed_english_vietnamese_pair() -> None:
    request = {
        "protocol_version": engine_runner.PROTOCOL_VERSION,
        "lang_in": engine_runner.TRANSLATION_LANG_IN,
        "lang_out": engine_runner.TRANSLATION_LANG_OUT,
        "input_pdf": "/tmp/source.pdf",
        "output_dir": "/tmp/output",
        "working_dir": "/tmp/work",
        "layout_model": "/tmp/layout.onnx",
        "source_pdf_sha256": "a" * 64,
    }

    assert engine_runner._validate_request(request) is request
    with pytest.raises(engine_runner.SafeEngineError) as error:
        engine_runner._validate_request({**request, "lang_out": "English"})

    assert error.value.code == "INVALID_ENGINE_REQUEST"


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


def test_checkpoint_hooks_reconcile_fake_babeldoc_lifecycle_without_quote_leaks(
    monkeypatch,
) -> None:
    """Exercise the pinned hook seams without importing BabelDOC or calling a provider."""
    helper_name = "babeldoc.format.pdf.document_il.utils.paragraph_helper"
    helper = types.ModuleType(helper_name)
    helper.is_placeholder_only_paragraph = lambda _paragraph: False
    helper.is_pure_numeric_paragraph = lambda paragraph: (
        bool(paragraph.unicode.strip())
        and all(char.isdigit() or char in " .,%+-" for char in paragraph.unicode.strip())
    )
    monkeypatch.setitem(sys.modules, helper_name, helper)

    prose = _paragraph("The proposed method reduces attention cost.", "translated")
    prose.pdf_paragraph_composition = [SimpleNamespace(pdf_line=True)]
    equation = _paragraph("E = mc^2", "equation")
    equation.layout_label = "equation"
    equation.pdf_paragraph_composition = [SimpleNamespace(pdf_formula=True)]
    numeric = _paragraph("93.5%", "numeric")
    numeric.pdf_paragraph_composition = [SimpleNamespace(pdf_line=True)]
    declined = _paragraph("A prose unit declined during preprocessing.", "declined")
    declined.pdf_paragraph_composition = []
    uncheckpointed = _paragraph(
        "A selected prose unit without a validated result.", "uncheckpointed"
    )
    uncheckpointed.pdf_paragraph_composition = [SimpleNamespace(pdf_line=True)]
    split_footnote = [
        _paragraph("5We used values of 2.8, 3.", "footnote-1"),
        _paragraph("7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100", "footnote-2"),
        _paragraph(", respectively.", "footnote-3"),
    ]
    split_footnote[0].layout_label = "fallback_line"
    split_footnote[1].layout_label = "abandon"
    split_footnote[2].layout_label = "fallback_line"
    next_order = 100
    for paragraph in split_footnote:
        characters = [
            SimpleNamespace(render_order=order)
            for order in range(next_order, next_order + len(paragraph.unicode))
        ]
        paragraph.pdf_paragraph_composition = [
            SimpleNamespace(pdf_line=SimpleNamespace(pdf_character=characters))
        ]
        next_order += len(characters)
    paragraphs = [prose, equation, numeric, declined, uncheckpointed, *split_footnote]
    docs = SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=paragraphs)])
    translations = {id(prose): "Phương pháp đề xuất giảm chi phí chú ý."}
    events: list[dict] = []
    monkeypatch.setattr(engine_runner, "_emit", events.append)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )

    class FakeILTranslator:
        def __init__(self, translation_config, llm_translator):
            self.translation_config = translation_config
            self.llm_translator = llm_translator
            self.inputs: dict[int, str | None] = {}

        def translate(self, docs):
            return docs

        def get_translate_input(self, paragraph, page_font_map, disable_rich_text_translate):
            del page_font_map, disable_rich_text_translate
            if paragraph is equation or paragraph is numeric or paragraph is declined:
                return None
            return paragraph.unicode

        def pre_translate_paragraph(self, paragraph, tracker, page_font_map, xobj_font_map):
            del xobj_font_map
            translate_input = self.get_translate_input(paragraph, page_font_map, False)
            self.inputs[id(paragraph)] = translate_input
            return translate_input, tracker

        def post_translate_paragraph(self, paragraph, tracker, translate_input, translated_text):
            del paragraph, tracker, translate_input
            return translated_text

        def translate_paragraph(self, paragraph, page):
            del page
            return self.llm_translator.translate_paragraph([paragraph])

    class FakeLLMTranslator:
        def __init__(self, translation_config, il_translator):
            self.translation_config = translation_config
            self.il_translator = il_translator

        def translate(self, docs):
            for page in docs.page:
                for paragraph in page.pdf_paragraph:
                    translate_input, tracker = self.il_translator.pre_translate_paragraph(
                        paragraph, object(), None, None
                    )
                    if translate_input is None or paragraph is uncheckpointed:
                        continue
                    self.il_translator.translate_paragraph(paragraph, page)
            return docs

        def process_cross_page_paragraph(
            self, docs, executor, pbar, tracker, executor2, translated_ids
        ):
            del docs, executor, pbar, tracker, executor2, translated_ids
            return None

        def translate_paragraph(self, batch_paragraph):
            results = []
            for paragraph in batch_paragraph:
                translate_input = self.il_translator.inputs[id(paragraph)]
                translated_text = translations[id(paragraph)]
                results.append(
                    self.il_translator.post_translate_paragraph(
                        paragraph, object(), translate_input, translated_text
                    )
                )
            return results

    config = SimpleNamespace(
        min_text_length=5,
        glossaries=[],
    )
    il_translator = FakeILTranslator(config, None)
    llm_translator = FakeLLMTranslator(config, il_translator)
    il_translator.llm_translator = llm_translator

    with engine_runner._checkpoint_hooks(FakeILTranslator, FakeLLMTranslator, recorder):
        llm_translator.translate(docs)
        counts = recorder.finish()
        recorder.emit_summary()

    assert counts == {"total": 8, "completed": 1, "skipped": 5, "failed": 2}
    assert counts["total"] == counts["completed"] + counts["skipped"] + counts["failed"]
    assert recorder.get(prose)["status"] == "completed"
    assert recorder.get(equation)["skip_reason"] == "protected_scientific_content"
    assert recorder.get(numeric)["skip_reason"] == "numeric_or_symbol_only"
    assert recorder.get(declined)["failure_reason"] == "no_composition"
    assert recorder.get(declined)["preprocess_decline_reason"] == "no_composition"
    assert recorder.get(uncheckpointed)["failure_reason"] == "missing_validated_checkpoint"
    assert all(recorder.get(p)["status"] == "skipped" for p in split_footnote)
    assert "5We used values of 2.8, 3." in [p.unicode for p in split_footnote]
    assert [event["type"] for event in events] == ["checkpoint", "segment_summary"]

    checkpoint, summary = events
    assert checkpoint["segment"]["source_quote"] == prose.unicode
    assert checkpoint["segment"]["translated_text"] == translations[id(prose)]
    assert summary["skip_reasons"] == {
        "protected_scientific_content": 1,
        "numeric_or_symbol_only": 1,
        "preserved_split_footnote_layout_content": 3,
    }
    assert summary["failure_causes"] == {
        "no_composition": 1,
        "missing_validated_checkpoint": 1,
    }
    summary_wire = json.dumps(summary)
    protocol_wire = json.dumps(events)
    for paragraph in paragraphs:
        assert paragraph.unicode not in summary_wire
    assert all("source_quote" not in unit for unit in summary["failure_units"])
    assert prose.unicode in protocol_wire  # only the validated checkpoint carries its source span
    for paragraph in paragraphs[1:]:
        assert paragraph.unicode not in protocol_wire


def test_checkpoint_hooks_keep_cross_page_paragraphs_in_page_local_batches(monkeypatch) -> None:
    """The pinned engine's cross-page batch is bypassed without losing either unit."""
    helper_name = "babeldoc.format.pdf.document_il.utils.paragraph_helper"
    helper = types.ModuleType(helper_name)
    helper.is_placeholder_only_paragraph = lambda _paragraph: False
    helper.is_pure_numeric_paragraph = lambda _paragraph: False
    monkeypatch.setitem(sys.modules, helper_name, helper)

    paragraphs = [
        _paragraph("The first paragraph appears at the end of page one.", "page-one"),
        _paragraph("The second paragraph appears at the start of page two.", "page-two"),
    ]
    for paragraph in paragraphs:
        paragraph.pdf_paragraph_composition = [SimpleNamespace(pdf_line=True)]
    pages = [
        SimpleNamespace(page_number=1, pdf_paragraph=[paragraphs[0]]),
        SimpleNamespace(page_number=2, pdf_paragraph=[paragraphs[1]]),
    ]
    docs = SimpleNamespace(page=pages)
    events: list[dict] = []
    batches: list[list[int]] = []
    monkeypatch.setattr(engine_runner, "_emit", events.append)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="b" * 64,
        glossary=[],
        checkpoint_results=[],
    )

    class FakeILTranslator:
        def __init__(self, translation_config, llm_translator):
            self.translation_config = translation_config
            self.llm_translator = llm_translator
            self.inputs: dict[int, str] = {}

        def translate(self, docs):
            return docs

        def get_translate_input(self, paragraph, page_font_map, disable_rich_text_translate):
            del page_font_map, disable_rich_text_translate
            return paragraph.unicode

        def pre_translate_paragraph(self, paragraph, tracker, page_font_map, xobj_font_map):
            del xobj_font_map
            value = self.get_translate_input(paragraph, page_font_map, False)
            self.inputs[id(paragraph)] = value
            return value, tracker

        def post_translate_paragraph(self, paragraph, tracker, translate_input, translated_text):
            del paragraph, tracker, translate_input
            return translated_text

        def translate_paragraph(self, paragraph, page):
            del page
            return self.llm_translator.translate_paragraph([paragraph])

    class FakeLLMTranslator:
        def __init__(self, translation_config, il_translator):
            self.translation_config = translation_config
            self.il_translator = il_translator
            self.cross_page_calls = 0

        def translate(self, docs):
            self.process_cross_page_paragraph(docs, object(), None, object(), object(), set())
            for page in docs.page:
                for paragraph in page.pdf_paragraph:
                    translate_input, tracker = self.il_translator.pre_translate_paragraph(
                        paragraph, object(), None, None
                    )
                    if translate_input is not None:
                        self.il_translator.translate_paragraph(paragraph, page)
            return docs

        def process_cross_page_paragraph(
            self, docs, executor, pbar, tracker, executor2, translated_ids
        ):
            del docs, executor, pbar, tracker, executor2, translated_ids
            self.cross_page_calls += 1

        def translate_paragraph(self, batch_paragraph):
            batches.append([id(paragraph) for paragraph in batch_paragraph])
            return [
                self.il_translator.post_translate_paragraph(
                    paragraph,
                    object(),
                    self.il_translator.inputs[id(paragraph)],
                    f"Bản dịch {index}",
                )
                for index, paragraph in enumerate(batch_paragraph, start=1)
            ]

    config = SimpleNamespace(min_text_length=5, glossaries=[])
    il_translator = FakeILTranslator(config, None)
    llm_translator = FakeLLMTranslator(config, il_translator)
    il_translator.llm_translator = llm_translator
    original_cross_page = llm_translator.process_cross_page_paragraph

    with engine_runner._checkpoint_hooks(FakeILTranslator, FakeLLMTranslator, recorder):
        assert llm_translator.process_cross_page_paragraph != original_cross_page
        llm_translator.translate(docs)
        counts = recorder.finish()

    checkpoints = [event["segment"] for event in events if event["type"] == "checkpoint"]
    assert llm_translator.cross_page_calls == 0
    assert batches == [[id(paragraphs[0])], [id(paragraphs[1])]]
    assert counts == {"total": 2, "completed": 2, "skipped": 0, "failed": 0}
    assert [(item["page_number"], item["page_ordinal"]) for item in checkpoints] == [(1, 0), (2, 0)]
    assert [item["source_quote"] for item in checkpoints] == [p.unicode for p in paragraphs]
    assert len({item["source_sha256"] for item in checkpoints}) == 2
    assert llm_translator.process_cross_page_paragraph == original_cross_page

    with pytest.raises(RuntimeError, match="test exception"):
        with engine_runner._checkpoint_hooks(FakeILTranslator, FakeLLMTranslator, recorder):
            assert llm_translator.process_cross_page_paragraph != original_cross_page
            raise RuntimeError("test exception")

    assert llm_translator.process_cross_page_paragraph == original_cross_page


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


def test_checkpoint_recorder_keeps_unclassified_vertical_text_unresolved() -> None:
    metadata = _paragraph("arXiv:1901.02860v3 [cs.LG] 2 Jun 2019")
    prose = _paragraph("A meaningful paragraph set vertically in the page margin.")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[metadata, prose])])
    )
    recorder.get(metadata)["preprocess_decline_reason"] = "vertical_paragraph"
    recorder.get(prose)["preprocess_decline_reason"] = "no_composition"

    counts = recorder.finish()

    assert counts == {"total": 2, "completed": 0, "skipped": 0, "failed": 2}
    assert recorder.skip_reasons() == {}
    assert recorder.failure_causes() == {
        "vertical_paragraph": 1,
        "no_composition": 1,
    }


def test_checkpoint_recorder_preserves_vertical_text_only_on_reviewed_attention_pages() -> None:
    source_hash = "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697"
    embedded_pages = [
        _paragraph(f"Embedded attention visualization labels on page {page_number}.")
        for page_number in (12, 13, 14)
    ]
    for paragraph in embedded_pages:
        paragraph.vertical = True
    caption = _paragraph("Figure 3: Attention visualization.")
    caption.vertical = True
    caption.layout_label = "caption"
    figure_caption = _paragraph("Figure 4: A second caption.")
    figure_caption.vertical = True
    figure_caption.layout_label = "figure_caption"
    title = _paragraph("Attention Visualizations")
    title.vertical = True
    title.layout_label = "title"
    outside = _paragraph("Vertical prose outside the reviewed pages.")
    outside.vertical = True
    body = _paragraph("Nonvertical prose on a reviewed page remains translatable.")

    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256=source_hash,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(
            page=[
                SimpleNamespace(
                    page_number=12,
                    pdf_paragraph=[embedded_pages[0], caption, figure_caption, title, body],
                ),
                SimpleNamespace(page_number=13, pdf_paragraph=[embedded_pages[1]]),
                SimpleNamespace(page_number=14, pdf_paragraph=[embedded_pages[2]]),
                SimpleNamespace(page_number=11, pdf_paragraph=[outside]),
            ]
        )
    )

    assert all(
        recorder.get(paragraph)["preserve_reason"] == "preserved_embedded_figure_text"
        for paragraph in embedded_pages
    )
    assert "preserve_reason" not in recorder.get(caption)
    assert "preserve_reason" not in recorder.get(figure_caption)
    assert "preserve_reason" not in recorder.get(title)
    assert "preserve_reason" not in recorder.get(outside)
    assert "preserve_reason" not in recorder.get(body)

    counts = recorder.finish()

    assert counts == {"total": 8, "completed": 0, "skipped": 3, "failed": 5}
    assert recorder.skip_reasons() == {"preserved_embedded_figure_text": 3}


@pytest.mark.parametrize(
    ("source_hash", "page_number"),
    [
        ("a" * 64, 12),
        ("bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697", 11),
        ("bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697", 15),
    ],
)
def test_checkpoint_recorder_does_not_preserve_unreviewed_vertical_text(
    source_hash: str, page_number: int
) -> None:
    paragraph = _paragraph("Meaningful vertical prose that must remain unresolved.")
    paragraph.vertical = True
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256=source_hash,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=page_number, pdf_paragraph=[paragraph])])
    )

    assert "preserve_reason" not in recorder.get(paragraph)
    assert recorder.finish() == {"total": 1, "completed": 0, "skipped": 0, "failed": 1}
    assert recorder.failure_causes() == {"preprocessing_not_selected": 1}


@pytest.mark.parametrize(
    ("page_number", "ordinal", "quote", "expected", "layout_label"),
    [
        (5, 75, "O(1)", "preserved_scientific_table_content", "fallback_line"),
        (7, 117, "BLEU", "preserved_scientific_table_content", "fallback_line"),
        (8, 189, "drop", "preserved_scientific_table_content", "fallback_line"),
        (12, 375, "r5", "preserved_embedded_figure_text", "fallback_line"),
        (
            3,
            39,
            "Scaled Dot-Product Attention",
            "preserved_figure_or_citation_metadata",
            "abandon",
        ),
        (
            9,
            314,
            "Vinyals & Kaiser el al. (2014) [37]",
            "preserved_figure_or_citation_metadata",
            "fallback_line",
        ),
        (
            9,
            332,
            "Huang & Harper (2009) [14]",
            "preserved_figure_or_citation_metadata",
            "fallback_line",
        ),
    ],
)
def test_reviewed_attention_preservation_matches_only_exact_source_units(
    page_number: int, ordinal: int, quote: str, expected: str, layout_label: str
) -> None:
    assert (
        engine_runner._reviewed_attention_preserve_reason(
            source_sha256="bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
            page_number=page_number,
            ordinal=ordinal,
            source_quote=quote,
            is_vertical=False,
            layout_label=layout_label,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("source_hash", "page_number", "ordinal", "quote", "is_vertical", "label"),
    [
        ("a" * 64, 5, 75, "O(1)", False, "fallback_line"),
        (
            "bdfaa68d8984f0dc02beaca527b76f207d99b666d1d1da728ee0728182df697",
            9,
            350,
            "incr",
            False,
            "fallback_line",
        ),
        (
            "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
            5,
            75,
            "O(2)",
            False,
            "fallback_line",
        ),
        (
            "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
            12,
            448,
            "Figure 3 caption",
            True,
            "caption",
        ),
        (
            "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
            9,
            314,
            "Vinyals & Kaiser el al. (2014) [37]",
            False,
            "text",
        ),
        (
            "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697",
            9,
            314,
            "Vinyals & Kaiser et al. (2014) [37]",
            False,
            "fallback_line",
        ),
    ],
)
def test_reviewed_attention_preservation_rejects_near_matches(
    source_hash: str,
    page_number: int,
    ordinal: int,
    quote: str,
    is_vertical: bool,
    label: str,
) -> None:
    assert (
        engine_runner._reviewed_attention_preserve_reason(
            source_sha256=source_hash,
            page_number=page_number,
            ordinal=ordinal,
            source_quote=quote,
            is_vertical=is_vertical,
            layout_label=label,
        )
        is None
    )


def test_checkpoint_recorder_terminally_skips_formula_and_short_engine_declines() -> None:
    formula = _paragraph("E = mc squared")
    short = _paragraph("abc")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[formula, short])])
    )
    recorder.get(formula)["preprocess_decline_reason"] = "formula_only"
    recorder.get(short)["preprocess_decline_reason"] = "below_minimum_length"

    counts = recorder.finish()
    summary = _safe_event(
        {
            "type": "segment_summary",
            **counts,
            "skip_reasons": recorder.skip_reasons(),
        }
    )

    assert counts == {"total": 2, "completed": 0, "skipped": 2, "failed": 0}
    assert summary["skip_reasons"] == {
        "protected_scientific_content": 1,
        "below_engine_minimum": 1,
    }


def test_checkpoint_recorder_skips_only_recognized_rotated_arxiv_stamp() -> None:
    stamp = _paragraph("2023 ug CL] 2 A 7 [cs. 03762v :1706. iv arX")
    stamp.vertical = True
    stamp.layout_label = "abandon"
    prose = _paragraph("A meaningful paragraph set vertically in the page margin.")
    prose.vertical = True
    prose.layout_label = "abandon"
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[stamp, prose])])
    )
    recorder.get(prose)["preprocess_decline_reason"] = "vertical_paragraph"

    counts = recorder.finish()

    assert counts == {"total": 2, "completed": 0, "skipped": 1, "failed": 1}
    assert recorder.skip_reasons() == {"arxiv_version_stamp": 1}
    assert recorder.failure_causes() == {"vertical_paragraph": 1}


@pytest.mark.parametrize(("vertical", "layout_label"), [(False, "abandon"), (True, "text")])
def test_checkpoint_recorder_requires_vertical_abandon_arxiv_stamp(
    vertical: bool, layout_label: str
) -> None:
    paragraph = _paragraph("2023 ug CL] 2 A 7 [cs. 03762v :1706. iv arX")
    paragraph.vertical = vertical
    paragraph.layout_label = layout_label
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
    assert recorder.skip_reasons() == {}


def test_checkpoint_recorder_preserves_only_known_metadata_and_numeric_content() -> None:
    numeric = _paragraph("1.0 · 1020")
    reference = _paragraph("Vinyals & Kaiser el al. (2014) [37]")
    author_contact = _paragraph("Ashish Vaswani∗ Google Brain avaswani@google.com")
    short_label = _paragraph("Scaled Dot-Product Attention")
    required = _paragraph("A required prose paragraph missing its validated translation.")
    reference.layout_label = "fallback_line"
    short_label.layout_label = "abandon"
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(
            page=[
                SimpleNamespace(
                    page_number=1,
                    pdf_paragraph=[numeric, reference, author_contact, short_label, required],
                )
            ]
        )
    )
    recorder.get(numeric)["preprocess_decline_reason"] = "pure_numeric"
    recorder.note_preprocessed(numeric, None)
    recorder.note_preprocessed(reference, None)
    recorder.note_preprocessed(author_contact, author_contact.unicode)
    recorder.complete(recorder.get(author_contact), author_contact.unicode)
    recorder.note_preprocessed(short_label, short_label.unicode)
    recorder.complete(recorder.get(short_label), short_label.unicode)
    recorder.note_preprocessed(required, required.unicode)

    assert recorder.finish() == {"total": 5, "completed": 0, "skipped": 2, "failed": 3}
    assert recorder.skip_reasons() == {
        "numeric_or_symbol_only": 1,
        "author_contact_metadata": 1,
    }
    assert recorder.failure_causes() == {
        "preprocessing_not_selected": 1,
        "unchanged_prose": 1,
        "missing_validated_checkpoint": 1,
    }


def test_checkpoint_recorder_allows_short_unselected_fallback_line_as_visible_omission() -> None:
    paragraph = _paragraph("from right to left")
    paragraph.layout_label = "fallback_line"
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )

    assert recorder.finish() == {"total": 1, "completed": 0, "skipped": 1, "failed": 0}
    assert recorder.skip_reasons() == {"untranslated_short_fallback": 1}


def test_checkpoint_recorder_skips_exact_reviewed_attention_metadata() -> None:
    source_hash = "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697"
    paragraphs = [_paragraph(str(index)) for index in range(39)]
    figure_heading = _paragraph("Scaled Dot-Product Attention")
    figure_heading.layout_label = "abandon"
    paragraphs.append(figure_heading)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256=source_hash,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(SimpleNamespace(page=[SimpleNamespace(page_number=3, pdf_paragraph=paragraphs)]))

    assert recorder.get(figure_heading)["preserve_reason"] == (
        "preserved_figure_or_citation_metadata"
    )
    recorder.note_preprocessed(figure_heading, None)

    assert recorder.finish() == {"total": 40, "completed": 0, "skipped": 40, "failed": 0}
    assert recorder.get(figure_heading)["skip_reason"] == ("preserved_figure_or_citation_metadata")
    assert recorder.failure_causes() == {}


def test_checkpoint_recorder_keeps_short_unclassified_text_unresolved() -> None:
    paragraph = _paragraph("abc")
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )

    assert recorder.finish() == {"total": 1, "completed": 0, "skipped": 0, "failed": 1}
    assert recorder.skip_reasons() == {}
    assert recorder.failure_causes() == {"preprocessing_not_selected": 1}


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


def test_checkpoint_recorder_skips_only_numeric_abandoned_layout_fragment() -> None:
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
    assert recorder.skip_reasons() == {"numeric_or_symbol_only": 1}


def test_checkpoint_recorder_rejects_unchanged_english_sentence_in_translated_prose(
    monkeypatch,
) -> None:
    output = io.StringIO()
    monkeypatch.setattr(engine_runner, "PROTOCOL_STDOUT", output)
    source = "The proposed method reduces attention cost. Wepresentthese results in Table 3."
    paragraph = _paragraph(source)
    paragraph.layout_label = "plain text"
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=9, pdf_paragraph=[paragraph])])
    )

    record = recorder.get(paragraph)
    assert record is not None
    recorder.note_preprocessed(paragraph, paragraph.unicode)
    recorder.complete(
        record, "Phương pháp đề xuất giảm chi phí chú ý. Wepresentthese results in Table 3."
    )

    assert recorder.finish() == {"total": 1, "completed": 0, "skipped": 0, "failed": 1}
    assert record["status"] == "unchanged_prose"
    assert record["failure_reason"] == "unchanged_prose"
    assert output.getvalue() == ""


def test_unchanged_sentence_guard_does_not_treat_citation_title_as_prose() -> None:
    source = "We evaluate our model against a baseline. Attention Is All You Need, 2017."
    translated = "Chúng tôi đánh giá mô hình theo đường cơ sở. Attention Is All You Need, 2017."

    assert not engine_runner._contains_unchanged_english_sentence(source, translated)
    assert engine_runner._english_function_word_count("Attention") == 0


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
    assert request_texts[1].startswith("instructions\n\n## Here is the input:\n\n")
    assert "previous output left these prose items in English" not in request_texts[1]
    assert "first_occurrence_glossary_terms" in request_texts[1]
    assert translated == "Mô hình attention (chú ý) {v1} cải thiện kết quả trong thực tiễn."
    assert record["status"] == "completed"
    assert recorder.finish()["completed"] == 1


def test_llm_adapter_retries_cached_prose_with_an_unchanged_sentence() -> None:
    source = "The proposed method reduces attention cost. Wepresentthese results clearly."
    paragraph = _paragraph(source)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=9, pdf_paragraph=[paragraph])])
    )
    record = recorder.get(paragraph)
    assert record is not None
    prefix = "instructions\n\n## Here is the input:\n\n"
    item = {"id": 0, "input": source, "layout_label": "text"}
    checkpoint_key = recorder.key_for(record, engine_runner._context_hash(prefix, "", source))
    recorder.checkpoints[checkpoint_key] = (
        "Phương pháp đề xuất giảm chi phí chú ý. Wepresentthese results clearly."
    )
    requests: list[str] = []
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )

    def translate(text: str) -> str:
        requests.append(text)
        return json.dumps(
            [
                {
                    "id": 0,
                    "output": "Phương pháp đề xuất giảm chi phí chú ý và trình bày rõ các kết quả.",
                }
            ],
            ensure_ascii=False,
        )

    translator._request = translate
    with recorder.batch_scope():
        recorder.note_preprocessed(paragraph, source)
        translated = json.loads(translator.llm_translate(prefix + json.dumps([item])))[0]["output"]
    recorder.complete(record, translated)

    assert len(requests) == 1
    assert "Wepresentthese" in requests[0]
    assert translated == "Phương pháp đề xuất giảm chi phí chú ý và trình bày rõ các kết quả."
    assert record["status"] == "completed"
    assert recorder.finish()["completed"] == 1


def test_llm_adapter_locks_scientific_identifiers_and_numbers_for_translation() -> None:
    source = "PosUnk and Deep-Att reached 2.8, 3.1 BLEU on K80."
    response = (
        "[[MYRA_KEEP_0]] và [[MYRA_KEEP_1]] đạt [[MYRA_KEEP_2]], "
        "[[MYRA_KEEP_3]] [[MYRA_KEEP_4]] trên [[MYRA_KEEP_5]]."
    )
    translated, request_texts, _, _ = _translate_batch_with_responses(source, [response])

    assert translated == "PosUnk và Deep-Att đạt 2.8, 3.1 BLEU trên K80."
    assert len(request_texts) == 1
    for source_token in ("PosUnk", "Deep-Att", "2.8", "3.1", "BLEU", "K80"):
        assert source_token not in request_texts[0]


def test_llm_adapter_rejects_missing_or_reordered_scientific_tokens() -> None:
    source = "The model scored 2.8 then 3.1 BLEU."
    response = "Mô hình đạt [[MYRA_KEEP_1]] rồi [[MYRA_KEEP_0]] [[MYRA_KEEP_2]]."

    with pytest.raises(engine_runner.SafeEngineError, match="PROVIDER_SCIENTIFIC_TOKEN_MISMATCH"):
        _translate_batch_with_responses(source, [response])


def test_scientific_token_locking_preserves_repeated_values_and_existing_markers() -> None:
    source = "Result 2.8, then 2.8 again; cite {v1} and 2.8."
    protected, replacements = engine_runner._lock_scientific_tokens(source)

    assert tuple(replacements.values()) == ("2.8", "2.8", "2.8")
    assert "{v1}" in protected
    assert (
        engine_runner._restore_scientific_tokens(protected, replacements, source=source) == source
    )


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


def test_llm_single_paragraph_fallback_reuses_existing_prompt_checkpoint(monkeypatch) -> None:
    source = "The output contains {v1} and {v2} values."
    paragraph = _paragraph(source)
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
    recorder.note_preprocessed(paragraph, source)
    fallback_prompt = "BabelDOC's single-paragraph fallback prompt"
    key = recorder.key_for(record, hashlib.sha256(fallback_prompt.encode()).hexdigest())
    translated_checkpoint = "Đầu ra chứa {v1} và {v2} giá trị."
    recorder.checkpoints[key] = translated_checkpoint
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )
    monkeypatch.setattr(
        translator,
        "_request",
        lambda _text: pytest.fail("valid checkpoint should avoid a provider call"),
    )

    with recorder.paragraph_scope(paragraph):
        translated = translator.llm_translate(fallback_prompt)

    assert translated == translated_checkpoint


def test_llm_single_paragraph_fallback_retries_stale_unchanged_sentence_checkpoint(
    monkeypatch,
) -> None:
    source = "The proposed method reduces attention cost. Wepresentthese results clearly."
    paragraph = _paragraph(source)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=9, pdf_paragraph=[paragraph])])
    )
    record = recorder.get(paragraph)
    assert record is not None
    recorder.note_preprocessed(paragraph, source)
    fallback_prompt = "The pinned single-paragraph provider request"
    key = recorder.key_for(record, hashlib.sha256(fallback_prompt.encode()).hexdigest())
    recorder.checkpoints[key] = (
        "Phương pháp đề xuất giảm chi phí chú ý. Wepresentthese results clearly."
    )
    requests: list[str] = []
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )

    def translate(text: str) -> str:
        requests.append(text)
        return "Phương pháp đề xuất giảm chi phí chú ý và trình bày rõ các kết quả."

    monkeypatch.setattr(translator, "_request", translate)
    with recorder.paragraph_scope(paragraph):
        translated = translator.llm_translate(fallback_prompt)

    assert requests == [fallback_prompt]
    assert translated == "Phương pháp đề xuất giảm chi phí chú ý và trình bày rõ các kết quả."


def test_direct_translation_retries_stale_unchanged_sentence_checkpoint(monkeypatch) -> None:
    source = "The proposed method reduces attention cost. Wepresentthese results clearly."
    paragraph = _paragraph(source)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=9, pdf_paragraph=[paragraph])])
    )
    record = recorder.get(paragraph)
    assert record is not None
    key = recorder.key_for(record, hashlib.sha256(source.encode()).hexdigest())
    recorder.checkpoints[key] = (
        "Phương pháp đề xuất giảm chi phí chú ý. Wepresentthese results clearly."
    )
    requests: list[str] = []
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )

    def translate(text: str) -> str:
        requests.append(text)
        return "Phương pháp đề xuất giảm chi phí chú ý và trình bày rõ các kết quả."

    monkeypatch.setattr(translator, "_request", translate)
    with recorder.paragraph_scope(paragraph):
        translated = translator.translate(source)

    assert requests == [source]
    assert translated == "Phương pháp đề xuất giảm chi phí chú ý và trình bày rõ các kết quả."


def test_llm_single_paragraph_fallback_keeps_provider_prompt_and_classifies_marker_loss(
    monkeypatch,
) -> None:
    source = "The result contains {v1} and {v2} values."
    paragraph = _paragraph(source)
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
    recorder.note_preprocessed(paragraph, source)
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )
    fallback_prompt = "The pinned BabelDOC single-paragraph request"
    requests: list[str] = []

    def fake_request(text: str) -> str:
        requests.append(text)
        return "Kết quả không còn dấu phân cách."

    monkeypatch.setattr(translator, "_request", fake_request)
    with (
        recorder.paragraph_scope(paragraph),
        pytest.raises(engine_runner.SafeEngineError, match="PROVIDER_MARKER_MISMATCH"),
    ):
        translator.llm_translate(fallback_prompt)

    assert requests == [fallback_prompt]
    assert translator.failure_code == "PROVIDER_MARKER_MISMATCH"
    assert record["status"] == "invalid"
    assert record["failure_reason"] == "protected_placeholder_mismatch"


def test_llm_batch_marker_mismatch_identifies_only_the_failed_unit(monkeypatch) -> None:
    source = "The result contains {v1} and remains protected."
    paragraph = _paragraph(source)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=4, pdf_paragraph=[paragraph])])
    )
    record = recorder.get(paragraph)
    assert record is not None
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )
    prefix = "instructions\n\n## Here is the input:\n\n"
    item = {"id": 0, "input": source, "layout_label": "text"}
    monkeypatch.setattr(
        translator,
        "_request",
        lambda _text: json.dumps([{"id": 0, "output": "Kết quả được bảo vệ."}]),
    )

    with (
        recorder.batch_scope(),
        recorder.paragraph_scope(paragraph),
        pytest.raises(engine_runner.SafeEngineError, match="PROVIDER_MARKER_MISMATCH"),
    ):
        recorder.note_preprocessed(paragraph, source)
        translator.llm_translate(prefix + json.dumps([item]))

    assert record["status"] == "invalid"
    assert record["failure_reason"] == "protected_placeholder_mismatch"
    assert recorder.failure_units() == [
        {
            "page_number": 4,
            "ordinal": 0,
            "status": "invalid",
            "failure_reason": "protected_placeholder_mismatch",
            "source_chars": len(source),
            "layout_label": "text",
        }
    ]
    summary = {
        "type": "segment_summary",
        **recorder.finish(),
        "failure_reasons": recorder.failure_reasons(),
        "failure_causes": recorder.failure_causes(),
        "skip_reasons": recorder.skip_reasons(),
        "failure_units": recorder.failure_units(),
    }
    parsed_summary = _safe_event(summary)
    assert parsed_summary["failed"] == 1
    assert parsed_summary["failure_causes"] == {"protected_placeholder_mismatch": 1}


def test_llm_batch_malformed_provider_json_sets_safe_failure_code(monkeypatch) -> None:
    source = "The formula is {v1} and the result is {v2}."
    paragraph = _paragraph(source)
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
    recorder.note_preprocessed(paragraph, source)
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )
    attempts = 0

    def invalid_json(_text: str) -> str:
        nonlocal attempts
        attempts += 1
        return "not valid JSON"

    monkeypatch.setattr(translator, "_request", invalid_json)
    prefix = "instructions\n\n## Here is the input:\n\n"
    item = {"id": 0, "input": source, "layout_label": "text"}

    with (
        recorder.batch_scope(),
        recorder.paragraph_scope(paragraph),
        pytest.raises(engine_runner.SafeEngineError, match="PROVIDER_INVALID_JSON"),
    ):
        recorder.note_preprocessed(paragraph, source)
        translator.llm_translate(prefix + json.dumps([item]))

    assert translator.failure_code == "PROVIDER_INVALID_JSON"
    assert attempts == 3
    assert record["status"] == "pending"


def test_llm_adapter_retries_transient_malformed_single_item_json(monkeypatch) -> None:
    source = "The model improves the result for this task."
    paragraph = _paragraph(source)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )
    responses = iter(
        [
            "not valid JSON",
            json.dumps([{"id": 0, "output": "Mô hình cải thiện kết quả cho nhiệm vụ này."}]),
        ]
    )
    attempts = 0

    def request(_text: str) -> str:
        nonlocal attempts
        attempts += 1
        return next(responses)

    monkeypatch.setattr(translator, "_request", request)
    prefix = "instructions\n\n## Here is the input:\n\n"
    item = {"id": 0, "input": source, "layout_label": "text"}

    with recorder.batch_scope():
        recorder.note_preprocessed(paragraph, source)
        result = json.loads(translator.llm_translate(prefix + json.dumps([item])))

    assert result == [{"id": 0, "output": "Mô hình cải thiện kết quả cho nhiệm vụ này."}]
    assert attempts == 2


def test_llm_adapter_retries_single_item_with_invalid_result_schema(monkeypatch) -> None:
    source = "The model improves the result for this task."
    paragraph = _paragraph(source)
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(
        SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=[paragraph])])
    )
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )
    responses = iter(
        [
            json.dumps([{}]),
            json.dumps([{"id": 0, "output": "Mô hình cải thiện kết quả cho nhiệm vụ này."}]),
        ]
    )
    attempts = 0

    def request(_text: str) -> str:
        nonlocal attempts
        attempts += 1
        return next(responses)

    monkeypatch.setattr(translator, "_request", request)
    prefix = "instructions\n\n## Here is the input:\n\n"
    item = {"id": 0, "input": source, "layout_label": "text"}

    with recorder.batch_scope():
        recorder.note_preprocessed(paragraph, source)
        result = json.loads(translator.llm_translate(prefix + json.dumps([item])))

    assert result == [{"id": 0, "output": "Mô hình cải thiện kết quả cho nhiệm vụ này."}]
    assert attempts == 2


def test_llm_adapter_splits_multi_item_invalid_result_schema(monkeypatch) -> None:
    paragraphs = [
        _paragraph("The first model improves the result.", "p-1"),
        _paragraph("The second model improves the result.", "p-2"),
    ]
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=paragraphs)]))
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )
    responses = iter(
        [
            json.dumps([{}]),
            json.dumps([{"id": 0, "output": "Mô hình thứ nhất cải thiện kết quả."}]),
            json.dumps([{"id": 1, "output": "Mô hình thứ hai cải thiện kết quả."}]),
        ]
    )
    prompts: list[str] = []

    def request(prompt: str) -> str:
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr(translator, "_request", request)
    prefix = "instructions\n\n## Here is the input:\n\n"
    items = [
        {"id": 0, "input": paragraphs[0].unicode, "layout_label": "text"},
        {"id": 1, "input": paragraphs[1].unicode, "layout_label": "text"},
    ]

    with recorder.batch_scope():
        for paragraph in paragraphs:
            recorder.note_preprocessed(paragraph, paragraph.unicode)
        result = json.loads(translator.llm_translate(prefix + json.dumps(items)))

    assert [item["id"] for item in result] == [0, 1]
    assert len(prompts) == 3
    assert '"id": 0' in prompts[0] and '"id": 1' in prompts[0]
    assert '"id": 0' in prompts[1] and '"id": 1' not in prompts[1]
    assert '"id": 1' in prompts[2] and '"id": 0' not in prompts[2]


def test_llm_adapter_keeps_normal_batch_response_on_single_request() -> None:
    translated, request_texts, record, recorder = _translate_batch_with_responses(
        "The model improves the result in practice for this task.",
        ["该模型在此任务中提高了实际效果。"],
    )

    assert len(request_texts) == 1
    assert translated == "该模型在此任务中提高了实际效果。"
    assert record["status"] == "completed"
    assert recorder.finish()["completed"] == 1


def test_llm_adapter_retries_only_items_that_lose_protected_markers() -> None:
    source_with_marker = "The value <b1> matters."
    source_plain = "The model improves the result."
    paragraphs = [_paragraph(source_with_marker, "p-1"), _paragraph(source_plain, "p-2")]
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=paragraphs)]))
    records = [recorder.get(paragraph) for paragraph in paragraphs]
    assert all(record is not None for record in records)

    class FakeResponse:
        status_code = 200

        def __init__(self, translated_items: list[dict[str, object]]) -> None:
            self.content = json.dumps(translated_items, ensure_ascii=False)

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"content": self.content}

    class FakeClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.responses = [
                FakeResponse(
                    [
                        {"id": 0, "output": "Giá trị quan trọng."},
                        {"id": 1, "output": "Mô hình cải thiện kết quả."},
                    ]
                ),
                FakeResponse([{"id": 0, "output": "Giá trị rất quan trọng."}]),
                FakeResponse([{"id": 0, "output": "Giá trị <b1> vẫn quan trọng."}]),
            ]

        def post(self, _url: str, **kwargs) -> FakeResponse:
            prompt = kwargs["json"]["text"]
            self.prompts.append(prompt)
            return self.responses.pop(0)

    prefix = "instructions\n\n## Here is the input:\n\n"
    items = [
        {"id": 0, "input": source_with_marker, "layout_label": "text"},
        {"id": 1, "input": source_plain, "layout_label": "text"},
    ]
    client = FakeClient()
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )
    translator.client.close()
    translator.client = client

    with recorder.batch_scope():
        recorder.note_preprocessed(paragraphs[0], source_with_marker)
        recorder.note_preprocessed(paragraphs[1], source_plain)
        result = json.loads(translator.llm_translate(prefix + json.dumps(items)))

    assert [item["output"] for item in result] == [
        "Giá trị <b1> vẫn quan trọng.",
        "Mô hình cải thiện kết quả.",
    ]
    assert len(client.prompts) == 3
    for retry_prompt in client.prompts[1:]:
        assert retry_prompt.startswith(prefix)
        assert '"id": 0' in retry_prompt
        assert '"id": 1' not in retry_prompt


def test_llm_adapter_caps_marker_and_unchanged_retries_at_three_total_requests() -> None:
    source = "The value <b1> matters."
    translated, requests, record, _ = _translate_batch_with_responses(
        source,
        ["Giá trị quan trọng.", "Giá trị vẫn quan trọng.", source],
    )

    assert translated == source
    assert len(requests) == 3
    assert record["status"] in {"completed", "unchanged_prose"}


def test_llm_adapter_splits_malformed_multi_item_response() -> None:
    paragraphs = [
        _paragraph("First source sentence.", "p-1"),
        _paragraph("Second sentence.", "p-2"),
    ]
    recorder = engine_runner.TranslationCheckpointRecorder(
        source_sha256="a" * 64,
        glossary=[],
        checkpoint_results=[],
    )
    recorder.begin(SimpleNamespace(page=[SimpleNamespace(page_number=1, pdf_paragraph=paragraphs)]))

    class FakeResponse:
        status_code = 200

        def __init__(self, content: str) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"content": self.content}

    class FakeClient:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.responses = [
                FakeResponse("not valid JSON"),
                FakeResponse(json.dumps([{"id": 0, "output": "Câu thứ nhất."}])),
                FakeResponse(json.dumps([{"id": 1, "output": "Câu thứ hai."}])),
            ]

        def post(self, _url: str, **kwargs) -> FakeResponse:
            self.prompts.append(kwargs["json"]["text"])
            return self.responses.pop(0)

    items = [
        {"id": 0, "input": "First source sentence.", "layout_label": "text"},
        {"id": 1, "input": "Second sentence.", "layout_label": "text"},
    ]
    client = FakeClient()
    translator = engine_runner.SiliconFlowFreeTranslator(
        "English", "Vietnamese", engine_runner.SharedRateLimiter(), recorder
    )
    translator.client.close()
    translator.client = client

    with recorder.batch_scope():
        recorder.note_preprocessed(paragraphs[0], items[0]["input"])
        recorder.note_preprocessed(paragraphs[1], items[1]["input"])
        result = json.loads(
            translator.llm_translate(
                "instructions\n\n## Here is the input:\n\n" + json.dumps(items)
            )
        )

    assert [item["output"] for item in result] == ["Câu thứ nhất.", "Câu thứ hai."]
    assert len(client.prompts) == 3
    assert '"id": 0' in client.prompts[1] and '"id": 1' not in client.prompts[1]
    assert '"id": 1' in client.prompts[2] and '"id": 0' not in client.prompts[2]


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


def test_siliconflow_invalid_response_is_distinguished_from_outage(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            raise ValueError("private response body")

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

    assert exc_info.value.code == "PROVIDER_INVALID_RESPONSE"
    assert translator.failure_code == "PROVIDER_INVALID_RESPONSE"
    assert "private response body" not in str(exc_info.value)
    assert fake.calls == 3


def test_siliconflow_rejected_request_is_not_retried_or_exposed(monkeypatch) -> None:
    class FakeResponse:
        status_code = 400

        def raise_for_status(self) -> None:
            raise AssertionError("permanent client errors are classified before this call")

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            self.calls = 0

        def post(self, *args, **kwargs) -> FakeResponse:
            self.calls += 1
            return FakeResponse()

    fake = FakeClient()
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: fake)
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

    assert exc_info.value.code == "PROVIDER_REJECTED"
    assert translator.failure_code == "PROVIDER_REJECTED"
    assert "private paper text" not in str(exc_info.value)
    assert fake.calls == 1


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


@pytest.mark.parametrize(
    ("raw_event", "expected_error", "diagnostic", "unknown_error_code"),
    [
        (
            {"type": "segment_summary", "skip_reasons": {"private event text": 1}},
            "ENGINE_PROTOCOL_ERROR",
            True,
            False,
        ),
        ({"type": "error", "code": "ENGINE_INCOMPLETE"}, "ENGINE_INCOMPLETE", False, False),
        ({"type": "error", "code": "private error message"}, "ENGINE_FAILURE", True, True),
        ({"type": "error", "code": []}, "ENGINE_FAILURE", True, True),
    ],
)
def test_subprocess_protocol_rejection_logs_metadata_without_event_content(
    tmp_path: Path,
    raw_event: dict,
    expected_error: str,
    diagnostic: bool,
    unknown_error_code: bool,
) -> None:
    runner = tmp_path / "invalid_summary_runner.py"
    encoded_event = json.dumps(raw_event, separators=(",", ":"))
    runner.write_text(
        f"import json, sys\nsys.stdin.readline()\nprint({encoded_event!r})\n",
        encoding="utf-8",
    )

    async def run() -> None:
        process = TranslationEngineProcess(
            python=sys.executable,
            runner=runner,
            timeout_seconds=5,
        )
        await process.run({})

    logger = logging.getLogger("myra")
    engine_logger = logging.getLogger("myra.translation.engine")
    original_handlers = logger.handlers
    original_level = logger.level
    original_propagate = logger.propagate
    original_disabled = logger.disabled
    original_engine_handlers = engine_logger.handlers
    original_engine_level = engine_logger.level
    original_engine_propagate = engine_logger.propagate
    original_engine_disabled = engine_logger.disabled
    original_engine_filters = engine_logger.filters
    original_global_disable = logging.root.manager.disable
    output = io.StringIO()
    try:
        with redirect_stderr(output):
            logging.disable(logging.NOTSET)
            logger.disabled = False
            engine_logger.handlers = []
            engine_logger.setLevel(logging.NOTSET)
            engine_logger.propagate = True
            engine_logger.disabled = False
            engine_logger.filters = []
            configure_logging("INFO")
            with pytest.raises(TranslationEngineError, match=expected_error):
                asyncio.run(run())
    finally:
        logger.handlers = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate
        logger.disabled = original_disabled
        engine_logger.handlers = original_engine_handlers
        engine_logger.setLevel(original_engine_level)
        engine_logger.propagate = original_engine_propagate
        engine_logger.disabled = original_engine_disabled
        engine_logger.filters = original_engine_filters
        logging.disable(original_global_disable)

    output = output.getvalue()
    records = [json.loads(line) for line in output.splitlines()]
    diagnostics = [
        record for record in records if record["message"] == "translation_engine_protocol_rejected"
    ]
    assert bool(diagnostics) is diagnostic
    if diagnostic:
        record = diagnostics[0]
        assert record["event_type"] == raw_event["type"]
        assert record["event_fields"] == sorted(raw_event)
        assert record["unknown_error_code"] is unknown_error_code
        if "skip_reasons" in raw_event:
            assert record["unknown_skip_reason_count"] == 1
    assert "private event text" not in output
    assert "private error message" not in output


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
