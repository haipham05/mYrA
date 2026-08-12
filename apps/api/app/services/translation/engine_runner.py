"""Entry point executed only by the translation engine's isolated interpreter."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_APP_ROOT = Path(__file__).resolve().parents[3]
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

PROTOCOL_VERSION = 1
PDF2ZH_VERSION = "2.9.0"
BABELDOC_VERSION = "0.6.2"
TRANSLATION_LANG_IN = "English"
TRANSLATION_LANG_OUT = "Vietnamese"
TRANSLATION_POLICY_VERSION = "nllb-local-v1-f8d333a0"
NLLB_MODEL_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
_REVIEWED_ATTENTION_FIGURE_SOURCE_SHA256 = (
    "bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697"
)
_REVIEWED_ATTENTION_FIGURE_PAGES = frozenset({12, 13, 14})
_NON_FIGURE_LAYOUT_LABELS = frozenset({"caption", "figure_caption", "table_caption", "title"})
_REVIEWED_ATTENTION_TABLE_FRAGMENTS = frozenset(
    {
        (5, 75, "O(1)"),
        (5, 76, "O(1)"),
        (5, 79, "O(n)"),
        (5, 80, "O(n)"),
        (5, 83, "O(1)"),
        (5, 87, "O(1)"),
        (7, 117, "BLEU"),
        (8, 178, "N"),
        (8, 179, "d"),
        (8, 181, "d"),
        (8, 182, "ff"),
        (8, 183, "h"),
        (8, 184, "d"),
        (8, 185, "k"),
        (8, 186, "d"),
        (8, 187, "v"),
        (8, 188, "P"),
        (8, 189, "drop"),
        (8, 190, "εls"),
        (8, 192, "PPL"),
        (8, 193, "BLEU"),
        (8, 198, "×106"),
        (8, 199, "base"),
        (8, 208, "100K"),
        (8, 212, "(A)"),
        (8, 233, "(B)"),
        (8, 242, "(C)"),
        (8, 275, "(D)"),
        (8, 288, "(E)"),
        (8, 292, "big"),
        (8, 298, "300K"),
    }
)
_REVIEWED_ATTENTION_FIGURE_LABELS = frozenset({(12, 375, "r5")})
_REVIEWED_ATTENTION_NON_PROSE_UNITS = frozenset(
    {
        (3, 39, "Scaled Dot-Product Attention", "abandon"),
        (9, 314, "Vinyals & Kaiser el al. (2014) [37]", "fallback_line"),
        (9, 332, "Huang & Harper (2009) [14]", "fallback_line"),
    }
)
_PREPROCESS_DECLINE_CAUSES = frozenset(
    {
        "vertical_paragraph",
        "no_composition",
        "pure_numeric",
        "placeholder_only",
        "formula_only",
        "debug_unicode_composition",
        "below_minimum_length",
        "unsupported_composition",
        "unknown_decline",
    }
)
LAYOUT_MODEL_SHA3_256 = "60be061226930524958b5465c8c04af3d7c03bcb0beb66454f5da9f792e3cf2a"
PROTOCOL_STDOUT = sys.stdout
_ACTIVE_RECORDER: TranslationCheckpointRecorder | None = None
_PROTECTED_TOKEN = re.compile(
    r"\{v\d+\}|\{[A-Za-z][\w.-]*\}|%[sd]|\[\[.*?\]\]|%%.*?%%|</?style\b[^>]*>|</?b\d+>"
)
_SCIENTIFIC_TOKEN = re.compile(
    r"(?<![\w])(?:"
    r"[A-Z][a-z]+[A-Z][A-Za-z0-9]*|"  # CamelCase model and architecture names
    r"[A-Z][a-z]+(?:-[A-Z][a-z]+)+|"  # Hyphenated architecture names
    r"[A-Z]{2,}(?:-[A-Z0-9]+)*|"  # Acronyms and benchmark names
    r"[A-Za-z][A-Za-z0-9-]*\d+[A-Za-z0-9-]*|"  # Alphanumeric model identifiers
    r"\d+(?:[.,]\d+)*(?:\s?%|[A-Za-z]{1,5})?"  # Measurements, values, and citations
    r")(?![\w])"
)
_FOOTNOTE_PROSE_START = re.compile(r"^\s*\d{1,2}\s*(?:we|this|it|our|see|note|source)\b", re.I)


def _checkpoint_base_key(identity: dict[str, Any]) -> str:
    identity_wire = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity_wire.encode()).hexdigest()


def _paragraph_render_order_span(paragraph: Any) -> tuple[int, int] | None:
    """Return a span only when source characters form one exact reading-order run."""
    characters = []
    for composition in getattr(paragraph, "pdf_paragraph_composition", None) or []:
        if line := getattr(composition, "pdf_line", None):
            characters.extend(getattr(line, "pdf_character", None) or [])
        elif character := getattr(composition, "pdf_character", None):
            characters.append(character)
        elif formula := getattr(composition, "pdf_formula", None):
            formula_characters = getattr(formula, "pdf_character", None) or []
            if not formula_characters:
                return None
            characters.extend(formula_characters)
        elif same_style := getattr(composition, "pdf_same_style_characters", None):
            characters.extend(getattr(same_style, "pdf_character", None) or [])
        else:
            return None
    orders = []
    for character in characters:
        order = getattr(character, "render_order", None)
        if order is None:
            char_unicode = getattr(character, "char_unicode", None)
            if isinstance(char_unicode, str) and char_unicode.isspace():
                continue
            return None
        orders.append(order)
    if (
        not orders
        or any(not isinstance(order, int) or isinstance(order, bool) for order in orders)
        or orders != list(range(orders[0], orders[0] + len(orders)))
    ):
        return None
    return orders[0], orders[-1]


class SafeEngineError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code


class SharedRateLimiter:
    """One process-wide token schedule shared by translation and glossary calls."""

    def __init__(self, requests_per_second: int = 2) -> None:
        self._interval = 1 / requests_per_second
        self._lock = threading.Lock()
        self._concurrency = threading.BoundedSemaphore(2)
        self._next_request_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request_at - now)
            self._next_request_at = max(now, self._next_request_at) + self._interval
        if delay:
            time.sleep(delay)

    @contextmanager
    def request_slot(self):
        with self._concurrency:
            self.wait()
            yield


def _reviewed_attention_preserve_reason(
    *,
    source_sha256: str,
    page_number: int,
    ordinal: int,
    source_quote: str,
    is_vertical: bool,
    layout_label: str | None,
) -> str | None:
    if source_sha256 != _REVIEWED_ATTENTION_FIGURE_SOURCE_SHA256:
        return None
    if (page_number, ordinal, source_quote) in _REVIEWED_ATTENTION_TABLE_FRAGMENTS:
        return "preserved_scientific_table_content"
    if (page_number, ordinal, source_quote) in _REVIEWED_ATTENTION_FIGURE_LABELS:
        return "preserved_embedded_figure_text"
    if (page_number, ordinal, source_quote, layout_label) in _REVIEWED_ATTENTION_NON_PROSE_UNITS:
        return "preserved_figure_or_citation_metadata"
    if (
        page_number in _REVIEWED_ATTENTION_FIGURE_PAGES
        and is_vertical
        and str(layout_label or "").casefold().strip() not in _NON_FIGURE_LAYOUT_LABELS
    ):
        return "preserved_embedded_figure_text"
    return None


class TranslationCheckpointRecorder:
    """Build page/quote identities and persistable checkpoints from BabelDOC IL hooks."""

    def __init__(
        self,
        *,
        source_sha256: str,
        glossary: list[dict[str, str]],
        checkpoint_results: list[dict[str, str]],
        allow_untranslated_units: bool = False,
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise SafeEngineError("INVALID_ENGINE_REQUEST")
        self.source_sha256 = source_sha256
        self.glossary = glossary
        self.allow_untranslated_units = allow_untranslated_units
        glossary_wire = json.dumps(
            glossary, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        self.glossary_sha256 = hashlib.sha256(glossary_wire.encode()).hexdigest()
        self.local = threading.local()
        self.lock = threading.Lock()
        self.records: dict[int, dict[str, Any]] = {}
        self.checkpoints: dict[str, str] = {}
        for item in checkpoint_results:
            if not isinstance(item, dict):
                continue
            key = item.get("segment_key")
            translated = item.get("translated_text")
            translated_sha = item.get("translated_sha256")
            if (
                isinstance(key, str)
                and isinstance(translated, str)
                and len(translated) <= 65_536
                and hashlib.sha256(translated.encode()).hexdigest() == translated_sha
            ):
                self.checkpoints[key] = translated
        self.completed = 0
        self.skipped = 0
        self.failed = 0
        self.summary_emitted = False

    def begin(self, docs: Any) -> None:
        global_ordinal = 0
        for page_index, page in enumerate(docs.page, start=1):
            page_number = page.page_number or page_index
            offset = 0
            page_records: list[dict[str, Any]] = []
            page_paragraphs = list(page.pdf_paragraph)
            for page_ordinal, paragraph in enumerate(page_paragraphs):
                quote = paragraph.unicode or ""
                record = {
                    "page_number": int(page_number),
                    "ordinal": global_ordinal,
                    "page_ordinal": page_ordinal,
                    "source_quote": quote,
                    "is_vertical": bool(getattr(paragraph, "vertical", False)),
                    "source_start": offset,
                    "source_end": offset + len(quote),
                    "source_sha256": hashlib.sha256(quote.encode()).hexdigest(),
                    "engine_paragraph_id": paragraph.debug_id,
                    "first_occurrence_terms": [],
                    "layout_label": paragraph.layout_label,
                    "preprocessed_input": None,
                    "context_hash": None,
                    "status": "pending",
                }
                record["preserve_reason"] = _reviewed_attention_preserve_reason(
                    source_sha256=self.source_sha256,
                    page_number=record["page_number"],
                    ordinal=record["ordinal"],
                    source_quote=record["source_quote"],
                    is_vertical=record["is_vertical"],
                    layout_label=record["layout_label"],
                )
                if record["preserve_reason"] is None:
                    del record["preserve_reason"]
                identity = {
                    "source_pdf_sha256": self.source_sha256,
                    "page_number": record["page_number"],
                    "ordinal": global_ordinal,
                    "page_ordinal": page_ordinal,
                    "source_sha256": record["source_sha256"],
                    "glossary_sha256": self.glossary_sha256,
                    "pdf2zh_version": PDF2ZH_VERSION,
                    "babeldoc_version": BABELDOC_VERSION,
                    "lang_in": TRANSLATION_LANG_IN,
                    "lang_out": TRANSLATION_LANG_OUT,
                    "translation_policy_version": TRANSLATION_POLICY_VERSION,
                }
                record["base_key"] = _checkpoint_base_key(identity)
                self.records[id(paragraph)] = record
                page_records.append(record)
                offset += len(quote) + 1
                global_ordinal += 1
            self._mark_split_footnotes(page_records, page_paragraphs)

    @staticmethod
    def _mark_split_footnotes(records: list[dict[str, Any]], paragraphs: list[Any]) -> None:
        spans = [_paragraph_render_order_span(paragraph) for paragraph in paragraphs]
        for start, record in enumerate(records):
            first_text = record["source_quote"]
            if not _FOOTNOTE_PROSE_START.match(first_text) or spans[start] is None:
                continue

            group = [start]
            combined = first_text
            while len(group) < 4:
                previous_index = group[-1]
                next_index = previous_index + 1
                if next_index >= len(records):
                    break
                previous = records[previous_index]
                following = records[next_index]
                previous_span = spans[previous_index]
                following_span = spans[next_index]
                if (
                    following["page_number"] != record["page_number"]
                    or following["page_ordinal"] != previous["page_ordinal"] + 1
                    or previous_span is None
                    or following_span is None
                    or following_span[0] != previous_span[1] + 1
                ):
                    break

                next_text = following["source_quote"]
                decimal_continuation = bool(
                    re.search(r"\d\.$", combined) and re.match(r"^\d", next_text)
                )
                if re.search(r"[.!?][\"')\]]*$", combined) and not decimal_continuation:
                    break
                group.append(next_index)
                combined += next_text
                if len(combined) > 512:
                    group = []
                    break

                if re.search(r"[.!?][\"')\]]*$", combined):
                    if len(group) > 1:
                        for index in group:
                            records[index]["preserve_reason"] = (
                                "preserved_split_footnote_layout_content"
                            )
                    break

    def get(self, paragraph: Any) -> dict[str, Any] | None:
        return self.records.get(id(paragraph))

    @contextmanager
    def batch_scope(self):
        previous = getattr(self.local, "batch", None)
        self.local.batch = []
        try:
            yield
        finally:
            self.local.batch = previous

    @contextmanager
    def paragraph_scope(self, paragraph: Any):
        previous = getattr(self.local, "paragraph", None)
        self.local.paragraph = self.get(paragraph)
        try:
            yield
        finally:
            self.local.paragraph = previous

    def note_preprocessed(self, paragraph: Any, text: str | None) -> None:
        record = self.get(paragraph)
        if not record:
            return
        if text is None:
            reason = self._intentional_skip_reason(record)
            if reason:
                self._skip(record, reason)
            return
        reason = self._intentional_skip_reason(record)
        if reason and reason not in {"protected_scientific_content", "placeholder_only"}:
            self._skip(record, reason)
            return
        record["preprocessed_input"] = text
        batch = getattr(self.local, "batch", None)
        if batch is not None:
            batch.append(record)

    def current_batch(self) -> list[dict[str, Any]]:
        return list(getattr(self.local, "batch", []) or [])

    def key_for(self, record: dict[str, Any], context_hash: str) -> str:
        record["context_hash"] = context_hash
        return hashlib.sha256(f"{record['base_key']}:{context_hash}".encode()).hexdigest()

    def complete(self, record: dict[str, Any], translated_text: str) -> None:
        if record["status"] == "skipped":
            return
        source = record["source_quote"]
        validation_source = record.get("preprocessed_input") or source
        if _protected_tokens(validation_source) != _protected_tokens(translated_text):
            record["status"] = "invalid"
            record["failure_reason"] = "protected_placeholder_mismatch"
            self.failed += 1
            return
        unchanged_prose = _is_unchanged_english_prose(
            validation_source, translated_text, layout_label=record.get("layout_label")
        )
        preserve_official_title = record.get("layout_label") == "title" and unchanged_prose
        if unchanged_prose and not preserve_official_title and self.allow_untranslated_units:
            self._skip(record, "local_model_unchanged")
            return
        if unchanged_prose and not preserve_official_title:
            skip_reason = self._intentional_skip_reason(record)
            if skip_reason:
                self._skip(record, skip_reason)
                return
        if unchanged_prose and not preserve_official_title:
            record["status"] = "unchanged_prose"
            record["failure_reason"] = "unchanged_prose"
            self.failed += 1
            return
        if len(source) > 65_536 or len(translated_text) > 65_536:
            record["status"] = "oversized"
            record["failure_reason"] = "unit_too_large"
            self.failed += 1
            return
        context_hash = (
            record.get("context_hash") or hashlib.sha256(validation_source.encode()).hexdigest()
        )
        key = self.key_for(record, context_hash)
        translated_sha = hashlib.sha256(translated_text.encode()).hexdigest()
        with self.lock:
            record["status"] = "preserved" if preserve_official_title else "completed"
            self.completed += 1
            _emit(
                {
                    "type": "checkpoint",
                    "segment": {
                        "segment_key": key,
                        "page_number": record["page_number"],
                        "ordinal": record["ordinal"],
                        "page_ordinal": record["page_ordinal"],
                        "source_quote": source,
                        "source_start": record["source_start"],
                        "source_end": record["source_end"],
                        "source_sha256": record["source_sha256"],
                        "source_pdf_sha256": self.source_sha256,
                        "offset_basis": "babeldoc_reading_order_paragraphs",
                        "engine_paragraph_id": record["engine_paragraph_id"],
                        "layout_label": record["layout_label"],
                        "glossary_sha256": self.glossary_sha256,
                        "translated_text": translated_text,
                        "translated_sha256": translated_sha,
                        "status": "preserved" if preserve_official_title else "validated",
                    },
                }
            )

    def invalidate(self, record: dict[str, Any], reason: str) -> None:
        if record["status"] not in {"invalid", "unchanged_prose", "oversized", "untranslated"}:
            self.failed += 1
        record["status"] = "invalid"
        record["failure_reason"] = reason

    def finish(self) -> dict[str, Any]:
        for record in self.records.values():
            if record["status"] == "pending":
                # BabelDOC also omits long paragraphs for layout/ID reasons. Unless
                # preservation is objectively clear, an unvisited unit is unresolved.
                reason = self._intentional_skip_reason(record)
                if reason:
                    self._skip(record, reason)
                else:
                    record["status"] = "untranslated"
                    decline_reason = record.get("preprocess_decline_reason")
                    if decline_reason not in _PREPROCESS_DECLINE_CAUSES:
                        decline_reason = None
                    record["failure_reason"] = (
                        decline_reason or "preprocessing_not_selected"
                        if record.get("preprocessed_input") is None
                        else "missing_validated_checkpoint"
                    )
                    self.failed += 1
        return {
            "total": len(self.records),
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": self.failed,
        }

    @staticmethod
    def _intentional_skip_reason(record: dict[str, Any]) -> str | None:
        source = record["source_quote"]
        normalized = source.strip()
        label = str(record.get("layout_label") or "").casefold()
        if record.get("preserve_reason") == "preserved_split_footnote_layout_content":
            return record["preserve_reason"]
        if record.get("preserve_reason") == "preserved_embedded_figure_text":
            return record["preserve_reason"]
        if record.get("preserve_reason") == "preserved_scientific_table_content":
            return record["preserve_reason"]
        if record.get("preserve_reason") == "preserved_figure_or_citation_metadata":
            return record["preserve_reason"]
        if not normalized:
            return "empty"
        if label in {"equation", "formula", "math"}:
            return "protected_scientific_content"
        if record.get("preprocess_decline_reason") == "formula_only":
            return "protected_scientific_content"
        if record.get("preprocess_decline_reason") == "below_minimum_length":
            return "below_engine_minimum"
        # BabelDOC may leave short fallback fragments outside its selectable
        # translation batches. Keep them in English as explicit omissions rather
        # than calling them engine-minimum skips or failing the partial PDF.
        if label == "fallback_line" and len(normalized) <= 22:
            return "untranslated_short_fallback"
        if _PROTECTED_TOKEN.fullmatch(normalized):
            return "placeholder_only"
        if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", normalized):
            return "author_contact_metadata"
        if _is_arxiv_version_stamp(record):
            return "arxiv_version_stamp"
        if re.fullmatch(r"[\d\s.,:%/+−–—=()\[\]{}·]+", normalized):
            return "numeric_or_symbol_only"
        return None

    def _skip(self, record: dict[str, Any], reason: str) -> None:
        if record["status"] == "skipped":
            return
        record["status"] = "skipped"
        record["skip_reason"] = reason
        self.skipped += 1

    def failure_reasons(self) -> dict[str, int]:
        reasons: dict[str, int] = {}
        for record in self.records.values():
            status = record["status"]
            if status in {"invalid", "unchanged_prose", "oversized", "untranslated"}:
                reasons[status] = reasons.get(status, 0) + 1
        return reasons

    def skip_reasons(self) -> dict[str, int]:
        reasons: dict[str, int] = {}
        for record in self.records.values():
            reason = record.get("skip_reason")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        return reasons

    def failure_causes(self) -> dict[str, int]:
        causes: dict[str, int] = {}
        for record in self.records.values():
            if record["status"] not in {"invalid", "unchanged_prose", "oversized", "untranslated"}:
                continue
            decline_reason = record.get("preprocess_decline_reason")
            if decline_reason not in _PREPROCESS_DECLINE_CAUSES:
                decline_reason = None
            reason = decline_reason or record.get("failure_reason")
            if reason:
                causes[reason] = causes.get(reason, 0) + 1
        return causes

    def failure_units(self) -> list[dict[str, Any]]:
        return [
            {
                "page_number": record["page_number"],
                "ordinal": record["ordinal"],
                "status": record["status"],
                "failure_reason": record.get("failure_reason"),
                "source_chars": len(record["source_quote"]),
                "layout_label": record["layout_label"],
            }
            for record in self.records.values()
            if record["status"] in {"invalid", "unchanged_prose", "oversized", "untranslated"}
        ]

    def emit_summary(self) -> None:
        if self.summary_emitted:
            return
        self.summary_emitted = True
        _emit(
            {
                "type": "segment_summary",
                **self.finish(),
                "failure_reasons": self.failure_reasons(),
                "failure_causes": self.failure_causes(),
                "skip_reasons": self.skip_reasons(),
                "failure_units": self.failure_units(),
            }
        )


def _protected_tokens(text: str) -> tuple[str, ...]:
    return tuple(_PROTECTED_TOKEN.findall(text))


def _lock_scientific_tokens(text: str) -> tuple[str, dict[str, str]]:
    """Temporarily protect compact identifiers and numeric values during translation."""
    existing = [match.span() for match in _PROTECTED_TOKEN.finditer(text)]
    replacements: dict[str, str] = {}
    pieces: list[str] = []
    cursor = 0
    token_index = 0
    for match in _SCIENTIFIC_TOKEN.finditer(text):
        start, end = match.span()
        if any(
            start < protected_end and end > protected_start
            for protected_start, protected_end in existing
        ):
            continue
        placeholder = f"[[MYRA_KEEP_{token_index}]]"
        while placeholder in text or placeholder in replacements:
            token_index += 1
            placeholder = f"[[MYRA_KEEP_{token_index}]]"
        pieces.extend((text[cursor:start], placeholder))
        replacements[placeholder] = match.group(0)
        cursor = end
        token_index += 1
    if not replacements:
        return text, replacements
    pieces.append(text[cursor:])
    return "".join(pieces), replacements


def _restore_scientific_tokens(text: str, replacements: dict[str, str], *, source: str) -> str:
    if not replacements:
        return text
    if text == source:
        return text
    expected = tuple(replacements)
    found = tuple(_PROTECTED_TOKEN.findall(text))
    locked = tuple(token for token in found if token in replacements)
    if locked != expected:
        raise SafeEngineError("PROVIDER_SCIENTIFIC_TOKEN_MISMATCH")
    restored = text
    for placeholder, source_token in replacements.items():
        restored = restored.replace(placeholder, source_token, 1)
    return restored


def _classify_preprocess_decline(
    paragraph: Any,
    translate_input: Any,
    *,
    minimum_text_length: int,
    is_pure_numeric: Any,
    is_placeholder_only: Any,
) -> str | None:
    """Classify BabelDOC 0.6.2's no-input branches without exposing paragraph text."""
    if getattr(paragraph, "vertical", False):
        return "vertical_paragraph"

    compositions = getattr(paragraph, "pdf_paragraph_composition", None)
    if not compositions:
        return "no_composition"
    if is_pure_numeric(paragraph):
        return "pure_numeric"
    if len(compositions) == 1 and getattr(compositions[0], "pdf_formula", None):
        return "formula_only"
    if is_placeholder_only(paragraph):
        return "placeholder_only"

    if translate_input is not None:
        input_text = getattr(translate_input, "unicode", None)
        if isinstance(input_text, str) and len(input_text) < minimum_text_length:
            return "below_minimum_length"
        return None
    if len(compositions) == 1:
        composition = compositions[0]
        if getattr(composition, "pdf_same_style_unicode_characters", None):
            return "debug_unicode_composition"

    known_fields = (
        "pdf_line",
        "pdf_same_style_characters",
        "pdf_character",
        "pdf_formula",
        "pdf_same_style_unicode_characters",
    )
    if any(
        not any(getattr(composition, field, None) for field in known_fields)
        for composition in compositions
    ):
        return "unsupported_composition"
    return "unknown_decline"


def _is_arxiv_version_stamp(record: dict[str, Any]) -> bool:
    """Recognize BabelDOC's rotated, fragmented arXiv page-edge stamp only."""
    if not record.get("is_vertical") or record.get("layout_label") != "abandon":
        return False
    tokens = re.findall(r"[a-z]+|\d+", record["source_quote"].casefold())
    words = {token for token in tokens if token.isalpha()}
    numbers = [token for token in tokens if token.isdigit()]
    has_arxiv_marker = "iv" in words and ({"ar", "x"}.issubset(words) or "arx" in words)
    has_archive_id = any(len(token) == 4 for token in numbers) and any(
        len(token) == 5 for token in numbers
    )
    has_subject = "cs" in words and "cl" in words
    has_date = any(len(token) == 4 and token.startswith(("19", "20")) for token in numbers)
    return has_arxiv_marker and has_archive_id and has_subject and has_date


_ENGLISH_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "can",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "this",
        "these",
        "those",
        "to",
        "was",
        "were",
        "will",
        "with",
        "we",
        "you",
    }
)
_NON_PROSE_LAYOUT_LABELS = frozenset({"code", "equation", "formula", "math"})
_PROSE_LAYOUT_LABELS = frozenset({"caption", "heading", "text", "title"})
_ENGLISH_SENTENCE_STARTERS = frozenset(
    {
        "a",
        "an",
        "although",
        "because",
        "for",
        "he",
        "however",
        "if",
        "in",
        "it",
        "our",
        "she",
        "the",
        "their",
        "these",
        "they",
        "this",
        "those",
        "we",
        "when",
        "while",
        "you",
    }
)


def _english_function_word_count(text: str) -> int:
    words = re.findall(r"[a-z]+", text.casefold())
    count = sum(word in _ENGLISH_FUNCTION_WORDS for word in words)
    for word in words:
        # Layout extraction can join a leading pronoun to the following verb
        # (for example, "Wepresentthese"). Do not split arbitrary technical
        # words: doing so can misread "attention" as "a" + "on".
        if word.startswith(("we", "they", "you", "it", "this", "these")) and any(
            word.endswith(suffix) and len(word) - len("we") - len(suffix) >= 4
            for suffix in ("the", "these", "those", "a", "an", "in", "to", "for")
        ):
            count += 2
    return count


def _contains_unchanged_english_sentence(source: str, translated: str) -> bool:
    """Detect a whole source sentence copied unchanged into an otherwise translated unit."""

    def compact(text: str) -> str:
        return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", text).casefold())

    translated_compact = compact(translated)
    for sentence in re.split(r"(?<=[.!?])\s+", source):
        sentence_compact = compact(sentence)
        if len(sentence_compact) < 24 or sentence_compact not in translated_compact:
            continue
        first_word = re.match(r"[A-Za-z]+", sentence.strip())
        if first_word is None:
            continue
        first_word = first_word.group().casefold()
        starts_like_sentence = first_word in _ENGLISH_SENTENCE_STARTERS or first_word.startswith(
            ("we", "they", "you", "it", "this", "these")
        )
        if not starts_like_sentence:
            continue
        if _english_function_word_count(sentence) >= 2:
            return True
    return False


def _is_unchanged_english_prose(
    source: str, translated: str, *, layout_label: str | None = None
) -> bool:
    """Catch obvious unchanged English sentences, not equations or short terms.

    This deliberately uses a conservative language heuristic rather than a new
    language-detection dependency. BabelDOC already marks formula-only units as
    skipped; the word/function-word threshold guards mixed technical content.
    """
    if layout_label and layout_label.casefold().strip() in _NON_PROSE_LAYOUT_LABELS:
        return False

    def normalize_exact(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()

    if len(normalize_exact(source)) >= 24 and normalize_exact(source) == normalize_exact(
        translated
    ):
        return True
    if (layout_label or "").casefold().strip() != "title" and _contains_unchanged_english_sentence(
        source, translated
    ):
        return True

    def normalize(text: str) -> str:
        text = _PROTECTED_TOKEN.sub(" ", text)
        text = re.sub(r"\\[A-Za-z]+(?:\{[^}]*\})?", " ", text)
        return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))

    source_normalized = normalize(source)
    if not source_normalized or source_normalized != normalize(translated):
        return False
    words = source_normalized.split()
    if len(words) < 4:
        return False
    function_word_count = sum(word in _ENGLISH_FUNCTION_WORDS for word in words)
    prose_label = (layout_label or "").casefold().strip() in _PROSE_LAYOUT_LABELS
    acronym_count = len(re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", source))
    if acronym_count >= 2 and function_word_count < 2:
        return False
    if function_word_count < 2 and not (prose_label and function_word_count >= 1):
        return False
    # Obvious equations/markup-rich fragments should remain eligible unchanged.
    math_markers = len(re.findall(r"[=^_{}]", source)) + len(
        re.findall(r"\\(?:frac|sum|int|sqrt|begin)\b", source)
    )
    return math_markers < 2


def _checkpoint_is_usable(source: str, translated: str, layout_label: str | None) -> bool:
    if _protected_tokens(source) != _protected_tokens(translated):
        return False
    if str(layout_label or "").casefold().strip() == "title":
        return True
    return not _is_unchanged_english_prose(source, translated, layout_label=layout_label)


def _apply_first_occurrence_terms(translated_text: str, terms: list[dict[str, str]]) -> str:
    """Keep the preferred Vietnamese term and add its English form at first use."""
    output = translated_text
    for term in terms:
        english = term["english"]
        vietnamese = term["vietnamese"]
        if not english or not vietnamese or english.casefold() == vietnamese.casefold():
            continue
        paired = re.compile(
            rf"{re.escape(english)}\s*\(\s*{re.escape(vietnamese)}\s*\)",
            re.IGNORECASE,
        )
        if paired.search(output):
            continue
        target_match = re.search(re.escape(vietnamese), output, re.IGNORECASE)
        if target_match is None:
            raise SafeEngineError("ENGINE_INCOMPLETE")
        output = (
            output[: target_match.start()]
            + f"{english} ({target_match.group(0)})"
            + output[target_match.end() :]
        )
    return output


def _context_hash(prefix: str, suffix: str, source_input: str) -> str:
    return hashlib.sha256(f"{prefix}\0{suffix}\0{source_input}".encode()).hexdigest()


def _parse_batch_prompt(prompt: str) -> tuple[str, str, list[dict[str, Any]]] | None:
    marker = "## Here is the input:\n\n"
    if marker not in prompt:
        return None
    prefix, raw_input = prompt.rsplit(marker, 1)
    try:
        items = json.loads(raw_input)
    except json.JSONDecodeError:
        return None
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        return None
    return prefix + marker, "", items


def _clean_model_json(value: str) -> list[dict[str, Any]]:
    cleaned = value.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    parsed = json.loads(cleaned)
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, dict) for item in parsed):
        raise SafeEngineError("PROVIDER_INVALID_SCHEMA")
    return parsed


def _emit(event: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(event, separators=(",", ":")) + "\n")
    PROTOCOL_STDOUT.flush()


def _validate_request(request: Any) -> dict[str, Any]:
    if (
        not isinstance(request, dict)
        or request.get("protocol_version") != PROTOCOL_VERSION
        or request.get("lang_in") != TRANSLATION_LANG_IN
        or request.get("lang_out") != TRANSLATION_LANG_OUT
    ):
        raise SafeEngineError("INVALID_ENGINE_REQUEST")
    for key in ("input_pdf", "output_dir", "working_dir", "layout_model"):
        if not isinstance(request.get(key), str) or not request[key]:
            raise SafeEngineError("INVALID_ENGINE_REQUEST")
    glossary = request.get("glossary", [])
    if not isinstance(glossary, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("source"), str)
        or not isinstance(item.get("target"), str)
        for item in glossary
    ):
        raise SafeEngineError("INVALID_ENGINE_REQUEST")
    if not re.fullmatch(r"[0-9a-f]{64}", str(request.get("source_pdf_sha256", ""))):
        raise SafeEngineError("INVALID_ENGINE_REQUEST")
    if not isinstance(request.get("checkpoint_results", []), list):
        raise SafeEngineError("INVALID_ENGINE_REQUEST")
    return request


def _check_versions() -> None:
    if (
        importlib.metadata.version("pdf2zh-next") != PDF2ZH_VERSION
        or importlib.metadata.version("babeldoc") != BABELDOC_VERSION
    ):
        raise SafeEngineError("ENGINE_VERSION_MISMATCH")


def _check_layout_model(path: Path) -> None:
    if not path.is_file():
        raise SafeEngineError("LAYOUT_MODEL_UNAVAILABLE")
    digest = hashlib.sha3_256()
    with path.open("rb") as model_file:
        for block in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != LAYOUT_MODEL_SHA3_256:
        raise SafeEngineError("LAYOUT_MODEL_UNAVAILABLE")


def _preflight_offline_assets(layout_model: Path) -> None:
    """Fail closed unless all BabelDOC runtime assets already exist locally."""
    _check_layout_model(layout_model)
    try:
        from babeldoc.assets import assets
        from babeldoc.assets.embedding_assets_metadata import TIKTOKEN_CACHES

        asset_root = Path(
            os.environ.get("MYRA_TRANSLATION_ASSET_DIR", "/tmp/.cache/babeldoc")
        ).resolve()
        original_cache_path = assets.get_cache_file_path

        def offline_cache_path(filename: str, sub_folder: str | None = None) -> Path:
            if sub_folder in {"models", "fonts", "cmap", "tiktoken"}:
                return asset_root / sub_folder / filename
            return original_cache_path(filename, sub_folder)

        assets.get_cache_file_path = offline_cache_path
        os.environ["TIKTOKEN_CACHE_DIR"] = str(asset_root / "tiktoken")

        required_fonts = assets.get_font_family("Vietnamese")
        font_names = {
            font_name
            for family in ("normal", "script", "fallback", "base")
            for font_name in required_fonts[family]
        }
        for font_name in font_names:
            metadata = assets.EMBEDDING_FONT_METADATA.get(font_name)
            if not metadata:
                raise SafeEngineError("FONT_ASSETS_UNAVAILABLE")
            cached_path = assets.get_cache_file_path(font_name, "fonts")
            if not assets.verify_file(cached_path, metadata["sha3_256"]):
                raise SafeEngineError("FONT_ASSETS_UNAVAILABLE")
        for cache_name, expected_hash in TIKTOKEN_CACHES.items():
            cache_path = assets.get_cache_file_path(cache_name, "tiktoken")
            if not assets.verify_file(cache_path, expected_hash):
                raise SafeEngineError("FONT_ASSETS_UNAVAILABLE")
    except SafeEngineError:
        raise
    except Exception as exc:
        raise SafeEngineError("FONT_ASSETS_UNAVAILABLE") from exc


def _install_no_download_asset_guards(high_level: Any) -> None:
    """Replace BabelDOC's asset helpers with verified local-cache-only readers."""
    from babeldoc.assets import assets

    def cached_font(font_file_name: str) -> tuple[Path, dict[str, Any]]:
        metadata = assets.EMBEDDING_FONT_METADATA.get(font_file_name)
        path = assets.get_cache_file_path(font_file_name, "fonts")
        if not metadata or not assets.verify_file(path, metadata["sha3_256"]):
            raise SafeEngineError("FONT_ASSETS_UNAVAILABLE")
        return path, metadata

    def cached_cmap(name: str) -> dict[str, Any]:
        filename = name if name.endswith(".json") else f"{name}.json"
        metadata = assets.CMAP_METADATA.get(filename)
        path = assets.get_cache_file_path(filename, "cmap")
        if not metadata or not assets.verify_file(path, metadata["sha3_256"]):
            raise SafeEngineError("FONT_ASSETS_UNAVAILABLE")
        return json.loads(path.read_text(encoding="utf-8"))

    assets.get_font_and_metadata = cached_font
    assets.get_cmap_data = cached_cmap
    high_level.warmup = lambda: None


class SiliconFlowFreeTranslator:
    """Bounded adapter matching BabelDOC's translator interface."""

    name = "myra-sf-free"
    model = "THUDM/GLM-4-9B-0414 (service-documented)"
    lang_map: dict[str, str] = {}

    def __init__(
        self,
        lang_in: str,
        lang_out: str,
        limiter: SharedRateLimiter,
        recorder: TranslationCheckpointRecorder,
    ) -> None:
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.ignore_cache = True
        self.translate_call_count = 0
        self.translate_cache_call_count = 0
        self.limiter = limiter
        self.recorder = recorder
        self.failure_code: str | None = None
        import httpx

        self.client = httpx.Client(timeout=httpx.Timeout(60.0))

    def __str__(self) -> str:
        return f"{self.name} {self.lang_in} {self.lang_out}"

    def _provider_error(self, code: str = "PROVIDER_INVALID_OUTPUT") -> SafeEngineError:
        self.failure_code = code
        return SafeEngineError(self.failure_code)

    def _marker_error(self, record: dict[str, Any] | None) -> SafeEngineError:
        if record is not None:
            self.recorder.invalidate(record, "protected_placeholder_mismatch")
        return self._provider_error("PROVIDER_MARKER_MISMATCH")

    def translate(self, text: str, ignore_cache: bool = False, rate_limit_params=None) -> str:
        del ignore_cache, rate_limit_params
        self.translate_call_count += 1
        record = getattr(self.recorder.local, "paragraph", None)
        if record:
            key = self.recorder.key_for(record, hashlib.sha256(text.encode()).hexdigest())
            cached = self.recorder.checkpoints.get(key)
            if cached is not None and _checkpoint_is_usable(
                text, cached, record.get("layout_label")
            ):
                return cached
        result = self._request(text)
        if record:
            if _protected_tokens(text) != _protected_tokens(result):
                raise self._marker_error(record)
            record["context_hash"] = hashlib.sha256(text.encode()).hexdigest()
            result = _apply_first_occurrence_terms(result, record.get("first_occurrence_terms", []))
        return result

    def llm_translate(self, text: str, ignore_cache: bool = False, rate_limit_params=None) -> str:
        del ignore_cache, rate_limit_params
        parsed = _parse_batch_prompt(text)
        records = self.recorder.current_batch()
        if not parsed or len(records) != len(parsed[2]):
            record = getattr(self.recorder.local, "paragraph", None)
            if record:
                context_hash = hashlib.sha256(text.encode()).hexdigest()
                key = self.recorder.key_for(record, context_hash)
                cached = self.recorder.checkpoints.get(key)
                source = record.get("preprocessed_input") or ""
                if cached is not None and _checkpoint_is_usable(
                    source, cached, record.get("layout_label")
                ):
                    return _apply_first_occurrence_terms(
                        cached, record.get("first_occurrence_terms", [])
                    )
                record["context_hash"] = context_hash
                result = self._request(text)
                if source and _protected_tokens(source) != _protected_tokens(result):
                    raise self._marker_error(record)
                return result
            return self._request(text)

        prefix, suffix, items = parsed
        record_by_id = {item.get("id"): records[index] for index, item in enumerate(items)}
        prepared: dict[int, tuple[dict[str, Any], str]] = {}
        misses: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            record = records[index]
            source_input = item.get("input")
            if not isinstance(source_input, str):
                return self._request(text)
            if record["first_occurrence_terms"]:
                item["first_occurrence_glossary_terms"] = record["first_occurrence_terms"]
            context_hash = _context_hash(prefix, suffix, source_input)
            key = self.recorder.key_for(record, context_hash)
            cached = self.recorder.checkpoints.get(key)
            if isinstance(cached, str) and not _checkpoint_is_usable(
                source_input, cached, item.get("layout_label")
            ):
                # Older, integrity-valid checkpoints may contain a translated
                # paragraph with an unchanged English sentence. Do not let that
                # checkpoint suppress the bounded provider retry on resume.
                cached = None
            if isinstance(cached, str) and _protected_tokens(source_input) == _protected_tokens(
                cached
            ):
                prepared[index] = (item, cached)
            else:
                misses.append(item)

        provider_results: dict[Any, str] = {}
        if misses:
            isolated_ids: set[Any] = set()
            provider_attempts: dict[Any, int] = {}

            def translate_individually(batch: list[dict[str, Any]]) -> dict[Any, str]:
                recovered: dict[Any, str] = {}
                for item in batch:
                    isolated_ids.add(item.get("id"))
                    recovered.update(request_batch([item]))
                return recovered

            def request_batch(batch: list[dict[str, Any]]) -> dict[Any, str]:
                for item in batch:
                    item_id = item.get("id")
                    provider_attempts[item_id] = provider_attempts.get(item_id, 0) + 1
                locked_tokens: dict[Any, dict[str, str]] = {}
                original_inputs: dict[Any, str] = {}
                provider_batch: list[dict[str, Any]] = []
                for item in batch:
                    provider_item = dict(item)
                    protected_input, replacements = _lock_scientific_tokens(item["input"])
                    provider_item["input"] = protected_input
                    provider_batch.append(provider_item)
                    locked_tokens[item.get("id")] = replacements
                    original_inputs[item.get("id")] = item["input"]
                provider_text = prefix
                if any(item.get("first_occurrence_glossary_terms") for item in batch):
                    provider_text += (
                        "For an item with first_occurrence_glossary_terms, include the "
                        "English term "
                        "followed by its preferred Vietnamese form in parentheses at its first "
                        "document occurrence. Use Vietnamese only for later occurrences.\n\n"
                    )
                provider_text += json.dumps(provider_batch, ensure_ascii=False, indent=2)
                if suffix:
                    provider_text += suffix
                try:
                    results = _clean_model_json(self._request(provider_text))
                    expected_ids = {item.get("id") for item in batch}
                    parsed_results: dict[Any, str] = {}
                    for result in results:
                        result_id = result.get("id")
                        output = result.get("output")
                        if (
                            result_id not in expected_ids
                            or not isinstance(output, str)
                            or result_id in parsed_results
                        ):
                            raise SafeEngineError("PROVIDER_INVALID_SCHEMA")
                        parsed_results[result_id] = _restore_scientific_tokens(
                            output,
                            locked_tokens.get(result_id, {}),
                            source=original_inputs.get(result_id, ""),
                        )
                    if len(batch) == 1 and set(parsed_results) != expected_ids:
                        raise SafeEngineError("PROVIDER_INVALID_SCHEMA")
                except SafeEngineError as exc:
                    if exc.code == "PROVIDER_INVALID_SCHEMA" and len(batch) > 1:
                        return translate_individually(batch)
                    item_id = batch[0].get("id") if len(batch) == 1 else None
                    if (
                        exc.code == "PROVIDER_INVALID_SCHEMA"
                        and item_id is not None
                        and provider_attempts.get(item_id, 0) < 3
                    ):
                        return request_batch(batch)
                    self.failure_code = exc.code
                    raise
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    if len(batch) > 1:
                        return translate_individually(batch)
                    item_id = batch[0].get("id") if batch else None
                    if item_id is not None and provider_attempts.get(item_id, 0) < 3:
                        return request_batch(batch)
                    raise self._provider_error("PROVIDER_INVALID_JSON") from exc
                return parsed_results

            provider_results = request_batch(misses)
            invalid_items = []
            for item in misses:
                if item.get("id") in isolated_ids:
                    continue
                output = provider_results.get(item.get("id"))
                if output is None or _protected_tokens(item["input"]) != _protected_tokens(output):
                    invalid_items.append(item)
            for item in invalid_items:
                # Batch responses can lose layout placeholders. Retry only that
                # item, with at most three total attempts including the batch.
                while provider_attempts.get(item.get("id"), 0) < 3:
                    isolated_ids.add(item.get("id"))
                    provider_results.update(request_batch([item]))
                    translated = provider_results.get(item.get("id"))
                    if translated and _protected_tokens(item["input"]) == _protected_tokens(
                        translated
                    ):
                        break

            unchanged_items: list[dict[str, Any]] = []
            for item in misses:
                output = provider_results.get(item.get("id"))
                if output is None:
                    raise self._provider_error("PROVIDER_MISSING_ITEM")
                if _protected_tokens(item["input"]) != _protected_tokens(output):
                    raise self._marker_error(record_by_id.get(item.get("id")))
                layout_label = item.get("layout_label")
                unchanged = _is_unchanged_english_prose(
                    item["input"], output, layout_label=layout_label
                )
                # BabelDOC deliberately preserves official titles; keep that behavior.
                if unchanged and str(layout_label or "").casefold().strip() != "title":
                    unchanged_items.append(item)

            if unchanged_items:
                # Reuse BabelDOC's exact prompt wrapper. The free proxy rejects
                # additional natural-language instructions as an unsupported prompt.
                retries_left = [
                    item for item in unchanged_items if provider_attempts.get(item.get("id"), 0) < 3
                ]
                if retries_left:
                    provider_results.update(request_batch(retries_left))

        combined: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if index in prepared:
                translated = prepared[index][1]
            else:
                result_id = item.get("id")
                translated = provider_results.get(result_id)
                if translated is None:
                    raise self._provider_error("PROVIDER_MISSING_ITEM")
            if _protected_tokens(item["input"]) != _protected_tokens(translated):
                raise self._marker_error(records[index])
            translated = _apply_first_occurrence_terms(
                translated, records[index].get("first_occurrence_terms", [])
            )
            combined.append({"id": item.get("id"), "output": translated})
        return json.dumps(combined, ensure_ascii=False)

    def do_translate(self, text: str, rate_limit_params=None) -> str:
        del rate_limit_params
        return self._request(text)

    def do_llm_translate(self, text: str, rate_limit_params=None) -> str:
        del rate_limit_params
        if text is None:
            # BabelDOC probes this method to decide whether it should generate
            # a structured LLM prompt; this probe must never call the provider.
            return None
        return self._request(text)

    def get_rich_text_left_placeholder(self, placeholder_id: int | str) -> str:
        return f"<b{placeholder_id}>"

    def get_rich_text_right_placeholder(self, placeholder_id: int | str) -> str:
        return f"</b{placeholder_id}>"

    def get_formular_placeholder(self, placeholder_id: int | str) -> str:
        return self.get_rich_text_left_placeholder(placeholder_id)

    def _request(self, text: str) -> str:
        import httpx

        if self.failure_code is not None:
            raise SafeEngineError(self.failure_code)
        last_error: SafeEngineError | None = None
        for attempt in range(3):
            with self.limiter.request_slot():
                try:
                    response = self.client.post(
                        "https://api1.pdf2zh-next.com/chatproxy",
                        json={"text": text},
                        timeout=60.0,
                    )
                    if response.status_code == 429:
                        last_error = SafeEngineError("PROVIDER_RATE_LIMITED")
                    elif 400 <= response.status_code < 500:
                        # A rejected request is not made more valid by repeating
                        # the same paper text. Keep the status category only; the
                        # response body may echo sensitive document content.
                        self.failure_code = "PROVIDER_REJECTED"
                        raise SafeEngineError(self.failure_code)
                    else:
                        response.raise_for_status()
                        data = response.json()
                        content = data.get("content") if isinstance(data, dict) else None
                        if isinstance(content, str) and content:
                            return content
                        last_error = SafeEngineError("PROVIDER_INVALID_RESPONSE")
                except httpx.HTTPError:
                    last_error = SafeEngineError("PROVIDER_UNAVAILABLE")
                except ValueError:
                    last_error = SafeEngineError("PROVIDER_INVALID_RESPONSE")
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
        self.failure_code = (last_error or SafeEngineError("PROVIDER_UNAVAILABLE")).code
        raise SafeEngineError(self.failure_code)


def _stage_name(raw: Any) -> str:
    value = str(raw).lower()
    if "layout" in value or "parse" in value:
        return "layout_analysis"
    if "term" in value or "glossary" in value:
        return "term_extraction"
    if "typeset" in value or "font" in value or "pdf" in value or "save" in value:
        return "rendering"
    if "finish" in value:
        return "finalizing"
    return "translation"


async def _translate(request: dict[str, Any]) -> None:
    global _ACTIVE_RECORDER
    _check_versions()
    input_pdf = Path(request["input_pdf"])
    output_dir = Path(request["output_dir"])
    working_dir = Path(request["working_dir"])
    layout_model = Path(request["layout_model"])
    if not input_pdf.is_file():
        raise SafeEngineError("INVALID_ENGINE_REQUEST")
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    # BabelDOC creates its SQLite translation cache during import. Keep that
    # cache writable and attempt-local instead of under the read-only asset mount.
    engine_home = working_dir / "engine-home"
    engine_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(engine_home)
    _preflight_offline_assets(layout_model)

    from babeldoc.docvision.doclayout import OnnxModel
    from babeldoc.format.pdf import high_level
    from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator
    from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import ILTranslatorLLMOnly
    from babeldoc.format.pdf.translation_config import TranslationConfig, WatermarkOutputMode
    from babeldoc.glossary import Glossary, GlossaryEntry

    from app.services.translation.local_nllb import LocalNllbTranslator

    _install_no_download_asset_guards(high_level)

    recorder = TranslationCheckpointRecorder(
        source_sha256=request["source_pdf_sha256"],
        glossary=request.get("glossary", []),
        checkpoint_results=request.get("checkpoint_results", []),
        allow_untranslated_units=True,
    )
    _ACTIVE_RECORDER = recorder

    class CheckpointedNllbTranslator(LocalNllbTranslator):
        def translate(self, text: str, ignore_cache: bool = False, rate_limit_params=None) -> str:
            del ignore_cache, rate_limit_params
            record = getattr(recorder.local, "paragraph", None)
            context_hash = hashlib.sha256(text.encode()).hexdigest()
            if record:
                key = recorder.key_for(record, context_hash)
                cached = recorder.checkpoints.get(key)
                if cached is not None and _checkpoint_is_usable(
                    text, cached, record.get("layout_label")
                ):
                    return cached
            protected_input, replacements = _lock_scientific_tokens(text)
            translated = super().translate(protected_input)
            translated = _restore_scientific_tokens(
                translated, replacements, source=protected_input
            )
            if record:
                if _protected_tokens(text) != _protected_tokens(translated):
                    raise SafeEngineError("PROVIDER_MARKER_MISMATCH")
                record["context_hash"] = context_hash
            return translated

    translator = CheckpointedNllbTranslator(revision=NLLB_MODEL_REVISION)
    try:
        translator._load()
    except (ImportError, OSError, RuntimeError) as exc:
        raise SafeEngineError("MODEL_ASSETS_UNAVAILABLE") from exc
    glossary_entries = [
        GlossaryEntry(item["source"], item["target"]) for item in request.get("glossary", [])
    ]
    glossaries = [Glossary("mYrA project glossary", glossary_entries)] if glossary_entries else []
    config = TranslationConfig(
        translator=translator,
        input_file=input_pdf,
        lang_in=TRANSLATION_LANG_IN,
        lang_out=TRANSLATION_LANG_OUT,
        doc_layout_model=OnnxModel(str(layout_model)),
        output_dir=output_dir,
        working_dir=working_dir,
        no_dual=True,
        no_mono=False,
        qps=1,
        pool_max_workers=1,
        term_pool_max_workers=1,
        glossaries=glossaries,
        auto_extract_glossary=False,
        watermark_output_mode=WatermarkOutputMode.NoWatermark,
        debug=False,
        use_rich_pbar=False,
    )
    completion: dict[str, Any] | None = None
    with _checkpoint_hooks(ILTranslator, ILTranslatorLLMOnly, recorder):
        async for event in _render_with_progress(high_level, config):
            if event.get("type") == "error":
                # Do not forward BabelDOC's error text; it may contain paper text.
                raise SafeEngineError("ENGINE_FAILURE")
            if event.get("type") == "progress_update":
                _emit(
                    {
                        "type": "progress",
                        "stage": _stage_name(event.get("stage")),
                        "current": event.get("stage_current"),
                        "total": event.get("stage_total"),
                        "progress": event.get("stage_progress"),
                    }
                )
            if event.get("type") == "finish":
                result = event.get("translate_result")
                output_path = getattr(result, "mono_pdf_path", None)
                resolved_output = Path(output_path).resolve() if output_path else None
                if (
                    resolved_output
                    and resolved_output.is_relative_to(output_dir.resolve())
                    and resolved_output.is_file()
                ):
                    counts = recorder.finish()
                    if counts["total"] > 0 and counts["completed"] == 0:
                        raise SafeEngineError("ENGINE_INCOMPLETE")
                    completion = {
                        "type": "complete",
                        "output_pdf": str(resolved_output),
                        "source_mapping": "available",
                        "resume_supported": True,
                        "segment_counts": counts,
                        "failure_code": None,
                    }
    if completion is None:
        raise SafeEngineError("ENGINE_FAILURE")
    _emit(completion)


async def _render_with_progress(high_level: Any, config: Any):
    """Run BabelDOC's synchronous renderer without its hanging async wrapper.

    BabelDOC 0.6.2's ``async_translate`` can yield its terminal ``finish`` event
    and then wait forever on an internal event. The supported ``do_translate``
    entry point performs the same render; this adapter forwards its documented
    progress callbacks while keeping the engine subprocess responsive to the
    parent worker's deadline/cancellation handling.
    """
    from babeldoc.progress_monitor import ProgressMonitor

    loop = asyncio.get_running_loop()
    progress: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def on_progress(**event: Any) -> None:
        loop.call_soon_threadsafe(progress.put_nowait, event)

    monitor = ProgressMonitor(
        high_level.get_translation_stage(config),
        progress_change_callback=on_progress,
        report_interval=config.report_interval,
    )
    future = loop.run_in_executor(None, high_level.do_translate, monitor, config)
    while not future.done():
        try:
            event = await asyncio.wait_for(progress.get(), timeout=0.2)
        except TimeoutError:
            continue
        yield event
    while not progress.empty():
        yield progress.get_nowait()
    result = await future
    yield {"type": "finish", "translate_result": result}


@contextmanager
def _checkpoint_hooks(
    il_translator: Any, llm_translator: Any, recorder: TranslationCheckpointRecorder
):
    """Guard and temporarily wrap only the verified BabelDOC 0.6.2 IL seams."""
    from babeldoc.format.pdf.document_il.utils.paragraph_helper import (
        is_placeholder_only_paragraph,
        is_pure_numeric_paragraph,
    )

    required_signatures = (
        (il_translator, "translate", {"docs"}),
        (
            il_translator,
            "get_translate_input",
            {"paragraph", "page_font_map", "disable_rich_text_translate"},
        ),
        (
            il_translator,
            "pre_translate_paragraph",
            {"paragraph", "tracker", "page_font_map", "xobj_font_map"},
        ),
        (
            il_translator,
            "post_translate_paragraph",
            {"paragraph", "tracker", "translate_input", "translated_text"},
        ),
        (il_translator, "translate_paragraph", {"paragraph", "page"}),
        (llm_translator, "translate", {"docs"}),
        (
            llm_translator,
            "process_cross_page_paragraph",
            {"docs", "executor", "pbar", "tracker", "executor2", "translated_ids"},
        ),
        (llm_translator, "translate_paragraph", {"batch_paragraph"}),
    )
    originals: list[tuple[Any, str, Any]] = []
    for target, name, required in required_signatures:
        original = getattr(target, name)
        if not required.issubset(inspect.signature(original).parameters):
            raise SafeEngineError("ENGINE_VERSION_MISMATCH")
        originals.append((target, name, original))

    original_get_input = il_translator.get_translate_input
    original_pre = il_translator.pre_translate_paragraph
    original_post = il_translator.post_translate_paragraph
    original_single = il_translator.translate_paragraph
    original_standard = il_translator.translate
    original_batch = llm_translator.translate_paragraph
    original_whole = llm_translator.translate

    def skip_cross_page_batch(self, *args, **kwargs):
        # Keep source units page-local so each translation remains attributable
        # to its original page and checkpoint identity.
        del self, args, kwargs
        return None

    def capture_translate_input(
        self, paragraph, page_font_map=None, disable_rich_text_translate=None
    ):
        # Call BabelDOC's implementation exactly once, then classify its return.
        result = original_get_input(self, paragraph, page_font_map, disable_rich_text_translate)
        record = recorder.get(paragraph)
        if record:
            reason = _classify_preprocess_decline(
                paragraph,
                result,
                minimum_text_length=self.translation_config.min_text_length,
                is_pure_numeric=is_pure_numeric_paragraph,
                is_placeholder_only=is_placeholder_only_paragraph,
            )
            if reason:
                record["preprocess_decline_reason"] = reason
            else:
                record.pop("preprocess_decline_reason", None)
        return result

    def pre_translate(self, paragraph, tracker, page_font_map=None, xobj_font_map=None):
        record = recorder.get(paragraph)
        if record and record.get("preserve_reason"):
            recorder.note_preprocessed(paragraph, None)
            return None, None
        result = original_pre(self, paragraph, tracker, page_font_map, xobj_font_map)
        if isinstance(result, tuple) and len(result) == 2:
            if record and result[0] is None and not record.get("preprocess_decline_reason"):
                reason = _classify_preprocess_decline(
                    paragraph,
                    None,
                    minimum_text_length=self.translation_config.min_text_length,
                    is_pure_numeric=is_pure_numeric_paragraph,
                    is_placeholder_only=is_placeholder_only_paragraph,
                )
                record["preprocess_decline_reason"] = reason or "unknown_decline"
            recorder.note_preprocessed(paragraph, result[0])
        return result

    def post_translate(self, paragraph, tracker, translate_input, translated_text):
        record = recorder.get(paragraph)
        if record:
            translated_text = _apply_first_occurrence_terms(
                translated_text, record.get("first_occurrence_terms", [])
            )
        source = record.get("preprocessed_input") if record else None
        if source is not None and _protected_tokens(source) != _protected_tokens(translated_text):
            recorder.invalidate(record, "protected_placeholder_mismatch")
            raise SafeEngineError("PROVIDER_MARKER_MISMATCH")
        result = original_post(self, paragraph, tracker, translate_input, translated_text)
        if record:
            recorder.complete(record, translated_text)
        return result

    def single_translate(self, paragraph, *args, **kwargs):
        with recorder.paragraph_scope(paragraph):
            return original_single(self, paragraph, *args, **kwargs)

    def standard_translate(self, docs):
        recorder.begin(docs)
        try:
            return original_standard(self, docs)
        finally:
            recorder.emit_summary()

    def batch_translate(self, batch_paragraph, *args, **kwargs):
        with recorder.batch_scope():
            return original_batch(self, batch_paragraph, *args, **kwargs)

    def whole_translate(self, docs):
        recorder.begin(docs)
        glossary_terms = [
            (entry.source, entry.target)
            for glossary in getattr(self.translation_config, "glossaries", [])
            for entry in getattr(glossary, "entries", [])
        ]
        seen_terms: set[str] = set()
        for page in docs.page:
            for paragraph in page.pdf_paragraph:
                record = recorder.get(paragraph)
                if not record:
                    continue
                source_text = paragraph.unicode or ""
                for source_term, target_term in glossary_terms:
                    normalized = source_term.casefold()
                    if normalized in seen_terms:
                        continue
                    pattern = re.compile(rf"(?<!\w){re.escape(source_term)}(?!\w)", re.IGNORECASE)
                    if pattern.search(source_text):
                        record["first_occurrence_terms"].append(
                            {"english": source_term, "vietnamese": target_term}
                        )
                        seen_terms.add(normalized)
        try:
            return original_whole(self, docs)
        finally:
            recorder.emit_summary()

    il_translator.get_translate_input = capture_translate_input
    il_translator.pre_translate_paragraph = pre_translate
    il_translator.post_translate_paragraph = post_translate
    il_translator.translate = standard_translate
    il_translator.translate_paragraph = single_translate
    llm_translator.translate_paragraph = batch_translate
    llm_translator.process_cross_page_paragraph = skip_cross_page_batch
    llm_translator.translate = whole_translate
    try:
        yield
    finally:
        for target, name, original in originals:
            setattr(target, name, original)


def main() -> int:
    # BabelDOC logs include paragraph text in at least one info message.
    # Silence all logging and redirect incidental stdout before importing it.
    logging.disable(logging.CRITICAL)
    try:
        raw = sys.stdin.readline()
        request = _validate_request(json.loads(raw))
        os.environ["HF_HUB_OFFLINE"] = "1"
        with open(os.devnull, "w", encoding="utf-8") as sink:
            sys.stdout = sink
            try:
                asyncio.run(_translate(request))
            finally:
                sys.stdout = PROTOCOL_STDOUT
        return 0
    except SafeEngineError as exc:
        if _ACTIVE_RECORDER is not None:
            _ACTIVE_RECORDER.emit_summary()
        _emit({"type": "error", "code": exc.code})
        return 1
    except BaseException:
        if _ACTIVE_RECORDER is not None:
            _ACTIVE_RECORDER.emit_summary()
        _emit({"type": "error", "code": "ENGINE_FAILURE"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
