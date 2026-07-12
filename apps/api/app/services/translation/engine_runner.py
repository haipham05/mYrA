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
from contextlib import contextmanager
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
PDF2ZH_VERSION = "2.9.0"
BABELDOC_VERSION = "0.6.2"
LAYOUT_MODEL_SHA3_256 = "60be061226930524958b5465c8c04af3d7c03bcb0beb66454f5da9f792e3cf2a"
PROTOCOL_STDOUT = sys.stdout
_PROTECTED_TOKEN = re.compile(
    r"\{v\d+\}|\{[A-Za-z][\w.-]*\}|%[sd]|\[\[.*?\]\]|%%.*?%%|</?style\b[^>]*>|</?b\d+>"
)


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


class TranslationCheckpointRecorder:
    """Build page/quote identities and persistable checkpoints from BabelDOC IL hooks."""

    def __init__(
        self,
        *,
        source_sha256: str,
        glossary: list[dict[str, str]],
        checkpoint_results: list[dict[str, str]],
    ) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
            raise SafeEngineError("INVALID_ENGINE_REQUEST")
        self.source_sha256 = source_sha256
        self.glossary = glossary
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

    def begin(self, docs: Any) -> None:
        global_ordinal = 0
        for page_index, page in enumerate(docs.page, start=1):
            page_number = page.page_number or page_index
            offset = 0
            for page_ordinal, paragraph in enumerate(page.pdf_paragraph):
                quote = paragraph.unicode or ""
                record = {
                    "page_number": int(page_number),
                    "ordinal": global_ordinal,
                    "page_ordinal": page_ordinal,
                    "source_quote": quote,
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
                if record["layout_label"] == "abandon" and len(quote.strip()) <= 3:
                    record["status"] = "skipped"
                    self.skipped += 1
                identity = {
                    "source_pdf_sha256": self.source_sha256,
                    "page_number": record["page_number"],
                    "ordinal": global_ordinal,
                    "page_ordinal": page_ordinal,
                    "source_sha256": record["source_sha256"],
                    "glossary_sha256": self.glossary_sha256,
                    "pdf2zh_version": PDF2ZH_VERSION,
                    "babeldoc_version": BABELDOC_VERSION,
                }
                record["base_key"] = hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest()
                self.records[id(paragraph)] = record
                offset += len(quote) + 1
                global_ordinal += 1

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
            if record["status"] != "skipped":
                record["status"] = "skipped"
                self.skipped += 1
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
        source = record["source_quote"]
        validation_source = record.get("preprocessed_input") or source
        if _protected_tokens(validation_source) != _protected_tokens(translated_text):
            record["status"] = "invalid"
            self.failed += 1
            return
        unchanged_prose = _is_unchanged_english_prose(
            validation_source, translated_text, layout_label=record.get("layout_label")
        )
        preserve_official_title = record.get("layout_label") == "title" and unchanged_prose
        if unchanged_prose and not preserve_official_title:
            record["status"] = "unchanged_prose"
            self.failed += 1
            return
        if len(source) > 65_536 or len(translated_text) > 65_536:
            record["status"] = "oversized"
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

    def finish(self) -> dict[str, Any]:
        for record in self.records.values():
            if record["status"] == "pending":
                record["status"] = "untranslated"
                self.failed += 1
        return {
            "total": len(self.records),
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": self.failed,
        }

    def failure_reasons(self) -> dict[str, int]:
        reasons: dict[str, int] = {}
        for record in self.records.values():
            status = record["status"]
            if status in {"invalid", "unchanged_prose", "oversized", "untranslated"}:
                reasons[status] = reasons.get(status, 0) + 1
        return reasons

    def failure_units(self) -> list[dict[str, Any]]:
        return [
            {
                "page_number": record["page_number"],
                "ordinal": record["ordinal"],
                "status": record["status"],
                "source_chars": len(record["source_quote"]),
                "layout_label": record["layout_label"],
            }
            for record in self.records.values()
            if record["status"] in {"invalid", "unchanged_prose", "oversized", "untranslated"}
        ]


def _protected_tokens(text: str) -> tuple[str, ...]:
    return tuple(_PROTECTED_TOKEN.findall(text))


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
        raise SafeEngineError("PROVIDER_UNAVAILABLE")
    return parsed


def _emit(event: dict[str, Any]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(event, separators=(",", ":")) + "\n")
    PROTOCOL_STDOUT.flush()


def _validate_request(request: Any) -> dict[str, Any]:
    if (
        not isinstance(request, dict)
        or request.get("protocol_version") != PROTOCOL_VERSION
        or request.get("lang_in") != "English"
        or request.get("lang_out") != "Vietnamese"
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

    def translate(self, text: str, ignore_cache: bool = False, rate_limit_params=None) -> str:
        del ignore_cache, rate_limit_params
        self.translate_call_count += 1
        record = getattr(self.recorder.local, "paragraph", None)
        if record:
            key = self.recorder.key_for(record, hashlib.sha256(text.encode()).hexdigest())
            cached = self.recorder.checkpoints.get(key)
            if cached is not None and _protected_tokens(text) == _protected_tokens(cached):
                return cached
        result = self._request(text)
        if record:
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
                if cached is not None and _protected_tokens(source) == _protected_tokens(cached):
                    return _apply_first_occurrence_terms(
                        cached, record.get("first_occurrence_terms", [])
                    )
                record["context_hash"] = context_hash
            return self._request(text)

        prefix, suffix, items = parsed
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
            if isinstance(cached, str) and _protected_tokens(source_input) == _protected_tokens(
                cached
            ):
                prepared[index] = (item, cached)
            else:
                misses.append(item)

        provider_results: dict[Any, str] = {}
        if misses:
            provider_text = prefix
            if any(item.get("first_occurrence_glossary_terms") for item in misses):
                provider_text += (
                    "For an item with first_occurrence_glossary_terms, include the English term "
                    "followed by its preferred Vietnamese form in parentheses at its first "
                    "document occurrence. Use Vietnamese only for later occurrences.\n\n"
                )
            provider_text += json.dumps(misses, ensure_ascii=False, indent=2)
            if suffix:
                provider_text += suffix
            for result in _clean_model_json(self._request(provider_text)):
                result_id = result.get("id")
                output = result.get("output")
                if not isinstance(output, str):
                    raise SafeEngineError("PROVIDER_UNAVAILABLE")
                provider_results[result_id] = output

        combined: list[dict[str, Any]] = []
        for index, item in enumerate(items):
            if index in prepared:
                translated = prepared[index][1]
            else:
                result_id = item.get("id")
                translated = provider_results.get(result_id)
                if translated is None:
                    raise SafeEngineError("PROVIDER_UNAVAILABLE")
            if _protected_tokens(item["input"]) != _protected_tokens(translated):
                raise SafeEngineError("PROVIDER_UNAVAILABLE")
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
                    else:
                        response.raise_for_status()
                        data = response.json()
                        content = data.get("content") if isinstance(data, dict) else None
                        if isinstance(content, str) and content:
                            return content
                        last_error = SafeEngineError("PROVIDER_UNAVAILABLE")
                except (httpx.HTTPError, ValueError):
                    last_error = SafeEngineError("PROVIDER_UNAVAILABLE")
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
    _check_versions()
    input_pdf = Path(request["input_pdf"])
    output_dir = Path(request["output_dir"])
    working_dir = Path(request["working_dir"])
    layout_model = Path(request["layout_model"])
    if not input_pdf.is_file():
        raise SafeEngineError("INVALID_ENGINE_REQUEST")
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir.mkdir(parents=True, exist_ok=True)
    _preflight_offline_assets(layout_model)

    from babeldoc.docvision.doclayout import OnnxModel
    from babeldoc.format.pdf import high_level
    from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator
    from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import ILTranslatorLLMOnly
    from babeldoc.format.pdf.translation_config import TranslationConfig
    from babeldoc.glossary import Glossary, GlossaryEntry

    _install_no_download_asset_guards(high_level)

    recorder = TranslationCheckpointRecorder(
        source_sha256=request["source_pdf_sha256"],
        glossary=request.get("glossary", []),
        checkpoint_results=request.get("checkpoint_results", []),
    )
    limiter = SharedRateLimiter(2)
    translator = SiliconFlowFreeTranslator("English", "Vietnamese", limiter, recorder)
    glossary_entries = [
        GlossaryEntry(item["source"], item["target"]) for item in request.get("glossary", [])
    ]
    glossaries = [Glossary("mYrA project glossary", glossary_entries)] if glossary_entries else []
    config = TranslationConfig(
        translator=translator,
        input_file=input_pdf,
        lang_in="English",
        lang_out="Vietnamese",
        doc_layout_model=OnnxModel(str(layout_model)),
        output_dir=output_dir,
        working_dir=working_dir,
        no_dual=True,
        no_mono=False,
        qps=2,
        pool_max_workers=2,
        term_pool_max_workers=2,
        glossaries=glossaries,
        auto_extract_glossary=False,
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
                    if translator.failure_code:
                        raise SafeEngineError(translator.failure_code)
                    if counts["failed"] > 0 or (counts["total"] > 0 and counts["completed"] == 0):
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
    required_signatures = (
        (il_translator, "translate", {"docs"}),
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
        (llm_translator, "translate_paragraph", {"batch_paragraph"}),
    )
    originals: list[tuple[Any, str, Any]] = []
    for target, name, required in required_signatures:
        original = getattr(target, name)
        if not required.issubset(inspect.signature(original).parameters):
            raise SafeEngineError("ENGINE_VERSION_MISMATCH")
        originals.append((target, name, original))

    original_pre = il_translator.pre_translate_paragraph
    original_post = il_translator.post_translate_paragraph
    original_single = il_translator.translate_paragraph
    original_batch = llm_translator.translate_paragraph
    original_whole = llm_translator.translate

    def pre_translate(self, paragraph, tracker, page_font_map=None, xobj_font_map=None):
        result = original_pre(self, paragraph, tracker, page_font_map, xobj_font_map)
        if isinstance(result, tuple) and len(result) == 2:
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
            raise SafeEngineError("ENGINE_FAILURE")
        result = original_post(self, paragraph, tracker, translate_input, translated_text)
        if record:
            recorder.complete(record, translated_text)
        return result

    def single_translate(self, paragraph, *args, **kwargs):
        with recorder.paragraph_scope(paragraph):
            return original_single(self, paragraph, *args, **kwargs)

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
            counts = recorder.finish()
            _emit(
                {
                    "type": "segment_summary",
                    **counts,
                    "failure_reasons": recorder.failure_reasons(),
                    "failure_units": recorder.failure_units(),
                }
            )

    il_translator.pre_translate_paragraph = pre_translate
    il_translator.post_translate_paragraph = post_translate
    il_translator.translate_paragraph = single_translate
    llm_translator.translate_paragraph = batch_translate
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
        _emit({"type": "error", "code": exc.code})
        return 1
    except BaseException:
        _emit({"type": "error", "code": "ENGINE_FAILURE"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
