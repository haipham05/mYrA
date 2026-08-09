"""Subprocess boundary for the separately installed PDF translation runtime."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
MAX_JOB_SECONDS = 30 * 60
_SAFE_PROGRESS_STAGES = {
    "layout_analysis",
    "term_extraction",
    "translation",
    "rendering",
    "finalizing",
}


class TranslationEngineError(RuntimeError):
    """A safe, classified translation-runtime failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]
CheckpointCallback = Callable[[dict[str, Any]], Awaitable[None]]


def _minimal_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Pass only runtime paths and locale settings, never the application's secrets."""
    allowed = {
        "PATH",
        "HOME",
        "TMPDIR",
        "LANG",
        "LC_ALL",
        "HF_HOME",
        "MYRA_TRANSLATION_ASSET_DIR",
    }
    source = dict(os.environ)
    if extra:
        source.update(extra)
    return {key: value for key, value in source.items() if key in allowed}


def _safe_event(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("type"), str):
        raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
    event_type = value["type"]
    if event_type == "progress":
        stage = value.get("stage")
        if stage not in _SAFE_PROGRESS_STAGES:
            stage = "translation"
        current = value.get("current")
        total = value.get("total")
        progress = value.get("progress")
        return {
            "type": "progress",
            "stage": stage,
            "current": current if isinstance(current, int) and current >= 0 else None,
            "total": total if isinstance(total, int) and total >= 0 else None,
            "progress": (
                max(0.0, min(100.0, float(progress)))
                if isinstance(progress, (int, float))
                else None
            ),
        }
    if event_type == "complete":
        output_pdf = value.get("output_pdf")
        if not isinstance(output_pdf, str) or not output_pdf:
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        counts = value.get("segment_counts")
        if not isinstance(counts, dict) or not all(
            isinstance(counts.get(name), int) and counts[name] >= 0
            for name in ("total", "completed", "skipped", "failed")
        ):
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        if counts["failed"] > 0 or (counts["total"] > 0 and counts["completed"] == 0):
            raise TranslationEngineError("ENGINE_INCOMPLETE")
        failure_code = value.get("failure_code")
        if failure_code not in {
            None,
            "PROVIDER_RATE_LIMITED",
            "PROVIDER_UNAVAILABLE",
        }:
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        return {
            "type": "complete",
            "output_pdf": output_pdf,
            "source_mapping": value.get("source_mapping", "unsupported"),
            "resume_supported": value.get("resume_supported") is True,
            "segment_counts": counts,
            "failure_code": failure_code,
        }
    if event_type == "checkpoint":
        segment = value.get("segment")
        if not isinstance(segment, dict):
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        text_fields = (
            "segment_key",
            "source_quote",
            "source_sha256",
            "translated_text",
            "translated_sha256",
            "status",
        )
        integer_fields = ("page_number", "ordinal", "source_start", "source_end")
        if not all(isinstance(segment.get(key), str) for key in text_fields):
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        if not all(isinstance(segment.get(key), int) for key in integer_fields):
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        if max(len(segment["source_quote"]), len(segment["translated_text"])) > 65_536:
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        return {"type": "checkpoint", "segment": segment}
    if event_type == "segment_summary":
        counts = {name: value.get(name) for name in ("total", "completed", "skipped", "failed")}
        if not all(isinstance(item, int) and item >= 0 for item in counts.values()):
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        reasons = value.get("failure_reasons", {})
        if not isinstance(reasons, dict) or not all(
            key in {"invalid", "unchanged_prose", "oversized", "untranslated"}
            and isinstance(count, int)
            and count >= 0
            for key, count in reasons.items()
        ):
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        failure_causes = value.get("failure_causes", {})
        allowed_failure_causes = {
            "protected_placeholder_mismatch",
            "unchanged_prose",
            "unit_too_large",
            "preprocessing_not_selected",
            "missing_validated_checkpoint",
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
        if not isinstance(failure_causes, dict) or not all(
            key in allowed_failure_causes and isinstance(count, int) and count >= 0
            for key, count in failure_causes.items()
        ):
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        skip_reasons = value.get("skip_reasons", {})
        if not isinstance(skip_reasons, dict) or not all(
            key
            in {
                "empty",
                "below_engine_minimum",
                "protected_scientific_content",
                "placeholder_only",
                "numeric_or_symbol_only",
                "preserved_vertical_layout_content",
                "author_contact_metadata",
                "preserved_fallback_layout_content",
                "preserved_short_layout_label",
            }
            and isinstance(count, int)
            and count >= 0
            for key, count in skip_reasons.items()
        ):
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        units = value.get("failure_units", [])
        if not isinstance(units, list) or any(
            not isinstance(unit, dict)
            or not isinstance(unit.get("page_number"), int)
            or not isinstance(unit.get("ordinal"), int)
            or not isinstance(unit.get("source_chars"), int)
            or unit.get("status") not in {"invalid", "unchanged_prose", "oversized", "untranslated"}
            or unit.get("failure_reason") not in allowed_failure_causes | {None}
            or (unit.get("layout_label") is not None and not isinstance(unit["layout_label"], str))
            for unit in units
        ):
            raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")
        return {
            "type": "segment_summary",
            **counts,
            "failure_reasons": reasons,
            "failure_causes": failure_causes,
            "skip_reasons": skip_reasons,
            "failure_units": [
                {
                    "page_number": unit["page_number"],
                    "ordinal": unit["ordinal"],
                    "status": unit["status"],
                    "failure_reason": unit.get("failure_reason"),
                    "source_chars": unit["source_chars"],
                    "layout_label": (
                        unit["layout_label"][:100]
                        if isinstance(unit.get("layout_label"), str)
                        else None
                    ),
                }
                for unit in units
            ],
        }
    if event_type == "error":
        code = value.get("code")
        allowed_codes = {
            "ENGINE_FAILURE",
            "ENGINE_VERSION_MISMATCH",
            "LAYOUT_MODEL_UNAVAILABLE",
            "FONT_ASSETS_UNAVAILABLE",
            "PROVIDER_RATE_LIMITED",
            "PROVIDER_UNAVAILABLE",
            "ENGINE_INCOMPLETE",
            "INVALID_ENGINE_REQUEST",
        }
        raise TranslationEngineError(
            code if isinstance(code, str) and code in allowed_codes else "ENGINE_FAILURE"
        )
    raise TranslationEngineError("ENGINE_PROTOCOL_ERROR")


class TranslationEngineProcess:
    """Run one engine job with a strict JSON-lines protocol and hard deadline."""

    def __init__(
        self,
        *,
        python: str | Path,
        runner: str | Path,
        env: Mapping[str, str] | None = None,
        timeout_seconds: int = MAX_JOB_SECONDS,
    ) -> None:
        if not 1 <= timeout_seconds <= MAX_JOB_SECONDS:
            raise ValueError("timeout_seconds must be between 1 and 1800")
        self.python = str(python)
        self.runner = str(runner)
        self.env = _minimal_environment(env)
        self.timeout_seconds = timeout_seconds

    async def run(
        self,
        request: Mapping[str, Any],
        *,
        on_progress: ProgressCallback | None = None,
        on_checkpoint: CheckpointCallback | None = None,
    ) -> dict[str, Any]:
        payload = {"protocol_version": PROTOCOL_VERSION, **request}
        process = await asyncio.create_subprocess_exec(
            self.python,
            self.runner,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=self.env,
            start_new_session=True,
        )
        assert process.stdin is not None
        assert process.stdout is not None
        process.stdin.write(json.dumps(payload, separators=(",", ":")).encode() + b"\n")
        await process.stdin.drain()
        process.stdin.close()

        async def read_protocol() -> dict[str, Any]:
            completion: dict[str, Any] | None = None
            async for line in process.stdout:
                try:
                    raw = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    raise TranslationEngineError("ENGINE_PROTOCOL_ERROR") from exc
                event = _safe_event(raw)
                if event["type"] == "progress":
                    if on_progress:
                        await on_progress(event)
                elif event["type"] == "checkpoint":
                    if on_checkpoint:
                        await on_checkpoint(event["segment"])
                elif event["type"] == "segment_summary":
                    if on_progress:
                        await on_progress(event)
                else:
                    completion = event
            returncode = await process.wait()
            if returncode != 0 or completion is None:
                raise TranslationEngineError("ENGINE_FAILURE")
            return completion

        try:
            return await asyncio.wait_for(read_protocol(), timeout=self.timeout_seconds)
        except TimeoutError as exc:
            await self._terminate(process)
            raise TranslationEngineError("ENGINE_DEADLINE_EXCEEDED") from exc
        except asyncio.CancelledError:
            await self._terminate(process)
            raise
        except BaseException:
            await self._terminate(process)
            raise

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=2)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
