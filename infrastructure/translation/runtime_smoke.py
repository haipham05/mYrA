"""Exercise the pinned renderer on a fixture with every provider call mocked.

Run this only in the translation image with the verified asset cache mounted;
use ``docker run --network none`` so a regression cannot call the real service.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import fitz

RUNNER_PATH = Path("/app/app/services/translation/engine_runner.py")
LAYOUT_PATH = Path(os.environ.get("MYRA_TRANSLATION_LAYOUT_MODEL", ""))


class FakeResponse:
    status_code = 200

    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {"content": self.content}


class FakeClient:
    calls = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:
        del url, timeout
        type(self).calls += 1
        prompt = json["text"]

        def translate(source: str) -> str:
            # Keep a stable length for tiny labels/numbers that the upstream
            # engine rejects when a test prefix makes them expand too much.
            return source if len(source.strip()) <= 24 else f"Bản dịch: {source}"

        parsed = runner._parse_batch_prompt(prompt)
        if parsed:
            outputs = [
                {"id": item.get("id"), "output": translate(item["input"])} for item in parsed[2]
            ]
            content = __import__("json").dumps(outputs, ensure_ascii=False)
        else:
            content = translate(prompt)
        return FakeResponse(content)

    def close(self) -> None:
        return None


spec = importlib.util.spec_from_file_location("myra_translation_runtime", RUNNER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load isolated translation runner")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
_original_emit = runner._emit


def report_emitted_event(event: dict[str, Any]) -> None:
    if event.get("type") == "progress":
        print(f"renderer stage: {event.get('stage')}", flush=True)
    elif event.get("type") == "segment_summary":
        print(
            json.dumps(
                {
                    key: event.get(key)
                    for key in (
                        "total",
                        "completed",
                        "skipped",
                        "failed",
                        "failure_reasons",
                        "failure_units",
                    )
                },
                sort_keys=True,
            ),
            flush=True,
        )
    elif event.get("type") != "checkpoint":
        print(f"engine emitted: {event.get('type')}", flush=True)
    _original_emit(event)


runner._emit = report_emitted_event


async def run_once(
    root: Path, fixture_path: Path, checkpoint_results: list[dict[str, str]]
) -> tuple[Path, list[dict[str, Any]]]:
    attempt = root / f"attempt-{len(checkpoint_results)}-{FakeClient.calls}"
    output = attempt / "output"
    work = attempt / "work"
    output.mkdir(parents=True)
    work.mkdir()
    engine_home = work / "engine-home"
    engine_home.mkdir()
    events = io.StringIO()
    original_stdout = runner.PROTOCOL_STDOUT
    runner.PROTOCOL_STDOUT = events
    request = {
        "protocol_version": runner.PROTOCOL_VERSION,
        "input_pdf": str(fixture_path),
        "output_dir": str(output),
        "working_dir": str(work),
        "layout_model": str(LAYOUT_PATH),
        "lang_in": "English",
        "lang_out": "Vietnamese",
        "source_pdf_sha256": runner.hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "glossary": [],
        "checkpoint_results": checkpoint_results,
    }
    try:
        os.environ["HOME"] = str(engine_home)
        print(f"renderer run started: {attempt.name}", flush=True)
        await asyncio.wait_for(runner._translate(request), timeout=1800)
        print(f"renderer run returned: {attempt.name}", flush=True)
    finally:
        runner.PROTOCOL_STDOUT = original_stdout
    parsed = [json.loads(line) for line in events.getvalue().splitlines() if line.strip()]
    completion = next((event for event in parsed if event.get("type") == "complete"), None)
    if completion is None:
        raise RuntimeError("Pinned renderer did not emit completion")
    return Path(completion["output_pdf"]), parsed


async def main() -> None:
    logging.disable(logging.CRITICAL)
    import httpx

    httpx.Client = FakeClient
    with tempfile.TemporaryDirectory(prefix="myra-translation-smoke-") as temp:
        root = Path(temp)
        fixture_path = root / "scientific-two-column.pdf"
        original_document = fitz.open()
        paragraph = (
            "Attention-based scientific models compare representations across positions. "
            "The encoder computes stable features while preserving the input sequence. "
            "We report reproducible measurements and discuss the limitations of the method. "
        ) * 3
        for page_index in range(1):
            page = original_document.new_page(width=612, height=792)
            page.insert_text(
                (52, 32), f"Scientific Layout Fixture - Page {page_index + 1}", fontsize=14
            )
            for x0 in (52, 318):
                page.insert_textbox(
                    fitz.Rect(x0, 52, x0 + 238, 445),
                    paragraph,
                    fontsize=9,
                    lineheight=1.2,
                )
            page.insert_text((52, 474), "Equation: E = mc2; Q = K W_Q", fontsize=10)
            page.draw_line((60, 530), (285, 530), color=(0, 0, 0), width=1)
            page.draw_line((60, 530), (60, 640), color=(0, 0, 0), width=1)
            for bar_x, bar_height in ((90, 36), (130, 82), (170, 58), (210, 102)):
                page.draw_rect(fitz.Rect(bar_x, 530 - bar_height, bar_x + 18, 530), color=(0, 0, 0))
            page.insert_text(
                (62, 662), "Figure 1. Measured attention response by input position.", fontsize=9
            )
            for row in range(3):
                for column in range(3):
                    x, y = 345 + column * 70, 565 + row * 24
                    page.draw_rect(fitz.Rect(x, y, x + 68, y + 22), color=(0, 0, 0), width=0.5)
                    if row == 0:
                        page.insert_text((x + 4, y + 15), f"Metric {column + 1}", fontsize=7)
            page.insert_text((345, 652), "Table 1. Example scientific measurements.", fontsize=8)
        original_document.save(fixture_path)
        original_document.close()
        original = fitz.open(fixture_path)
        first_output, first_events = await run_once(root, fixture_path, [])
        checkpoints = [
            {
                "segment_key": event["segment"]["segment_key"],
                "translated_text": event["segment"]["translated_text"],
                "translated_sha256": event["segment"]["translated_sha256"],
            }
            for event in first_events
            if event.get("type") == "checkpoint"
        ]
        first_calls = FakeClient.calls
        if not checkpoints or first_calls == 0:
            raise RuntimeError("Smoke did not exercise translation and checkpoint output")

        before_resume_calls = FakeClient.calls
        second_output, second_events = await run_once(root, fixture_path, checkpoints)
        if FakeClient.calls != before_resume_calls:
            raise RuntimeError("Resume retransmitted completed segments")

        first_pdf = fitz.open(first_output)
        second_pdf = fitz.open(second_output)
        first_text = "\n".join(page.get_text() for page in first_pdf)
        normalized_text = first_text.replace("\x03", " ")
        if first_pdf.page_count != original.page_count or "Bản dịch" not in normalized_text:
            raise RuntimeError(
                "Rendered PDF did not preserve pages and selectable Vietnamese text; "
                f"pages={first_pdf.page_count}, text_sample={normalized_text[:240]!r}"
            )

        original_drawings = sum(len(page.get_drawings()) for page in original)
        output_drawings = sum(len(page.get_drawings()) for page in first_pdf)
        if output_drawings < original_drawings:
            raise RuntimeError("Rendered PDF dropped vector figure/table content")
        completion = next(event for event in first_events if event.get("type") == "complete")
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "fixture_pages": original.page_count,
                    "output_pages": first_pdf.page_count,
                    "source_drawings": original_drawings,
                    "output_drawings": output_drawings,
                    "checkpoints": len(checkpoints),
                    "provider_calls_first_run": first_calls,
                    "provider_calls_during_resume": FakeClient.calls - before_resume_calls,
                    "resume_completion": any(
                        event.get("type") == "complete" for event in second_events
                    ),
                    "segments": completion["segment_counts"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        original.close()
        first_pdf.close()
        second_pdf.close()


if __name__ == "__main__":
    asyncio.run(main())
