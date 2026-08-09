from __future__ import annotations

import asyncio
import hashlib
import io

import httpx
import pytest
from PIL import Image

from app.config import Settings
from app.schemas.vision import VisualAnalysis
from app.services.cache import JsonCache
from app.services.vision import (
    DeepSeekVisionProvider,
    StructuredVisionResult,
    VisionProviderError,
    VisionRequest,
    VisualSourceReference,
    _visual_cache_key,
    analyze_figure_cached,
)


def _png_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), color="white").save(output, format="PNG")
    return output.getvalue()


class _Budget:
    def __init__(self) -> None:
        self.calls = []
        self.settlements = []
        self.unknown = []

    def reserve(self, **kwargs):
        self.calls.append(kwargs)
        return type("Reservation", (), {"reservation_id": "reservation"})()

    def settle(self, reservation_id, **kwargs):
        self.settlements.append((reservation_id, kwargs))

    def mark_unknown(self, reservation_id):
        self.unknown.append(reservation_id)


class _Response:
    def __init__(self, data: dict):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class _Client:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.payload = None
        self.headers = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def post(self, _url, *, headers, json):
        self.headers = headers
        self.payload = json
        if self.error:
            raise self.error
        return self.response


def _provider(client, budget=None):
    return DeepSeekVisionProvider(
        settings=Settings(deepseek_api_key="test-key"),
        budget_manager=budget,
        http_client_factory=lambda **_kwargs: client,
    )


class _Redis:
    def __init__(self):
        self.values = {}

    def info(self, _section):
        return {"evicted_keys": 0}

    def get(self, key):
        return self.values.get(key)

    def set(self, key, value, *, ex):
        self.values[key] = value
        return True


class _Vision:
    def __init__(self):
        self.calls = 0

    async def analyze_structured(self, request):
        self.calls += 1
        return StructuredVisionResult(
            generation=type("Generation", (), {"content": "{}", "usage": None})(),
            analysis=VisualAnalysis(
                observations=[{"statement": f"Observed {self.calls}"}],
                interpretation="A descriptive result.",
            ),
        )


def _source_metadata(image: bytes):
    return {
        "project_id": "7c19c7db-a9ef-4e1e-8b24-768c57a2ee20",
        "paper_id": "26f06423-7fd5-40ee-9796-a7b8c207c51e",
        "document_sha256": "a" * 64,
        "page_number": 4,
        "crop_sha256": hashlib.sha256(image).hexdigest(),
        "crop_box_normalized_top_left": {"left": 0.1, "top": 0.2, "right": 0.8, "bottom": 0.9},
        "caption": "Figure 2. Example.",
        "text_citation": False,
    }


def test_image_request_uses_inline_image_and_reports_provider_usage() -> None:
    client = _Client(
        _Response(
            {
                "id": "response-1",
                "model": "deepseek-flash",
                "choices": [{"message": {"content": "A line rises over the x-axis."}}],
                "usage": {"prompt_tokens": 130, "completion_tokens": 15, "total_tokens": 145},
            }
        )
    )
    budget = _Budget()

    result = asyncio.run(
        _provider(client, budget).analyze(
            VisionRequest(question="Describe the trend.", image_bytes=_png_bytes())
        )
    )

    assert result.content == "A line rises over the x-axis."
    assert result.usage.prompt_tokens == 130
    assert result.reported_model == "deepseek-flash"
    assert client.payload["model"] == "deepseek-flash"
    assert client.payload["thinking"] == {"type": "disabled"}
    image_part = client.payload["messages"][0]["content"][1]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert "test-key" not in str(client.payload)
    assert budget.calls[0]["estimated_input_tokens"] <= 1024 + 4000 * 2
    assert budget.settlements[0][1]["prompt_tokens"] == 130
    assert budget.unknown == []


def test_image_request_reserves_before_sending_and_keeps_unknown_on_transport_error() -> None:
    budget = _Budget()
    provider = _provider(_Client(error=httpx.ConnectError("secret transport detail")), budget)

    with pytest.raises(VisionProviderError) as error:
        asyncio.run(
            provider.analyze(VisionRequest(question="What is shown?", image_bytes=_png_bytes()))
        )

    assert error.value.code == "PROVIDER_REQUEST_FAILED"
    assert "secret transport detail" not in str(error.value)
    assert len(budget.calls) == 1
    assert budget.unknown == ["reservation"]


def test_invalid_or_oversized_image_is_rejected_before_budget_or_network() -> None:
    budget = _Budget()
    client = _Client()
    provider = _provider(client, budget)

    with pytest.raises(VisionProviderError, match="could not be read"):
        asyncio.run(
            provider.analyze(VisionRequest(question="Read it", image_bytes=b"not an image"))
        )
    with pytest.raises(VisionProviderError, match="size limit"):
        asyncio.run(
            provider.analyze(
                VisionRequest(question="Read it", image_bytes=b"x" * (4 * 1024 * 1024 + 1))
            )
        )

    assert budget.calls == []
    assert client.payload is None


def test_missing_key_and_oversized_prompt_fail_without_reservation() -> None:
    budget = _Budget()
    provider = DeepSeekVisionProvider(settings=Settings(), budget_manager=budget)

    with pytest.raises(VisionProviderError) as missing_key:
        asyncio.run(provider.analyze(VisionRequest(question="Read it", image_bytes=_png_bytes())))
    assert missing_key.value.code == "PROVIDER_UNAVAILABLE"

    provider = _provider(_Client(), budget)
    with pytest.raises(VisionProviderError) as long_prompt:
        asyncio.run(provider.analyze(VisionRequest(question="q" * 1001, image_bytes=_png_bytes())))
    assert long_prompt.value.code == "TEXT_TOO_LONG"
    assert budget.calls == []


def test_structured_output_keeps_observation_inference_and_uncertainty_separate() -> None:
    client = _Client(
        _Response(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"type":"json_object",'
                                '"observations":[{"statement":"The blue series rises."}],'
                                '"readings":[{"label":"At x=2","value":"about 4",'
                                '"unit":null,"kind":"plot_estimate"}],'
                                '"interpretation":"The series trends upward.",'
                                '"uncertainty_notes":["The y-axis labels are blurred."]}'
                            )
                        }
                    }
                ]
            }
        )
    )
    provider = _provider(client)

    result = asyncio.run(
        provider.analyze_structured(
            VisionRequest(question="Describe the curve.", image_bytes=_png_bytes())
        )
    )

    assert result.analysis.observations[0].statement == "The blue series rises."
    assert result.analysis.readings[0].kind == "plot_estimate"
    assert result.analysis.uncertainty_notes == ["The y-axis labels are blurred."]
    assert client.payload["response_format"] == {"type": "json_object"}


def test_malformed_structured_output_fails_with_safe_classification() -> None:
    client = _Client(_Response({"choices": [{"message": {"content": '{"unexpected":"field"}'}}]}))

    with pytest.raises(VisionProviderError) as error:
        asyncio.run(
            _provider(client).analyze_structured(
                VisionRequest(question="Describe this.", image_bytes=_png_bytes())
            )
        )

    assert error.value.code == "INVALID_ANALYSIS"
    assert "unexpected" not in str(error.value)


def test_visual_analysis_cache_reuses_only_matching_source_and_question() -> None:
    cache = JsonCache(_Redis())
    provider = _Vision()
    image = _png_bytes()
    request = VisionRequest(question="Describe the curve.", image_bytes=image)

    first = asyncio.run(
        analyze_figure_cached(
            provider=provider,
            cache=cache,
            request=request,
            source_metadata=_source_metadata(image),
        )
    )
    second = asyncio.run(
        analyze_figure_cached(
            provider=provider,
            cache=cache,
            request=request,
            source_metadata=_source_metadata(image),
        )
    )
    changed = asyncio.run(
        analyze_figure_cached(
            provider=provider,
            cache=cache,
            request=VisionRequest(question="Read the labels.", image_bytes=image),
            source_metadata=_source_metadata(image),
        )
    )

    assert first.cache_status == "miss"
    assert first.generation is not None
    assert second.cache_status == "hit"
    assert second.generation is None
    assert changed.cache_status == "miss"
    assert provider.calls == 2
    assert second.source.text_citation is False
    assert all(b"base64" not in str(value).encode() for value in cache._client.values.values())


def test_visual_cache_key_changes_when_prompt_version_changes(monkeypatch) -> None:
    import app.services.vision as vision

    image = _png_bytes()
    source = VisualSourceReference.from_source_metadata(_source_metadata(image))
    request = VisionRequest(question="Describe the curve.", image_bytes=image)
    original = _visual_cache_key(source=source, question=request.question, context=request.context)

    monkeypatch.setattr(vision, "VISUAL_PROMPT_VERSION", "figure-analysis-next")

    changed = _visual_cache_key(source=source, question=request.question, context=request.context)

    assert changed != original


def test_visual_cache_refuses_crop_bytes_that_do_not_match_source_identity() -> None:
    provider = _Vision()
    image = _png_bytes()
    metadata = _source_metadata(image)
    metadata["crop_sha256"] = "b" * 64

    with pytest.raises(VisionProviderError) as error:
        asyncio.run(
            analyze_figure_cached(
                provider=provider,
                cache=JsonCache(_Redis()),
                request=VisionRequest(question="Describe.", image_bytes=image),
                source_metadata=metadata,
            )
        )

    assert error.value.code == "SOURCE_MISMATCH"
    assert provider.calls == 0
