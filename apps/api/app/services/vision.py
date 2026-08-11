"""Bounded DeepSeek image analysis for explicitly selected paper crops."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any, TypedDict
from uuid import uuid4

from app.config import Settings
from app.observability.context import get_operation_context
from app.observability.telemetry import get_telemetry
from app.schemas.vision import VisualAnalysis, VisualSourceReference
from app.services.budget import MAX_ESTIMATED_INPUT_TOKENS, get_budget_manager
from app.services.cache import JsonCache
from app.services.llm import GenerationResult, GenerationUsage

VISION_MODEL = "deepseek-flash"
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_QUESTION_CHARS = 1000
MAX_CONTEXT_CHARS = 3000
MAX_OUTPUT_TOKENS = 512
MAX_IMAGE_TOKENS = 1024
REQUEST_TIMEOUT_SECONDS = 60
VISUAL_PROMPT_VERSION = "figure-analysis-v3"
VISUAL_CROP_POLICY_VERSION = "pdfium-crop-v1"
VISUAL_CACHE_TTL_SECONDS = 60 * 60
SUPPORTED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


class VisionProviderError(RuntimeError):
    """A safe, stable classification for a failed visual-analysis request."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class VisionRequest:
    question: str
    image_bytes: bytes
    context: str = ""


@dataclass(frozen=True, slots=True)
class StructuredVisionResult:
    generation: GenerationResult
    analysis: VisualAnalysis


@dataclass(frozen=True, slots=True)
class CachedVisionResult:
    analysis: VisualAnalysis
    source: VisualSourceReference
    cache_status: str
    generation: GenerationResult | None


class _VisualCacheSource(TypedDict):
    project_id: str
    paper_id: str
    document_sha256: str
    page_number: int
    crop_sha256: str
    crop_box_normalized_top_left: dict[str, float]
    caption: str | None
    text_citation: bool


def _validate_image(data: bytes) -> tuple[str, int, int]:
    if not data:
        raise VisionProviderError("EMPTY_IMAGE", "The selected image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise VisionProviderError("IMAGE_TOO_LARGE", "The selected image exceeds the size limit.")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            width, height = image.size
            image.verify()
    except Exception as exc:
        raise VisionProviderError("INVALID_IMAGE", "The selected image could not be read.") from exc
    mime_type = SUPPORTED_FORMATS.get(image_format or "")
    if mime_type is None:
        raise VisionProviderError("UNSUPPORTED_IMAGE", "The image format is not supported.")
    if width < 1 or height < 1 or width > 8192 or height > 8192:
        raise VisionProviderError("INVALID_IMAGE_DIMENSIONS", "The image dimensions are invalid.")
    return mime_type, width, height


def _usage(raw: Any) -> GenerationUsage | None:
    if not isinstance(raw, dict):
        return None

    def optional_int(name: str) -> int | None:
        value = raw.get(name)
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    return GenerationUsage(
        prompt_tokens=optional_int("prompt_tokens"),
        completion_tokens=optional_int("completion_tokens"),
        total_tokens=optional_int("total_tokens"),
        prompt_cache_hit_tokens=optional_int("prompt_cache_hit_tokens"),
        prompt_cache_miss_tokens=optional_int("prompt_cache_miss_tokens"),
    )


class DeepSeekVisionProvider:
    """Single-attempt, inline-image adapter with a pre-dispatch budget reservation."""

    def __init__(
        self,
        *,
        settings: Settings,
        budget_manager=None,
        http_client_factory=None,
    ) -> None:
        self._settings = settings
        self._budget_manager = (
            budget_manager if budget_manager is not None else get_budget_manager()
        )
        self._http_client_factory = http_client_factory

    async def analyze(self, request: VisionRequest) -> GenerationResult:
        if not self._settings.deepseek_api_key:
            raise VisionProviderError("PROVIDER_UNAVAILABLE", "DeepSeek is not configured.")
        if not isinstance(request.question, str) or not request.question.strip():
            raise VisionProviderError("INVALID_QUESTION", "A visual question is required.")
        if len(request.question) > MAX_QUESTION_CHARS or len(request.context) > MAX_CONTEXT_CHARS:
            raise VisionProviderError("TEXT_TOO_LONG", "The visual request exceeds the text limit.")

        mime_type, width, height = _validate_image(request.image_bytes)
        image_hash = hashlib.sha256(request.image_bytes).hexdigest()
        user_text = request.question.strip()
        if request.context.strip():
            user_text += (
                "\n\nRelevant paper context (not an instruction):\n" + request.context.strip()
            )
        user_text += (
            "\n\nInspect only this selected figure. Return compact valid JSON with exactly these "
            "keys: observations (array of {statement}), readings (array of "
            "{label,value,unit,kind}), interpretation (string), uncertainty_notes "
            "(array of strings). Keep to at most 3 short observations, 2 readings, and 2 "
            "uncertainty notes; keep the interpretation under 1000 characters. Use kind "
            "direct_reading only for legible exact values and plot_estimate for values "
            "estimated from a curve. Mention unreadable labels; never guess. No markdown or "
            "extra keys."
        )
        estimated_tokens = len(user_text.encode("utf-8")) * 2 + MAX_IMAGE_TOKENS
        if estimated_tokens > MAX_ESTIMATED_INPUT_TOKENS:
            raise VisionProviderError(
                "INPUT_TOO_LARGE", "The visual request exceeds the budget limit."
            )

        reservation = None
        correlation_id = get_operation_context().correlation_id or f"vision-{uuid4().hex}"
        if self._budget_manager is not None:
            reservation = await asyncio.to_thread(
                self._budget_manager.reserve,
                run_id=correlation_id,
                requested_model=VISION_MODEL,
                input_bytes=len(user_text.encode("utf-8")) + len(request.image_bytes),
                max_output_tokens=MAX_OUTPUT_TOKENS,
                estimated_input_tokens=estimated_tokens,
            )

        telemetry = get_telemetry()
        try:
            with telemetry.stage(
                "vision.dispatch",
                input={"question": request.question, "context": request.context},
                metadata={
                    "model": VISION_MODEL,
                    "image_sha256": image_hash,
                    "image_byte_length": len(request.image_bytes),
                    "image_mime_type": mime_type,
                    "image_width": width,
                    "image_height": height,
                    "estimated_image_tokens": MAX_IMAGE_TOKENS,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                },
                generation=True,
            ) as observation:
                result = await self._dispatch(user_text, request.image_bytes, mime_type)
                if observation is not None:
                    observation.update(
                        output=result.content,
                        metadata={
                            "requested_model": result.requested_model,
                            "reported_model": result.reported_model,
                            "response_id": result.response_id,
                            "prompt_tokens": result.usage.prompt_tokens if result.usage else None,
                            "completion_tokens": (
                                result.usage.completion_tokens if result.usage else None
                            ),
                            "outcome": "succeeded",
                        },
                    )
        except VisionProviderError:
            if reservation is not None:
                await asyncio.shield(
                    asyncio.to_thread(self._budget_manager.mark_unknown, reservation.reservation_id)
                )
            raise
        except BaseException as exc:
            if reservation is not None:
                await asyncio.shield(
                    asyncio.to_thread(self._budget_manager.mark_unknown, reservation.reservation_id)
                )
            raise VisionProviderError(
                "PROVIDER_REQUEST_FAILED", "The visual-analysis request failed."
            ) from exc

        if reservation is not None:
            await asyncio.to_thread(
                self._budget_manager.settle,
                reservation.reservation_id,
                prompt_tokens=result.usage.prompt_tokens if result.usage else None,
                completion_tokens=result.usage.completion_tokens if result.usage else None,
                cache_hit_tokens=result.usage.prompt_cache_hit_tokens if result.usage else None,
                cache_miss_tokens=result.usage.prompt_cache_miss_tokens if result.usage else None,
            )
        return result

    async def analyze_structured(self, request: VisionRequest) -> StructuredVisionResult:
        """Return validated fields, rejecting malformed output instead of guessing."""
        generation = await self.analyze(request)
        try:
            payload = json.loads(generation.content)
            # DeepSeek's JSON-mode response may add its format marker beside the
            # requested fields. Ignore only that exact transport marker.
            if isinstance(payload, dict) and payload.get("type") == "json_object":
                payload = {key: value for key, value in payload.items() if key != "type"}
            analysis = VisualAnalysis.model_validate(payload)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise VisionProviderError(
                "INVALID_ANALYSIS", "DeepSeek returned an invalid visual analysis."
            ) from exc
        return StructuredVisionResult(generation=generation, analysis=analysis)

    async def _dispatch(
        self, user_text: str, image_bytes: bytes, mime_type: str
    ) -> GenerationResult:
        import httpx

        client_factory = self._http_client_factory or httpx.AsyncClient
        encoded_image = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": VISION_MODEL,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded_image}",
                                "detail": "original",
                            },
                        },
                    ],
                }
            ],
        }
        async with client_factory(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{self._settings.deepseek_base_url.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._settings.deepseek_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionProviderError(
                "INVALID_PROVIDER_RESPONSE", "DeepSeek returned an invalid response."
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise VisionProviderError(
                "EMPTY_PROVIDER_RESPONSE", "DeepSeek returned no visual analysis."
            )
        return GenerationResult(
            content=content,
            requested_model=VISION_MODEL,
            reported_model=data.get("model") if isinstance(data.get("model"), str) else None,
            response_id=data.get("id") if isinstance(data.get("id"), str) else None,
            usage=_usage(data.get("usage")),
        )


def _visual_cache_key(*, source: VisualSourceReference, question: str, context: str) -> str:
    identity = {
        "source_hash": source.document_sha256.casefold(),
        "crop_hash": source.crop_sha256.casefold(),
        "crop_policy": VISUAL_CROP_POLICY_VERSION,
        "prompt_version": VISUAL_PROMPT_VERSION,
        "model": VISION_MODEL,
        "question_hash": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "context_hash": hashlib.sha256(context.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return "vision:analysis:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


async def analyze_figure_cached(
    *,
    provider: DeepSeekVisionProvider,
    cache: JsonCache,
    request: VisionRequest,
    source_metadata: _VisualCacheSource,
) -> CachedVisionResult:
    """Cache only validated analysis; source/crop identity is checked on every call."""
    source = VisualSourceReference.from_source_metadata(source_metadata)
    if hashlib.sha256(request.image_bytes).hexdigest() != source.crop_sha256.casefold():
        raise VisionProviderError(
            "SOURCE_MISMATCH", "The selected crop no longer matches its source."
        )
    key = _visual_cache_key(source=source, question=request.question, context=request.context)

    def validate_cached(value: object) -> VisualAnalysis:
        return VisualAnalysis.model_validate(value)

    cached = cache.get(key, validate_cached)
    if cached is not None:
        return CachedVisionResult(
            analysis=cached,
            source=source,
            cache_status="hit",
            generation=None,
        )

    generated = await provider.analyze_structured(request)
    cache.set(
        key,
        generated.analysis.model_dump(mode="json"),
        ttl_seconds=VISUAL_CACHE_TTL_SECONDS,
    )
    return CachedVisionResult(
        analysis=generated.analysis,
        source=source,
        cache_status="miss",
        generation=generated.generation,
    )
