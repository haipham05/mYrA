import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.config import Settings

DEFAULT_MAX_OUTPUT_TOKENS = 4096


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    """Provider-reported token usage for one generation call."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Immutable result and provider metadata for one generation call."""

    content: str
    requested_model: str | None = None
    reported_model: str | None = None
    response_id: str | None = None
    usage: GenerationUsage | None = None


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    """Supported per-call controls for compatible chat-completion providers.

    ``None`` means to retain the provider's configured default. Keeping the
    request options immutable prevents concurrent calls from changing one
    another's model or response mode.
    """

    model_name: str | None = None
    max_output_tokens: int | None = None
    structured_json: bool | None = None
    disable_thinking: bool | None = None

    def __post_init__(self) -> None:
        if self.model_name is not None and not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.max_output_tokens is not None and (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or not 1 <= self.max_output_tokens <= 65_536
        ):
            raise ValueError("max_output_tokens must be between 1 and 65536")
        if self.structured_json is not None and not isinstance(self.structured_json, bool):
            raise TypeError("structured_json must be a boolean")
        if self.disable_thinking is not None and not isinstance(self.disable_thinking, bool):
            raise TypeError("disable_thinking must be a boolean")


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass

    async def generate_result(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        """Backward-compatible metadata adapter for existing providers."""
        return GenerationResult(content=await self.generate(system_prompt, user_prompt))


async def generate_with_metadata(
    provider: object,
    system_prompt: str,
    user_prompt: str,
    *,
    options: GenerationOptions | None = None,
) -> GenerationResult:
    """Use the metadata API for real providers and preserve duck-typed test doubles.

    Unspecced AsyncMock instances intentionally follow the legacy ``generate`` path:
    their dynamically-created ``generate_result`` child would otherwise return a mock.
    """
    if options is not None and isinstance(provider, DeepSeekLLMProvider):
        return await provider.generate_result(system_prompt, user_prompt, options=options)
    if options is not None:
        raise TypeError("Generation options are supported only by DeepSeekLLMProvider")
    if isinstance(provider, LLMProvider):
        return await provider.generate_result(system_prompt, user_prompt)

    content = await provider.generate(system_prompt=system_prompt, user_prompt=user_prompt)  # type: ignore[attr-defined]
    if isinstance(content, GenerationResult):
        return content
    if not isinstance(content, str):
        raise TypeError("LLM provider generate() must return text")
    return GenerationResult(content=content)


class DeepSeekLLMProvider(LLMProvider):
    """DeepSeek API client."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        *,
        model_name: str = "deepseek-chat",
        max_output_tokens: int | None = None,
        structured_json: bool = False,
        disable_thinking: bool = False,
        budget_manager=None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_options = GenerationOptions(
            model_name=model_name,
            max_output_tokens=max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
            structured_json=structured_json,
            disable_thinking=disable_thinking,
        )
        self.model_name = model_name
        self._budget_manager = budget_manager

    @property
    def provider_name(self) -> str:
        return "deepseek"

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (await self.generate_result(system_prompt, user_prompt)).content

    async def generate_result(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        options: GenerationOptions | None = None,
    ) -> GenerationResult:
        import httpx

        request_options = options or GenerationOptions()
        model_name = (
            request_options.model_name or self.default_options.model_name or self.model_name
        )
        max_output_tokens = (
            request_options.max_output_tokens
            if request_options.max_output_tokens is not None
            else (self.default_options.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS)
        )
        structured_json = (
            request_options.structured_json
            if request_options.structured_json is not None
            else self.default_options.structured_json
        )
        disable_thinking = (
            request_options.disable_thinking
            if request_options.disable_thinking is not None
            else self.default_options.disable_thinking
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        if structured_json:
            payload["response_format"] = {"type": "json_object"}
        if disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        reservation = None
        if self._budget_manager is not None:
            from app.observability.context import get_operation_context

            context = get_operation_context()
            run_id = context.correlation_id or f"direct-{uuid4().hex}"
            max_tokens = max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS
            reservation = await asyncio.to_thread(
                self._budget_manager.reserve,
                run_id=run_id,
                requested_model=model_name,
                input_bytes=len(system_prompt.encode("utf-8")) + len(user_prompt.encode("utf-8")),
                max_output_tokens=max_tokens,
            )

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                raw_usage = data.get("usage")
                usage = None
                if isinstance(raw_usage, dict):
                    usage = GenerationUsage(
                        prompt_tokens=_optional_int(raw_usage.get("prompt_tokens")),
                        completion_tokens=_optional_int(raw_usage.get("completion_tokens")),
                        total_tokens=_optional_int(raw_usage.get("total_tokens")),
                        prompt_cache_hit_tokens=_optional_int(
                            raw_usage.get("prompt_cache_hit_tokens")
                        ),
                        prompt_cache_miss_tokens=_optional_int(
                            raw_usage.get("prompt_cache_miss_tokens")
                        ),
                    )
            result = GenerationResult(
                content=data["choices"][0]["message"]["content"],
                requested_model=model_name,
                reported_model=data.get("model") if isinstance(data.get("model"), str) else None,
                response_id=data.get("id") if isinstance(data.get("id"), str) else None,
                usage=usage,
            )
        except BaseException:
            if reservation is not None:
                await asyncio.shield(
                    asyncio.to_thread(self._budget_manager.mark_unknown, reservation.reservation_id)
                )
            raise

        if reservation is not None:
            await asyncio.to_thread(
                self._budget_manager.settle,
                reservation.reservation_id,
                prompt_tokens=usage.prompt_tokens if usage is not None else None,
                completion_tokens=usage.completion_tokens if usage is not None else None,
                cache_hit_tokens=usage.prompt_cache_hit_tokens if usage is not None else None,
                cache_miss_tokens=usage.prompt_cache_miss_tokens if usage is not None else None,
            )
        return result


class FakeLLMProvider(LLMProvider):
    """Deterministic LLM for testing and development without API keys."""

    def __init__(self, fixed_response: str | None = None) -> None:
        self.fixed_response = fixed_response

    @property
    def provider_name(self) -> str:
        return "test-fake"

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.fixed_response is not None:
            return self.fixed_response

        # Check if evidence exists in system_prompt or user_prompt
        if "[E1]" in user_prompt:
            import re

            m = re.search(r'\[E1\][^\n]*\nEvidence quote: "([^"]+)"', user_prompt)
            if not m:
                m = re.search(r'\[E1\][^\n]*\n"([^"]+)"', user_prompt)
            if m:
                quote_text = m.group(1).strip()
                if quote_text:
                    return f"{quote_text.rstrip('.')} [E1]."
            return "Insufficient evidence available in the uploaded papers to answer this question."
        return "Insufficient evidence available in the uploaded papers to answer this question."


_llm_instance: LLMProvider | None = None


def get_llm_provider(settings: Settings | None = None, mode: str | None = None) -> LLMProvider:
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    import os

    effective_mode = (mode or os.getenv("MYRA_LLM_MODE", "deepseek")).lower()

    if settings is None:
        settings = Settings.from_environment()

    if effective_mode in ("deepseek", "production"):
        if not settings.deepseek_api_key:
            raise RuntimeError(
                "Production mode requires DeepSeek API key, but DEEPSEEK_API_KEY is not "
                "configured. Set DEEPSEEK_API_KEY or use test fake provider."
            )
        from app.services.budget import get_budget_manager

        _llm_instance = DeepSeekLLMProvider(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            model_name=settings.deepseek_model_name,
            max_output_tokens=settings.deepseek_max_output_tokens,
            budget_manager=get_budget_manager(),
        )
    elif effective_mode in ("test", "demo"):
        _llm_instance = FakeLLMProvider()
    else:
        raise ValueError(f"Unknown MYRA_LLM_MODE: {effective_mode}")

    return _llm_instance


def set_llm_provider(provider: LLMProvider | None) -> None:
    global _llm_instance
    _llm_instance = provider
