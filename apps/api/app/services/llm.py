from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.config import Settings


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
    provider: object, system_prompt: str, user_prompt: str
) -> GenerationResult:
    """Use the metadata API for real providers and preserve duck-typed test doubles.

    Unspecced AsyncMock instances intentionally follow the legacy ``generate`` path:
    their dynamically-created ``generate_result`` child would otherwise return a mock.
    """
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

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = "deepseek-chat"

    @property
    def provider_name(self) -> str:
        return "deepseek"

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        return (await self.generate_result(system_prompt, user_prompt)).content

    async def generate_result(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        }
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
                    prompt_cache_hit_tokens=_optional_int(raw_usage.get("prompt_cache_hit_tokens")),
                    prompt_cache_miss_tokens=_optional_int(
                        raw_usage.get("prompt_cache_miss_tokens")
                    ),
                )
            return GenerationResult(
                content=data["choices"][0]["message"]["content"],
                requested_model=self.model_name,
                reported_model=data.get("model") if isinstance(data.get("model"), str) else None,
                response_id=data.get("id") if isinstance(data.get("id"), str) else None,
                usage=usage,
            )


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
        _llm_instance = DeepSeekLLMProvider(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    elif effective_mode in ("test", "demo"):
        _llm_instance = FakeLLMProvider()
    else:
        raise ValueError(f"Unknown MYRA_LLM_MODE: {effective_mode}")

    return _llm_instance


def set_llm_provider(provider: LLMProvider | None) -> None:
    global _llm_instance
    _llm_instance = provider
