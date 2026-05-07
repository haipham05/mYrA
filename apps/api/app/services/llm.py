from abc import ABC, abstractmethod

from app.config import Settings


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        pass


class DeepSeekLLMProvider(LLMProvider):
    """DeepSeek API client."""

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    @property
    def provider_name(self) -> str:
        return "deepseek"

    async def generate(self, system_prompt: str, user_prompt: str) -> str:
        import httpx

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "deepseek-chat",
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
            return data["choices"][0]["message"]["content"]


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
                words = [w for w in re.findall(r"\w+", quote_text) if len(w) > 3]
                snippet = " ".join(words[:4]) if words else "the cited evidence"
                return f"The research demonstrates {snippet} [E1]."
            return "Based on the provided research papers, significant results were found [E1]."
        return "Insufficient evidence available in the uploaded papers to answer this question."


_llm_instance: LLMProvider | None = None


def get_llm_provider(settings: Settings | None = None, mode: str | None = None) -> LLMProvider:
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    import os

    effective_mode = mode or os.getenv("MYRA_LLM_MODE", "auto").lower()

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
    elif settings.deepseek_api_key:
        _llm_instance = DeepSeekLLMProvider(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    else:
        _llm_instance = FakeLLMProvider()

    return _llm_instance


def set_llm_provider(provider: LLMProvider | None) -> None:
    global _llm_instance
    _llm_instance = provider
