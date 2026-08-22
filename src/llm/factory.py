from __future__ import annotations

from src.core.config import get_settings
from src.llm.base import LLMClient


class ResilientLLMClient(LLMClient):
    """Wraps primary LLM client with automatic fallback for local test/demo reliability."""

    def __init__(self, primary: LLMClient) -> None:
        from src.llm.fallback_client import FallbackClient
        self._primary = primary
        self._fallback = FallbackClient()

    def complete(self, prompt: str, system: str = "") -> str:
        try:
            return self._primary.complete(prompt, system)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Primary LLM request failed: {exc} — using local fallback engine")
            return self._fallback.complete(prompt, system)

    def stream(self, prompt: str, system: str = ""):
        try:
            yield from self._primary.stream(prompt, system)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Primary LLM stream failed: {exc} — using local fallback stream")
            yield from self._fallback.stream(prompt, system)

    def embed(self, text: str) -> list[float]:
        try:
            return self._primary.embed(text)
        except Exception:
            return self._fallback.embed(text)


def get_llm_client() -> LLMClient:
    backend = get_settings().llm_backend
    if backend == "openai":
        from src.llm.openai_client import OpenAIClient
        return ResilientLLMClient(OpenAIClient())
    from src.llm.ollama_client import OllamaClient
    return ResilientLLMClient(OllamaClient())

