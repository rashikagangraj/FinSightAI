from __future__ import annotations

from src.core.config import get_settings
from src.llm.base import LLMClient


def get_llm_client() -> LLMClient:
    backend = get_settings().llm_backend
    if backend == "openai":
        from src.llm.openai_client import OpenAIClient
        return OpenAIClient()
    from src.llm.ollama_client import OllamaClient
    return OllamaClient()
