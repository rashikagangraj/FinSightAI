from __future__ import annotations

from typing import Iterator

import ollama

from src.core.config import get_settings
from src.core.logging import get_logger
from src.llm.base import LLMClient

logger = get_logger(__name__)


class OllamaClient(LLMClient):
    def __init__(self) -> None:
        cfg = get_settings()
        self._client = ollama.Client(host=cfg.ollama_base_url, timeout=2.0)
        self._model = cfg.ollama_model
        self._embed_model = cfg.ollama_embed_model
        self._options = {
            "temperature": cfg.ollama_temperature,
            "num_ctx": cfg.ollama_num_ctx,
        }


    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self._model,
            messages=messages,
            options=self._options,
        )
        content: str = response["message"]["content"]
        logger.debug(f"Ollama complete | model={self._model} chars={len(content)}")
        return content

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        for chunk in self._client.chat(
            model=self._model,
            messages=messages,
            options=self._options,
            stream=True,
        ):
            token: str = chunk["message"]["content"]
            if token:
                yield token

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings(model=self._embed_model, prompt=text)
        return response["embedding"]
