from __future__ import annotations

from typing import Iterator

from openai import OpenAI

from src.core.config import get_settings
from src.core.logging import get_logger
from src.llm.base import LLMClient

logger = get_logger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self) -> None:
        cfg = get_settings()
        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.openai_model
        self._embed_model = cfg.openai_embed_model
        self._temperature = cfg.openai_temperature
        self._max_tokens = cfg.openai_max_tokens

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.debug(
            f"OpenAI complete | model={self._model} "
            f"tokens={response.usage.total_tokens if response.usage else '?'}"
        )
        return content

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        with self._client.chat.completions.stream(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def embed(self, text: str) -> list[float]:
        response = self._client.embeddings.create(
            model=self._embed_model,
            input=text,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._embed_model,
            input=texts,
        )
        return [item.embedding for item in response.data]
