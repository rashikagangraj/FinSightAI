from __future__ import annotations

from typing import Iterator
import httpx
from openai import OpenAI

from src.core.config import get_settings
from src.core.logging import get_logger
from src.llm.base import LLMClient

logger = get_logger(__name__)


class GeminiClient(LLMClient):
    """Gemini client supporting official OpenAI-compatible endpoint and direct REST fallback."""

    def __init__(self) -> None:
        cfg = get_settings()
        if not cfg.gemini_api_key or cfg.gemini_api_key.startswith("placeholder"):
            raise ValueError("Gemini API key is missing or set to placeholder.")
        self._api_key = cfg.gemini_api_key
        self._model = cfg.gemini_model
        self._embed_model = cfg.gemini_embed_model
        self._temperature = cfg.gemini_temperature
        self._max_tokens = cfg.gemini_max_tokens

        # Initialize OpenAI-compatible client for Google Gemini
        self._client = OpenAI(
            api_key=self._api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=10.0,
        )

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
            content = response.choices[0].message.content or ""
            logger.debug(f"Gemini complete | model={self._model}")
            return content
        except Exception as exc:
            logger.warning(f"Gemini OpenAI-compat call failed: {exc}, attempting direct REST API...")
            # Fallback to direct Google Generative Language REST API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={self._api_key}"
            contents = []
            if system:
                contents.append({"role": "user", "parts": [{"text": f"System instructions: {system}"}]})
            contents.append({"role": "user", "parts": [{"text": prompt}]})
            
            with httpx.Client(timeout=10.0) as http_client:
                res = http_client.post(
                    url,
                    json={"contents": contents, "generationConfig": {"temperature": self._temperature, "maxOutputTokens": self._max_tokens}},
                )
                res.raise_for_status()
                data = res.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            with self._client.chat.completions.stream(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as exc:
            logger.warning(f"Gemini stream failed ({exc}), falling back to complete()...")
            yield self.complete(prompt, system)

    def embed(self, text: str) -> list[float]:
        try:
            response = self._client.embeddings.create(
                model=self._embed_model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.warning(f"Gemini embed OpenAI-compat call failed: {exc}, trying REST...")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._embed_model}:embedContent?key={self._api_key}"
            with httpx.Client(timeout=10.0) as http_client:
                res = http_client.post(
                    url,
                    json={"model": f"models/{self._embed_model}", "content": {"parts": [{"text": text}]}},
                )
                res.raise_for_status()
                data = res.json()
                return data["embedding"]["values"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(
                model=self._embed_model,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception:
            return [self.embed(t) for t in texts]
