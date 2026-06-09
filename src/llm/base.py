from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class LLMClient(ABC):
    """Unified interface for all LLM backends."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Return full completion as a single string."""

    @abstractmethod
    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        """Yield completion tokens one at a time."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return embedding vector for the given text."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]
