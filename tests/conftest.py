"""Shared test fixtures for FinAgent RAG test suite."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Force test environment settings before any imports
os.environ.setdefault("LLM_BACKEND", "ollama")
os.environ.setdefault("OLLAMA_MODEL", "llama3.2:3b")
os.environ.setdefault("OLLAMA_EMBED_MODEL", "nomic-embed-text")
os.environ.setdefault("CHROMA_PERSIST_DIR", tempfile.mkdtemp())


@pytest.fixture
def mock_llm_client():
    """A mock LLMClient that returns deterministic answers."""
    client = MagicMock()
    client.complete.return_value = (
        "Apple's revenue was $383.3 billion in fiscal 2023. "
        "This represents a 3% decline from $394.3 billion in fiscal 2022."
    )
    client.stream.return_value = iter(["Apple's ", "revenue ", "was ", "$383.3 billion."])
    client.embed.return_value = [0.1] * 384
    client.embed_batch.return_value = [[0.1] * 384]
    return client


@pytest.fixture
def sample_texts():
    return [
        "Apple's total revenue was $383.3 billion in fiscal 2023, down 3% year-over-year.",
        "Tesla delivered 1,808,581 vehicles in 2023, up 38% year-over-year.",
        "The S&P 500 gained 24.2% in 2023, led by Information Technology at +57.8%.",
    ]


@pytest.fixture
def sample_metadatas():
    return [
        {"source": "apple_10k_excerpt.txt"},
        {"source": "tesla_earnings_q4.txt"},
        {"source": "sp500_overview.md"},
    ]


@pytest.fixture
def tmp_text_file(tmp_path):
    f = tmp_path / "test_doc.txt"
    f.write_text(
        "Revenue: $100 million. Net income: $20 million. EPS: $2.50. "
        "Gross margin: 40%. Total assets: $500 million.",
        encoding="utf-8",
    )
    return f
