from __future__ import annotations

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from llama_index.core.embeddings import MockEmbedding


@pytest.fixture(autouse=True)
def isolate_chroma(tmp_path):
    os.environ["CHROMA_PERSIST_DIR"] = str(tmp_path / "chroma")


def _make_mock_embed():
    return MockEmbedding(embed_dim=384)



@patch("src.rag.indexer._get_embed_model")
def test_ingest_texts_returns_node_count(mock_embed, sample_texts, sample_metadatas):
    mock_embed.return_value = _make_mock_embed()
    from src.rag.indexer import ingest_texts

    count = ingest_texts(sample_texts, sample_metadatas)
    assert isinstance(count, int)
    assert count >= len(sample_texts)


@patch("src.rag.indexer._get_embed_model")
def test_ingest_file_txt(mock_embed, tmp_text_file):
    mock_embed.return_value = _make_mock_embed()
    from src.rag.indexer import ingest_file

    count = ingest_file(tmp_text_file)
    assert count >= 1


@patch("src.rag.indexer._get_embed_model")
def test_ingest_file_missing_raises(mock_embed):
    mock_embed.return_value = _make_mock_embed()
    from src.rag.indexer import ingest_file

    with pytest.raises(FileNotFoundError):
        ingest_file("/nonexistent/path/file.txt")


@patch("src.rag.indexer._get_embed_model")
def test_list_indexed_sources(mock_embed, sample_texts, sample_metadatas):
    mock_embed.return_value = _make_mock_embed()
    from src.rag.indexer import ingest_texts, list_indexed_sources

    ingest_texts(sample_texts, sample_metadatas)
    sources = list_indexed_sources()
    assert isinstance(sources, list)
