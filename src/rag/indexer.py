from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb
from llama_index.core import (
    Document,
    Settings as LlamaSettings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


def _get_embed_model() -> Any:
    cfg = get_settings()
    if cfg.llm_backend == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        return OpenAIEmbedding(model=cfg.openai_embed_model, api_key=cfg.openai_api_key)
    from llama_index.embeddings.ollama import OllamaEmbedding
    return OllamaEmbedding(model_name=cfg.ollama_embed_model, base_url=cfg.ollama_base_url)


def _get_chroma_collection() -> chromadb.Collection:
    cfg = get_settings()
    client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
    return client.get_or_create_collection(cfg.chroma_collection_name)


def _build_storage_context(collection: chromadb.Collection) -> StorageContext:
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return StorageContext.from_defaults(vector_store=vector_store)


def ingest_texts(texts: list[str], metadatas: list[dict] | None = None) -> int:
    """Index a list of plain strings. Returns number of nodes stored."""
    cfg = get_settings()
    LlamaSettings.embed_model = _get_embed_model()
    LlamaSettings.llm = None  # we handle LLM ourselves

    splitter = SentenceSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    docs = []
    for i, text in enumerate(texts):
        meta = (metadatas[i] if metadatas else {}) or {}
        doc_id = hashlib.md5((text[:200]).encode()).hexdigest()
        docs.append(Document(text=text, metadata=meta, id_=doc_id))

    nodes = splitter.get_nodes_from_documents(docs)
    collection = _get_chroma_collection()
    storage_ctx = _build_storage_context(collection)
    VectorStoreIndex(nodes, storage_context=storage_ctx)

    logger.info(f"Indexed {len(nodes)} nodes from {len(docs)} documents")
    return len(nodes)


def ingest_file(file_path: str | Path) -> int:
    """Read a file (txt, md, pdf) and index its content."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".pdf":
        from llama_index.readers.file import PDFReader
        reader = PDFReader()
        llama_docs = reader.load_data(file=path)
        texts = [d.text for d in llama_docs]
        metas = [{"source": str(path), "page": i + 1} for i in range(len(texts))]
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        texts = [text]
        metas = [{"source": str(path)}]

    return ingest_texts(texts, metas)


def ingest_directory(dir_path: str | Path, glob: str = "**/*") -> int:
    """Recursively index all supported files in a directory."""
    dir_path = Path(dir_path)
    supported = {".txt", ".md", ".pdf"}
    total = 0
    for p in dir_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in supported:
            try:
                n = ingest_file(p)
                total += n
                logger.info(f"Ingested {p.name} → {n} nodes")
            except Exception as exc:
                logger.warning(f"Skipping {p.name}: {exc}")
    return total


def list_indexed_sources() -> list[str]:
    """Return unique source filenames stored in ChromaDB."""
    collection = _get_chroma_collection()
    results = collection.get(include=["metadatas"])
    sources: set[str] = set()
    for meta in results.get("metadatas") or []:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sorted(sources)


def get_document_count() -> int:
    return _get_chroma_collection().count()
