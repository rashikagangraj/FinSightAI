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


from functools import lru_cache


@lru_cache(maxsize=1)
def _check_ollama_online(base_url: str) -> bool:
    import socket
    import urllib.parse
    try:
        parsed = urllib.parse.urlparse(base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or 11434
        sock = socket.create_connection((host, port), timeout=0.8)
        sock.close()
        return True
    except Exception:
        return False


def _get_embed_model() -> Any:
    from llama_index.core.embeddings import MockEmbedding

    cfg = get_settings()
    if cfg.llm_backend == "openai" and cfg.openai_api_key and not cfg.openai_api_key.startswith("sk-placeholder"):
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
            return OpenAIEmbedding(model=cfg.openai_embed_model, api_key=cfg.openai_api_key)
        except Exception as exc:
            logger.warning(f"OpenAI embedding unavailable: {exc} — using MockEmbedding")
            return MockEmbedding(embed_dim=384)

    if cfg.llm_backend == "ollama":
        if _check_ollama_online(cfg.ollama_base_url):
            try:
                from llama_index.embeddings.ollama import OllamaEmbedding
                return OllamaEmbedding(model_name=cfg.ollama_embed_model, base_url=cfg.ollama_base_url)
            except Exception as exc:
                logger.warning(f"Ollama embedding failed: {exc} — using MockEmbedding")
                return MockEmbedding(embed_dim=384)
        else:
            logger.debug(f"Ollama server offline at {cfg.ollama_base_url} — using MockEmbedding fallback")
            return MockEmbedding(embed_dim=384)

    return MockEmbedding(embed_dim=384)




def _get_chroma_collection() -> chromadb.Collection:
    cfg = get_settings()
    client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
    return client.get_or_create_collection(cfg.chroma_collection_name)


def reset_collection() -> None:
    """Clear all documents in the ChromaDB collection."""
    cfg = get_settings()
    client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
    try:
        client.delete_collection(cfg.chroma_collection_name)
    except Exception:
        pass
    client.get_or_create_collection(cfg.chroma_collection_name)


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


def ingest_file(file_path: str | Path, source_override: str | None = None) -> int:
    """Read a file (txt, md, pdf, csv, json) and index its content."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    source_name = source_override or path.name
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from llama_index.readers.file import PDFReader
        reader = PDFReader()
        llama_docs = reader.load_data(file=path)
        texts = [d.text for d in llama_docs if d.text.strip()]
        metas = [{"source": source_name, "page": i + 1} for i in range(len(texts))]
    elif suffix == ".csv":
        import csv
        text = path.read_text(encoding="utf-8", errors="ignore")
        # Format CSV rows as readable lines
        rows = list(csv.reader(text.splitlines()))
        if rows:
            header = rows[0]
            row_texts = [
                ", ".join(f"{h}: {val}" for h, val in zip(header, r))
                for r in rows[1:] if r
            ]
            texts = ["\n".join(row_texts)] if row_texts else [text]
        else:
            texts = [text]
        metas = [{"source": source_name}]
    elif suffix == ".json":
        import json
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            parsed = json.loads(text)
            formatted = json.dumps(parsed, indent=2)
            texts = [formatted]
        except Exception:
            texts = [text]
        metas = [{"source": source_name}]
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")
        texts = [text]
        metas = [{"source": source_name}]

    return ingest_texts(texts, metas)


def ingest_directory(dir_path: str | Path, glob: str = "**/*") -> int:
    """Recursively index all supported files in a directory."""
    dir_path = Path(dir_path)
    supported = {".txt", ".md", ".pdf", ".csv", ".json"}
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


def delete_source(source_name: str) -> int:
    """Delete all chunks belonging to a specific source from ChromaDB."""
    collection = _get_chroma_collection()
    # Search for matching IDs with where metadata filter or substring match
    results = collection.get(include=["metadatas"])
    ids_to_delete = []
    for doc_id, meta in zip(results.get("ids", []), results.get("metadatas", [])):
        if meta and meta.get("source"):
            src = meta["source"]
            if src == source_name or Path(src).name == source_name:
                ids_to_delete.append(doc_id)

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        logger.info(f"Deleted {len(ids_to_delete)} chunks for source: {source_name}")
    return len(ids_to_delete)


def clear_all_documents() -> int:
    """Clear all documents from the Chroma collection."""
    collection = _get_chroma_collection()
    count = collection.count()
    if count > 0:
        results = collection.get()
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            logger.info(f"Cleared all {len(ids)} chunks from collection")
    return count


def seed_sample_documents() -> dict[str, int]:
    """Ingest bundled sample financial documents if available."""
    sample_dir = Path(__file__).parent.parent.parent / "examples" / "sample_docs"
    if not sample_dir.exists():
        return {}

    supported = {".txt", ".md", ".pdf", ".csv", ".json"}
    results = {}
    for p in sample_dir.iterdir():
        if p.is_file() and p.suffix.lower() in supported:
            try:
                nodes = ingest_file(p, source_override=p.name)
                results[p.name] = nodes
            except Exception as exc:
                logger.warning(f"Failed to seed sample doc {p.name}: {exc}")
                results[p.name] = 0
    return results


def list_indexed_sources() -> list[str]:
    """Return unique source filenames stored in ChromaDB."""
    collection = _get_chroma_collection()
    results = collection.get(include=["metadatas"])
    sources: set[str] = set()
    for meta in results.get("metadatas") or []:
        if meta and "source" in meta:
            # Clean path to display readable filename
            src = meta["source"]
            sources.add(Path(src).name if ("/" in src or "\\" in src) else src)
    return sorted(sources)


def get_document_count() -> int:
    return _get_chroma_collection().count()

