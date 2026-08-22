from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import chromadb
from llama_index.core import Settings as LlamaSettings, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.core.config import get_settings
from src.core.logging import get_logger
from src.rag.indexer import _get_chroma_collection, _get_embed_model

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


def _reciprocal_rank_fusion(
    results_a: list[RetrievedChunk],
    results_b: list[RetrievedChunk],
    weight_a: float,
    weight_b: float,
    k: int = 60,
) -> list[RetrievedChunk]:
    """Combine two ranked lists via Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(results_a):
        key = chunk.text[:100]
        scores[key] = scores.get(key, 0.0) + weight_a / (k + rank + 1)
        chunks[key] = chunk

    for rank, chunk in enumerate(results_b):
        key = chunk.text[:100]
        scores[key] = scores.get(key, 0.0) + weight_b / (k + rank + 1)
        if key not in chunks:
            chunks[key] = chunk

    sorted_keys = sorted(scores, key=lambda k_: scores[k_], reverse=True)
    fused = []
    for key in sorted_keys:
        c = chunks[key]
        fused.append(
            RetrievedChunk(
                text=c.text,
                source=c.source,
                score=round(scores[key], 4),
                metadata=c.metadata,
            )
        )
    return fused


def _vector_search(query: str, top_k: int) -> list[RetrievedChunk]:
    cfg = get_settings()
    LlamaSettings.embed_model = _get_embed_model()
    LlamaSettings.llm = None

    collection = _get_chroma_collection()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(vector_store)
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    results = []
    for n in nodes:
        results.append(
            RetrievedChunk(
                text=n.get_content(),
                source=n.metadata.get("source", "unknown"),
                score=float(n.score or 0.0),
                metadata=n.metadata,
            )
        )
    return results


def _bm25_search(query: str, top_k: int) -> list[RetrievedChunk]:
    """Keyword search via ChromaDB's built-in full-text (no external BM25 dep)."""
    cfg = get_settings()
    collection = _get_chroma_collection()

    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]

    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append(
            RetrievedChunk(
                text=doc,
                source=(meta or {}).get("source", "unknown"),
                score=round(1.0 / (1.0 + float(dist)), 4),
                metadata=meta or {},
            )
        )
    return chunks


def _extract_query_entities(query: str) -> list[str]:
    """Extract company/entity names mentioned in the query."""
    known_entities = {
        "aikart": "AIKART",
        "tesla": "Tesla",
        "apple": "Apple",
        "sp500": "S&P 500",
        "s&p 500": "S&P 500",
        "s&p": "S&P 500",
    }
    lower_query = query.lower()
    found = []
    for k, v in known_entities.items():
        if re.search(r"\b" + re.escape(k) + r"\b", lower_query):
            if v not in found:
                found.append(v)

    # Also discover capitalized company names from query
    words = re.findall(r"\b[A-Z][a-zA-Z0-9_\-\&]+\b", query)
    stopwords = {
        "Calculate", "Compute", "Compare", "Comparison", "Versus", "Vs", "What", "How", "Why",
        "The", "And", "With", "Between", "From", "In", "Of", "Total", "Revenue", "Margin",
        "Profit", "Income", "Cash", "Ebitda", "Assets", "Debt", "Eps", "Fiscal", "Year",
        "Gross", "Net", "Diluted", "Operating", "Fcf", "Roe", "Roa", "D/e", "Ratio",
        "Q1", "Q2", "Q3", "Q4", "Fy", "Fy2023", "Fy2024", "Fy2025", "Fy2026",
    }
    for w in words:
        if w not in stopwords and w.title() not in stopwords and w not in found:
            found.append(w)

    return found


def _entity_search(entity: str, top_k: int) -> list[RetrievedChunk]:
    """Retrieve chunks specifically matching an entity via ChromaDB document and metadata filtering."""
    collection = _get_chroma_collection()
    chunks = []
    seen = set()

    for ent_variant in [entity.upper(), entity.title(), entity.lower()]:
        # 1. Document content search
        try:
            res = collection.get(
                where_document={"$contains": ent_variant},
                limit=top_k * 2,
                include=["documents", "metadatas"],
            )
            for doc, meta in zip(res.get("documents", []), res.get("metadatas", [])):
                key = doc[:100]
                if key not in seen:
                    seen.add(key)
                    chunks.append(
                        RetrievedChunk(
                            text=doc,
                            source=(meta or {}).get("source", "unknown"),
                            score=1.0,
                            metadata=meta or {},
                        )
                    )
        except Exception as exc:
            logger.debug(f"Entity document search for '{ent_variant}' failed: {exc}")

        # 2. Source filename metadata match
        try:
            all_data = collection.get(
                limit=top_k * 5,
                include=["documents", "metadatas"],
            )
            for doc, meta in zip(all_data.get("documents", []), all_data.get("metadatas", [])):
                src = (meta or {}).get("source", "").lower()
                if entity.lower() in src:
                    key = doc[:100]
                    if key not in seen:
                        seen.add(key)
                        chunks.append(
                            RetrievedChunk(
                                text=doc,
                                source=(meta or {}).get("source", "unknown"),
                                score=1.0,
                                metadata=meta or {},
                            )
                        )
        except Exception as exc:
            logger.debug(f"Entity metadata search failed: {exc}")

    return chunks


def hybrid_search(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Combine vector + keyword + entity-filtered search via Reciprocal Rank Fusion."""
    cfg = get_settings()
    k = top_k or cfg.top_k_retrieval

    if get_document_count() == 0:
        logger.warning("Vector store is empty — no documents indexed yet")
        return []

    entities = _extract_query_entities(query)

    # 1. Single-entity query: strictly filter and prioritize chunks for this company
    if len(entities) == 1:
        target_ent = entities[0]
        entity_chunks = _entity_search(target_ent, k * 2)
        if entity_chunks:
            # Rank entity chunks by question keyword relevance
            scored = []
            q_words = [w for w in query.lower().split() if len(w) > 2]
            for c in entity_chunks:
                text_lower = c.text.lower()
                keyword_hits = sum(1 for w in q_words if w in text_lower)
                scored.append((keyword_hits, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored[:k]]

    # 2. Multi-entity comparison retrieval: retrieve for each entity separately
    if len(entities) >= 2:
        multi_chunks = []
        q_words = [w for w in query.lower().split() if len(w) > 2]
        for ent in entities:
            ent_results = _entity_search(ent, k)
            scored = []
            for c in ent_results:
                text_lower = c.text.lower()
                keyword_hits = sum(1 for w in q_words if w in text_lower)
                scored.append((keyword_hits, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            multi_chunks.extend([item[1] for item in scored[: max(2, k // len(entities) + 1)]])
        if multi_chunks:
            return multi_chunks[:k]

    # 3. General hybrid search (when no specific company entity is targeted)
    vector_results = _vector_search(query, k * 2)
    bm25_results = _bm25_search(query, k * 2)

    fused = _reciprocal_rank_fusion(
        vector_results,
        bm25_results,
        weight_a=cfg.vector_weight,
        weight_b=cfg.bm25_weight,
    )
    return fused[:k]


def get_document_count() -> int:
    return _get_chroma_collection().count()



