from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.agents.graph import run_agent
from src.api.schemas import QueryRequest, QueryResponse
from src.core.config import get_settings
from src.core.logging import get_logger
from src.rag.retriever import get_document_count

router = APIRouter(prefix="/query", tags=["query"])
logger = get_logger(__name__)


@router.post("/", response_model=QueryResponse)
async def query_agent(request: QueryRequest) -> QueryResponse:
    if get_document_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. POST a file to /documents/ingest first.",
        )

    try:
        state = await asyncio.get_event_loop().run_in_executor(
            None, run_agent, request.query
        )
        return QueryResponse(
            query=request.query,
            answer=state.get("answer", ""),
            sources=state.get("sources", []),
            intent=state.get("intent", "unknown"),
            agent_trace=state.get("agent_trace", []),
            tokens_estimated=len(state.get("answer", "").split()) * 2,
        )
    except Exception as exc:
        logger.error(f"Agent error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/stream")
async def query_agent_stream(q: str) -> StreamingResponse:
    """SSE streaming endpoint. Usage: GET /query/stream?q=your+question"""
    if get_document_count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet.")

    cfg = get_settings()

    async def event_generator():
        from src.rag.retriever import hybrid_search
        from src.llm.factory import get_llm_client

        chunks = hybrid_search(q, top_k=cfg.top_k_retrieval)
        if not chunks:
            yield f"data: {json.dumps({'token': 'No relevant documents found.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        context = "\n\n---\n\n".join(
            f"[Source: {c.source}]\n{c.text}" for c in chunks
        )
        prompt = (
            f"Answer using only the provided financial documents.\n\n"
            f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer:"
        )
        system = (
            "You are FinAgent, an expert financial analyst. "
            "Ground your answers in the provided context."
        )

        llm = get_llm_client()
        for token in llm.stream(prompt, system=system):
            yield f"data: {json.dumps({'token': token})}\n\n"
            await asyncio.sleep(cfg.stream_chunk_delay if hasattr(cfg, 'stream_chunk_delay') else 0.02)

        sources = list({c.source for c in chunks})
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
