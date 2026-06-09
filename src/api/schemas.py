from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Financial question to answer")
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = Field(default=False)


class SourceDoc(BaseModel):
    source: str
    score: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    intent: str
    agent_trace: list[str]
    tokens_estimated: int | None = None


class DocumentIngestResponse(BaseModel):
    message: str
    nodes_indexed: int
    filename: str


class DocumentListResponse(BaseModel):
    sources: list[str]
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    llm_backend: str
    model: str
    embed_model: str
    document_chunks: int
    version: str = "0.1.0"
