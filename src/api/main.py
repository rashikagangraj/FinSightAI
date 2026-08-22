from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.documents import router as documents_router
from src.api.routes.query import router as query_router
from src.api.schemas import HealthResponse
from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging
from src.rag.retriever import get_document_count

setup_logging(get_settings().log_level)
logger = get_logger(__name__)


def create_app() -> FastAPI:
    cfg = get_settings()
    app = FastAPI(
        title="FinAgent RAG API",
        description=(
            "Agentic RAG system for financial document intelligence. "
            "Supports OpenAI and local Ollama backends."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(documents_router)
    app.include_router(query_router)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            llm_backend=cfg.llm_backend,
            model=cfg.active_model,
            embed_model=cfg.active_embed_model,
            document_chunks=get_document_count(),
        )

    @app.get("/api/info", tags=["system"])
    async def api_info():
        return {
            "name": "FinAgent RAG API",
            "version": "0.1.0",
            "status": "online",
            "llm_backend": cfg.llm_backend,
            "model": cfg.active_model,
            "embed_model": cfg.active_embed_model,
            "document_chunks": get_document_count(),
            "endpoints": [
                "/health",
                "/api/info",
                "/documents/",
                "/documents/ingest",
                "/documents/{source_name}",
                "/documents/clear",
                "/documents/seed-samples",
                "/query/",
                "/query/ratio",
                "/query/stream",
                "/docs",
            ],
        }

    @app.get("/", tags=["system"])
    async def root():
        import os
        from fastapi.responses import HTMLResponse
        html_path = os.path.join(os.path.dirname(__file__), "index.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>FinAgent API is running</h1>")

    logger.info(f"FinAgent API ready | backend={cfg.llm_backend} model={cfg.active_model}")
    return app



app = create_app()


def run() -> None:
    cfg = get_settings()
    uvicorn.run("src.api.main:app", host=cfg.api_host, port=cfg.api_port, reload=True)


if __name__ == "__main__":
    run()
