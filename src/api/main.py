from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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
        title="FinSight AI API",
        description=(
            "Financial Intelligence Agent — Turn financial documents into business decisions. "
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
            "name": "FinSight AI API",
            "subtitle": "Financial Intelligence Agent",
            "tagline": "Turn financial documents into business decisions.",
            "version": "0.1.0",
            "status": "online",
            "llm_backend": cfg.llm_backend,
            "model": cfg.active_model,
            "embed_model": cfg.active_embed_model,
            "document_chunks": get_document_count(),
            "endpoints": {
                "health": "/health",
                "info": "/api/info",
                "query": "/query/",
                "query_stream": "/query/stream",
                "ratio": "/query/ratio",
                "documents": "/documents/",
                "upload": "/documents/upload",
                "seed": "/documents/seed",
                "docs": "/docs",
            },
        }

    # Serve Static Dashboard at Root
    static_html_path = Path(__file__).parent / "index.html"
    if static_html_path.exists():
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def serve_dashboard():
            return HTMLResponse(content=static_html_path.read_text(encoding="utf-8"))
    else:
        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def serve_fallback():
            return HTMLResponse(content="<h1>FinSight AI API is running</h1>")

    logger.info(f"FinSight AI API ready | backend={cfg.llm_backend} model={cfg.active_model}")
    return app



app = create_app()


def run() -> None:
    cfg = get_settings()
    uvicorn.run("src.api.main:app", host=cfg.api_host, port=cfg.api_port, reload=True)


if __name__ == "__main__":
    run()
