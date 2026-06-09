from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from src.api.schemas import DocumentIngestResponse, DocumentListResponse
from src.core.logging import get_logger
from src.rag.indexer import get_document_count, ingest_file, list_indexed_sources

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf"}


@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> DocumentIngestResponse:
    suffix = Path(file.filename or "file.txt").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        nodes = ingest_file(tmp_path)
        logger.info(f"Ingested '{file.filename}' → {nodes} nodes")
        return DocumentIngestResponse(
            message=f"Successfully indexed '{file.filename}'",
            nodes_indexed=nodes,
            filename=file.filename or "unknown",
        )
    except Exception as exc:
        logger.error(f"Ingest failed for '{file.filename}': {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    sources = list_indexed_sources()
    return DocumentListResponse(sources=sources, total_chunks=get_document_count())
