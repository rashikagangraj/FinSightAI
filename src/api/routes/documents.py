from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from src.api.schemas import (
    DocumentActionResponse,
    DocumentIngestResponse,
    DocumentListResponse,
    SeedSampleResponse,
)
from src.core.logging import get_logger
from src.rag.indexer import get_document_count, ingest_file, list_indexed_sources

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)

ALLOWED_EXTENSIONS = {".txt", ".md", ".pdf", ".csv", ".json"}


@router.post("/ingest", response_model=DocumentIngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> DocumentIngestResponse:
    suffix = Path(file.filename or "file.txt").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        nodes = ingest_file(tmp_path, source_override=file.filename)
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


@router.delete("/{source_name}", response_model=DocumentActionResponse)
async def delete_document(source_name: str) -> DocumentActionResponse:
    from src.rag.indexer import delete_source
    deleted = delete_source(source_name)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"Document '{source_name}' not found in vector index")
    return DocumentActionResponse(
        message=f"Deleted document '{source_name}'",
        deleted_chunks=deleted,
    )


@router.post("/clear", response_model=DocumentActionResponse)
async def clear_documents() -> DocumentActionResponse:
    from src.rag.indexer import clear_all_documents
    cleared = clear_all_documents()
    return DocumentActionResponse(
        message="Cleared all indexed documents",
        cleared_chunks=cleared,
    )


@router.post("/seed-samples", response_model=SeedSampleResponse)
async def seed_samples() -> SeedSampleResponse:
    from src.rag.indexer import seed_sample_documents
    try:
        seeded = seed_sample_documents()
        total = get_document_count()
        return SeedSampleResponse(
            message=f"Successfully seeded {len(seeded)} sample files into vector store",
            seeded_files=seeded,
            total_chunks=total,
        )
    except Exception as exc:
        logger.error(f"Failed to seed samples: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

