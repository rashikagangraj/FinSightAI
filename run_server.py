"""
FinSight AI — Local Server Runner
-----------------------------------
Run:
    python run_server.py
"""
from __future__ import annotations

import sys
import webbrowser
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn
from rich.console import Console
from rich.panel import Panel

from src.core.config import get_settings
from src.rag.indexer import get_document_count, seed_sample_documents

console = Console(force_terminal=True, legacy_windows=False)


def main():
    cfg = get_settings()

    console.print(
        Panel.fit(
            "[bold cyan]FinSight AI - Financial Intelligence Agent[/bold cyan]\n"
            "[italic white]Turn financial documents into business decisions.[/italic white]\n\n"
            f"[dim]LLM Backend:[/dim] [green]{cfg.llm_backend.upper()}[/green]  "
            f"[dim]Model:[/dim] [yellow]{cfg.active_model}[/yellow]  "
            f"[dim]Embeddings:[/dim] [yellow]{cfg.active_embed_model}[/yellow]\n"
            f"[dim]Host:[/dim] [white]{cfg.api_host}:{cfg.api_port}[/white]\n"
            f"[dim]Dashboard:[/dim] [bold underline blue]http://localhost:{cfg.api_port}[/bold underline blue]\n"
            f"[dim]API Docs:[/dim] [bold underline blue]http://localhost:{cfg.api_port}/docs[/bold underline blue]",
            title="Local Server Starting",
            border_style="cyan",
        )
    )


    doc_count = get_document_count()
    if doc_count == 0:
        console.print("[yellow][i] Vector database is empty. Auto-seeding bundled sample documents...[/yellow]")
        try:
            results = seed_sample_documents()
            console.print(f"[green][OK] Successfully seeded {len(results)} sample reports ({get_document_count()} chunks total)![/green]")
        except Exception as exc:
            console.print(f"[dim]Sample seeding skipped ({exc}). You can upload documents from the dashboard.[/dim]")
    else:
        console.print(f"[green][OK] ChromaDB loaded with {doc_count} document chunks ready for retrieval.[/green]")

    console.print(f"\n[bold green]>>> Server is LIVE at: http://localhost:{cfg.api_port}[/bold green]\n")


    # Start FastAPI with Uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=False,
        log_level=cfg.log_level.lower(),
    )


if __name__ == "__main__":
    main()
