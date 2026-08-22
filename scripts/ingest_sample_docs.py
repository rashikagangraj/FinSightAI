"""
Ingest all sample documents into the vector store.
Run: python scripts/ingest_sample_docs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.core.logging import setup_logging
from src.rag.indexer import get_document_count, ingest_file, reset_collection

setup_logging("INFO")
console = Console()

SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "examples" / "sample_docs"


def main() -> None:
    console.print("\n[bold cyan]FinSight AI — Ingesting Sample Financial Documents[/bold cyan]\n")
    reset_collection()



    files = list(SAMPLE_DOCS_DIR.iterdir())
    supported = {".txt", ".md", ".pdf"}
    files = [f for f in files if f.is_file() and f.suffix.lower() in supported]

    if not files:
        console.print(f"[red]No files found in {SAMPLE_DOCS_DIR}[/red]")
        sys.exit(1)

    table = Table(title="Ingestion Results", show_header=True, header_style="bold magenta")
    table.add_column("File")
    table.add_column("Nodes Indexed", justify="right")
    table.add_column("Status", justify="center")

    total_nodes = 0
    for f in sorted(files):
        try:
            nodes = ingest_file(f, source_override=f.name)
            total_nodes += nodes
            table.add_row(f.name, str(nodes), "[green]SUCCESS[/green]")
        except Exception as exc:
            table.add_row(f.name, "0", f"[red]FAIL ({exc})[/red]")

    console.print(table)
    console.print(f"\n[bold green]Total nodes in vector store: {get_document_count()}[/bold green]")
    console.print("\n[dim]Ready! Run: python examples/demo_cli.py demo[/dim]\n")



if __name__ == "__main__":
    main()
