"""
FinAgent CLI Demo
-----------------
Run: python examples/demo_cli.py

Make sure to ingest sample docs first:
    python scripts/ingest_sample_docs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from src.agents.graph import run_agent
from src.core.logging import setup_logging

app = typer.Typer(help="FinAgent RAG — Financial Document Intelligence Demo")
console = Console()

DEMO_QUERIES = [
    "What was Apple's total revenue in fiscal year 2023?",
    "Compare Apple and Tesla's gross margin in 2023",
    "What are the main risks facing Tesla according to their earnings call?",
    "Generate a brief financial summary of the S&P 500 performance in 2023",
]


@app.command()
def ask(
    query: str = typer.Argument(..., help="Financial question to ask"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show agent trace"),
):
    """Ask a financial question to the FinAgent."""
    setup_logging("INFO")
    console.print(Panel(f"[bold cyan]Query:[/bold cyan] {query}", expand=False))

    with console.status("[bold green]Agent is thinking...[/bold green]"):
        state = run_agent(query)

    console.print("\n[bold green]Answer:[/bold green]")
    console.print(Markdown(state["answer"]))

    if state.get("sources"):
        table = Table(title="Sources", show_header=True, header_style="bold magenta")
        table.add_column("Document")
        for src in state["sources"]:
            table.add_row(Path(src).name if src != "unknown" else src)
        console.print(table)

    if verbose and state.get("agent_trace"):
        console.print("\n[dim]Agent Trace:[/dim]")
        for step in state["agent_trace"]:
            console.print(f"  [dim]{step}[/dim]")

    console.print(f"\n[dim]Intent detected: {state.get('intent', '?')}[/dim]")


@app.command()
def demo(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run all 4 demo queries to showcase the agent."""
    setup_logging("WARNING")
    console.print(Panel(
        "[bold]FinAgent RAG — Demo Run[/bold]\nRunning 4 sample queries...",
        style="cyan",
    ))

    for i, q in enumerate(DEMO_QUERIES, 1):
        console.rule(f"[bold]Query {i}/{len(DEMO_QUERIES)}[/bold]")
        console.print(f"[cyan]Q: {q}[/cyan]\n")
        try:
            state = run_agent(q)
            console.print(Markdown(state["answer"][:600] + ("..." if len(state["answer"]) > 600 else "")))
            console.print(f"[dim]Sources: {', '.join(Path(s).name for s in state['sources'][:3])}[/dim]")
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
        console.print()


@app.command()
def evaluate():
    """Run evaluation on demo_queries.json and print a score table."""
    setup_logging("WARNING")
    queries_path = Path(__file__).parent / "demo_queries.json"
    queries = json.loads(queries_path.read_text())

    table = Table(title="FinAgent Evaluation Results", show_header=True, header_style="bold")
    table.add_column("#", style="dim")
    table.add_column("Category")
    table.add_column("Query (truncated)")
    table.add_column("Keywords Found", justify="center")
    table.add_column("Pass", justify="center")

    passed = 0
    for item in queries:
        state = run_agent(item["query"])
        answer_lower = state["answer"].lower()
        found = sum(1 for kw in item["expected_keywords"] if kw.lower() in answer_lower)
        total = len(item["expected_keywords"])
        passed_this = found >= (total // 2)
        if passed_this:
            passed += 1
        table.add_row(
            str(item["id"]),
            item["category"],
            item["query"][:50] + "...",
            f"{found}/{total}",
            "[green]PASS[/green]" if passed_this else "[red]FAIL[/red]",
        )

    console.print(table)
    console.print(f"\n[bold]Score: {passed}/{len(queries)} queries passed[/bold]")



if __name__ == "__main__":
    app()
