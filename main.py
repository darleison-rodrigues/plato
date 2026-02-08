#!/usr/bin/env python3
"""
Contexter - Local-first PDF to Knowledge Graph Pipeline
Production-ready TUI with real backend integration.
"""
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
import sys
from pathlib import Path
from typing import Optional, List

# -----------------------------------------------------------------------------
# PATCH: Pydantic V1 Compatibility (Python 3.14 fix)
# Must be imported BEFORE any module that uses ChromaDB/Pydantic
# -----------------------------------------------------------------------------
try:
    import contexter.patch
except ImportError:
    pass 

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt

# Real backend imports
try:
    from contexter.core import Pipeline
    from contexter.queue import QueueManager
    from contexter.config import get_config
except ImportError as e:
    print(f"Critical Error: Failed to import Contexter modules: {e}")
    sys.exit(1)

# Initialize Rich Console
console = Console()
app = typer.Typer(help="PDF to Knowledge Graph for researchers", add_completion=False)

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def handle_error(message: str, exception: Exception = None):
    """Unified error handling"""
    console.print(f"[bold red]✗ Error:[/bold red] {message}")
    if exception:
        console.print(f"[dim]{exception}[/dim]")
    raise typer.Exit(1)

def get_pipeline() -> Pipeline:
    """Initialize pipeline with error handling"""
    try:
        return Pipeline()
    except Exception as e:
        handle_error("Failed to initialize pipeline. Check config/Ollama.", e)

def validate_pdf(path: Path) -> Path:
    """Validate PDF file existence"""
    if not path.exists():
        handle_error(f"File not found: {path}")
    if path.suffix.lower() != '.pdf':
        handle_error(f"Not a PDF file: {path}")
    return path

# -----------------------------------------------------------------------------
# CLI COMMANDS
# -----------------------------------------------------------------------------

@app.command()
def add(
    pdf: Path,
    tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Tags for organization"),
    priority: str = typer.Option("medium", "--priority", "-p", help="Reading priority (low/medium/high)"),
    queue: bool = typer.Option(True, "--queue/--no-queue", help="Add to reading queue")
):
    """Add a PDF to your collection (Parse -> Vector Store -> KG)"""
    pdf = validate_pdf(pdf)
    pipeline = get_pipeline()
    
    console.print(f"[bold cyan]Processing {pdf.name}...[/bold cyan]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Running pipeline...", total=None)
            
            # Real processing call
            result = pipeline.process_pdf(str(pdf))
            
            if result.get("errors"):
                progress.stop()
                console.print(f"[bold red]Pipeline encountered errors:[/bold red]")
                for err in result["errors"]:
                    console.print(f"  - {err}")
                raise typer.Exit(1)

        # Success Output
        entity_count = sum(len(v) for v in result.get('entities', {}).values() if isinstance(v, list))
        relation_count = len(result.get('relations', []))
        
        console.print(f"[bold green]✓[/bold green] Successfully added {pdf.name}")
        console.print(f"  • Extracted {entity_count} entities")
        console.print(f"  • Found {relation_count} relations")
        console.print(f"  • Saved to: {result.get('output_dir', 'output/')}")

        # Add to Queue
        if queue:
            qm = QueueManager()
            if qm.add(pdf.stem, priority, tags):
                console.print(f"  • Added to reading queue ([yellow]{priority}[/yellow])")
            else:
                console.print(f"  • Already in queue")

    except Exception as e:
        handle_error("Processing failed", e)

@app.command()
def search(
    query: str,
    limit: int = typer.Option(5, "--limit", "-n", help="Max results"),
    graph: bool = typer.Option(True, "--graph/--no-graph", help="Include knowledge graph relations")
):
    """Search your collection (Hybrid: Vector + Graph)"""
    pipeline = get_pipeline()
    
    console.print(f"Searching for: [bold cyan]{query}[/bold cyan]...")
    
    try:
        results = pipeline.search_documents(query, n_results=limit)
        
        combined = results.get("combined_results", [])
        if not combined:
            console.print("[yellow]No results found.[/yellow]")
            return

        # Display Results
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Type", width=12)
        table.add_column("Source", width=20)
        table.add_column("Content / Relation", style="white")
        table.add_column("Score", justify="right", style="green")

        for item in combined:
            score = f"{item.get('relevance_score', 0):.2f}"
            doc_source = item.get('source_document', 'Unknown')
            
            if item['type'] == 'document':
                content = item.get('content', '')[:100].replace('\n', ' ') + "..."
                table.add_row("📄 Document", doc_source, content, score)
            
            elif item['type'] == 'relationship' and graph:
                sub = item.get('subject', '?')
                rel = item.get('relation', 'related')
                obj = item.get('object', '?')
                relation_str = f"{sub} --[{rel}]--> {obj}"
                table.add_row("🔗 Graph", doc_source, relation_str, score)

        console.print(table)

    except Exception as e:
        handle_error("Search failed", e)

@app.command()
def queue(
    action: str = typer.Argument("show", help="show | next | clear"),
    priority: str = typer.Option("medium", "--priority", "-p")
):
    """Manage reading queue"""
    qm = QueueManager()
    
    if action == "show":
        items = qm.list()
        if not items:
            console.print("[yellow]Queue is empty.[/yellow]")
            return
            
        table = Table(title="Reading Queue")
        table.add_column("Priority", style="bold")
        table.add_column("Paper")
        table.add_column("Tags", style="dim")
        table.add_column("Added")

        # Sort by priority for display
        priority_map = {"high": 0, "medium": 1, "low": 2}
        items.sort(key=lambda x: priority_map.get(x.get('priority', 'medium'), 3))

        for item in items:
            prio_color = {"high": "red", "medium": "yellow", "low": "green"}.get(item['priority'], "white")
            table.add_row(
                f"[{prio_color}]{item['priority'].upper()}[/{prio_color}]",
                item['path'],
                ", ".join(item.get('tags', [])),
                item.get('added_at', '')[:10]
            )
        console.print(table)

    elif action == "next":
        next_item = qm.get_next()
        if next_item:
            console.print(f"[bold green]Next up:[/bold green] {next_item['path']}")
            console.print(f"Priority: {next_item['priority']}")
        else:
            console.print("[yellow]No pending items in queue![/yellow]")

    elif action == "clear":
        if Confirm.ask("Clear entire reading queue?"):
            qm.clear()
            console.print("[green]Queue cleared.[/green]")

@app.command()
def stats():
    """Show collection statistics"""
    pipeline = get_pipeline()
    try:
        stats = pipeline.get_vector_store_stats() # Returns dict
        
        console.print("[bold cyan]Collection Statistics[/bold cyan]")
        console.print(f"• Total Documents: [bold]{stats.get('total_documents', 'N/A')}[/bold]")
        console.print(f"• Vector Store Path: {stats.get('persist_directory', 'N/A')}")
        
    except Exception as e:
        handle_error("Failed to fetch stats", e)

@app.command()
def start(
    folder: Optional[Path] = typer.Option(None, "--folder", "-f", help="Folder to process on start"),
):
    """Start interactive session (optional: auto-process a folder)"""
    pipeline = get_pipeline()
    qm = QueueManager() # Initialize queue manager for interactive use
    
    if folder:
        if not folder.exists() or not folder.is_dir():
             handle_error(f"Invalid folder: {folder}")
        
        console.print(f"[bold cyan]Processing folder: {folder}[/bold cyan]")
        results = pipeline.process_directory_parallel(str(folder))
        console.print(f"[green]Processed {len(results)} files.[/green]")

    # Interactive loop
    console.print(Panel.fit(
        "[bold cyan]Contexter Interactive Session[/bold cyan]\n"
        "[dim]Type 'help' for commands, 'quit' to exit[/dim]",
        border_style="cyan"
    ))
    
    while True:
        try:
            cmd_input = Prompt.ask("\n[bold green]contexter>[/bold green]").strip()
            if not cmd_input: continue
            
            cmd_parts = cmd_input.split()
            action = cmd_parts[0].lower()
            args = cmd_parts[1:]
            
            if action in ('quit', 'exit', 'q'):
                console.print("[yellow]Goodbye![/yellow]")
                break

            elif action == 'help':
                console.print(Panel(
                    "Available Commands:\n"
                    "  add <file> [tags]     : Add a PDF to collection\n"
                    "  search <query>        : Search knowledge base\n"
                    "  queue [show|next|clear]: Manage reading queue\n"
                    "  stats                 : Show collection stats\n"
                    "  quit                  : Exit session",
                    title="Help"
                ))

            elif action == 'add':
                if not args:
                    console.print("[red]Usage: add <file.pdf> [tags][/red]")
                    continue
                pdf_path = Path(args[0])
                # Interactive mode simple add (defaults priority=medium, queue=True)
                if not pdf_path.exists():
                     console.print(f"[red]File not found: {pdf_path}[/red]")
                     continue
                
                # We can reuse the typer command logic here or call pipeline directly
                # Direct call for simplicity in loop
                console.print(f"[cyan]Processing {pdf_path.name}...[/cyan]")
                res = pipeline.process_pdf(str(pdf_path))
                if not res.get("errors"):
                    console.print(f"[green]Success![/green] Entities: {sum(len(v) for v in res.get('entities', {}).values())}")
                    qm.add(pdf_path.stem)
                    console.print("Added to reading queue.")
                else:
                     console.print(f"[red]Errors during processing:[/red]")
                     for e in res["errors"]: console.print(f"- {e}")

            elif action == 'search':
                if not args:
                    console.print("[red]Usage: search <query>[/red]")
                    continue
                query = " ".join(args)
                results = pipeline.search_documents(query, n_results=3)
                combined = results.get("combined_results", [])
                if not combined:
                    console.print("[yellow]No results.[/yellow]")
                else:
                    for r in combined:
                         score = r.get('relevance_score', 0)
                         content = r.get('content', '')[:100].replace('\n', ' ')
                         console.print(f"[green]{score:.2f}[/green]: {content}...")

            elif action == 'stats':
                stats = pipeline.get_vector_store_stats()
                console.print(f"Total Docs: {stats.get('total_documents')}")

            elif action == 'queue':
                subcmd = args[0] if args else "show"
                if subcmd == "show":
                    items = qm.list()
                    if not items: console.print("Queue empty.")
                    else:
                        for i in items: console.print(f"- [{i['priority']}] {i['path']}")
                elif subcmd == "next":
                     n = qm.get_next()
                     if n: console.print(f"Next: {n['path']}")
                     else: console.print("Nothing pending.")
                elif subcmd == "clear":
                    qm.clear()
                    console.print("Cleared.")

            else:
                console.print(f"[red]Unknown command: {action}[/red]")
                
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'quit' to exit[/yellow]")
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")

if __name__ == "__main__":
    app()
