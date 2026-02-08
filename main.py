#!/usr/bin/env python3
"""
Platograph - Local-first Context Preparation Assistant 🦫
"""
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import sys
from pathlib import Path

# Real backend imports
try:
    from plato.graph_rag import GraphRAGPipeline
    from plato.parser import DocumentParser
    from plato.config import get_config
    from plato.agents.ascii_art import plato, PLATYPUS_MINI
except ImportError as e:
    print(f"Critical Error: Failed to import Platograph modules: {e}")
    sys.exit(1)

# Initialize Rich Console
console = Console()
app = typer.Typer(help="Platograph: Research Context Assistant", add_completion=False)

# --- Helper Functions ---

def handle_error(message: str, exception: Exception = None):
    """Unified error handling"""
    console.print(f"[bold red]✗ Error:[/bold red] {message}")
    if exception:
        console.print(f"[dim]{exception}[/dim]")
    raise typer.Exit(1)

# --- CLI Commands ---

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Platograph Entry Point"""
    if ctx.invoked_subcommand is None:
        console.print(plato, style="cyan")
        console.print(Panel.fit(
            f"[bold cyan]Platograph {PLATYPUS_MINI}[/bold cyan]\n"
            "[dim]A local-first, schema-driven knowledge extraction tool.[/dim]",
            border_style="cyan"
        ))
        console.print("\n[bold]Quick Start:[/bold]")
        console.print("  1. Place documents in a folder (e.g., 'my_papers/')")
        console.print("  2. Process them into the knowledge graph:")
        console.print("     [cyan]plato process ./my_papers/[/cyan]")
        console.print("  3. Chat with your new knowledge graph:")
        console.print("     [cyan]plato chat[/cyan]")

@app.command()
def process(
    input_dir: Path = typer.Argument(..., help="Directory containing documents to process (PDFs, etc.)"),
):
    """
    🦫 Process documents from a directory into the knowledge graph.
    """
    if not input_dir.is_dir():
        handle_error(f"Invalid input directory: {input_dir}")

    console.print(f"[cyan]{PLATYPUS_MINI} Initializing pipeline...[/cyan]")
    try:
        parser = DocumentParser()
        pipeline = GraphRAGPipeline()
    except Exception as e:
        handle_error("Failed to initialize system. Is Ollama running?", e)

    console.print(f"[cyan]Reading documents from {input_dir}...[/cyan]")
    documents = parser.load_data(input_dir)
    
    if not documents:
        console.print("[yellow]No new documents to process.[/yellow]")
        return

    pipeline.insert_documents(documents)
    
    console.print(f"\n[bold green]✓ Processing complete![/bold green]")
    console.print(f"  - Indexed [bold]{len(documents)}[/bold] documents.")
    console.print("  - You can now run [bold cyan]plato chat[/bold cyan] to talk to your data.")

@app.command()
def chat():
    """
    🦫 Start an interactive chat session with your knowledge graph.
    """
    console.print(f"[cyan]{PLATYPUS_MINI} Loading knowledge graph...[/cyan]")
    
    try:
        pipeline = GraphRAGPipeline()
        console.print("[bold green]✓[/bold green] [dim]Knowledge graph loaded. Type 'exit' to quit.[/dim]")
    except Exception as e:
         handle_error("Failed to load knowledge graph. Have you run 'plato process' yet?", e)

    while True:
        question = Prompt.ask("\n[bold green]You[/bold green]").strip()
        if not question:
            continue
        if question.lower() in ('exit', 'quit'):
            break
        
        with console.status("[bold cyan]Thinking...[/bold cyan]"):
            try:
                answer = pipeline.query(question)
                console.print(f"\n[bold magenta]Plato[/bold magenta]: {answer}")
            except Exception as e:
                console.print(f"[red]Query failed: {e}[/red]")

    console.print("\n[yellow]Chat session ended.[/yellow]")

if __name__ == "__main__":
    app()
