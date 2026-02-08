#!/usr/bin/env python3
"""
plato - Local-first Context Preparation Assistant 🦫
"""
import sys
import asyncio
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text

try:
    # Try importing from the package (when installed or run via python -m)
    from plato.graph_rag import GraphRAGPipeline
    from plato.parser import DocumentParser
    from plato.config import get_config
except ImportError as e:
    # Try adding src to path if running directly
    current_dir = Path(__file__).resolve().parent
    src_dir = current_dir.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        from plato.graph_rag import GraphRAGPipeline
        from plato.parser import DocumentParser
        from plato.config import get_config
    except ImportError as e2:
        print(f"Critical Error: Failed to import plato modules: {e2}")
        sys.exit(1)

# Initialize Rich Console
console = Console()
app = typer.Typer(help="plato: Research Context Assistant", add_completion=False)

# Constants
PLATYPUS_MINI = "🦫"
LOGO = """
               d8b                                       
               88P                      d8P              
              d88                    d888888P            
?88,.d88b,    888       d888b8b        ?88'       d8888b 
`?88'  ?88    ?88      d8P' ?88        88P       d8P' ?88
  88b  d8P     88b     88b  ,88b       88b       88b  d88
  888888P'      88b    `?88P'`88b      `?8b      `?8888P'
  88P'                                                   
 d88                                                     
 ?8P    
"""

def handle_error(message: str, exception: Exception = None):
    """Unified error handling"""
    console.print(f"[bold red]✗ Error:[/bold red] {message}")
    if exception:
        console.print(f"[dim]{exception}[/dim]")
    raise typer.Exit(1)

# Import backend lazily to allow fast CLI startup/help
def get_pipeline():
    try:
        from plato.core import Pipeline
        return Pipeline()
    except Exception as e:
        handle_error("Failed to initialize system.", e)

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Platograph Entry Point"""
    if ctx.invoked_subcommand is None:
        console.print(LOGO, style="cyan")
        console.print(Panel.fit(
            f"[bold cyan]Platograph {PLATYPUS_MINI}[/bold cyan]\n"
            "[dim]A local-first, schema-driven knowledge extraction tool.[/dim]",
            border_style="cyan"
        ))
        console.print("\n[bold]Available Commands:[/bold]")
        console.print("  [cyan]scan [dir][/cyan]    - List documents found in a directory")
        console.print("  [cyan]process [dir][/cyan] - Process documents into the Knowledge Graph")
        console.print("  [cyan]chat[/cyan]           - Chat with your knowledge base")
        console.print("  [cyan]show[/cyan]           - Show graph statistics")

@app.command()
def scan(
    directory: Path = typer.Argument(
        "./plato/documents", 
        help="Directory to scan",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True
    )
):
    """Scan a directory for supported documents."""
    console.print(f"[cyan]Scanning {directory}...[/cyan]")
    files = list(directory.glob("*.pdf")) + list(directory.glob("*.md"))
    
    if not files:
        console.print("[yellow]No PDF or Markdown files found.[/yellow]")
        return

    table = Table(title=f"Found {len(files)} Documents")
    table.add_column("File Name", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Type", style="green")

    for f in files:
        size_kb = f.stat().st_size / 1024
        table.add_row(f.name, f"{size_kb:.1f} KB", f.suffix)

    console.print(table)

@app.command()
def process(
    input_dir: Path = typer.Argument(
        "./plato/documents",
        help="Directory containing documents to process"
    )
):
    """
    Process documents into the Knowledge Graph.
    Uses async pipeline with live progress tracking.
    """
    if not input_dir.exists():
        console.print(f"[yellow]Directory {input_dir} does not exist. Creating it...[/yellow]")
        input_dir.mkdir(parents=True, exist_ok=True)
        console.print("[yellow]Please add PDF/MD files to this directory and run again.[/yellow]")
        return

    pipeline = get_pipeline()
    files = list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.md"))
    
    if not files:
        console.print("[yellow]No documents found to process.[/yellow]")
        return

    console.print(f"[bold cyan]Starting Processing Pipeline for {len(files)} files...[/bold cyan]")
    
    # Progress Bar Context
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:
        
        main_task = progress.add_task("[green]Total Progress", total=len(files))
        
        async def process_all():
            for file_path in files:
                task_id = progress.add_task(f"Processing {file_path.name}...", total=None)
                try:
                    # Run async processing
                    result = await pipeline.process_pdf_async(str(file_path))
                    
                    if result.get("success"):
                        stats = f"[Ent: {result['entities_found']} | Rel: {result['relations_found']}]"
                        progress.print(f"[green]✓ {file_path.name}[/green] {stats}")
                    else:
                        progress.print(f"[red]✗ {file_path.name}[/red] - {result.get('error')}")
                        
                except Exception as e:
                    progress.print(f"[red]Error processing {file_path.name}: {e}[/red]")
                
                progress.remove_task(task_id)
                progress.advance(main_task)

        # Run async loop
        asyncio.run(process_all())

    console.print("\n[bold green]Processing Complete![/bold green]")
    console.print("Run [cyan]plato chat[/cyan] to interact with your data.")

@app.command()
def chat():
    """
    Start an interactive chat session.
    """
    pipeline = get_pipeline()
    console.print(f"[cyan]{PLATYPUS_MINI} Knowledge Graph Loaded.[/cyan]")
    console.print("[dim]Type 'exit' or 'quit' to end session.[/dim]\n")
    
    while True:
        question = Prompt.ask("[bold green]You[/bold green]")
        
        if not question.strip():
            continue
            
        if question.lower().strip() in ('exit', 'quit'):
            break
            
        with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
            try:
                response = pipeline.query(question)
                console.print(f"\n[bold magenta]Plato[/bold magenta]: {response}\n")
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {e}\n")

    console.print("[yellow]Goodbye![/yellow]")

@app.command()
def show():
    """Show knowledge graph statistics."""
    pipeline = get_pipeline()
    console.print("[bold cyan]Reading Graph Stats...[/bold cyan]")
    
    # Simple hack to get stats from the underlying index if possible,
    # or just show where the storage is.
    storage_dir = pipeline.output_dir / "storage"
    
    if storage_dir.exists():
        size = sum(f.stat().st_size for f in storage_dir.glob('**/*') if f.is_file()) / (1024*1024)
        console.print(f"Storage Directory: [blue]{storage_dir}[/blue]")
        console.print(f"Total Size: [magenta]{size:.2f} MB[/magenta]")
        
        # We could add more stats if GraphRAGPipeline exposed them
        # stats = pipeline.graph_rag.get_stats() ...
    else:
        console.print("[yellow]No knowledge graph found yet.[/yellow]")

if __name__ == "__main__":
    app()
