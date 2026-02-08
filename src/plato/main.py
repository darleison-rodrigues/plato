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
    from plato.visualize import GraphVisualizer
    import json
    import yaml
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

@app.command()
def visualize(
    output: Path = typer.Option("graph.html", help="Output file path"),
    format: str = typer.Option("mermaid", help="Format: mermaid, graphviz, or json"),
    limit: int = typer.Option(50, help="Max nodes to display"),
    filter_type: str = typer.Option(None, help="Filter by entity type: PERSON, ORG, CONCEPT")
):
    """
    Generate interactive graph visualization.
    
    Examples:
        plato visualize                           # Default: graph.html
        plato visualize --limit 30                # Limit to 30 nodes
        plato visualize --filter-type PERSON      # Only show people
        plato visualize --format graphviz -o graph.dot
    """
    pipeline = get_pipeline()
    
    console.print("[cyan]Extracting graph data...[/cyan]")
    
    try:
        # Get data from GraphRAG
        data = pipeline.graph_rag.export_graph_data()
        
        entities = data['entities']
        relations = data['relations']
        
        # Filter if requested
        if filter_type:
            entities = [e for e in entities if e['type'] == filter_type.upper()]
            entity_names = {e['name'] for e in entities}
            relations = [r for r in relations 
                        if r['source'] in entity_names and r['target'] in entity_names]
        
        if not entities:
            console.print("[yellow]No entities found. Process some documents first.[/yellow]")
            console.print("[dim]Run: plato process ./documents[/dim]")
            return
        
        console.print(f"[green]Found {len(entities)} entities, {len(relations)} relations[/green]")
        
        # Generate diagram
        with console.status(f"[cyan]Generating {format} diagram...[/cyan]"):
            viz = GraphVisualizer()
            
            if format == "mermaid":
                mermaid_code = viz.generate_mermaid(entities, relations, max_nodes=limit)
                html = viz.wrap_mermaid_html(mermaid_code, title="Plato Knowledge Graph")
                output.write_text(html, encoding='utf-8')
                
            elif format == "graphviz":
                dot_code = viz.generate_graphviz(entities, relations)
                output.write_text(dot_code, encoding='utf-8')
                
            elif format == "json":
                output.write_text(json.dumps(data, indent=2), encoding='utf-8')
            
            else:
                console.print(f"[red]Unknown format: {format}[/red]")
                return
        
        console.print(f"\n[bold green]✓ Visualization created:[/bold green] {output}")
        
        if format == "mermaid":
            console.print(f"[dim]Open {output} in your browser to view the interactive graph[/dim]")
        elif format == "graphviz":
            console.print(f"[dim]Render with: dot -Tpng {output} -o graph.png[/dim]")
        
        # Show stats
        table = Table(title="Graph Statistics")
        table.add_column("Type", style="cyan")
        table.add_column("Count", style="magenta")
        
        entity_types = {}
        for e in entities:
            entity_types[e['type']] = entity_types.get(e['type'], 0) + 1
        
        for ent_type, count in sorted(entity_types.items()):
            table.add_row(ent_type, str(count))
        
        table.add_row("—" * 15, "—" * 5, style="dim")
        table.add_row("TOTAL ENTITIES", str(len(entities)), style="bold")
        table.add_row("TOTAL RELATIONS", str(len(relations)), style="bold")
        
        console.print(table)
        
    except Exception as e:
        handle_error(f"Failed to generate visualization", e)


@app.command()
def export(
    output: Path = typer.Option("context.md", help="Output file"),
    format: str = typer.Option("markdown", help="markdown, json, or yaml"),
    include_graph: bool = typer.Option(True, help="Include graph diagram in MD")
):
    """
    Export knowledge base to file.
    
    Examples:
        plato export                              # context.md with embedded graph
        plato export --format json -o data.json
        plato export --no-include-graph           # Just text, no diagram
    """
    pipeline = get_pipeline()
    
    console.print(f"[cyan]Exporting to {format}...[/cyan]")
    
    try:
        data = pipeline.graph_rag.export_graph_data()
        
        if format == "markdown":
            # Generate comprehensive MD file
            lines = [
                "# Knowledge Graph Export",
                f"\nGenerated by Plato 🦫",
                f"\n## Summary",
                f"- **Entities:** {len(data['entities'])}",
                f"- **Relations:** {len(data['relations'])}",
            ]
            
            # Add graph diagram
            if include_graph:
                viz = GraphVisualizer()
                mermaid = viz.generate_mermaid(data['entities'], data['relations'], max_nodes=30)
                lines.extend([
                    "\n## Graph Visualization",
                    "```mermaid",
                    mermaid,
                    "```"
                ])
            
            # Entities by type
            lines.append("\n## Entities")
            entity_types = {}
            for e in data['entities']:
                if e['type'] not in entity_types:
                    entity_types[e['type']] = []
                entity_types[e['type']].append(e['name'])
            
            for ent_type, names in sorted(entity_types.items()):
                lines.append(f"\n### {ent_type}")
                for name in sorted(names):
                    lines.append(f"- {name}")
            
            # Relations
            lines.append("\n## Relations")
            for rel in data['relations']:
                lines.append(f"- **{rel['source']}** → *{rel['relation']}* → **{rel['target']}**")
            
            output.write_text("\n".join(lines), encoding='utf-8')
            
        elif format == "json":
            output.write_text(json.dumps(data, indent=2), encoding='utf-8')
            
        elif format == "yaml":
            output.write_text(yaml.dump(data, default_flow_style=False), encoding='utf-8')
        
        console.print(f"[bold green]✓ Exported to {output}[/bold green]")
        
    except Exception as e:
        handle_error(f"Failed to export", e)

if __name__ == "__main__":
    app()
