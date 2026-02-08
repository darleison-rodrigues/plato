# PLATO TUI Integration Example

Showing how the refactored `OllamaClient` connects to the terminal interface for real-time feedback.

```python
"""
PLATO TUI: Terminal interface for context preparation
Integrates refactored OllamaClient for streaming extraction feedback
"""
import asyncio
import json
from pathlib import Path
from typing import List
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
import typer

from ollama_client import OllamaClient, ExtractionResult


class PlatoTUI:
    """Terminal UI for context preparation workflow"""
    
    def __init__(self, instance: str = "localhost:2222"):
        """
        Args:
            instance: Ollama backend address (local or remote workstation)
        """
        self.console = Console()
        self.client = OllamaClient()
        self.instance = instance
        self.knowledge_base = []  # Store results for downstream processing
    
    async def process_pdf(self, pdf_path: str) -> None:
        """
        Process a single PDF with streaming extraction feedback
        Shows real-time progress as extraction completes
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            self.console.print(f"[red]Error: {pdf_path} not found[/red]")
            return
        
        doc_id = pdf_path.stem
        
        self.console.print(f"\n[bold]📚 Processing: {pdf_path.name}[/bold]")
        self.console.print(f"   ID: {doc_id}")
        
        # Progress layout with multiple sections
        layout = Layout()
        layout.split_column(
            Layout(name="entities", size=10),
            Layout(name="relations", size=10),
            Layout(name="summary", size=5)
        )
        
        entity_table = Table(title="🏷️  Entities Discovered")
        entity_table.add_column("Type", style="cyan")
        entity_table.add_column("Count", style="magenta")
        entity_table.add_column("Samples", style="green")
        
        relation_table = Table(title="🔗 Relations Found")
        relation_table.add_column("Subject", style="cyan")
        relation_table.add_column("Relation", style="yellow")
        relation_table.add_column("Object", style="green")
        
        try:
            # Process document in chunks
            results = await self.client.process_document_complete(
                str(pdf_path), 
                doc_id
            )
            
            all_entities = {}
            all_relations = []
            all_summaries = []
            
            # Aggregate results across chunks
            for i, result in enumerate(results):
                chunk_num = f"{result.doc_context.chunk_index + 1}/{result.doc_context.total_chunks}"
                
                # Update entities table
                for entity_type, entity_list in result.entities.items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = []
                    all_entities[entity_type].extend(entity_list)
                    
                    # Show first 3 examples
                    samples = ", ".join(entity_list[:3])
                    entity_table.add_row(
                        entity_type,
                        str(len(entity_list)),
                        samples
                    )
                
                # Accumulate relations
                all_relations.extend(result.relations)
                for rel in result.relations[:3]:  # Show first 3
                    relation_table.add_row(
                        rel['subject'],
                        rel['relation'],
                        rel['object']
                    )
                
                # Collect summaries
                if result.summary:
                    all_summaries.append(result.summary)
                
                # Log chunk completion
                self.console.print(
                    f"   [blue]✓ Chunk {chunk_num}[/blue] - "
                    f"{len(result.entities)} entity types, "
                    f"{len(result.relations)} relations"
                )
            
            # Display final tables
            self.console.print("\n")
            self.console.print(entity_table)
            self.console.print("\n")
            self.console.print(relation_table)
            
            # Summary section
            if all_summaries:
                combined_summary = " ".join(all_summaries[:2])  # First 2 chunks
                self.console.print(Panel(
                    combined_summary[:300] + "...",
                    title="📝 Summary (from first chunks)",
                    border_style="blue"
                ))
            
            # Statistics
            total_entities = sum(len(v) for v in all_entities.values())
            self.console.print(f"\n[bold green]✅ Extraction Complete[/bold green]")
            self.console.print(f"   Total entities: {total_entities}")
            self.console.print(f"   Total relations: {len(all_relations)}")
            self.console.print(f"   Chunks processed: {len(results)}")
            
            # Store for knowledge graph
            self.knowledge_base.append({
                "doc_id": doc_id,
                "doc_path": str(pdf_path),
                "entities": all_entities,
                "relations": all_relations,
                "chunks": len(results)
            })
            
        except Exception as e:
            self.console.print(f"[red]Error processing PDF: {e}[/red]")
    
    async def process_directory(self, dir_path: str, pattern: str = "*.pdf") -> None:
        """
        Process multiple PDFs with progress tracking
        Shows batch processing status
        """
        dir_path = Path(dir_path)
        pdf_files = list(dir_path.glob(pattern))
        
        if not pdf_files:
            self.console.print(f"[yellow]No PDFs found in {dir_path}[/yellow]")
            return
        
        self.console.print(f"\n[bold]📂 Found {len(pdf_files)} PDFs[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", justify="right"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing PDFs...",
                total=len(pdf_files)
            )
            
            for pdf_file in pdf_files:
                await self.process_pdf(str(pdf_file))
                progress.update(task, advance=1)
    
    def suggest_workflows(self) -> None:
        """
        Analyze extracted knowledge and suggest next steps
        Implements the "Suggest" phase of PLATO workflow
        """
        if not self.knowledge_base:
            self.console.print("[yellow]No documents processed yet[/yellow]")
            return
        
        # Analyze what was extracted
        total_entities = sum(
            sum(len(v) for v in doc['entities'].values())
            for doc in self.knowledge_base
        )
        total_relations = sum(len(doc['relations']) for doc in self.knowledge_base)
        
        self.console.print("\n[bold]💡 Suggested Workflows[/bold]")
        
        suggestions = []
        
        if total_entities > 50:
            suggestions.append(
                "🏢 [cyan]Build Organization Map[/cyan] - "
                "You have many entities. Create a visual org structure."
            )
        
        if total_relations > 30:
            suggestions.append(
                "🔗 [cyan]Create Knowledge Graph[/cyan] - "
                "Strong relation density. Build an interactive graph."
            )
        
        if any('DATE' in doc['entities'] or 'TIME' in doc['entities'] 
               for doc in self.knowledge_base):
            suggestions.append(
                "📅 [cyan]Build Timeline[/cyan] - "
                "Date/time entities found. Extract chronological flow."
            )
        
        if total_entities > 20:
            suggestions.append(
                "📊 [cyan]Build Comparison Table[/cyan] - "
                "Compare entities across documents."
            )
        
        # Display suggestions
        for i, suggestion in enumerate(suggestions, 1):
            self.console.print(f"   {i}. {suggestion}")
        
        self.console.print("\n[dim]Run: plato build --workflow comparison_table[/dim]")
    
    async def export_context(self, output_path: str) -> None:
        """
        Export extracted knowledge as formatted context.md
        Ready to feed into ChatGPT/Claude for report generation
        """
        if not self.knowledge_base:
            self.console.print("[yellow]No documents to export[/yellow]")
            return
        
        output_path = Path(output_path)
        
        context_md = "# Research Context\n\n"
        
        # Entities section
        context_md += "## Discovered Entities\n\n"
        all_entities = {}
        for doc in self.knowledge_base:
            for entity_type, entities in doc['entities'].items():
                if entity_type not in all_entities:
                    all_entities[entity_type] = set()
                all_entities[entity_type].update(entities)
        
        for entity_type, entities in sorted(all_entities.items()):
            context_md += f"### {entity_type}\n"
            for entity in sorted(entities)[:10]:  # Limit to top 10
                context_md += f"- {entity}\n"
            if len(entities) > 10:
                context_md += f"- ... and {len(entities) - 10} more\n"
            context_md += "\n"
        
        # Relations section
        context_md += "## Key Relationships\n\n"
        all_relations = []
        for doc in self.knowledge_base:
            all_relations.extend(doc['relations'])
        
        for rel in all_relations[:20]:  # Show top 20 relations
            context_md += f"- **{rel['subject']}** {rel['relation']} **{rel['object']}**\n"
        
        if len(all_relations) > 20:
            context_md += f"\n_... and {len(all_relations) - 20} more relations_\n"
        
        # Write file
        output_path.write_text(context_md)
        self.console.print(f"\n[green]✅ Context exported to {output_path}[/green]")
    
    def show_knowledge_graph(self) -> None:
        """
        Display knowledge base as JSON for import into graph DB
        """
        if not self.knowledge_base:
            self.console.print("[yellow]No knowledge base[/yellow]")
            return
        
        # Pretty print JSON
        output = json.dumps(self.knowledge_base, indent=2)
        self.console.print("[bold]Knowledge Base (JSON)[/bold]")
        self.console.print(output)


# CLI Commands
app = typer.Typer(help="PLATO - Context Preparation Assistant")


@app.command()
def chat(
    instance: str = typer.Option("localhost:2222", help="Ollama instance (IP:port)")
):
    """Interactive chat with extracted knowledge"""
    tui = PlatoTUI(instance=instance)
    # TODO: Implement interactive chat loop
    tui.console.print("[yellow]Chat mode coming soon[/yellow]")


@app.command()
def scan(
    pdf_dir: str = typer.Argument("./library", help="Directory with PDFs"),
    pattern: str = typer.Option("*.pdf", help="File pattern"),
):
    """Scan and extract from PDF directory"""
    tui = PlatoTUI()
    asyncio.run(tui.process_directory(pdf_dir, pattern))
    tui.suggest_workflows()


@app.command()
def process(
    pdf_path: str = typer.Argument(..., help="PDF file path"),
):
    """Process a single PDF"""
    tui = PlatoTUI()
    asyncio.run(tui.process_pdf(pdf_path))


@app.command()
def export(
    output: str = typer.Option("context.md", help="Output file path"),
):
    """Export knowledge as context.md"""
    tui = PlatoTUI()
    asyncio.run(tui.export_context(output))


@app.command()
def show():
    """Display current knowledge base"""
    tui = PlatoTUI()
    tui.show_knowledge_graph()


if __name__ == "__main__":
    app()
```

## Usage Examples

```bash
# Scan all PDFs in library/
plato scan ./library

# Process single PDF with streaming feedback
plato process ~/Downloads/research.pdf

# Export knowledge as context.md for ChatGPT/Claude
plato export context.md

# View knowledge base as JSON
plato show
```

## Output Example

```
📚 Processing: research.pdf
   ID: research

   ✓ Chunk 1/10 - 15 entity types, 23 relations
   ✓ Chunk 2/10 - 12 entity types, 19 relations
   ✓ Chunk 3/10 - 18 entity types, 25 relations
   ...

🏷️  Entities Discovered
┌──────────┬───────┬────────────────────────────────┐
│ Type     │ Count │ Samples                        │
├──────────┼───────┼────────────────────────────────┤
│ PERSON   │ 47    │ Smith, Johnson, Chen           │
│ ORG      │ 23    │ MIT, Stanford, Google          │
│ GPE      │ 12    │ USA, China, Germany            │
│ DATE     │ 34    │ 2024, Q1 2023, January 2024    │
└──────────┴───────┴────────────────────────────────┘

🔗 Relations Found
┌─────────────────────┬──────────────────┬────────────────────┐
│ Subject             │ Relation         │ Object             │
├─────────────────────┼──────────────────┼────────────────────┤
│ Smith               │ works_at         │ MIT                │
│ MIT                 │ located_in       │ USA                │
│ research_paper      │ published_by     │ Google Research    │
└─────────────────────┴──────────────────┴────────────────────┘

📝 Summary (from first chunks)
┌────────────────────────────────────────────────────────┐
│ This paper investigates novel approaches to language   │
│ model optimization. The authors propose a new...       │
└────────────────────────────────────────────────────────┘

✅ Extraction Complete
   Total entities: 156
   Chunks processed: 10
   Total relations: 89

💡 Suggested Workflows
   1. 🏢 Build Organization Map - You have many entities.
   2. 🔗 Create Knowledge Graph - Strong relation density.
   3. 📅 Build Timeline - Date entities found.

Run: plato build --workflow comparison_table
```

## Integration with Knowledge Graph

```python
# Once extraction is done, feed into a graph DB
import neo4j

driver = neo4j.GraphDatabase.driver("bolt://localhost:7687")

async def build_knowledge_graph(tui: PlatoTUI):
    """Convert extracted knowledge to graph"""
    with driver.session() as session:
        for doc in tui.knowledge_base:
            # Add entities
            for entity_type, entities in doc['entities'].items():
                for entity in entities:
                    session.run(
                        "CREATE (n:Entity {name: $name, type: $type}) "
                        "RETURN n",
                        name=entity,
                        type=entity_type
                    )
            
            # Add relations
            for rel in doc['relations']:
                session.run(
                    "MATCH (a:Entity {name: $subject}), (b:Entity {name: $object}) "
                    "CREATE (a)-[r {relation: $rel_type}]->(b) "
                    "RETURN r",
                    subject=rel['subject'],
                    object=rel['object'],
                    rel_type=rel['relation']
                )
```

This shows:
- ✅ Real-time streaming feedback
- ✅ Batch processing with progress bars
- ✅ Suggestion engine (recommending next steps)
- ✅ Knowledge export for downstream (reports, graph DBs)
- ✅ Clean async integration with refactored client
