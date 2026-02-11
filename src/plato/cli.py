import cmd
import shlex
import asyncio
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import random
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.layout import Layout
from rich.text import Text
from rich import box
from rich.syntax import Syntax
from rich.live import Live
from rich.align import Align
from rich.columns import Columns
import time

from plato.config import get_config
from plato.core.pdf import PDFProcessor
from plato.core.retriever import VectorRetriever
from plato.ollama.client import OllamaClient, OllamaEmbeddingFunction
from plato.core.template import TemplateEngine

console = Console()

# Enhanced ASCII art with animation-ready design
BANNER = """
[bold cyan]
 ██████╗ ██╗      █████╗ ████████╗ ██████╗ 
 ██╔══██╗██║     ██╔══██╗╚══██╔══╝██╔═══██╗
 ██████╔╝██║     ███████║   ██║   ██║   ██║
 ██╔═══╝ ██║     ██╔══██║   ██║   ██║   ██║
 ██║     ███████╗██║  ██║   ██║   ╚██████╔╝
 ╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ 
[/bold cyan]
[bold magenta]🧠 Philosophical Learning & Analysis Tool 🧠[/bold magenta]
[italic dim]Your AI-Powered Document Intelligence Engine[/italic dim]
"""

# Motivational tips and fun facts
TIPS = [
    "💡 Pro tip: Use 'query' with specific keywords for better results!",
    "🎯 Did you know? PLATO can analyze multiple PDFs simultaneously!",
    "🚀 Fun fact: Vector embeddings help find semantic similarities!",
    "📚 Tip: The more documents you index, the smarter your insights become!",
    "⚡ Speed tip: Query specific topics instead of broad searches!",
    "🎨 Try using 'insights' to generate beautiful markdown reports!",
    "🔍 Pro search: Combine multiple keywords for precise results!",
    "💪 PLATO gets better with more diverse documents!",
]

# Fun loading messages
LOADING_MESSAGES = [
    "Consulting the ancient texts...",
    "Summoning the AI spirits...",
    "Crunching through the knowledge...",
    "Teaching robots to read...",
    "Decoding the mysteries...",
    "Brewing some intelligence...",
    "Waking up the neurons...",
    "Philosophizing with electrons...",
]

class PlatoShell(cmd.Cmd):
    prompt = '[bold magenta]plato[/bold magenta][cyan]>[/cyan] '

    def __init__(self):
        super().__init__()
        self.config = get_config()
        self.retriever: Optional[VectorRetriever] = None
        self.embedding_fn = None
        self.session_start = datetime.now()
        self.queries_run = 0
        self.files_processed = 0
        self.insights_generated = 0
        self._display_animated_banner()
        self._setup_retriever()
        self._show_random_tip()
        
    def _display_animated_banner(self):
        """Display welcome banner with animation."""
        console.clear()
        
        # Animated banner appearance
        with Live(console=console, refresh_per_second=10) as live:
            for i in range(3):
                live.update(Align.center("✨" * (i + 1)))
                time.sleep(0.15)
        
        console.print(Align.center(BANNER))
        
        # Welcome message with current time
        welcome_panel = Panel(
            f"[yellow]Welcome back! 👋[/yellow]\n"
            f"[dim]Session started: {self.session_start.strftime('%I:%M %p on %B %d, %Y')}[/dim]\n\n"
            f"[cyan]💡 Type [bold]help[/bold] to explore commands[/cyan]\n"
            f"[cyan]📚 Type [bold]tutorial[/bold] for a quick start guide[/cyan]\n"
            f"[cyan]🎯 Type [bold]examples[/bold] to see use cases[/cyan]",
            border_style="cyan",
            box=box.DOUBLE,
            title="[bold magenta]🌟 PLATO v2.0[/bold magenta]",
            subtitle="[dim]Powered by AI Magic[/dim]"
        )
        console.print(Align.center(welcome_panel))
        console.print()
        
    def _show_random_tip(self):
        """Display a random helpful tip."""
        tip = random.choice(TIPS)
        console.print(f"[dim italic]{tip}[/dim italic]\n")
        
    def _setup_retriever(self):
        """Initialize vectors with enhanced progress display."""
        try:
            steps = [
                ("🔌 Connecting to Ollama", 0.3),
                ("📦 Loading embedding model", 0.5),
                ("🗄️  Initializing vector database", 0.4),
                ("✅ Finalizing setup", 0.2),
            ]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Starting up...", total=len(steps))
                
                for step_desc, delay in steps:
                    progress.update(task, description=f"[cyan]{step_desc}")
                    time.sleep(delay)
                    
                    if "embedding" in step_desc:
                        self.embedding_fn = OllamaEmbeddingFunction(
                            model=self.config.ollama.embedding_model,
                            base_url=self.config.ollama.base_url
                        )
                    elif "database" in step_desc:
                        self.retriever = VectorRetriever(
                            persist_dir=self.config.pipeline.output_dir,
                            embedding_fn=self.embedding_fn
                        )
                    
                    progress.advance(task)
            
            stats = self.retriever.get_stats()
            
            # Enhanced stats display with visual indicators
            stats_table = Table(box=box.ROUNDED, border_style="green", show_header=False)
            stats_table.add_column("Metric", style="cyan bold", justify="right")
            stats_table.add_column("Value", style="green")
            stats_table.add_column("Visual", style="yellow")
            
            chunks = stats['total_chunks']
            chunk_visual = "📄 " * min(chunks // 10, 10) if chunks > 0 else "🔍 Ready to index!"
            
            stats_table.add_row("📊 Indexed Chunks", str(chunks), chunk_visual)
            stats_table.add_row("🤖 AI Model", self.config.ollama.embedding_model, "✨")
            stats_table.add_row("💾 Status", "Ready to go!", "🟢")
            
            console.print(Panel(
                stats_table,
                title="[bold green]⚡ System Ready[/bold green]",
                border_style="green",
                box=box.DOUBLE
            ))
            console.print()
            
        except Exception as e:
            console.print(Panel(
                f"[red bold]❌ Oops! Something went wrong[/red bold]\n\n"
                f"[yellow]Error: {e}[/yellow]\n\n"
                f"[cyan]💡 Suggestions:[/cyan]\n"
                f"  • Check if Ollama is running\n"
                f"  • Verify your configuration\n"
                f"  • Try 'stats' command for diagnostics",
                border_style="red",
                box=box.HEAVY,
                title="[bold red]⚠️ Initialization Failed[/bold red]"
            ))

    def cmdloop(self, intro=None):
        """Override to use Rich rendering for prompt."""
        self.preloop()
        if intro is not None:
            self.intro = intro
        if self.intro:
            console.print(self.intro)
        
        stop = None
        while not stop:
            try:
                # Custom prompt rendering with Rich
                line = console.input(f'\n[bold magenta]plato[/bold magenta][cyan]>[/cyan] ')
                line = self.precmd(line)
                stop = self.onecmd(line)
                stop = self.postcmd(stop, line)
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️  Interrupted! Press Ctrl+C again or type 'exit' to leave[/yellow]")
            except EOFError:
                console.print("\n[cyan]👋 Goodbye! Keep learning![/cyan]")
                break
        self.postloop()

    def do_tutorial(self, arg):
        """
        tutorial
        📖 Interactive tutorial for new users.
        """
        console.print(Panel(
            "[bold cyan]🎓 Welcome to PLATO Tutorial![/bold cyan]\n\n"
            "[yellow]Let's learn the basics in 3 simple steps:[/yellow]",
            border_style="cyan",
            box=box.DOUBLE
        ))
        
        steps = [
            {
                "title": "1️⃣ Index Your Documents",
                "desc": "Use [bold cyan]process[/bold cyan] to analyze PDFs",
                "example": "process ~/Documents/research",
                "emoji": "📁"
            },
            {
                "title": "2️⃣ Search for Knowledge",
                "desc": "Use [bold cyan]query[/bold cyan] to find relevant information",
                "example": "query machine learning concepts",
                "emoji": "🔍"
            },
            {
                "title": "3️⃣ Generate Insights",
                "desc": "Use [bold cyan]insights[/bold cyan] to create AI-powered reports",
                "example": "insights my_report.md",
                "emoji": "💡"
            }
        ]
        
        for step in steps:
            panel = Panel(
                f"{step['desc']}\n\n"
                f"[dim]Example:[/dim] [green]{step['example']}[/green]",
                title=f"[bold magenta]{step['emoji']} {step['title']}[/bold magenta]",
                border_style="magenta",
                box=box.ROUNDED
            )
            console.print(panel)
            time.sleep(0.3)
        
        console.print("\n[bold green]🎉 You're ready to go! Type 'help' anytime for more info.[/bold green]\n")

    def do_examples(self, arg):
        """
        examples
        💼 Show real-world usage examples.
        """
        examples_table = Table(title="💼 Real-World Use Cases", box=box.DOUBLE, border_style="cyan")
        examples_table.add_column("Scenario", style="bold yellow", width=30)
        examples_table.add_column("Commands", style="green", width=50)
        
        examples_table.add_row(
            "📚 Research Paper Analysis",
            "process ./papers\nquery methodology discussion\ninsights research_summary.md"
        )
        examples_table.add_row(
            "📊 Business Report Review",
            "process quarterly_reports.pdf\nquery revenue growth trends\ninsights executive_brief.md"
        )
        examples_table.add_row(
            "📖 Book Summarization",
            "process ./books\nquery main themes and ideas\ninsights book_notes.md"
        )
        examples_table.add_row(
            "🔬 Technical Documentation",
            "process ./docs/technical\nquery API endpoints\nquery authentication methods"
        )
        
        console.print()
        console.print(examples_table)
        console.print()

    def do_process(self, arg):
        """
        process <path>
        📁 Index all PDFs in the specified directory or a single PDF file.
        
        Examples:
          process ~/documents/papers
          process report.pdf
          process .  (current directory)
        """
        path_str = arg.strip()
        if not path_str:
            console.print(Panel(
                "[red]❌ Oops! I need a path to work with[/red]\n\n"
                "[yellow]Examples:[/yellow]\n"
                "[cyan]  • process ~/Documents[/cyan]\n"
                "[cyan]  • process research.pdf[/cyan]\n"
                "[cyan]  • process .[/cyan] [dim](current directory)[/dim]",
                border_style="red",
                box=box.ROUNDED,
                title="[bold red]Missing Path[/bold red]"
            ))
            return

        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            console.print(Panel(
                f"[red]❌ Path not found:[/red]\n[yellow]{path}[/yellow]\n\n"
                f"[cyan]💡 Tip: Use absolute paths or check your spelling![/cyan]",
                border_style="red"
            ))
            return

        files = []
        if path.is_file():
            if path.suffix.lower() == '.pdf':
                files = [path]
            else:
                console.print(Panel(
                    f"[red]❌ Not a PDF file:[/red] [yellow]{path.name}[/yellow]\n\n"
                    f"[cyan]PLATO only processes PDF files right now![/cyan]",
                    border_style="red"
                ))
                return
        else:
            files = list(path.glob("*.pdf"))

        if not files:
            console.print(Panel(
                "[yellow]🤷 No PDF files found in this location[/yellow]\n\n"
                "[cyan]Try another directory or add some PDFs first![/cyan]",
                border_style="yellow"
            ))
            return

        # Confirmation for large batches
        if len(files) > 10:
            if not Confirm.ask(f"[yellow]Found {len(files)} PDFs. Process all?[/yellow]"):
                console.print("[dim]Cancelled.[/dim]")
                return

        console.print(Panel(
            f"[bold cyan]🚀 Starting to process {len(files)} PDF file(s)[/bold cyan]\n"
            f"[dim]This might take a moment for large documents...[/dim]",
            border_style="cyan",
            box=box.ROUNDED
        ))
        
        asyncio.run(self._process_files(files))

    async def _process_files(self, files: List[Path]):
        start_time = time.time()
        successful = 0
        skipped = 0
        failed = 0
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            task = progress.add_task("[cyan]Processing PDFs...", total=len(files))
            
            for pdf in files:
                try:
                    progress.update(task, description=f"[cyan]📄 {pdf.name[:30]}...")
                    processor = PDFProcessor(pdf)
                    text, is_scanned = processor.analyze_content()
                    
                    if is_scanned:
                        console.print(f"[yellow]⏭️  Skipped (scanned): {pdf.name}[/yellow]")
                        skipped += 1
                        progress.advance(task)
                        continue
                    
                    doc_id = f"pdf_{pdf.name}_full"
                    
                    progress.update(task, description=f"[cyan]🔍 Indexing {pdf.name[:30]}...")
                    self.retriever.index_document(doc_id, text, {"filename": pdf.name})
                    
                    console.print(f"[green]✅ {pdf.name}[/green]")
                    successful += 1
                    progress.advance(task)
                    
                except Exception as e:
                    console.print(f"[red]❌ {pdf.name}: {str(e)[:50]}[/red]")
                    failed += 1
                    progress.advance(task)
        
        elapsed = time.time() - start_time
        self.files_processed += successful
        
        # Summary with emoji feedback
        status_emoji = "🎉" if failed == 0 else "⚠️"
        summary_table = Table(box=box.ROUNDED, show_header=False, border_style="green")
        summary_table.add_column("Metric", style="cyan bold")
        summary_table.add_column("Count", style="green bold", justify="right")
        
        summary_table.add_row("✅ Successful", str(successful))
        if skipped > 0:
            summary_table.add_row("⏭️  Skipped", str(skipped))
        if failed > 0:
            summary_table.add_row("❌ Failed", str(failed))
        summary_table.add_row("⏱️  Time", f"{elapsed:.1f}s")
        
        console.print(Panel(
            summary_table,
            title=f"[bold green]{status_emoji} Processing Complete![/bold green]",
            border_style="green",
            box=box.DOUBLE
        ))
        
        if successful > 0:
            console.print(f"\n[dim italic]💡 Try: [bold cyan]query your search term[/bold cyan][/dim italic]\n")

    def do_query(self, arg):
        """
        query <text>
        🔍 Search the index for relevant chunks.
        
        Examples:
          query machine learning concepts
          query conclusions and findings
          query methodology
        """
        if not arg.strip():
            console.print(Panel(
                "[red]❌ What should I search for?[/red]\n\n"
                "[yellow]Examples:[/yellow]\n"
                "[cyan]  • query neural networks[/cyan]\n"
                "[cyan]  • query key findings[/cyan]\n"
                "[cyan]  • query methodology[/cyan]",
                border_style="red",
                box=box.ROUNDED,
                title="[bold red]Missing Query[/bold red]"
            ))
            return
            
        if not self.retriever:
            console.print("[red]❌ Retriever not initialized. Try restarting![/red]")
            return
            
        try:
            loading_msg = random.choice(LOADING_MESSAGES)
            with console.status(f"[bold cyan]🔍 {loading_msg}", spinner="dots"):
                results = self.retriever.query(arg, n_results=3)
                time.sleep(0.3)  # Dramatic pause
            
            self.queries_run += 1
            
            if not results:
                console.print(Panel(
                    "[yellow]🤷 No matches found[/yellow]\n\n"
                    "[cyan]💡 Try:[/cyan]\n"
                    "  • Different keywords\n"
                    "  • Broader search terms\n"
                    "  • Index more documents with 'process'",
                    border_style="yellow",
                    box=box.ROUNDED,
                    title="[bold yellow]No Results[/bold yellow]"
                ))
                return
            
            # Enhanced results display
            console.print(f"\n[bold cyan]🎯 Found {len(results)} relevant results for:[/bold cyan] [yellow]\"{arg}\"[/yellow]\n")
            
            for i, res in enumerate(results, 1):
                fname = res.metadata.get('filename', 'Unknown')
                
                # Score-based visual feedback
                if res.distance < 0.5:
                    score_color = "green"
                    score_emoji = "🎯"
                    score_label = "Excellent match"
                elif res.distance < 1.0:
                    score_color = "yellow"
                    score_emoji = "👍"
                    score_label = "Good match"
                else:
                    score_color = "red"
                    score_emoji = "👌"
                    score_label = "Decent match"
                
                # Preview with smart truncation
                preview = res.content[:400]
                if len(res.content) > 400:
                    preview += "..."
                
                result_panel = Panel(
                    f"[bold cyan]📄 Source:[/bold cyan] {fname}\n"
                    f"[{score_color}]{score_emoji} {score_label} (score: {res.distance:.3f})[/{score_color}]\n\n"
                    f"[dim]{preview}[/dim]",
                    title=f"[bold magenta]Result #{i}[/bold magenta]",
                    border_style=score_color,
                    box=box.ROUNDED
                )
                console.print(result_panel)
            
            console.print(f"\n[dim italic]💡 Want more? Try [bold cyan]insights[/bold cyan] to generate a report![/dim italic]\n")
            
        except Exception as e:
            console.print(Panel(
                f"[red bold]❌ Search failed[/red bold]\n\n"
                f"[yellow]Error: {e}[/yellow]\n\n"
                f"[cyan]This might be a temporary issue. Try again![/cyan]",
                border_style="red",
                box=box.HEAVY
            ))

    def do_insights(self, arg):
        """
        insights [output_file]
        💡 Generate AI-powered insights from your indexed documents.
        
        Examples:
          insights report.md
          insights summary.md
          insights  (saves to insights.md)
        """
        output_file = arg.strip() or "insights.md"
        
        # Check if retriever has content
        if self.retriever and self.retriever.get_stats().get('total_chunks', 0) == 0:
            console.print(Panel(
                "[yellow]⚠️  No documents indexed yet![/yellow]\n\n"
                "[cyan]First, index some documents:[/cyan]\n"
                "[green]  process ~/Documents[/green]",
                border_style="yellow",
                box=box.ROUNDED
            ))
            return
        
        console.print(Panel(
            f"[cyan]🧠 Generating AI-powered insights...[/cyan]\n"
            f"[dim]This uses advanced language models to analyze your content[/dim]\n\n"
            f"[yellow]📝 Output:[/yellow] {output_file}",
            border_style="cyan",
            box=box.DOUBLE,
            title="[bold magenta]✨ Insight Generation[/bold magenta]"
        ))
        
        # Retrieve context with progress
        with console.status("[bold green]🔍 Gathering relevant context...", spinner="dots"):
            results = self.retriever.query("key findings executive summary conclusions main points", n_results=5)
            context = "\n---\n".join([r.content for r in results])
            time.sleep(0.3)
        
        console.print(f"[green]✅ Retrieved {len(results)} context chunks[/green]")
        
        # Generate
        asyncio.run(self._generate_insights("insights", context, output_file))

    async def _generate_insights(self, template_name: str, context: str, output_path: str):
        try:
            engine = TemplateEngine() 
            
            if template_name not in engine.list_templates():
                template_name = "summarize.j2"
                console.print("[yellow]ℹ️  Using default template[/yellow]")
                
            prompt = engine.render(template_name, {"context": context})
            
            # Enhanced AI generation with progress
            steps = ["🤔 Analyzing content", "📝 Drafting insights", "✨ Polishing output"]
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]AI is working...", total=len(steps))
                
                async with OllamaClient(base_url=self.config.ollama.base_url) as client:
                    for step in steps[:2]:
                        progress.update(task, description=f"[cyan]{step}")
                        await asyncio.sleep(0.5)
                        progress.advance(task)
                    
                    progress.update(task, description=f"[cyan]{steps[2]}")
                    response = await client.generate(
                        model=self.config.ollama.reasoning_model,
                        prompt=prompt
                    )
                    progress.advance(task)
            
            # Save with confirmation
            with open(output_path, "w") as f:
                f.write(response)
            
            self.insights_generated += 1
            
            # Success with stats
            file_size = len(response)
            word_count = len(response.split())
            
            success_table = Table(box=box.ROUNDED, show_header=False, border_style="green")
            success_table.add_column("Metric", style="cyan")
            success_table.add_column("Value", style="green bold")
            
            success_table.add_row("📄 File", output_path)
            success_table.add_row("📊 Words", str(word_count))
            success_table.add_row("💾 Size", f"{file_size:,} bytes")
            
            console.print(Panel(
                success_table,
                title="[bold green]✨ Insights Generated![/bold green]",
                border_style="green",
                box=box.DOUBLE
            ))
            
            # Preview
            console.print("\n[bold cyan]📄 Preview:[/bold cyan]")
            preview = response[:500] + ("..." if len(response) > 500 else "")
            console.print(Panel(Markdown(preview), border_style="cyan", box=box.ROUNDED))
            
            console.print(f"\n[dim italic]💡 Full content saved to: [bold cyan]{output_path}[/bold cyan][/dim italic]\n")
            
        except Exception as e:
            console.print(Panel(
                f"[red bold]❌ Generation failed[/red bold]\n\n"
                f"[yellow]Error: {e}[/yellow]\n\n"
                f"[cyan]Possible causes:[/cyan]\n"
                "  • Ollama service not running\n"
                "  • Model not available\n"
                "  • Network issues",
                border_style="red",
                box=box.HEAVY,
                title="[bold red]Error[/bold red]"
            ))

    def do_stats(self, arg):
        """
        stats
        📊 Display comprehensive system statistics and session info.
        """
        if not self.retriever:
            console.print("[red]❌ Retriever not initialized.[/red]")
            return
            
        try:
            stats = self.retriever.get_stats()
            session_duration = datetime.now() - self.session_start
            
            # Create multi-section stats display
            layout = Table.grid(padding=1)
            layout.add_column()
            layout.add_column()
            
            # System stats
            system_table = Table(title="🖥️  System Status", box=box.ROUNDED, border_style="cyan")
            system_table.add_column("Component", style="bold cyan")
            system_table.add_column("Status", style="green")
            
            system_table.add_row("📚 Indexed Chunks", str(stats.get('total_chunks', 0)))
            system_table.add_row("🔤 Embedding Model", self.config.ollama.embedding_model)
            system_table.add_row("🤖 Reasoning Model", self.config.ollama.reasoning_model)
            system_table.add_row("📁 Storage", str(self.config.pipeline.output_dir))
            system_table.add_row("🌐 Ollama URL", self.config.ollama.base_url)
            system_table.add_row("✅ Status", "[bold green]● Online[/bold green]")
            
            # Session stats
            session_table = Table(title="📈 Session Statistics", box=box.ROUNDED, border_style="magenta")
            session_table.add_column("Activity", style="bold magenta")
            session_table.add_column("Count", style="yellow bold", justify="right")
            
            session_table.add_row("⏱️  Duration", str(session_duration).split('.')[0])
            session_table.add_row("🔍 Queries Run", str(self.queries_run))
            session_table.add_row("📄 Files Processed", str(self.files_processed))
            session_table.add_row("💡 Insights Generated", str(self.insights_generated))
            
            layout.add_row(system_table, session_table)
            
            console.print()
            console.print(Panel(
                layout,
                title="[bold cyan]📊 PLATO Dashboard[/bold cyan]",
                border_style="cyan",
                box=box.DOUBLE
            ))
            console.print()
            
            # Progress bar for index size
            if stats.get('total_chunks', 0) > 0:
                chunks = stats['total_chunks']
                bar_width = min(chunks // 10, 50)
                progress_bar = "█" * bar_width
                console.print(f"[cyan]Index Size:[/cyan] [green]{progress_bar}[/green] [yellow]{chunks} chunks[/yellow]\n")
            
        except Exception as e:
            console.print(Panel(
                f"[red]❌ Failed to retrieve statistics[/red]\n\n[yellow]{e}[/yellow]",
                border_style="red"
            ))

    def do_clear(self, arg):
        """
        clear
        🧹 Clear the screen and redisplay the banner.
        """
        console.clear()
        self._display_animated_banner()
        self._show_random_tip()

    def do_help(self, arg):
        """
        help [command]
        ❓ Show help information.
        """
        if arg:
            super().do_help(arg)
        else:
            # Enhanced help display with categories
            console.print()
            
            # Quick Start
            quick_start = Table(title="⚡ Quick Start", box=box.ROUNDED, border_style="green")
            quick_start.add_column("Command", style="bold green", width=20)
            quick_start.add_column("What it does", style="cyan")
            quick_start.add_row("tutorial", "📖 Interactive beginner's guide")
            quick_start.add_row("examples", "💼 Real-world usage examples")
            
            # Core Commands
            core_table = Table(title="🎯 Core Commands", box=box.ROUNDED, border_style="cyan")
            core_table.add_column("Command", style="bold magenta", width=20)
            core_table.add_column("Description", style="cyan")
            
            core_table.add_row("process <path>", "📁 Index PDFs from folder or file")
            core_table.add_row("query <text>", "🔍 Search indexed documents")
            core_table.add_row("insights [file]", "💡 Generate AI insights")
            
            # Utility Commands
            util_table = Table(title="🛠️  Utilities", box=box.ROUNDED, border_style="yellow")
            util_table.add_column("Command", style="bold yellow", width=20)
            util_table.add_column("Description", style="cyan")
            
            util_table.add_row("stats", "📊 System & session statistics")
            util_table.add_row("clear", "🧹 Clear the screen")
            util_table.add_row("help [cmd]", "❓ Show help")
            util_table.add_row("exit / quit", "👋 Exit PLATO")
            
            console.print(quick_start)
            console.print()
            console.print(core_table)
            console.print()
            console.print(util_table)
            console.print()
            
            console.print(Panel(
                "[bold cyan]💡 Pro Tips:[/bold cyan]\n"
                "  • Type [bold]<command> --help[/bold] for detailed info\n"
                "  • Start with [bold]tutorial[/bold] if you're new\n"
                "  • Use [bold]examples[/bold] to see real use cases\n"
                "  • Check [bold]stats[/bold] to monitor your progress",
                border_style="cyan",
                box=box.ROUNDED
            ))
            console.print()

    def do_quit(self, arg):
        """
        quit
        👋 Exit the PLATO Shell.
        """
        session_duration = datetime.now() - self.session_start
        
        # Session summary
        summary = Table(box=box.ROUNDED, show_header=False, border_style="magenta")
        summary.add_column("Metric", style="cyan bold")
        summary.add_column("Value", style="yellow bold", justify="right")
        
        summary.add_row("⏱️  Session Duration", str(session_duration).split('.')[0])
        summary.add_row("🔍 Queries", str(self.queries_run))
        summary.add_row("📄 Files Processed", str(self.files_processed))
        summary.add_row("💡 Insights", str(self.insights_generated))
        
        console.print()
        console.print(Panel(
            summary,
            title="[bold cyan]📊 Session Summary[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        ))
        
        farewell_messages = [
            "Keep exploring! 🚀",
            "Happy learning! 📚",
            "See you next time! 🌟",
            "Stay curious! 🔍",
            "Keep analyzing! 💡"
        ]
        
        console.print(Panel(
            f"[bold cyan]👋 Thank you for using PLATO![/bold cyan]\n\n"
            f"[yellow]{random.choice(farewell_messages)}[/yellow]",
            border_style="magenta",
            box=box.DOUBLE,
            title="[bold magenta]Goodbye![/bold magenta]"
        ))
        console.print()
        return True
    
    def do_exit(self, arg):
        """
        exit
        👋 Exit the PLATO Shell.
        """
        return self.do_quit(arg)
    
    def emptyline(self):
        """Show a helpful tip on empty line instead of doing nothing."""
        if random.random() < 0.3:  # 30% chance
            self._show_random_tip()
    
    def default(self, line):
        """Handle unknown commands with helpful suggestions."""
        # Try to suggest similar commands
        suggestions = []
        commands = ['process', 'query', 'insights', 'stats', 'help', 'tutorial', 'examples']
        
        for cmd in commands:
            if cmd.startswith(line.lower()[:3]):
                suggestions.append(cmd)
        
        error_msg = f"[red]❌ Unknown command:[/red] [yellow]{line}[/yellow]\n\n"
        
        if suggestions:
            error_msg += "[cyan]Did you mean?[/cyan]\n"
            for sug in suggestions[:3]:
                error_msg += f"  • [green]{sug}[/green]\n"
        else:
            error_msg += "[cyan]Type [bold]help[/bold] to see available commands[/cyan]"
        
        console.print(Panel(error_msg, border_style="red", box=box.ROUNDED))
        console.print()

def run():
    """Entry point for the PLATO shell."""
    try:
        PlatoShell().cmdloop()
    except KeyboardInterrupt:
        console.print("\n[cyan]👋 Goodbye! Keep learning![/cyan]")
    except Exception as e:
        console.print(Panel(
            f"[red bold]💥 Fatal Error[/red bold]\n\n"
            f"[yellow]{e}[/yellow]\n\n"
            f"[cyan]Please report this issue if it persists![/cyan]",
            border_style="red",
            box=box.HEAVY
        ))