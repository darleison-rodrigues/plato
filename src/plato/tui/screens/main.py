from pathlib import Path
from typing import Optional
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static
from textual.containers import Horizontal, Vertical
from plato.tui.sidebar import Sidebar
from plato.tui.widgets.results import ResultPanel
from plato.tui.widgets.status import StatusBar, MemoryMonitor
from plato.core.pdf import PDFProcessor
from plato.core.retriever import VectorRetriever
from plato.ollama.client import OllamaClient
from plato.core.template import TemplateEngine
from plato.config import get_config
import asyncio

class MainScreen(Screen):
    """The primary interaction screen for PLATO."""
    def compose(self) -> ComposeResult:
        self.selected_pdf: Optional[Path] = None
        self.selected_action: Optional[Path] = None
        self.is_processing = False
        
        with Horizontal():
            yield Sidebar(id="sidebar")
            with Vertical(id="main-content"):
                yield ResultPanel(id="results")
                yield Static("Select a PDF and an Action to begin.", id="empty-state")
                with Horizontal(id="footer-status"):
                    yield StatusBar(id="status-bar")
                    yield MemoryMonitor(id="memory-monitor")

    async def on_sidebar_pdf_selected(self, message: Sidebar.PDFSelected) -> None:
        """PDF selected in sidebar."""
        self.selected_pdf = message.path
        self.query_one("#status-bar", StatusBar).set_status(f"Selected: {message.path.name}")
        await self.check_and_start()

    async def on_sidebar_action_selected(self, message: Sidebar.ActionSelected) -> None:
        """Action selected in sidebar."""
        self.selected_action = message.path
        self.query_one("#status-bar", StatusBar).set_status(f"Action: {message.path.stem.title()}")
        await self.check_and_start()

    async def check_and_start(self) -> None:
        """Triggers processing if both inputs are ready."""
        if self.selected_pdf and self.selected_action and not self.is_processing:
            self.run_worker(self.process_request())

    async def process_request(self) -> None:
        """The main orchestration logic for PDF-to-Context."""
        self.is_processing = True
        status = self.query_one("#status-bar", StatusBar)
        results = self.query_one("#results", ResultPanel)
        empty = self.query_one("#empty-state", Static)
        
        try:
            config = get_config()
            # UI preparation
            empty.display = False
            # 1. Extract Text
            status.set_status(f"Extracting text...", "cyan")
            processor = PDFProcessor(self.selected_pdf)
            full_text, is_scanned = processor.analyze_content()
            
            if is_scanned:
                results.update_content("Error: Document appears to be scanned (no selectable text).")
                return

            # 2. Setup Clients
            from plato.ollama.client import OllamaEmbeddingFunction
            async with OllamaClient(base_url=config.ollama.base_url) as ollama:
                # Use our hardened embedding function for vector operations
                embedding_fn = OllamaEmbeddingFunction(
                    model=config.ollama.embedding_model, 
                    base_url=config.ollama.base_url
                )
                
                status.set_status("Indexing chunks...", "cyan")
                retriever = VectorRetriever(
                    persist_dir=config.pipeline.output_dir,
                    embedding_fn=embedding_fn
                ) 

                # Index the document if not already indexed
                pdf_hash = f"pdf_{self.selected_pdf.name}" # Placeholder hash
                doc_id = f"{pdf_hash}_full"
                # Run heavy/blocking indexing in a separate thread to keep UI responsive
                await asyncio.to_thread(retriever.index_document, doc_id, full_text, {"filename": self.selected_pdf.name})

                # 3. Retrieve Context
                status.set_status("Retrieving context...", "cyan")
                # Querying with the action objective or just taking top chunks
                query_text = f"Objective: {self.selected_action.stem}"
                # Run heavy/blocking query in a separate thread
                search_results = await asyncio.to_thread(retriever.query, query_text, n_results=3)
                context = "\n---\n".join([r.content for r in search_results])

                # 4. Render Template
                status.set_status("Preparing prompt...", "cyan")
                engine = TemplateEngine(template_dir=str(self.selected_action.parent))
                rendered_prompt = await asyncio.to_thread(engine.render, self.selected_action.name, {"context": context})

                # 5. Stream Generation
                status.set_status("PLATO is thinking...", "magenta")
                results.update_content("") # Clear for streaming
                
                accumulated = ""
                async for chunk in ollama.generate_stream(
                    model=config.ollama.reasoning_model,
                    prompt=rendered_prompt
                ):
                    accumulated += chunk
                    results.update_content(accumulated)
            
            status.set_status("Done.", "green")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            results.update_content(f"## Pipeline Error\n{e}\n\n```\n{error_details}\n```")
            status.set_status(f"Error: {e}", "red")
        finally:
            self.is_processing = False
