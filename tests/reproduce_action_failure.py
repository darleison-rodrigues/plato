import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch
from textual.app import App, ComposeResult
from plato.tui.screens.main import MainScreen
from plato.tui.sidebar import Sidebar
# Need to mock the PDF processor to avoid PyMuPDF dependency if possible, 
# but better to use a real one since it was part of the traceback stack.

# Create a minimal valid PDF content
MINIMAL_PDF = b'%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Resources <<\n/Font <<\n/F1 4 0 R\n>>\n>>\n/Contents 5 0 R\n>>\nendobj\n4 0 obj\n<<\n/Type /Font\n/Subtype /Type1\n/BaseFont /Helvetica\n>>\nendobj\n5 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 24 Tf\n100 700 Td\n(Hello World) Tj\nET\nendstream\nendobj\nxref\n0 6\n0000000000 65535 f\n0000000010 00000 n\n0000000060 00000 n\n0000000157 00000 n\n0000000307 00000 n\n0000000394 00000 n\ntrailer\n<<\n/Size 6\n/Root 1 0 R\n>>\nstartxref\n490\n%%EOF\n'

async def reproduce_success_flow():
    """
    Simulates a FULL successful flow with mocked AI backend.
    """
    print("🚀 Starting reproduction script (Mocked AI)...")
    
    # Setup files
    pdf_path = Path("test_valid.pdf")
    pdf_path.write_bytes(MINIMAL_PDF)
    
    template_dir = Path("test_templates")
    template_dir.mkdir(exist_ok=True)
    template_file = template_dir / "summarize.j2"
    template_file.write_text("Summary requested.")
    
    # Mock Ollama Client
    mock_ollama = MagicMock()
    mock_ollama.__aenter__ = AsyncMock(return_value=mock_ollama)
    mock_ollama.__aexit__ = AsyncMock()
    
    # Mock generation stream
    async def mock_stream(*args, **kwargs):
        yield "This "
        await asyncio.sleep(0.1)
        yield "is "
        await asyncio.sleep(0.1)
        yield "a "
        await asyncio.sleep(0.1)
        yield "mocked "
        await asyncio.sleep(0.1)
        yield "response."
    
    # Mock embedding function (needed for retriever)
    async def mock_embed(*args, **kwargs):
        # Return list of floats
        return [0.1] * 768
    
    # We need to patch where MainScreen imports OllamaClient
    # It imports it as: from plato.ollama.client import OllamaClient
    
    with patch("plato.tui.screens.main.OllamaClient") as MockClientClass:
        # Configure the mock class to return our instance
        MockClientClass.return_value = mock_ollama
        mock_ollama.generate_stream = mock_stream
        
        # Also need to mock OllamaEmbeddingFunction
        with patch("plato.ollama.client.OllamaEmbeddingFunction") as MockEmbedFn:
             # This mock needs to be callable
             MockEmbedFn.return_value = MagicMock(side_effect=lambda x: [[0.1]*768 for _ in x])

             class TestApp(App):
                def compose(self) -> ComposeResult:
                    yield MainScreen()
            
             app = TestApp()
             async with app.run_test() as pilot:
                screen = app.query_one(MainScreen)
                
                print(f"👉 Selecting PDF: {pdf_path}")
                await screen.on_sidebar_pdf_selected(Sidebar.PDFSelected(pdf_path))
                
                print(f"👉 Selecting Action: {template_file}")
                await screen.on_sidebar_action_selected(Sidebar.ActionSelected(template_file))
                
                print("⏳ Waiting for processing...")
                # Wait enough time for the stream to finish (5 chunks * 0.1s + overhead)
                await pilot.pause(2.0) 
                
                results = screen.query_one("#results")
                from textual.widgets import Markdown
                result_text = results.query_one(Markdown).source
                
                print(f"📝 Result Panel Content:\n---\n{result_text}\n---")
                
                if "mocked response" in result_text:
                     print("✅ Success: Pipeline completed successfully.")
                else:
                     print("❌ Failure: Pipeline did not complete or output text.")
                     print(f"Current text: {result_text}")

    # Cleanup
    if pdf_path.exists(): pdf_path.unlink()
    import shutil
    if template_dir.exists(): shutil.rmtree(template_dir)

if __name__ == "__main__":
    try:
        asyncio.run(reproduce_success_flow())
    except Exception as e:
        import traceback
        traceback.print_exc()
