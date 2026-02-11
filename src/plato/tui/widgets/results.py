from textual.widgets import Static, Markdown
from textual.containers import ScrollableContainer

class ResultPanel(ScrollableContainer):
    """Container for the real-time LLM response."""
    
    def compose(self):
        yield Markdown("(Response will appear here...)", id="result-text")

    def update_content(self, md: str) -> None:
        """Update the displayed markdown content."""
        self.query_one("#result-text", Markdown).update(md)
