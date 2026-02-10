from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Static
from textual.binding import Binding
from plato.tui.screens.main import MainScreen

class PlatoApp(App):
    """PLATO: PDF-to-Context TUI"""
    
    CSS_PATH = "app.css"
    TITLE = "PLATO"
    SUB_TITLE = "Local PDF-to-Context Processor"
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", priority=True),
        Binding("f1", "show_help", "Help"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Footer()
    
    def on_mount(self) -> None:
        try:
            self.push_screen(MainScreen())
        except Exception as e:
            self.exit(message=f"Failed to load main screen: {e}")

    def action_show_help(self) -> None:
        """Show help dialog"""
        self.bell()  # Placeholder for help screen

if __name__ == "__main__":
    app = PlatoApp()
    app.run()
