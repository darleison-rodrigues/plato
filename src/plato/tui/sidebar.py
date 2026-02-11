from pathlib import Path
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widgets import DirectoryTree, Label, OptionList, Button
from textual.message import Message
from plato.tui.widgets.mascot import Mascot
from plato.tui.widgets.picker import TemplatePicker
from plato.tui.screens.folder_select import FolderSelectScreen
from plato.config import get_config

class Sidebar(Vertical):
    """Sidebar containing the mascot and file browser."""
    def compose(self) -> ComposeResult:
        config = get_config()
        self.source_path = Path(config.pipeline.source_dir).expanduser()
        self.source_path.mkdir(parents=True, exist_ok=True)
        
        yield Mascot()
        yield Label("SOURCE", classes="section-title")
        yield Label(str(self.source_path), id="source-path-label", classes="path-info")
        yield Button("📂 Open Folder", id="open-folder-btn", variant="default")
        yield Label("FILES", classes="section-title")
        tree = DirectoryTree(str(self.source_path), id="file-browser")
        yield tree
        yield Label("ACTIONS", classes="section-title")
        yield TemplatePicker(id="template-picker")

    def on_mount(self) -> None:
        """Focus the file browser and check for files."""
        self.query_one("#file-browser").focus()
        if not any(self.source_path.iterdir()):
             self.notify(f"Source folder is empty: {self.source_path}", severity="warning", timeout=10)

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle PDF selection."""
        if event.path.suffix.lower() == ".pdf":
            self.post_message(self.PDFSelected(event.path))
        else:
            self.notify("Please select a .pdf file.", severity="warning")

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Handle directory selection - process all PDFs in folder."""
        pdfs = list(event.path.glob("*.pdf"))
        if pdfs:
            self.notify(f"Found {len(pdfs)} PDFs in {event.path.name}. (Selecting first file for now.)", severity="info")
            self.post_message(self.PDFSelected(pdfs[0]))
        else:
            self.notify(f"Folder: {event.path.name} (No PDFs found)", severity="info")

    class PDFSelected(Message):
        """Emitted when a PDF is selected."""
        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle template selection."""
        if event.option_id:
            self.post_message(self.ActionSelected(Path(event.option_id)))

    class ActionSelected(Message):
        """Emitted when an action template is selected."""
        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "open-folder-btn":
            self.app.push_screen(FolderSelectScreen(self.source_path), self.on_folder_selected)

    def on_folder_selected(self, path: Path | None) -> None:
        """Callback from FolderSelectScreen."""
        if path:
            self.source_path = path
            self.query_one("#source-path-label", Label).update(str(path))
            
            # Update DirectoryTree
            tree = self.query_one("#file-browser", DirectoryTree)
            tree.path = str(path)
            tree.reload()
            
            self.notify(f"Source changed to: {path}")
