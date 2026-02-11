from pathlib import Path
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import DirectoryTree, Button, Label, Header, Static
from textual.containers import Vertical, Horizontal

class FolderSelectScreen(ModalScreen[Path]):
    """Modal screen to select a source folder."""
    
    CSS = """
    FolderSelectScreen {
        align: center middle;
    }
    
    #dialog {
        padding: 0 1;
        width: 80%;
        height: 80%;
        border: thick $background 80%;
        background: $surface;
    }
    
    #title {
        text-align: center;
        width: 100%;
        padding: 1;
        background: $primary;
        color: $text;
    }
    
    DirectoryTree {
        height: 1fr;
        border: solid $accent;
    }
    
    #buttons {
        height: auto;
        dock: bottom;
        padding: 1;
        align: center middle;
    }
    
    Button {
        margin: 0 1;
    }
    """

    def __init__(self, initial_path: Path):
        self.initial_path = initial_path
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Select Source Folder", id="title")
            yield DirectoryTree(str(self.initial_path), id="folder-tree")
            with Horizontal(id="buttons"):
                yield Button("Select Current", variant="primary", id="select")
                yield Button("Cancel", variant="error", id="cancel")

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        """Just navigate, don't close yet."""
        pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
        elif event.button.id == "select":
            tree = self.query_one("#folder-tree", DirectoryTree)
            # The current cursor node in the tree is what we want, or the root if nothing selected?
            # Textual's DirectoryTree doesn't easily expose "current directory" except via cursor path
            # We will return the path of the currently highlighted node if it's a dir, else parent
            
            # Simple approach: Return the root of the tree? No, user navigates down.
            # Use the path of the cursor node
            node = tree.cursor_node
            if node and node.data:
                path = node.data.path
                if path.is_file():
                    path = path.parent
                self.dismiss(path)
            else:
                self.dismiss(Path(tree.path))
