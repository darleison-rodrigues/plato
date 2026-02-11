from textual.app import ComposeResult
from textual.widgets import OptionList
from textual.widgets.option_list import Option
from pathlib import Path
import os

class TemplatePicker(OptionList):
    """A list of available templates/actions to run on a PDF."""
    
    def on_mount(self) -> None:
        self.refresh_templates()

    def refresh_templates(self) -> None:
        """Loads .j2 files from both internal and user templates directory."""
        user_dir = Path("~/.config/plato/templates").expanduser()
        internal_dir = Path(__file__).parent.parent.parent / "templates"
        
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect from both
        template_files = list(internal_dir.glob("*.j2")) + list(user_dir.glob("*.j2"))
        # De-duplicate by name (prefer user versions)
        unique_templates = {}
        for tf in template_files:
            unique_templates[tf.name] = tf
            
        self.clear_options()
        if not unique_templates:
            self.add_option(Option("No templates found", id="none", disabled=True))
        else:
            for name, path in sorted(unique_templates.items()):
                display_name = path.stem.replace("_", " ").title()
                self.add_option(Option(display_name, id=str(path)))
