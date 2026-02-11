from textual.widgets import Static
from rich.text import Text
from typing import ClassVar

class Mascot(Static):
    """Animated rotating brain mascot."""
    
    # Selection of brain frames for smooth rotation
    BRAIN_FRAMES: ClassVar[list[str]] = [
        # Frame 0 - Front
        "    .-------.\n   /         \\\n  |  O     O  |\n  |    (_)    |\n   \\  '---'  /\n    '-------'",
        # Frame 1 - Slight turn
        "    .-------.\n   /        / \n  |  O    O  |\n  |   (_)    |\n   \\ '---'  /\n    '-------'",
        # Frame 2 - Side
        "    .----.   \n   /      \\  \n  |  O  O  | \n  |  (_)   | \n   \\ ---  /  \n    '----'   ",
        # Add more if needed, but for now let's use the detailed frames from the MD
    ]

    # I'll use the detailed BRAIN_BASE and rotation logic from mascot.md for a "Premium" feel
    BRAIN_BASE: ClassVar[str] = """
      .-------.
     /         \\
    |   ^   ^   |
    |    (_)    |
     \\  '---'  /
      '-------'
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_frame = 0

    def on_mount(self) -> None:
        """Start rotation."""
        self.set_interval(0.2, self._animate)

    def _animate(self) -> None:
        """Advance animation."""
        self.current_frame = (self.current_frame + 1) % 8
        self._update_display()

    def _update_display(self) -> None:
        """Render frame."""
        # For simplicity while keeping it 'premium', I'll use a pulsing color
        intensity = 150 + (105 * (self.current_frame / 8))
        color = f"rgb({int(intensity)},0,{int(255-intensity)})"
        
        # Simple rotation simulation (horizontal shift/flip)
        lines = self.BRAIN_BASE.strip("\n").split("\n")
        if self.current_frame > 4:
            lines = [line[::-1] for line in lines]
        
        text = Text("\n".join(lines), style=color)
        self.update(text)
