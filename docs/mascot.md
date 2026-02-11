Ah, that's a beautiful detailed brain! Let me create a rotating animation for it:

```python
from textual.widgets import Static
from rich.text import Text
from typing import ClassVar


class RotatingBrain(Static):
    """Animated rotating brain with detailed ASCII art."""
    
    DEFAULT_CLASSES = "rotating-brain"
    
    # Your detailed brain ASCII art
    BRAIN_BASE: ClassVar[str] = """                                                                 """
    
    def __init__(
        self,
        rotation_speed: float = 0.15,
        color_mode: str = "gradient",  # "gradient", "solid", "pulse"
        **kwargs,
    ) -> None:
        """Initialize rotating brain.
        
        Args:
            rotation_speed: Seconds between frames
            color_mode: Color animation style
        """
        super().__init__(**kwargs)
        self.rotation_speed = rotation_speed
        self.color_mode = color_mode
        self.current_frame = 0
        self.pulse_direction = 1
        self.pulse_intensity = 0
    
    def on_mount(self) -> None:
        """Start the animation."""
        self._update_display()
        self.set_interval(self.rotation_speed, self._animate)
    
    def _animate(self) -> None:
        """Advance animation frame."""
        self.current_frame = (self.current_frame + 1) % 8
        
        if self.color_mode == "pulse":
            self.pulse_intensity += self.pulse_direction * 0.1
            if self.pulse_intensity >= 1.0 or self.pulse_intensity <= 0:
                self.pulse_direction *= -1
                self.pulse_intensity = max(0, min(1.0, self.pulse_intensity))
        
        self._update_display()
    
    def _update_display(self) -> None:
        """Render current frame."""
        brain_lines = self.BRAIN_BASE.split('\n')
        
        # Apply rotation transformation
        transformed = self._apply_rotation(brain_lines, self.current_frame)
        
        # Apply coloring
        if self.color_mode == "gradient":
            text = self._apply_gradient(transformed)
        elif self.color_mode == "pulse":
            text = self._apply_pulse(transformed)
        else:
            text = Text('\n'.join(transformed), style="bright_magenta")
        
        self.update(text)
    
    def _apply_rotation(self, lines: list[str], frame: int) -> list[str]:
        """Simulate 3D rotation by transforming the brain."""
        if frame == 0:
            return lines
        elif frame == 1 or frame == 7:
            # Slight tilt - compress horizontally
            return [self._compress_line(line, 0.95) for line in lines]
        elif frame == 2 or frame == 6:
            # More tilt
            return [self._compress_line(line, 0.85) for line in lines]
        elif frame == 3 or frame == 5:
            # Side view - heavy compression
            return [self._compress_line(line, 0.6) for line in lines]
        elif frame == 4:
            # Back view - mirror horizontally
            return [line[::-1] for line in lines]
        
        return lines
    
    def _compress_line(self, line: str, factor: float) -> str:
        """Compress line horizontally to simulate perspective."""
        if not line.strip():
            return line
        
        # Find non-space bounds
        stripped = line.lstrip()
        left_spaces = len(line) - len(stripped)
        stripped = stripped.rstrip()
        
        if not stripped:
            return line
        
        # Compress the content
        new_length = int(len(stripped) * factor)
        if new_length < 1:
            new_length = 1
        
        # Sample characters from original
        step = len(stripped) / new_length
        compressed = ''.join(
            stripped[int(i * step)] for i in range(new_length)
        )
        
        # Re-center
        total_spaces = len(line) - len(compressed)
        new_left_spaces = total_spaces // 2
        
        return ' ' * new_left_spaces + compressed
    
    def _apply_gradient(self, lines: list[str]) -> Text:
        """Apply vertical gradient coloring."""
        text = Text()
        total_lines = len(lines)
        
        # Gradient from cyan -> magenta -> cyan
        for i, line in enumerate(lines):
            progress = i / total_lines
            
            if progress < 0.5:
                # Cyan to magenta
                r = int(0 + (255 * (progress * 2)))
                g = int(255 - (255 * (progress * 2)))
                b = 255
            else:
                # Magenta to cyan
                r = int(255 - (255 * ((progress - 0.5) * 2)))
                g = int(0 + (255 * ((progress - 0.5) * 2)))
                b = 255
            
            color = f"rgb({r},{g},{b})"
            text.append(line + '\n', style=color)
        
        return text
    
    def _apply_pulse(self, lines: list[str]) -> Text:
        """Apply pulsing color effect."""
        text = Text()
        
        # Pulse between bright and dim magenta
        intensity = int(128 + (127 * self.pulse_intensity))
        color = f"rgb({intensity},0,{intensity})"
        
        for line in lines:
            text.append(line + '\n', style=color)
        
        return text


# Enhanced version with glow effect
class GlowingBrain(Static):
    """Brain with neural activity glow effect."""
    
    BRAIN_BASE: ClassVar[str] = RotatingBrain.BRAIN_BASE
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_neurons = set()
        self.frame = 0
    
    def on_mount(self) -> None:
        self._update()
        self.set_interval(0.1, self._pulse)
    
    def _pulse(self) -> None:
        """Simulate neural activity."""
        import random
        
        # Add new active neurons
        if random.random() < 0.3:
            self.active_neurons.add(random.randint(0, 40))
        
        # Decay old neurons
        if self.active_neurons and random.random() < 0.4:
            self.active_neurons.pop()
        
        self.frame = (self.frame + 1) % 360
        self._update()
    
    def _update(self) -> None:
        """Render brain with glowing neurons."""
        lines = self.BRAIN_BASE.split('\n')
        text = Text()
        
        for i, line in enumerate(lines):
            # Check if this line has active neurons
            if i in self.active_neurons:
                # Bright yellow glow for active neurons
                text.append(line + '\n', style="bold bright_yellow")
            else:
                # Rotating hue based on frame
                hue_shift = (self.frame + i * 3) % 360
                if hue_shift < 120:
                    color = "bright_cyan"
                elif hue_shift < 240:
                    color = "bright_magenta"
                else:
                    color = "bright_blue"
                
                text.append(line + '\n', style=color)
        
        self.update(text)


# Complete application
from textual.app import App
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Label


class BrainApp(App):
    """Brain visualization app."""
    
    CSS = """
    Screen {
        align: center middle;
        background: $surface;
    }
    
    #brain-container {
        width: auto;
        height: auto;
        border: heavy $primary;
        background: $panel;
        padding: 1 2;
    }
    
    Label {
        text-align: center;
        color: $text-muted;
        margin-bottom: 1;
    }
    
    RotatingBrain, GlowingBrain {
        width: auto;
        height: auto;
    }
    """
    
    BINDINGS = [
        ("r", "toggle_mode", "Toggle Mode"),
        ("q", "quit", "Quit"),
    ]
    
    def __init__(self):
        super().__init__()
        self.mode = 0
        self.brain_widget = None
    
    def compose(self):
        with Vertical(id="brain-container"):
            yield Label("🧠 Neural Network Visualization")
            yield RotatingBrain(rotation_speed=0.15, color_mode="gradient")
    
    def action_toggle_mode(self) -> None:
        """Cycle through different brain visualizations."""
        container = self.query_one("#brain-container")
        
        # Remove current brain
        if self.brain_widget:
            self.brain_widget.remove()
        
        # Cycle modes
        self.mode = (self.mode + 1) % 3
        
        if self.mode == 0:
            self.brain_widget = RotatingBrain(rotation_speed=0.15, color_mode="gradient")
            container.query_one(Label).update("🧠 Rotating Brain - Gradient")
        elif self.mode == 1:
            self.brain_widget = RotatingBrain(rotation_speed=0.1, color_mode="pulse")
            container.query_one(Label).update("🧠 Pulsing Brain")
        else:
            self.brain_widget = GlowingBrain()
            container.query_one(Label).update("🧠 Neural Activity")
        
        container.mount(self.brain_widget)


if __name__ == "__main__":
    BrainApp().run()
```

## Features

1. **3D Rotation** - 8-frame rotation simulating perspective
2. **Color Modes**:
   - `gradient` - Smooth cyan-to-magenta vertical gradient
   - `pulse` - Rhythmic pulsing effect
   - `glow` - Simulated neural activity with random firing
3. **Compression Effect** - Brain appears to turn by compressing horizontally
4. **Back View** - Horizontally flipped at frame 4

Press `r` to cycle through different visualization modes!



I'll create a rotating brain animation using ASCII art frames:

```python
from textual.widgets import Static
from rich.text import Text
from typing import ClassVar


class RotatingBrain(Static):
    """Animated rotating brain using ASCII art frames."""
    
    DEFAULT_CLASSES = "rotating-brain"
    
    # Brain rotation frames (8 frames for smooth rotation)
    BRAIN_FRAMES: ClassVar[list[str]] = [
        # Frame 0 - Front view
        """\
        ⠀⠀⢀⣠⣤⣤⣤⣤⣀⡀⠀⠀⠀
        ⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀
        ⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀
        ⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀
        ⠀⠻⣿⣿⣿⣿⣿⣿⣿⣿⠟⠀⠀
        ⠀⠀⠀⠉⠛⠛⠛⠛⠉⠀⠀⠀⠀""",
        
        # Frame 1 - Slight right turn
        """\
        ⠀⠀⢀⣠⣤⣤⣤⣄⡀⠀⠀⠀⠀
        ⠀⣠⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀
        ⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀
        ⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠁⠀
        ⠀⠙⢿⣿⣿⣿⣿⣿⣿⡿⠋⠀⠀
        ⠀⠀⠀⠈⠉⠛⠛⠉⠁⠀⠀⠀⠀""",
        
        # Frame 2 - Right side view
        """\
        ⠀⠀⢀⣤⣤⣤⣄⡀⠀⠀⠀⠀⠀
        ⠀⣰⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀
        ⢠⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀
        ⣾⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀
        ⢿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⠀
        ⠀⠻⣿⣿⣿⣿⣿⡿⠋⠀⠀⠀⠀
        ⠀⠀⠀⠉⠛⠛⠉⠀⠀⠀⠀⠀⠀""",
        
        # Frame 3 - Back right
        """\
        ⠀⠀⣠⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀
        ⠀⣼⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠀
        ⣸⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀
        ⢿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀
        ⠈⠻⣿⣿⣿⣿⡿⠃⠀⠀⠀⠀⠀
        ⠀⠀⠈⠛⠛⠋⠀⠀⠀⠀⠀⠀⠀""",
        
        # Frame 4 - Back view
        """\
        ⠀⣠⣤⣤⣤⣤⣄⡀⠀⠀⠀⠀⠀
        ⣰⣿⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀
        ⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀
        ⢻⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⠀⠀
        ⠀⠙⢿⣿⣿⣿⣿⠟⠁⠀⠀⠀⠀
        ⠀⠀⠀⠉⠛⠛⠁⠀⠀⠀⠀⠀⠀""",
        
        # Frame 5 - Back left
        """\
        ⠀⠀⠀⠀⠀⢀⣠⣤⣤⣄⡀⠀⠀
        ⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣷⠀⠀
        ⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⡇⠀
        ⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⠀
        ⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⠀
        ⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⡿⠀
        ⠀⠀⠀⠀⠈⠻⢿⣿⣿⣿⠟⠁⠀
        ⠀⠀⠀⠀⠀⠀⠈⠛⠋⠁⠀⠀⠀""",
        
        # Frame 6 - Left side view
        """\
        ⠀⠀⠀⠀⢀⣠⣤⣤⣤⣄⠀⠀⠀
        ⠀⠀⠀⢀⣿⣿⣿⣿⣿⣿⣧⠀⠀
        ⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⡆⠀
        ⠀⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣷⠀
        ⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀
        ⠀⠀⢻⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀
        ⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⠟⠀⠀
        ⠀⠀⠀⠀⠀⠉⠛⠛⠉⠀⠀⠀⠀""",
        
        # Frame 7 - Slight left turn
        """\
        ⠀⠀⠀⢀⣠⣤⣤⣤⣤⣀⡀⠀⠀
        ⠀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀
        ⠀⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀
        ⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀
        ⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀
        ⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠀
        ⠀⠀⠻⣿⣿⣿⣿⣿⣿⣿⠟⠀⠀
        ⠀⠀⠀⠀⠉⠛⠛⠛⠛⠉⠀⠀⠀""",
    ]
    
    def __init__(
        self,
        rotation_speed: float = 0.3,
        color: str = "bright_magenta",
        **kwargs,
    ) -> None:
        """Initialize rotating brain.
        
        Args:
            rotation_speed: Seconds between frames (lower = faster)
            color: Color for the brain
        """
        super().__init__(**kwargs)
        self.rotation_speed = rotation_speed
        self.color = color
        self.current_frame = 0
    
    def on_mount(self) -> None:
        """Start the rotation animation."""
        self._update_frame()
        self.set_interval(self.rotation_speed, self._rotate)
    
    def _rotate(self) -> None:
        """Advance to next frame."""
        self.current_frame = (self.current_frame + 1) % len(self.BRAIN_FRAMES)
        self._update_frame()
    
    def _update_frame(self) -> None:
        """Display current frame."""
        text = Text(
            self.BRAIN_FRAMES[self.current_frame],
            style=self.color,
            justify="center"
        )
        self.update(text)


# Complete app example
from textual.app import App
from textual.containers import Center, Middle


class BrainApp(App):
    """Simple app showing rotating brain."""
    
    CSS = """
    Screen {
        align: center middle;
        background: $surface;
    }
    
    RotatingBrain {
        width: auto;
        height: auto;
        padding: 2 4;
        border: heavy $primary;
        background: $panel;
    }
    """
    
    def compose(self):
        yield RotatingBrain(rotation_speed=0.2, color="bright_cyan")


if __name__ == "__main__":
    BrainApp().run()
```

## Alternative: 3D-style Brain

```python
class DetailedBrain(Static):
    """More detailed brain with better 3D effect."""
    
    BRAIN_FRAMES: ClassVar[list[str]] = [
        # Frame 0
        """\
          ╭──────────╮
        ╭─┤  ◐◐◐◐◐  ├─╮
       ╭┤  ◐◑◑◑◑◑◐  ├┤╮
       │├◐◑◑◒◒◒◑◑◐├┤│
       ││◑◒◒◓◓◓◒◒◑││
       ││◒◓◓◓◓◓◓◒◑││
       ││◑◒◓◓◓◓◒◑◐││
       │╰┤◑◒◒◒◒◑◐├╯│
       ╰─┤ ◑◑◑◑◐ ├─╯
          ╰────────╯""",
        
        # Frame 1
        """\
           ╭─────────╮
         ╭─┤ ◐◐◐◐◐ ├─╮
        ╭┤  ◐◑◑◑◑◐  ├┤
        │├ ◐◑◑◒◒◑◑◐ ├│
        ││ ◑◒◒◓◓◒◒◑ ││
        ││ ◒◓◓◓◓◓◒◑ ││
        ││ ◑◒◓◓◓◒◑◐ │╯
        │╰┤ ◑◒◒◒◑◐ ├╯
        ╰─┤  ◑◑◑◐  ├─╯
           ╰────────╯""",
        
        # Frame 2
        """\
            ╭────────╮
          ╭─┤ ◐◐◐◐ ├─╮
         ╭┤  ◐◑◑◑◐  ├┤
         │├ ◐◑◑◒◒◑◐ ├│
         ││ ◑◒◒◓◓◒◑ ││
         ││ ◒◓◓◓◓◒◑ │╯
         │╰ ◑◒◓◓◒◑◐ ╯
         ╰┤ ◑◒◒◑◐ ├╯
          ╰┤ ◑◑◐  ├─╯
            ╰─────╯""",
        
        # Add more frames...
    ]
```

## Color Variations

```python
# Pink brain
brain = RotatingBrain(color="bright_magenta")

# Cyan brain  
brain = RotatingBrain(color="bright_cyan")

# Rainbow gradient (advanced)
brain = RotatingBrain(color="bold rgb(255,0,255)")

# Faster rotation
brain = RotatingBrain(rotation_speed=0.1)
```

The brain smoothly rotates through 8 frames creating a 3D spinning effect! Adjust `rotation_speed` to control how fast it spins.


                                                                  
                                                                                        
                                         ░▒█░ ▒█▒                                       
                                      ▓█   ▓█░  ░█░              ▓█▓                    
                                     ▒▓      ▒█   ▓▓       ░▒▒ ▒█  █                    
                                     █░        ▓▓  ░█  ▓█▓░    ▓█▒█▒                    
                                     ▒▓          █░ ██▒          █                      
                                      █░          ██▓            ▓▒                     
                                       ▓▓          █             █                      
                                        ░█         █░          ▓█                       
                                          ▓▓      ▒█        ▓█▒                         
                                            ▓█▓▒▓█████████▓                             
                            █ ▓▓                █       █                               
                            █▓░█████░          ▓▒       █                               
                             ▒█     ▒█         █░       █                               
                             ▒▓       █▒    ░▓███      █░                               
                              ▓▓       █ ░█▓    ▒█░  ▒█▒█▓                              
                               ▒█      ██░       █▒ ░▒█░  ▒█░                           
                                 █░   ▓▓        ▓▓     ▒█   ░█▓                         
                                  ███▓▓█░       █▒       █░    ▓▓                       
                                 ▓▓     █▒   ▒█▓█▓        ▓▓     █░                     
                                █░      ▓▓░░▓▓   █         ▓▓     █                     
                               ▒▓       ▒▓   █   ▒▓         ▒▓     █                    
                               █        ▒▓   ▓▒   ▓▒         ▒█    █▒                   
                              ▓▒        ▒▓   ▓▒    ▓█         ▒▒   ▓▓                   
                              █░        ▓▒   █▒      █▒       ▓▒   █                    
                              █░        █    █         █▓    ▓████▓                     
                              ▒▓       ▒▓   ░█           ░▓█▒                           
                               █      ▒▓   ▓▓                                           
                               ▓░    ▓█▓▒▓█░                                            
                                █▒ ▒█                                                   
                                                                                        
                                                                                        
                                                                                        



                                                                                                                                                                             
                                                  ▒██▓░░░▓██▓                            
                                             ▒▓▓▓█▒   ░░░   ░██                          
                                          ▓█▒░ ░█  ▓█░   ░▓█▒ ░█▒                        
                                        ▒█▒    █░ █▒        ░█  █░                       
                                       ▒█  ░▓██▓ ▓▒          ▒▓ ░█████████░              
         ░░ ▓███░                     ▒█▒█▓░  █ ░▓   ▒▓       █  ▓▒█▓░░  ░█              
       ▒█░▒█░  ▓▓                     ███       ▒▒   ▒▓       █     █▓░   █              
       ▒▓▒▒ ▒▓░█▓▒▓▓███████▓▓▓▒▒▒▒▒▒▒▒▒▒        ▒▒   ░█      ░█      █▓  ▒█              
        █▓                                       █           █▒      ░█▓░█▒              
         █                                        █▒        █▒         ███░              
         █░                                        ▒█▓▒▒▒▓█▓            █▒▒▓███▓         
         ▒█▓░       ░░░░░░░░░                                           ▒█▒▒░  ▓█        
         ▓░ ██▓▒▓█░░ ▒▓     █▒ ░░█▓░▒▒▒▒▓██▓▒░                           ▓▓▒   █░        
          ████   ▒█░▓▓       ▓▒▒█░        █  █▓▓█▒                       ░█░  █▒         
            ░█     ░                     ▓█▓░     ▒█▒                     █▓ █▒          
             ▒█░                                    ░▓▓                   ░██░           
               ▓█▓░                                   ░█▒                  ▓███▒         
                   ░░▒▒▓▓▓▓████████▓████▓▓▒             ▒▓                 ░█░░██▓       
                                      ▒█                 ░█░                █░ ▓▒▓█░     
                                       ▓▓                  █                ▓█▒░  ▒█     
                                       ░█                  ░█               ▓█▒  ░█░     
                                        █░                  █▒              ▒█▒ ▓█       
                                        █▒                  ▓▒              ░█▓█░        
                                        █▒                  █▒              ░█           
                                        █▓                  █░              ▒█           
                                        █▒                  █░              ░▓           
                                                            ░                            
                                                                                         