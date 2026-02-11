from textual.widgets import Static
from textual.app import ComposeResult
import psutil
import threading
import time

class MemoryMonitor(Static):
    """Shows real-time RAM usage vs budget."""
    
    def on_mount(self) -> None:
        self.set_interval(2.0, self.update_memory)

    def update_memory(self) -> None:
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        
        # Simple health check based on used %
        color = "green"
        if mem.percent > 80: color = "yellow"
        if mem.percent > 95: color = "red"
        
        status = f"RAM: [{color}]{used_gb:.1f}GB[/] / {total_gb:.0f}GB ({mem.percent:.0f}%)"
        self.update(status)

class StatusBar(Static):
    """Progress and status message display."""
    
    def on_mount(self) -> None:
        self.update("Ready.")

    def set_status(self, message: str, color: str = "white") -> None:
        self.update(f"[{color}]{message}[/]")
