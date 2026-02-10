from textual.screen import Screen
from textual.widgets import Label

class MainScreen(Screen):
    def compose(self):
        yield Label("Welcome to PLATO")
