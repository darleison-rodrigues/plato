#!/usr/bin/env python3
import sys
from plato.tui.app import PlatoApp

def app():
    """Entry point for the application script."""
    plato = PlatoApp()
    plato.run()

if __name__ == "__main__":
    app()
