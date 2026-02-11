#!/usr/bin/env python3
import sys
import argparse
from plato.tui.app import PlatoApp

def app():
    """Entry point for the application script."""
    parser = argparse.ArgumentParser(description="PLATO: Local-first PDF-to-Context Processor")
    parser.add_argument("--chat", action="store_true", help="Launch interactive chat shell (REPL)")
    
    args = parser.parse_args()
    
    if args.chat:
        from plato.cli import run
        run()
    else:
        plato = PlatoApp()
        plato.run()

if __name__ == "__main__":
    app()
