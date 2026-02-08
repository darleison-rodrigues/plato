.PHONY: setup dev build docker-run test help clean

PYTHON := python3
VENV := .venv
BIN := $(VENV)/bin

# Default target
help:
	@echo "Contexter Management Commands (Native First)"
	@echo "============================================"
	@echo "make setup      - Create venv and install dependencies (Native)"
	@echo "make dev        - Run interactive TUI session (Native)"
	@echo "make test       - Run tests (Native)"
	@echo "make clean      - Clean up artifacts"
	@echo ""


test:
	$(BIN)/pytest tests/

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf output/*.md
	rm -rf output/*.json
