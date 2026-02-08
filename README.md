![Contexter Logo](logo.jpg)

# Contexter

Contexter is a local-first pipeline for extracting structured and unstructured knowledge from PDF documents. It combines semantic search (RAG) with Knowledge Graph (KG) extraction using local LLMs.

Designed for privacy and offline performance, Contexter processes your documents entirely on your machine.

## Key Features

*   **PDF to Markdown**: Uses [Docling](https://github.com/DS4SD/docling) for high-fidelity conversion.
*   **Hybrid Search**: Merges vector-based semantic search with relationship lookup in a Knowledge Graph.
*   **Local Processing**: Leverages [Ollama](https://ollama.ai) for embeddings and entity/relation extraction.
*   **Graph Visualization**: Generates relationship maps of your document corpus using NetworkX.

## Prerequisites

1.  **Ollama**: Install and serve the recommended models.
    ```bash
    ollama serve
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    ```
2.  **Python 3.10+**: Ensure you have a modern Python environment.

## Installation

```bash
pip install -r requirements.txt
# Or install as a package for CLI access
pip install -e .
```

## Usage

### Ingesting Documents

Process a single file or a batch to populate the vector store and knowledge graph:

```bash
# Process a single PDF
python main.py process document.pdf

# Batch process a directory
python main.py process ./my_pdfs/ --batch
```

### Searching

Perform a hybrid query that checks both semantic similarity and graph relationships:

```bash
python main.py search "How does X relate to Y?"
```

### Utilities

```bash
# List indexed documents
python main.py list

# View library statistics
python main.py stats
```

## Configuration

Settings are managed in `config.yaml`. Performance and extraction quality depend heavily on the Ollama model chosen.

*   **Extraction**: Higher-parameter models (e.g., Llama 3) produce better graphs but require more VRAM.
*   **Memory**: Both ChromaDB and NetworkX run locally; ensure your system has sufficient RAM for large corpora.

## Project Structure

*   `contexter/`: Core package logic (LLM, Store, Graph, Parser).
*   `main.py`: CLI entry point.
*   `config.yaml`: Global settings and LLM prompts.
*   `output/`: Generated markdown, entities, and visualizations.
*   `chroma_db/`: Local vector storage.

---
*Grounded in realistic information extraction. No cloud dependencies.*
