![Plato Logo](logo.png)

```ascii
               d8b                                       
               88P                      d8P              
              d88                    d888888P            
?88,.d88b,    888       d888b8b        ?88'       d8888b 
`?88'  ?88    ?88      d8P' ?88        88P       d8P' ?88
  88b  d8P     88b     88b  ,88b       88b       88b  d88
  888888P'      88b    `?88P'`88b      `?8b      `?8888P'
  88P'                                                   
 d88                                                     
 ?8P    
```

> **PLATO**: A high-integrity, local-first Context Preparation Assistant for independent scholars.

**PLATO** is designed to bridge the gap between static PDFs and AI-ready context. Unlike generic "Chat with PDF" tools, PLATO focuses on **Scholarship Workflows**—summarizing, auditing, and extracting insights using lightweight, local models that respect your privacy and hardware constraints.

## 🦫 Why PLATO?
- 🦫 **Hybrid Intelligence**: Vector search + Structured Context.
- 🏠 **Privacy-first**: All processing happens locally on your machine (via Ollama).
- 🧩 **Zero-Patch Philosophy**: 100% compatible with Python 3.14+ by using native NumPy vector storage (No ChromaDB).
- ⚡ **Lightweight**: Optimized for consumer hardware (M1/M2 Macs with 8GB RAM).
- 🏠 **Zero-Cloud**: Your documents never leave your machine.
- 🧩 **Schema-Driven**: Guaranteed structured output using JSON Schema validation.
- 🏛️ **Hardened Core**: Sandboxed template execution and multi-instance safe vector storage.

## 🖥️ The Interface (TUI)

PLATO features a rich Textual User Interface (TUI) designed for speed:
- **Scholar Mascot**: Your assistant is always visible in the sidebar.
- **File Browser**: Integrated directory tree linked to your research folder.
- **Action Picker**: Select from pre-defined tasks (Summary, Insights, Audit) using arrow keys.
- **Live Stream**: Watch the reasoning process in real-time with Markdown support.
- **Hardware Monitor**: Real-time RAM tracking vs your hardware budget.

## 🚀 Quick Start

### 1. Prerequisites
Install [Ollama](https://ollama.com) and ensure it's running.

### 2. Model Setup
PLATO uses specific model profiles to ensure stability on 8GB/16GB machines. Pull the default set:

```bash
ollama pull lfm2.5-thinking:1.2b
ollama pull qwen2.5-coder:1.5b
ollama pull embeddinggemma:latest
ollama pull deepseek-ocr:3b
```

### 3. Installation
```bash
git clone <repository_url>
cd plato
pip install -e .
```

### 4. Launch
Launch the Scholarship Interface:
```bash
PYTHONPATH=src python3 -m plato.tui.app
```

## 🛠️ Configuration

Edit `src/plato/core/models.yaml` to switch between hardware profiles:
- `m1_8gb`: Optimized for base Apple Silicon laptops (uses 1.2B/1.5B models).
- `performance`: Optimized for Mac Studios/Pros (uses 3B+ models).

Source directories can be configured in `src/plato/config.py` (Default: `~/Documents/plato`).

---
*Built with ❤️ for scholars who value privacy and local power.*