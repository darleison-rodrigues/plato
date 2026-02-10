an intelligent local-first document processor that creates
  structured knowledge for you to chat with and use in other projects.



# PLATO: PDF-to-Context TUI

**Version:** 1.0.0  
**Target Hardware:** 8GB RAM (Intel i3 10th Gen / Apple M1)  
**Philosophy:** Local-first, template-driven, hardware-aware context preparation

---

## 🎯 Project Definition

**PLATO** (PDF Learning Assistant for Template Operations) is a keyboard-centric Terminal User Interface for extracting, indexing, and templating PDF content for local LLM consumption. It replaces manual copy-paste workflows with a structured pipeline optimized for resource-constrained hardware.

### Non-Goals

- ❌ Cloud API integration (local-only)
- ❌ Real-time collaboration (single-user)
- ❌ Knowledge graph generation (deferred to v2.0)
- ❌ General-purpose chat interface (templates only)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         PLATO TUI                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ PDF Ingestion│→ │Vector Storage│→ │Template Engine│     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
         ↓                  ↓                    ↓
    PyMuPDF/Marker    ChromaDB/FAISS         Jinja2
         ↓                  ↓                    ↓
    DeepSeek-OCR      EmbeddingGemma      /api/generate
```

### Three-Phase Pipeline

1. **Extract** → PDF to structured chunks with metadata
2. **Index** → Hybrid BM25 + vector search
3. **Synthesize** → Template-driven context generation

---

## 📦 Technical Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| **TUI Framework** | Textual 0.50+ | Rich widgets, async workers, low overhead |
| **PDF Parser** | PyMuPDF (primary)<br>Marker (fallback) | Fast text extraction<br>High-accuracy OCR/markdown |
| **OCR Engine** | DeepSeek-OCR 3B | Local vision model for scanned PDFs |
| **Chunking** | LangChain TextSplitter | Recursive character splitting with overlap |
| **Vector Store** | ChromaDB (i3)<br>FAISS (M1) | Persistent on-disk<br>In-memory with Metal acceleration |
| **Embedding Model** | EmbeddingGemma 300M | 200MB RAM footprint, RAG-optimized |
| **Keyword Search** | Rank-BM25 | Exact match complement to semantic search |
| **LLM Interface** | Ollama `/api/generate` | Stateless, template-friendly |
| **Reasoning Models** | LFM2.5-Thinking 1.2B (i3)<br>Qwen2.5-Coder 3B (M1) | Low-latency inference |
| **Templating** | Jinja2 | Shareable, version-controlled templates |
| **Config Storage** | YAML | Human-readable, Git-friendly |
| **Clipboard** | pyperclip | Direct copy to IDE/terminal |

---

## 🧠 Memory Budget

### i3 10th Gen (8GB RAM)

```
OS + System Services:     2.5 GB
Browser (5 tabs):         1.0 GB  
VS Code:                  1.5 GB
──────────────────────────────────
Available for PLATO:      3.0 GB
```

**Allocation:**
- ChromaDB (10K chunks):  0.5 GB
- EmbeddingGemma:         0.2 GB
- LFM2.5-Thinking:        0.8 GB
- TUI + Python runtime:   0.3 GB
- **Buffer:**             1.2 GB

**Strategy:** Sequential model loading (never run embedder + LLM simultaneously)

### MacBook Air M1 (8GB RAM)

```
OS + System Services:     2.0 GB
Browser (10 tabs):        1.5 GB
VS Code:                  1.5 GB
──────────────────────────────────
Available for PLATO:      3.0 GB
```

**Allocation:**
- FAISS (in-memory):      0.8 GB
- EmbeddingGemma:         0.2 GB
- Qwen2.5-Coder 3B:       1.2 GB
- TUI + Python runtime:   0.3 GB
- **Buffer:**             0.5 GB

**Strategy:** Keep models warm with `keep_alive: 30s`, leverage Metal acceleration

---

## 📋 Functional Specification

### 1. PDF Ingestion

**Supported Formats:**
- Native PDFs (text-based)
- Scanned PDFs (OCR via DeepSeek-OCR)
- Multi-column layouts (Marker preprocessing)

**Processing Pipeline:**

```python
def ingest_pdf(filepath: Path) -> Document:
    """
    Returns Document with chunks and metadata.
    """
    # Step 1: Detect PDF type
    if is_scanned(filepath):
        markdown = ocr_with_deepseek(filepath)
    else:
        markdown = pymupdf_extract(filepath)
    
    # Step 2: Recursive chunking
    chunks = splitter.split_text(
        markdown,
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Step 3: Metadata attachment
    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            "source": filepath.name,
            "chunk_id": f"{filepath.stem}_{i:04d}",
            "page_range": extract_page_numbers(chunk),
            "char_count": len(chunk.content),
            "timestamp": datetime.utcnow()
        }
    
    return Document(chunks=chunks, metadata={...})
```

**Batch Processing:**
- Multi-select PDFs in file browser
- Merge into single document collection
- Maintain per-file metadata for provenance

### 2. Vector Indexing

**Hybrid Search Architecture:**

```python
class HybridRetriever:
    def __init__(self):
        self.vector_store = get_vector_db()  # Chroma or FAISS
        self.bm25 = BM25Okapi(corpus)
        
    def search(self, query: str, k: int = 5) -> List[Chunk]:
        # Parallel retrieval
        vector_results = self.vector_store.similarity_search(
            query, k=k*2
        )
        keyword_results = self.bm25.get_top_n(query, k=k*2)
        
        # Reciprocal Rank Fusion
        combined = self.rerank(vector_results, keyword_results)
        return combined[:k]
```

**Embedding Strategy:**
- **Cache embeddings** on disk keyed by PDF hash (SHA-256)
- Only re-embed if PDF changes
- Background worker for non-blocking UI

**Storage Schema (ChromaDB):**

```python
collection.add(
    ids=[chunk.id],
    embeddings=[embedding_vector],
    documents=[chunk.content],
    metadatas=[{
        "source": "paper.pdf",
        "page": 12,
        "section": "3.2",
        "confidence": 0.87
    }]
)
```

### 3. Template System

**Template Storage:** `~/.config/plato/templates/*.yaml`

**Template Schema:**

```yaml
name: "Security Audit"
version: "1.0"
description: "Extract CVEs and vulnerabilities"
author: "user@example.com"

# Hardware targeting
min_model: "lfm2.5-thinking:1.2b"
recommended_model: "qwen2.5-coder:3b"

# Ollama parameters
ollama_config:
  temperature: 0.1
  top_p: 0.9
  num_ctx: 4096

# System directive
system_prompt: |
  You are a cybersecurity researcher. Extract security-relevant information.
  Output ONLY valid JSON. No preamble.

# User template (Jinja2)
template: |
  Analyze the following document for security risks:
  
  {% for chunk in context %}
  [CHUNK {{ loop.index }} - Page {{ chunk.page }}]
  {{ chunk.content }}
  {% endfor %}
  
  Focus areas: {{ focus_areas | default("CVEs, authentication flaws, data leaks") }}
  
  Output format:
  {
    "vulnerabilities": [...],
    "risk_level": "low|medium|high|critical",
    "recommendations": [...]
  }

# Post-processing
output_format: "json"
validation_schema: "security_audit_v1.json"
```

**Built-in Templates:**

1. `executive_summary.yaml` - 500-word distillation
2. `technical_specs.yaml` - Extract architecture/APIs
3. `code_analysis.yaml` - Bug patterns, complexity metrics
4. `legal_review.yaml` - Contract clauses, obligations
5. `research_qa.yaml` - Q&A pair generation
6. `citation_extract.yaml` - Bibliography with page refs

**Template Variables:**

```jinja2
{{ pdf_title }}           # Auto-extracted from metadata
{{ page_count }}          # Total pages
{{ word_count }}          # Approximate word count
{{ context }}             # Retrieved chunks (List[Chunk])
{{ user_query }}          # Optional query parameter
{{ focus_areas }}         # Custom user input
{{ timestamp }}           # ISO 8601 format
```

### 4. Action System (Not Chat)

**Why `/api/generate` over `/api/chat`:**

| Consideration | `/api/generate` | `/api/chat` |
|--------------|----------------|-------------|
| Memory growth | ✅ Stateless | ❌ Accumulates KV cache |
| Latency (i3) | ✅ Fast pre-fill | ❌ Re-processes history |
| Output control | ✅ Strict templates | ⚠️ Conversational noise |
| Hardware fit | ✅ Low RAM pressure | ❌ Swaps to disk |

**Action Execution:**

```python
async def execute_template(
    template: Template,
    context_chunks: List[Chunk],
    user_vars: Dict[str, Any]
) -> str:
    """
    Stateless template execution via Ollama.
    """
    # Render Jinja2 template
    prompt = template.render(
        context=context_chunks,
        **user_vars
    )
    
    # Estimate token count
    estimated_tokens = len(prompt.split()) * 1.3
    if estimated_tokens > 4096:
        raise TokenLimitError(
            f"Prompt too large: {estimated_tokens} tokens. "
            f"Reduce chunk count or use summarization."
        )
    
    # Call Ollama (stateless)
    response = await ollama.generate(
        model=template.recommended_model,
        prompt=prompt,
        system=template.system_prompt,
        options={
            "temperature": template.ollama_config.temperature,
            "num_ctx": 4096
        },
        stream=False  # Get full response for parsing
    )
    
    # Validate output if schema provided
    if template.output_format == "json":
        validate_json(response, template.validation_schema)
    
    return response
```

**Progressive UI Pattern:**

```python
# In Textual app
with ui.loading_indicator("Processing..."):
    result = await self.run_worker(
        execute_template,
        template=selected_template,
        context_chunks=retrieved_chunks,
        user_vars={"focus_areas": user_input}
    )

# Display in dedicated widget
self.result_panel.update(result)
```

---

## 🎨 User Interface Specification

### Layout (80x24 minimum terminal)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ PLATO v1.0              [HW: M1/8GB] [Model: qwen2.5-coder:3b]  (F1) ?  │
├──────────────┬──────────────────────────────────────────────────────────┤
│              │ QUERY: _________________________________________________ │
│ 📁 BROWSER   │                                                          │
│              │ ┌──────────────────────────────────────────────────────┐ │
│ > ~/docs/    │ │ RETRIEVED CONTEXT (5 chunks)                         │ │
│   - paper.pdf│ │                                                      │ │
│   - spec.pdf │ │ [1] Page 3 | Similarity: 0.92                       │ │
│ > ~/research/│ │ The architecture consists of three layers...        │ │
│              │ │                                                      │ │
│ 📋 TEMPLATES │ │ [2] Page 7 | Similarity: 0.88                       │ │
│              │ │ Performance metrics show...                          │ │
│ > Built-in   │ │                                                      │ │
│   - Summary  │ │ [Show More] [Adjust K] [Re-search]                  │ │
│   - Audit    │ └──────────────────────────────────────────────────────┘ │
│ > Custom     │                                                          │
│   - MyTmpl   │ ┌──────────────────────────────────────────────────────┐ │
│              │ │ OUTPUT (JSON validated ✓)                            │ │
│ 📊 INDEX     │ │ {                                                    │ │
│              │ │   "summary": "This document describes...",           │ │
│ Docs: 12     │ │   "key_points": ["...", "..."]                       │ │
│ Chunks: 4.2K │ │ }                                                    │ │
│ Size: 890MB  │ │                                                      │ │
│              │ │ [Copy] [Save] [Export Markdown]                      │ │
│              │ └──────────────────────────────────────────────────────┘ │
├──────────────┴──────────────────────────────────────────────────────────┤
│ STATUS: Ready | RAM: 2.8/3.0 GB | Last action: 1.2s | Chunks cached: ✓ │
└─────────────────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+O` | Open PDF(s) |
| `Ctrl+I` | Index current PDF |
| `Ctrl+T` | Template picker |
| `Ctrl+R` | Run template action |
| `Ctrl+C` | Copy output to clipboard |
| `Ctrl+S` | Save output to file |
| `Ctrl+F` | Search query input |
| `Ctrl+L` | Clear output panel |
| `F2` | Hardware profile toggle |
| `F5` | Refresh index stats |
| `F10` / `Ctrl+Q` | Quit |
| `j/k` | Navigate lists (Vim bindings) |
| `Tab` | Cycle focus between panels |

### Hardware Profile Toggle (F2)

```python
class HardwareProfile(Enum):
    ECO = "eco"          # i3 HDD, aggressive unloading
    BALANCED = "balanced" # i3 SSD, moderate caching
    PERFORMANCE = "perf"  # M1, keep models warm

# Auto-detection
def detect_profile() -> HardwareProfile:
    ram_gb = psutil.virtual_memory().total / 1e9
    is_ssd = check_disk_type()
    is_arm = platform.machine() == 'arm64'
    
    if is_arm and ram_gb >= 7:
        return HardwareProfile.PERFORMANCE
    elif ram_gb < 6 or not is_ssd:
        return HardwareProfile.ECO
    else:
        return HardwareProfile.BALANCED
```

**Profile Behaviors:**

| Setting | ECO | BALANCED | PERFORMANCE |
|---------|-----|----------|-------------|
| Embedding model | `all-minilm` | `embeddinggemma` | `embeddinggemma` |
| LLM model | `smollm2:135m` | `lfm2.5-thinking:1.2b` | `qwen2.5-coder:3b` |
| `keep_alive` | `0s` | `15s` | `30s` |
| Vector DB | ChromaDB (disk) | ChromaDB (disk) | FAISS (memory) |
| Max chunks in RAM | 5K | 10K | 20K |
| Concurrent workers | 1 | 2 | 3 |

---

## 🔧 Implementation Details

### Chunk Metadata Schema

```python
@dataclass
class Chunk:
    id: str                    # {source_stem}_{index:04d}
    content: str               # Actual text
    embedding: Optional[List[float]]
    metadata: ChunkMetadata
    
@dataclass
class ChunkMetadata:
    source: str                # PDF filename
    page: int                  # Page number (1-indexed)
    section: Optional[str]     # Extracted heading/section
    char_start: int            # Character offset in original
    char_end: int
    confidence: float          # OCR confidence (0-1)
    created_at: datetime
    hash: str                  # SHA-256 of content
```

### Error Handling

```python
class PLATOError(Exception):
    """Base exception"""
    
class PDFProcessingError(PLATOError):
    """Failed to parse PDF"""
    
class ModelUnavailableError(PLATOError):
    """Ollama model not pulled"""
    
class TokenLimitError(PLATOError):
    """Prompt exceeds context window"""
    
class MemoryPressureError(PLATOError):
    """Insufficient RAM available"""

# Graceful degradation
try:
    result = await execute_template(...)
except TokenLimitError as e:
    # Automatically reduce chunk count
    chunks = chunks[:3]
    result = await execute_template(...)
except MemoryPressureError:
    # Switch to smaller model
    switch_to_profile(HardwareProfile.ECO)
    retry_action()
```

### Memory Pressure Monitor

```python
class MemoryMonitor:
    def __init__(self, threshold_gb: float = 1.0):
        self.threshold = threshold_gb * 1e9
        
    def check(self) -> bool:
        """Returns True if safe to proceed."""
        available = psutil.virtual_memory().available
        return available > self.threshold
    
    def get_status(self) -> str:
        mem = psutil.virtual_memory()
        used_gb = (mem.total - mem.available) / 1e9
        total_gb = mem.total / 1e9
        return f"{used_gb:.1f}/{total_gb:.1f} GB"

# In TUI status bar
monitor = MemoryMonitor()
status_bar.update(f"RAM: {monitor.get_status()}")
```

---

## 📊 Performance Targets

### i3 10th Gen (8GB RAM, SATA SSD)

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| PDF ingestion (100 pages) | < 10s | PyMuPDF parsing |
| OCR (scanned PDF, 100 pages) | < 60s | DeepSeek-OCR 3B |
| Chunk embedding (1000 chunks) | < 30s | EmbeddingGemma, batch size 32 |
| Vector search (10K index) | < 500ms | ChromaDB on SSD |
| Template execution | < 8s | LFM2.5-Thinking 1.2B, 2K context |
| Model cold start | < 12s | Loading from disk |

### MacBook Air M1 (8GB RAM, NVMe)

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| PDF ingestion (100 pages) | < 5s | Metal-accelerated |
| OCR (scanned PDF, 100 pages) | < 30s | DeepSeek-OCR 3B |
| Chunk embedding (1000 chunks) | < 15s | EmbeddingGemma, batch size 64 |
| Vector search (20K index) | < 200ms | FAISS in-memory |
| Template execution | < 3s | Qwen2.5-Coder 3B, 4K context |
| Model warm start | < 1s | Kept in memory |

---

## 🗂️ File Structure

```
plato/
├── pyproject.toml
├── README.md
├── .gitignore
│
├── plato/
│   ├── __init__.py
│   ├── __main__.py          # Entry point
│   │
│   ├── core/
│   │   ├── pdf.py           # PDF parsing/OCR
│   │   ├── chunker.py       # Text splitting
│   │   ├── embedder.py      # Embedding generation
│   │   ├── retriever.py     # Hybrid search
│   │   └── template.py      # Jinja2 engine
│   │
│   ├── models/
│   │   ├── document.py      # Data classes
│   │   ├── chunk.py
│   │   └── template_config.py
│   │
│   ├── ollama/
│   │   ├── client.py        # Ollama API wrapper
│   │   └── model_manager.py # Model loading/unloading
│   │
│   ├── storage/
│   │   ├── vector_db.py     # ChromaDB/FAISS abstraction
│   │   ├── cache.py         # Embedding cache
│   │   └── config.py        # User settings
│   │
│   ├── tui/
│   │   ├── app.py           # Main Textual app
│   │   ├── widgets/
│   │   │   ├── file_browser.py
│   │   │   ├── template_picker.py
│   │   │   ├── result_panel.py
│   │   │   └── status_bar.py
│   │   └── screens/
│   │       ├── main.py
│   │       └── settings.py
│   │
│   └── utils/
│       ├── hardware.py      # Profile detection
│       ├── memory.py        # RAM monitoring
│       └── validation.py    # JSON schema validation
│
├── templates/               # Shareable templates
│   ├── executive_summary.yaml
│   ├── security_audit.yaml
│   ├── technical_specs.yaml
│   └── research_qa.yaml
│
├── tests/
│   ├── test_pdf.py
│   ├── test_retriever.py
│   └── test_templates.py
│
└── docs/
    ├── TEMPLATES.md         # Template authoring guide
    ├── HARDWARE.md          # Optimization guide
    └── API.md               # Ollama integration
```

---

## 🚀 Installation

### Prerequisites

```bash
# System dependencies
brew install poppler tesseract  # macOS
sudo apt install poppler-utils tesseract-ocr  # Linux

# Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull embeddinggemma
ollama pull lfm2.5-thinking:1.2b    # i3
ollama pull qwen2.5-coder:3b        # M1
ollama pull deepseek-ocr:3b         # Optional: scanned PDFs
```

### Python Setup

```bash
# Clone repo
git clone https://github.com/yourusername/plato.git
cd plato

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

# Run
plato
```

### Configuration

First run creates `~/.config/plato/config.yaml`:

```yaml
version: "1.0"

hardware:
  profile: "auto"  # auto | eco | balanced | performance
  max_ram_gb: 3.0
  
ollama:
  host: "http://localhost:11434"
  timeout: 60
  
models:
  embedding: "embeddinggemma"
  reasoning: "lfm2.5-thinking:1.2b"  # Auto-adjusted per profile
  ocr: "deepseek-ocr:3b"
  
vector_db:
  backend: "chromadb"  # chromadb | faiss
  persist_directory: "~/.local/share/plato/vectors"
  
chunking:
  chunk_size: 512
  chunk_overlap: 50
  
retrieval:
  default_k: 5
  bm25_weight: 0.3
  vector_weight: 0.7
  
ui:
  vim_bindings: true
  theme: "monokai"
```

---

## 📈 Benchmarking

### Built-in Performance Test

```bash
plato benchmark --pdf sample.pdf --iterations 10
```

Output:

```
PLATO Performance Benchmark
Hardware: MacBook Air M1 (8GB)
Profile: PERFORMANCE

PDF Ingestion (100 pages):     4.2s
Chunking (1847 chunks):        0.8s
Embedding (batch=64):         12.1s
Index building:                1.3s
Vector search (k=5):           0.18s
Template execution:            2.7s
Total pipeline:               21.3s

Memory usage:
  Peak:      2.9 GB
  Average:   2.4 GB
  Available: 0.8 GB (CAUTION)
```

---

## 🔐 Security & Privacy

- **No telemetry:** Zero external network calls (except Ollama if remote)
- **Local storage:** All vectors/cache stored in `~/.local/share/plato/`
- **No logging of content:** Only metadata logged to `~/.cache/plato/logs/`
- **Encrypted templates:** (Optional) GPG-encrypt sensitive templates

---

## 🛣️ Roadmap

### v1.0 (MVP) - Current Spec
- ✅ PDF ingestion with OCR
- ✅ Hybrid vector + keyword search
- ✅ Template-driven actions
- ✅ Hardware-aware profiles

### v1.1 (Refinement)
- [ ] Multi-language OCR (Tesseract fallback)
- [ ] Table extraction (Camelot integration)
- [ ] Export to Obsidian/Notion format
- [ ] Template marketplace (community templates)

### v2.0 (Advanced)
- [ ] Knowledge graph generation (entity linking)
- [ ] Multi-document reasoning (cross-reference detection)
- [ ] Streaming template execution
- [ ] API server mode (headless operation)

---

## 📝 Contributing

### Template Contributions

1. Add your template to `templates/community/`
2. Include example output in `templates/community/examples/`
3. Submit PR with:
   - Template file
   - README with use case
   - Performance notes (hardware/model tested)

### Code Contributions

- Follow Black formatting
- Add type hints
- Write tests for new features
- Update docs

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- **Ollama** - Local LLM inference
- **Textual** - Beautiful TUI framework
- **ChromaDB** - Vector database
- **PyMuPDF** - Fast PDF parsing

---

**Status:** READY FOR IMPLEMENTATION  
**Last Updated:** 2025-02-10


To maximize the utility of **PLATO**, these use cases focus on your specific background as a **Senior Backend Technical Lead** and your personal interests (cycling, hydroponics, Salesforce).

Since you are on 8GB hardware, these cases highlight how PLATO "distills" heavy documents into small, manageable contexts that won't crash your local LLM.

---

## 1. The "Legacy Codebase" Onboarding

**User Scenario:** You are starting a new contract or project with a 400-page PDF manual and undocumented API specs.

* **The Input:** Technical PDFs, API documentation, and architecture diagrams.
* **The Template:** `technical_specs.yaml`
* **The Action:** Use **Hybrid Search** to query for specific endpoints (e.g., "OCAPI authentication flow").
* **PLATO Benefit:** Instead of the LLM hallucinating the API, PLATO pulls the exact JSON schemas and authentication headers from the PDF and presents them as a "System Context" for your IDE.

## 2. The "Gregario" Cycling Community Research

**User Scenario:** You are building your "Strava of Commerce" platform and need to analyze competitor terms of service or cycling event regulations.

* **The Input:** Competitor PDFs, race rulebooks, and e-commerce compliance documents.
* **The Template:** `legal_review.yaml`
* **The Action:** "Extract all clauses related to data privacy and user-generated content."
* **PLATO Benefit:** On the **i3 laptop**, you can't feed a 50-page Terms of Service into an LLM. PLATO chunks it, finds the 5 relevant privacy paragraphs, and provides a summary without exceeding your 3.0 GB RAM budget.

## 3. The Hydroponics "Kratky" Optimizer

**User Scenario:** You are researching scientific papers or advanced guides for your glass-jar Kratky setup.

* **The Input:** Academic papers on nutrient solution pH levels and light cycles.
* **The Template:** `research_qa.yaml`
* **The Action:** "What are the specific nutrient concentrations recommended for lettuce in non-circulated water?"
* **PLATO Benefit:** **EmbeddingGemma** (the 300M model) is perfect here. It will find the specific tables or numeric data in a dense paper that a general chat model might skip over or summarize too broadly.

## 4. Substack Content Distillation

**User Scenario:** You want to write a technical blog post for [darleison.substack.com](https://www.google.com/search?q=https://darleison.substack.com) based on a new whitepaper from NVIDIA or Salesforce.

* **The Input:** A new Salesforce Commerce Cloud (SCAPI) whitepaper.
* **The Template:** `executive_summary.yaml` (modified for "Blog Style").
* **The Action:** "Identify 3 key innovations and explain them for a mid-level developer audience."
* **PLATO Benefit:** Using the **PERF profile** on your **M1 Mac**, you can quickly cycle through 3 different templates to see which "angle" works best for your blog post before you ever start writing.

## 5. The "Smart Home" Server Audit

**User Scenario:** You are configuring your **PowerEdge R710** or **Jetson Nano** and need to reference specific hardware manuals for pinouts or BIOS settings.

* **The Input:** Scanned hardware manuals (PDFs from the early 2010s).
* **The Template:** `technical_specs.yaml`
* **The Action:** Trigger **DeepSeek-OCR** for specific pages containing motherboard diagrams.
* **PLATO Benefit:** Since older manuals are often "image-only," PLATO's OCR fallback ensures that technical data trapped in images becomes searchable text you can copy directly into your server setup script.

---

### Comparison of User Experience

| Use Case | Recommended Profile | Key Model | Why? |
| --- | --- | --- | --- |
| **API Onboarding** | PERF (M1) | `qwen2.5-coder:3b` | High accuracy for code/syntax. |
| **Legal/Compliance** | ECO (i3) | `lfm2.5-thinking:1.2b` | Needs reasoning but low memory. |
| **Manual OCR** | BALANCED (i3/SSD) | `deepseek-ocr:3b` | CPU-intensive, needs patience. |



# PLATO: User Cases & User Stories for TDD

**Version:** 1.0.0  
**Format:** Gherkin (Given-When-Then) + Acceptance Criteria  
**Purpose:** Test-Driven Development with coding agents

---

## 👥 User Personas

### Persona 1: Academic Researcher
- **Name:** Dr. Sarah Chen
- **Hardware:** MacBook Air M1 (8GB RAM)
- **Use Case:** Extracting research findings from 50+ academic papers
- **Pain Point:** Manual copy-paste loses citation context
- **Goal:** Generate literature review context with page references

### Persona 2: Indie Developer
- **Name:** Alex Kim
- **Hardware:** Laptop i3 10th Gen (8GB RAM, SSD)
- **Use Case:** Analyzing API documentation PDFs for integration work
- **Pain Point:** Context window limits when pasting full docs to LLM
- **Goal:** Extract only relevant API sections based on query

### Persona 3: Product Manager
- **Name:** Jamie Rodriguez
- **Hardware:** Laptop i5 11th Gen (16GB RAM)
- **Use Case:** Security audits of vendor documentation
- **Pain Point:** Manual search for compliance terms across 100+ page contracts
- **Goal:** Auto-extract security/privacy clauses into structured format

---

## 🎯 Epic 1: PDF Ingestion & Parsing

### User Story 1.1: Open Single PDF

```gherkin
Feature: Open and parse a single PDF file

  As a researcher
  I want to open a PDF file in the TUI
  So that I can extract its content for LLM consumption

  Scenario: Open a text-based PDF
    Given I have launched PLATO
    And I have a PDF file at "~/docs/research_paper.pdf"
    When I press "Ctrl+O"
    And I select "research_paper.pdf"
    Then the file browser should highlight the selected file
    And the status bar should show "Processing: research_paper.pdf"
    And within 10 seconds, the status should change to "Ready: 1 document indexed"
    And the index stats should show:
      | Metric | Value |
      | Docs   | 1     |
      | Chunks | >0    |
      | Size   | >0 MB |

  Scenario: Open a scanned PDF (OCR required)
    Given I have a scanned PDF at "~/docs/scanned_report.pdf"
    When I open the file
    Then PLATO should detect it requires OCR
    And show "OCR processing..." in the status bar
    And use the DeepSeek-OCR model
    And complete within 60 seconds for a 100-page document
```

**Acceptance Criteria:**
- ✅ File dialog opens with `Ctrl+O`
- ✅ Supports `.pdf` files only
- ✅ Auto-detects scanned vs. text-based PDFs
- ✅ Shows progress indicator during processing
- ✅ Updates index stats in sidebar
- ✅ Handles files up to 500 pages

**Test Cases:**

```python
# test_pdf_ingestion.py

def test_open_pdf_creates_chunks():
    """
    GIVEN: A 10-page text PDF
    WHEN: User opens the file
    THEN: System creates chunks with metadata
    """
    pdf_path = Path("tests/fixtures/sample_10page.pdf")
    document = ingest_pdf(pdf_path)
    
    assert document is not None
    assert len(document.chunks) > 0
    assert all(chunk.metadata.source == "sample_10page.pdf" 
               for chunk in document.chunks)
    assert all(chunk.metadata.page > 0 
               for chunk in document.chunks)

def test_scanned_pdf_uses_ocr():
    """
    GIVEN: A scanned PDF (image-based)
    WHEN: System processes the file
    THEN: OCR engine is invoked
    """
    pdf_path = Path("tests/fixtures/scanned_invoice.pdf")
    
    with patch('plato.core.pdf.ocr_with_deepseek') as mock_ocr:
        mock_ocr.return_value = "Extracted text..."
        document = ingest_pdf(pdf_path)
        
        mock_ocr.assert_called_once()
        assert document.chunks[0].content != ""

def test_pdf_ingestion_performance_i3():
    """
    GIVEN: Hardware profile is ECO (i3)
    WHEN: Processing a 100-page PDF
    THEN: Completes within 10 seconds
    """
    pdf_path = Path("tests/fixtures/large_100page.pdf")
    
    start = time.time()
    document = ingest_pdf(pdf_path)
    duration = time.time() - start
    
    assert duration < 10.0
    assert len(document.chunks) > 100  # Reasonable chunk count

def test_corrupted_pdf_raises_error():
    """
    GIVEN: A corrupted PDF file
    WHEN: User attempts to open it
    THEN: PDFProcessingError is raised with helpful message
    """
    pdf_path = Path("tests/fixtures/corrupted.pdf")
    
    with pytest.raises(PDFProcessingError) as exc_info:
        ingest_pdf(pdf_path)
    
    assert "Unable to parse PDF" in str(exc_info.value)
    assert pdf_path.name in str(exc_info.value)
```

---

### User Story 1.2: Batch PDF Selection

```gherkin
Feature: Select multiple PDFs for unified index

  As a researcher
  I want to select multiple related PDFs
  So that I can query across all documents simultaneously

  Scenario: Select 3 PDFs from file browser
    Given I have PDFs at:
      | Path                    |
      | ~/docs/paper1.pdf       |
      | ~/docs/paper2.pdf       |
      | ~/docs/paper3.pdf       |
    When I navigate to the file browser
    And I press "Space" on "paper1.pdf"
    And I press "Space" on "paper2.pdf"
    And I press "Space" on "paper3.pdf"
    Then the sidebar should show "3 selected"
    When I press "Ctrl+I" to index
    Then the status should show "Indexing 3 documents..."
    And the final index should contain chunks from all 3 sources
    And chunk metadata should preserve individual source filenames

  Scenario: Batch indexing respects memory limits
    Given I am on ECO hardware profile (i3, 3GB available RAM)
    When I select 10 PDFs totaling 500MB
    And I press "Ctrl+I"
    Then the system should process files sequentially
    And never exceed 3GB RAM usage
    And show progress like "Processing 3/10..."
```

**Acceptance Criteria:**
- ✅ `Space` key toggles file selection
- ✅ Visual indicator shows selected files (e.g., `[x]`)
- ✅ Batch indexing processes all selected files
- ✅ Chunks maintain source provenance via metadata
- ✅ Memory-aware processing (sequential on low RAM)

**Test Cases:**

```python
def test_batch_selection_creates_unified_index():
    """
    GIVEN: 3 PDF files selected
    WHEN: Batch indexing is triggered
    THEN: All chunks are in a single searchable index
    """
    pdf_paths = [
        Path("tests/fixtures/paper1.pdf"),
        Path("tests/fixtures/paper2.pdf"),
        Path("tests/fixtures/paper3.pdf"),
    ]
    
    index = batch_ingest(pdf_paths)
    
    # Verify chunks from all sources
    sources = {chunk.metadata.source for chunk in index.chunks}
    assert sources == {"paper1.pdf", "paper2.pdf", "paper3.pdf"}
    
    # Verify searchable
    results = index.search("neural networks")
    assert len(results) > 0

def test_batch_indexing_respects_memory_limit():
    """
    GIVEN: ECO profile with 3GB limit
    WHEN: Indexing 10 large PDFs
    THEN: Peak RAM usage stays below 3GB
    """
    pdf_paths = [Path(f"tests/fixtures/large_{i}.pdf") for i in range(10)]
    
    monitor = MemoryMonitor(threshold_gb=3.0)
    
    with monitor.track():
        batch_ingest(pdf_paths, profile=HardwareProfile.ECO)
    
    assert monitor.peak_usage_gb < 3.0

def test_batch_progress_callback():
    """
    GIVEN: A progress callback is registered
    WHEN: Batch processing 5 PDFs
    THEN: Callback is invoked with progress updates
    """
    pdf_paths = [Path(f"tests/fixtures/doc_{i}.pdf") for i in range(5)]
    progress_updates = []
    
    def on_progress(current, total):
        progress_updates.append((current, total))
    
    batch_ingest(pdf_paths, progress_callback=on_progress)
    
    assert len(progress_updates) == 5
    assert progress_updates[-1] == (5, 5)
```

---

## 🎯 Epic 2: Vector Search & Retrieval

### User Story 2.1: Semantic Search

```gherkin
Feature: Search indexed documents by semantic similarity

  As a developer
  I want to search for concepts, not just keywords
  So that I can find relevant context even with different terminology

  Scenario: Search returns semantically similar chunks
    Given I have indexed "aws_s3_documentation.pdf"
    When I enter the query "object storage pricing"
    And I press "Ctrl+F"
    Then the system should retrieve chunks about:
      | Topic            |
      | S3 storage costs |
      | Pricing tiers    |
      | Data transfer fees |
    And the results should be ranked by similarity score
    And each result should show:
      | Field      | Format                |
      | Page       | Integer               |
      | Similarity | Float (0.0-1.0)       |
      | Preview    | First 100 characters |

  Scenario: Adjust retrieval count (k parameter)
    Given I have indexed 3 documents
    And I have searched for "machine learning"
    When I press "k" to adjust retrieval count
    And I set k=10
    Then the UI should display 10 results instead of default 5
    And results should still be ordered by relevance
```

**Acceptance Criteria:**
- ✅ Query input field accepts natural language
- ✅ Returns top-k chunks (default k=5)
- ✅ Results show similarity score (0.0-1.0)
- ✅ Clicking a result shows full chunk content
- ✅ Search completes in <500ms for 10K chunk index (M1)

**Test Cases:**

```python
def test_semantic_search_returns_relevant_chunks():
    """
    GIVEN: Indexed document about cloud storage
    WHEN: User searches for "object storage pricing"
    THEN: Results include semantically related chunks
    """
    # Setup
    index = create_test_index("aws_s3_docs.pdf")
    
    # Execute
    results = index.search("object storage pricing", k=5)
    
    # Verify
    assert len(results) == 5
    assert all(0.0 <= r.similarity <= 1.0 for r in results)
    
    # Check semantic relevance (keywords may differ)
    content_combined = " ".join(r.content for r in results).lower()
    assert any(term in content_combined 
               for term in ["s3", "storage", "price", "cost", "tier"])

def test_search_performance_10k_chunks_m1():
    """
    GIVEN: Index with 10K chunks on M1 hardware
    WHEN: Performing vector search
    THEN: Query completes in under 500ms
    """
    index = create_test_index(chunk_count=10_000, backend="faiss")
    
    start = time.time()
    results = index.search("test query", k=5)
    duration = time.time() - start
    
    assert duration < 0.5  # 500ms
    assert len(results) == 5

def test_hybrid_search_combines_keyword_and_vector():
    """
    GIVEN: Document containing exact phrase "CVE-2024-1234"
    WHEN: Searching for "CVE-2024-1234"
    THEN: Hybrid retriever ranks exact match higher
    """
    index = HybridRetriever(
        vector_weight=0.7,
        bm25_weight=0.3
    )
    index.add_document("security_report.pdf")
    
    results = index.search("CVE-2024-1234", k=5)
    
    # Exact match should be in top 3
    top_3_content = " ".join(r.content for r in results[:3])
    assert "CVE-2024-1234" in top_3_content

def test_adjustable_k_parameter():
    """
    GIVEN: Default k=5
    WHEN: User changes k to 10
    THEN: Search returns 10 results
    """
    index = create_test_index("sample.pdf")
    
    # Default
    results_default = index.search("query", k=5)
    assert len(results_default) == 5
    
    # Adjusted
    results_adjusted = index.search("query", k=10)
    assert len(results_adjusted) == 10
```

---

### User Story 2.2: Chunk Provenance Display

```gherkin
Feature: Display chunk source metadata

  As a researcher
  I want to see which page and document each chunk came from
  So that I can cite sources properly

  Scenario: Search results show full metadata
    Given I have indexed "research_paper.pdf"
    When I search for "neural networks"
    Then each result should display:
      | Field       | Example Value              |
      | Source      | research_paper.pdf         |
      | Page        | 12                         |
      | Section     | 3.2 (if extractable)       |
      | Similarity  | 0.87                       |
      | Char Range  | 4500-5012                  |
    
  Scenario: Navigate to chunk in context
    Given I have a search result from page 12
    When I press "Enter" on that result
    Then the UI should show:
      | Element          | Content                        |
      | Full chunk       | Complete text of the chunk     |
      | Surrounding text | Previous and next chunks       |
      | Page indicator   | "Page 12 of 45"                |
```

**Acceptance Criteria:**
- ✅ Metadata displayed in result list
- ✅ Page numbers are 1-indexed (user-friendly)
- ✅ Section headers extracted when available
- ✅ Character range shows position in original document

**Test Cases:**

```python
def test_chunk_metadata_completeness():
    """
    GIVEN: A processed PDF
    WHEN: Chunks are created
    THEN: All metadata fields are populated
    """
    pdf_path = Path("tests/fixtures/research_paper.pdf")
    document = ingest_pdf(pdf_path)
    
    for chunk in document.chunks:
        assert chunk.metadata.source == "research_paper.pdf"
        assert chunk.metadata.page > 0  # 1-indexed
        assert chunk.metadata.char_start >= 0
        assert chunk.metadata.char_end > chunk.metadata.char_start
        assert chunk.metadata.created_at is not None
        assert chunk.metadata.hash is not None

def test_section_extraction_from_headers():
    """
    GIVEN: PDF with markdown headers (## Section 3.2)
    WHEN: Processing the PDF
    THEN: Chunks inherit section metadata
    """
    pdf_content = """
    ## 3.2 Neural Network Architecture
    
    The model consists of...
    """
    
    chunks = chunk_with_metadata(pdf_content, source="paper.pdf")
    
    # Chunks after the header should have section metadata
    chunk_in_section = next(
        c for c in chunks if "model consists" in c.content
    )
    assert chunk_in_section.metadata.section == "3.2"

def test_surrounding_context_retrieval():
    """
    GIVEN: A chunk at position N
    WHEN: User requests context
    THEN: System returns chunks N-1, N, N+1
    """
    document = create_test_document(chunk_count=20)
    
    target_chunk_id = document.chunks[10].id
    context = document.get_surrounding_context(target_chunk_id, window=1)
    
    assert len(context) == 3
    assert context[0].id == document.chunks[9].id
    assert context[1].id == document.chunks[10].id
    assert context[2].id == document.chunks[11].id
```

---

## 🎯 Epic 3: Template System

### User Story 3.1: Select and Execute Template

```gherkin
Feature: Execute predefined templates on retrieved context

  As a product manager
  I want to apply a "Security Audit" template
  So that I can extract compliance-relevant information automatically

  Scenario: Execute built-in template
    Given I have indexed "vendor_contract.pdf"
    And I have searched for "data privacy"
    And 5 relevant chunks are displayed
    When I press "Ctrl+T"
    Then a template picker should appear with:
      | Template Name      | Category  |
      | Executive Summary  | Built-in  |
      | Security Audit     | Built-in  |
      | Technical Specs    | Built-in  |
    When I select "Security Audit"
    And I press "Ctrl+R" to run
    Then the status should show "Processing with qwen2.5-coder:3b..."
    And within 5 seconds, the output panel should display JSON:
      """
      {
        "vulnerabilities": [...],
        "risk_level": "medium",
        "recommendations": [...]
      }
      """
    And the JSON should be validated against the template schema

  Scenario: Template execution with user variables
    Given I have selected the "Security Audit" template
    And the template has a variable {{ focus_areas }}
    When the template picker opens
    Then I should see an input field for "Focus Areas"
    When I enter "GDPR compliance, data encryption"
    And I execute the template
    Then the rendered prompt should include my custom focus areas
```

**Acceptance Criteria:**
- ✅ `Ctrl+T` opens template picker
- ✅ Templates are categorized (Built-in / Custom)
- ✅ Template picker shows description on hover
- ✅ Variable inputs are validated (required vs. optional)
- ✅ Output is validated against schema if specified
- ✅ Execution time is displayed in status bar

**Test Cases:**

```python
def test_template_execution_with_context():
    """
    GIVEN: A template and retrieved chunks
    WHEN: Template is executed
    THEN: Ollama is called with rendered prompt
    """
    template = load_template("security_audit.yaml")
    chunks = [
        Chunk(content="Section about encryption...", metadata=...),
        Chunk(content="Data retention policies...", metadata=...),
    ]
    
    with patch('plato.ollama.client.generate') as mock_ollama:
        mock_ollama.return_value = '{"vulnerabilities": []}'
        
        result = execute_template(
            template=template,
            context_chunks=chunks,
            user_vars={"focus_areas": "GDPR"}
        )
        
        # Verify Ollama was called
        mock_ollama.assert_called_once()
        call_args = mock_ollama.call_args
        
        # Check that chunks were included in prompt
        prompt = call_args.kwargs['prompt']
        assert "encryption" in prompt.lower()
        assert "retention" in prompt.lower()
        assert "GDPR" in prompt

def test_template_json_validation():
    """
    GIVEN: Template with JSON output schema
    WHEN: Execution returns invalid JSON
    THEN: Validation error is raised
    """
    template = load_template("security_audit.yaml")
    template.output_format = "json"
    template.validation_schema = "security_audit_v1.json"
    
    with patch('plato.ollama.client.generate') as mock_ollama:
        # Invalid: missing required field
        mock_ollama.return_value = '{"vulnerabilities": []}'  
        
        with pytest.raises(ValidationError) as exc_info:
            execute_template(template, chunks=[], user_vars={})
        
        assert "risk_level" in str(exc_info.value)

def test_template_token_limit_enforcement():
    """
    GIVEN: Context that exceeds 4096 token limit
    WHEN: Template execution is attempted
    THEN: TokenLimitError is raised with helpful message
    """
    template = load_template("executive_summary.yaml")
    
    # Create chunks that total ~6000 tokens
    huge_chunks = [
        Chunk(content="word " * 1500, metadata=...)
        for _ in range(4)
    ]
    
    with pytest.raises(TokenLimitError) as exc_info:
        execute_template(template, huge_chunks, {})
    
    error_msg = str(exc_info.value)
    assert "6000" in error_msg or "exceed" in error_msg.lower()
    assert "Reduce chunk count" in error_msg

def test_template_performance_target_m1():
    """
    GIVEN: M1 hardware with qwen2.5-coder:3b
    WHEN: Executing template with 2K token context
    THEN: Completes in under 3 seconds
    """
    template = load_template("technical_specs.yaml")
    chunks = create_test_chunks(total_tokens=2000)
    
    start = time.time()
    result = execute_template(template, chunks, {})
    duration = time.time() - start
    
    assert duration < 3.0
    assert result is not None
```

---

### User Story 3.2: Custom Template Creation

```gherkin
Feature: Create custom shareable templates

  As a power user
  I want to create my own extraction templates
  So that I can standardize my workflow and share with colleagues

  Scenario: Create template from TUI
    Given I press "Ctrl+T"
    When I select "New Template" option
    Then a template editor should open with fields:
      | Field             | Type       | Required |
      | Name              | Text       | Yes      |
      | Description       | Text       | No       |
      | System Prompt     | Textarea   | Yes      |
      | Template (Jinja2) | Textarea   | Yes      |
      | Output Format     | Dropdown   | Yes      |
    When I fill in:
      | Field          | Value                        |
      | Name           | Extract Pricing Tables       |
      | System Prompt  | You are a financial analyst  |
      | Template       | Find all pricing: {{ context }} |
      | Output Format  | json                         |
    And I save the template
    Then it should appear in "Custom" category
    And be saved to ~/.config/plato/templates/extract_pricing_tables.yaml

  Scenario: Template with validation schema
    Given I am creating a custom template
    When I specify output_format: "json"
    Then I should be able to upload a JSON schema file
    And the schema should be embedded in the template YAML
    And future executions should validate against this schema
```

**Acceptance Criteria:**
- ✅ Template editor validates Jinja2 syntax
- ✅ Variables in template are auto-detected ({{ var_name }})
- ✅ Templates saved to `~/.config/plato/templates/`
- ✅ YAML format for Git-friendliness
- ✅ Optional JSON schema attachment

**Test Cases:**

```python
def test_custom_template_creation():
    """
    GIVEN: User creates a new template
    WHEN: Template is saved
    THEN: File exists in config directory with correct schema
    """
    template_data = {
        "name": "Extract Pricing",
        "description": "Find pricing tables",
        "system_prompt": "You are a financial analyst",
        "template": "Extract pricing from: {{ context }}",
        "output_format": "json"
    }
    
    template_path = create_template(template_data)
    
    assert template_path.exists()
    assert template_path.parent == Path.home() / ".config/plato/templates"
    
    # Verify YAML structure
    loaded = yaml.safe_load(template_path.read_text())
    assert loaded["name"] == "Extract Pricing"
    assert "{{ context }}" in loaded["template"]

def test_jinja2_syntax_validation():
    """
    GIVEN: Template with invalid Jinja2 syntax
    WHEN: User attempts to save
    THEN: Validation error is raised
    """
    invalid_template = {
        "name": "Broken",
        "template": "{{ unclosed_variable }"  # Missing closing brace
    }
    
    with pytest.raises(TemplateSyntaxError):
        validate_template(invalid_template)

def test_template_variable_detection():
    """
    GIVEN: Template with multiple variables
    WHEN: Parsing the template
    THEN: All variables are detected and returned
    """
    template_text = """
    Analyze {{ context }} for {{ focus_areas }}.
    Author: {{ author | default("Unknown") }}
    """
    
    variables = detect_template_variables(template_text)
    
    assert set(variables) == {"context", "focus_areas", "author"}

def test_template_sharing_via_yaml():
    """
    GIVEN: A custom template YAML file
    WHEN: Another user copies it to their config directory
    THEN: Template should work identically
    """
    # Simulate sharing
    original_path = Path("tests/fixtures/shared_template.yaml")
    user_path = Path.home() / ".config/plato/templates/shared_template.yaml"
    
    shutil.copy(original_path, user_path)
    
    # Load and verify
    template = load_template("shared_template.yaml")
    assert template.name is not None
    assert template.template is not None
```

---

## 🎯 Epic 4: Hardware Awareness

### User Story 4.1: Auto-Detect Hardware Profile

```gherkin
Feature: Automatically detect optimal hardware profile

  As a user on varied hardware
  I want PLATO to optimize itself automatically
  So that I don't need to manually configure performance settings

  Scenario: M1 MacBook detected
    Given I launch PLATO on a MacBook Air M1 (8GB RAM)
    When the app initializes
    Then the status bar should show "HW: M1/8GB"
    And the active profile should be "PERFORMANCE"
    And the system should select:
      | Component       | Model                  |
      | Embedding       | embeddinggemma         |
      | Reasoning       | qwen2.5-coder:3b       |
      | Vector DB       | FAISS (in-memory)      |
      | keep_alive      | 30s                    |

  Scenario: i3 laptop detected (HDD)
    Given I launch PLATO on i3 10th Gen with HDD
    When the app initializes
    Then the status bar should show "HW: i3/8GB"
    And the active profile should be "ECO"
    And the system should select:
      | Component       | Model                  |
      | Embedding       | all-minilm             |
      | Reasoning       | smollm2:135m           |
      | Vector DB       | ChromaDB (disk)        |
      | keep_alive      | 0s                     |

  Scenario: Manual profile override
    Given the system auto-detected "ECO" profile
    When I press "F2" to toggle profile
    Then I should see options:
      | Profile     | Description                    |
      | ECO         | Minimal RAM, slow but stable   |
      | BALANCED    | Moderate RAM, good performance |
      | PERFORMANCE | Max RAM, best speed            |
    When I select "BALANCED"
    Then the models should switch accordingly
    And the setting should persist to config file
```

**Acceptance Criteria:**
- ✅ Auto-detection on first launch
- ✅ Profile shown in status bar
- ✅ `F2` key toggles profile selector
- ✅ Settings persist to `~/.config/plato/config.yaml`
- ✅ Graceful fallback if optimal model not available

**Test Cases:**

```python
def test_hardware_detection_m1():
    """
    GIVEN: Running on M1 Mac with 8GB RAM
    WHEN: Auto-detecting hardware
    THEN: PERFORMANCE profile is selected
    """
    with patch('platform.machine', return_value='arm64'), \
         patch('psutil.virtual_memory', return_value=Mock(total=8e9)):
        
        profile = detect_profile()
        
        assert profile == HardwareProfile.PERFORMANCE

def test_hardware_detection_i3_hdd():
    """
    GIVEN: i3 with 8GB RAM and HDD
    WHEN: Auto-detecting hardware
    THEN: ECO profile is selected
    """
    with patch('platform.machine', return_value='x86_64'), \
         patch('psutil.virtual_memory', return_value=Mock(total=8e9)), \
         patch('plato.utils.hardware.check_disk_type', return_value=False):
        
        profile = detect_profile()
        
        assert profile == HardwareProfile.ECO

def test_profile_model_mapping():
    """
    GIVEN: Each hardware profile
    WHEN: Querying recommended models
    THEN: Correct models are returned
    """
    eco_models = get_models_for_profile(HardwareProfile.ECO)
    assert eco_models.embedding == "all-minilm"
    assert eco_models.reasoning == "smollm2:135m"
    
    perf_models = get_models_for_profile(HardwareProfile.PERFORMANCE)
    assert perf_models.reasoning == "qwen2.5-coder:3b"

def test_profile_persistence():
    """
    GIVEN: User changes profile to BALANCED
    WHEN: App is restarted
    THEN: BALANCED profile is loaded from config
    """
    config_path = Path.home() / ".config/plato/config.yaml"
    
    # Set profile
    set_hardware_profile(HardwareProfile.BALANCED)
    
    # Reload config
    loaded_profile = load_config()["hardware"]["profile"]
    
    assert loaded_profile == "balanced"

def test_graceful_degradation_missing_model():
    """
    GIVEN: PERFORMANCE profile selected
    WHEN: qwen2.5-coder:3b is not available
    THEN: System falls back to next-best model
    """
    with patch('plato.ollama.client.list_models', return_value=[
        "lfm2.5-thinking:1.2b",
        "smollm2:1.7b"
    ]):
        models = get_available_models(HardwareProfile.PERFORMANCE)
        
        # Should fallback from qwen2.5-coder:3b
        assert models.reasoning in ["lfm2.5-thinking:1.2b", "smollm2:1.7b"]
```

---

### User Story 4.2: Memory Pressure Monitoring

```gherkin
Feature: Monitor RAM usage and prevent OOM crashes

  As a user on 8GB RAM
  I want PLATO to monitor memory usage
  So that my system doesn't freeze or swap excessively

  Scenario: Normal operation
    Given I am using 2.5 GB / 3.0 GB available RAM
    When I check the status bar
    Then it should show "RAM: 2.5/3.0 GB" in green

  Scenario: High memory warning
    Given I am using 2.8 GB / 3.0 GB available RAM
    When memory usage crosses 90% threshold
    Then the status bar should show "RAM: 2.8/3.0 GB" in yellow
    And a notification should appear: "High memory usage - consider closing apps"

  Scenario: Critical memory - auto-mitigation
    Given I am using 2.95 GB / 3.0 GB available RAM
    When memory usage crosses 98% threshold
    Then the system should:
      | Action                          | Result                      |
      | Show critical warning in red    | "RAM: 2.95/3.0 GB CRITICAL" |
      | Reduce retrieval k from 5 to 3  | Less context loaded         |
      | Unload Ollama models            | Free ~1-2 GB RAM            |
      | Suggest switching to ECO profile| User prompt                 |
    And future operations should use reduced settings until RAM drops below 80%
```

**Acceptance Criteria:**
- ✅ RAM usage updated every 5 seconds
- ✅ Color-coded status (green/yellow/red)
- ✅ Automatic mitigation at 98% usage
- ✅ User notification for manual intervention

**Test Cases:**

```python
def test_memory_monitor_normal_status():
    """
    GIVEN: RAM usage at 70%
    WHEN: Checking memory status
    THEN: Returns green status
    """
    with patch('psutil.virtual_memory', return_value=Mock(
        total=8e9,
        available=2.4e9  # 70% used
    )):
        monitor = MemoryMonitor(threshold_gb=1.0)
        status = monitor.get_status()
        
        assert "2.4" in status or "2.5" in status  # ~2.4 GB used
        assert monitor.check() is True  # Safe to proceed

def test_memory_monitor_high_warning():
    """
    GIVEN: RAM usage at 92%
    WHEN: Checking memory status
    THEN: Warning is triggered
    """
    with patch('psutil.virtual_memory', return_value=Mock(
        total=8e9,
        available=0.64e9  # 92% used
    )):
        monitor = MemoryMonitor(threshold_gb=1.0)
        
        assert monitor.is_warning() is True
        assert monitor.is_critical() is False

def test_memory_monitor_critical_mitigation():
    """
    GIVEN: RAM usage at 98%
    WHEN: Critical threshold crossed
    THEN: Auto-mitigation is triggered
    """
    with patch('psutil.virtual_memory', return_value=Mock(
        total=8e9,
        available=0.16e9  # 98% used
    )):
        monitor = MemoryMonitor(threshold_gb=1.0)
        
        assert monitor.is_critical() is True
        
        # Trigger mitigation
        actions = monitor.get_mitigation_actions()
        
        assert "reduce_k" in actions
        assert "unload_models" in actions
        assert "suggest_eco_profile" in actions

def test_memory_tracking_context_manager():
    """
    GIVEN: Long-running operation
    WHEN: Using memory tracker context manager
    THEN: Peak usage is recorded
    """
    monitor = MemoryMonitor()
    
    with monitor.track():
        # Simulate memory-intensive operation
        large_list = [0] * 10_000_000
        peak_usage = monitor.peak_usage_gb
    
    assert peak_usage > 0
    assert peak_usage < psutil.virtual_memory().total / 1e9
```

---

## 🎯 Epic 5: Output & Export

### User Story 5.1: Copy to Clipboard

```gherkin
Feature: Copy processed context to clipboard

  As a developer
  I want to copy the template output with one keystroke
  So that I can paste it directly into my IDE or LLM chat

  Scenario: Copy JSON output
    Given I have executed a template
    And the output panel shows:
      """
      {
        "summary": "The document describes...",
        "key_points": ["Point 1", "Point 2"]
      }
      """
    When I press "Ctrl+C"
    Then the JSON should be copied to system clipboard
    And a notification should appear: "Copied to clipboard"
    And I can paste it into any application

  Scenario: Copy with metadata
    Given the template output includes chunk references
    When I press "Ctrl+Shift+C"
    Then the clipboard should contain:
      """
      {
        "summary": "...",
        "sources": [
          {"page": 12, "source": "paper.pdf"},
          {"page": 15, "source": "paper.pdf"}
        ]
      }
      """
```

**Acceptance Criteria:**
- ✅ `Ctrl+C` copies current output panel
- ✅ `Ctrl+Shift+C` includes metadata
- ✅ Works on all platforms (macOS/Linux)
- ✅ Visual confirmation (toast notification)

**Test Cases:**

```python
def test_copy_to_clipboard():
    """
    GIVEN: Template output in result panel
    WHEN: User presses Ctrl+C
    THEN: Content is in system clipboard
    """
    output = {"summary": "Test", "key_points": ["A", "B"]}
    
    copy_to_clipboard(json.dumps(output, indent=2))
    
    # Verify clipboard content
    clipboard_content = pyperclip.paste()
    assert json.loads(clipboard_content) == output

def test_copy_with_metadata():
    """
    GIVEN: Output with chunk metadata
    WHEN: User presses Ctrl+Shift+C
    THEN: Sources are included in clipboard
    """
    output_with_meta = {
        "summary": "...",
        "sources": [
            {"page": 12, "source": "paper.pdf"},
            {"page": 15, "source": "paper.pdf"}
        ]
    }
    
    copy_with_metadata(output_with_meta)
    
    clipboard_content = json.loads(pyperclip.paste())
    assert "sources" in clipboard_content
    assert len(clipboard_content["sources"]) == 2
```

---

### User Story 5.2: Export Formats

```gherkin
Feature: Export context in multiple formats

  As a researcher
  I want to export the processed context
  So that I can include it in my notes or documentation

  Scenario: Export as Markdown
    Given I have template output
    When I press "Ctrl+S"
    And I select format "Markdown (.md)"
    Then a save dialog should appear
    When I save to "~/Documents/summary.md"
    Then the file should contain:
      """
      # Executive Summary
      
      ## Key Points
      - Point 1 (Source: paper.pdf, Page 12)
      - Point 2 (Source: paper.pdf, Page 15)
      
      ## Full Context
      [Chunk 1 content...]
      """

  Scenario: Export as JSON with schema
    Given I have JSON output
    When I export as "JSON (.json)"
    Then the file should be valid JSON
    And should include a "$schema" field if template had one
```

**Acceptance Criteria:**
- ✅ Supported formats: JSON, Markdown, Plain Text
- ✅ Markdown includes clickable source references
- ✅ Filename defaults to template name + timestamp
- ✅ File saved to last-used directory (persistent)

**Test Cases:**

```python
def test_export_markdown_with_citations():
    """
    GIVEN: Template output with chunk references
    WHEN: Exporting as Markdown
    THEN: File includes formatted citations
    """
    output_data = {
        "summary": "Test summary",
        "chunks": [
            {"content": "...", "page": 12, "source": "paper.pdf"}
        ]
    }
    
    md_content = export_as_markdown(output_data)
    
    assert "## Key Points" in md_content or "# " in md_content
    assert "(Source: paper.pdf, Page 12)" in md_content

def test_export_json_preserves_schema():
    """
    GIVEN: Template with validation schema
    WHEN: Exporting as JSON
    THEN: Schema reference is included
    """
    output_data = {"summary": "..."}
    schema_uri = "https://example.com/schema.json"
    
    json_content = export_as_json(output_data, schema_uri=schema_uri)
    
    parsed = json.loads(json_content)
    assert "$schema" in parsed
    assert parsed["$schema"] == schema_uri

def test_default_filename_generation():
    """
    GIVEN: Template named "Security Audit"
    WHEN: User exports without specifying filename
    THEN: Default is "security_audit_2025-02-10_14-30.md"
    """
    template_name = "Security Audit"
    
    with freeze_time("2025-02-10 14:30:00"):
        filename = generate_default_filename(template_name, format="md")
    
    assert filename == "security_audit_2025-02-10_14-30.md"
```

---

## 🎯 Epic 6: Error Handling & Edge Cases

### User Story 6.1: Graceful Degradation

```gherkin
Feature: Handle errors without crashing

  As a user
  I want meaningful error messages
  So that I can fix problems myself

  Scenario: Ollama not running
    Given Ollama service is stopped
    When I attempt to execute a template
    Then I should see an error:
      """
      ❌ Ollama Unavailable
      
      PLATO cannot connect to Ollama at http://localhost:11434
      
      Solutions:
      1. Start Ollama: `ollama serve`
      2. Check if Ollama is running: `ps aux | grep ollama`
      3. Verify the host in Settings
      
      [Retry] [Open Settings] [Dismiss]
      """
    And the app should not crash
    And I can still browse files or edit templates

  Scenario: Model not pulled
    Given I select template using "qwen2.5-coder:3b"
    And that model is not pulled
    When I execute the template
    Then I should see:
      """
      ❌ Model Unavailable: qwen2.5-coder:3b
      
      This model is not installed on your system.
      
      Pull it with: `ollama pull qwen2.5-coder:3b`
      
      Estimated size: 2.1 GB
      
      [Auto-pull] [Use fallback model] [Cancel]
      """

  Scenario: Out of disk space
    Given my disk is 98% full
    When I attempt to index a large PDF
    Then I should see:
      """
      ❌ Insufficient Disk Space
      
      Available: 500 MB
      Required: ~1.2 GB (for embeddings cache)
      
      Free up space or change cache location in Settings.
      
      [Open Settings] [Skip caching] [Cancel]
      """
```

**Acceptance Criteria:**
- ✅ All errors show actionable solutions
- ✅ App never crashes (errors are caught)
- ✅ User can retry or choose alternative action
- ✅ Errors logged to `~/.cache/plato/logs/errors.log`

**Test Cases:**

```python
def test_ollama_unavailable_error():
    """
    GIVEN: Ollama is not running
    WHEN: Attempting template execution
    THEN: ModelUnavailableError with helpful message
    """
    with patch('plato.ollama.client.generate', 
               side_effect=requests.ConnectionError):
        
        with pytest.raises(ModelUnavailableError) as exc_info:
            execute_template(template, chunks=[], user_vars={})
        
        error_msg = str(exc_info.value)
        assert "Ollama" in error_msg
        assert "localhost:11434" in error_msg
        assert "ollama serve" in error_msg.lower()

def test_model_not_pulled_suggests_auto_pull():
    """
    GIVEN: Template requires unpulled model
    WHEN: Executing template
    THEN: Error offers to auto-pull
    """
    with patch('plato.ollama.client.generate',
               side_effect=ollama.OllamaError("model not found")):
        
        with pytest.raises(ModelUnavailableError) as exc_info:
            execute_template(template, chunks=[], user_vars={})
        
        assert "ollama pull" in str(exc_info.value)

def test_disk_space_check_before_indexing():
    """
    GIVEN: Only 500 MB free disk space
    WHEN: Attempting to index 200 MB PDF
    THEN: Warning is shown about insufficient space
    """
    with patch('shutil.disk_usage', return_value=Mock(
        free=500 * 1024 * 1024  # 500 MB
    )):
        pdf_size_mb = 200
        
        with pytest.raises(PLATOError) as exc_info:
            check_disk_space_for_indexing(pdf_size_mb)
        
        assert "disk space" in str(exc_info.value).lower()
        assert "500 MB" in str(exc_info.value)
```

---

### User Story 6.2: Corrupted/Unusual PDFs

```gherkin
Feature: Handle problematic PDF files

  As a user
  I want PLATO to handle edge cases in PDFs
  So that one bad file doesn't block my workflow

  Scenario: Password-protected PDF
    Given I select "encrypted_doc.pdf"
    And it requires a password
    When PLATO attempts to open it
    Then I should see:
      """
      🔒 Password Required
      
      This PDF is encrypted. Enter password:
      [__________]
      
      [Unlock] [Skip] [Cancel]
      """
    When I enter the correct password
    Then the PDF should be processed normally

  Scenario: Corrupted PDF
    Given I select "corrupted.pdf"
    And the file is malformed
    When PLATO attempts to parse it
    Then I should see:
      """
      ❌ Unable to Parse PDF
      
      File: corrupted.pdf
      Issue: Invalid PDF structure (possibly corrupted)
      
      Try:
      1. Re-download the file
      2. Open in Adobe Reader and re-save
      3. Use an online PDF repair tool
      
      [Skip] [Retry] [Remove from queue]
      """
    And the file should be skipped in batch mode

  Scenario: Image-only PDF (no text layer)
    Given I select "scanned_old_book.pdf"
    And it contains only images, no text
    When PLATO processes it
    Then it should auto-invoke OCR
    And show progress: "OCR: Page 5/120"
    And allow cancellation with "Esc"
```

**Acceptance Criteria:**
- ✅ Password prompt for encrypted PDFs
- ✅ Graceful skip of corrupted files
- ✅ Auto-detection of image-only PDFs
- ✅ Cancellable long-running OCR
- ✅ Batch mode continues after failures

**Test Cases:**

```python
def test_password_protected_pdf():
    """
    GIVEN: Encrypted PDF file
    WHEN: Attempting to open
    THEN: Password prompt is shown
    """
    pdf_path = Path("tests/fixtures/encrypted.pdf")
    
    with pytest.raises(PDFProcessingError) as exc_info:
        ingest_pdf(pdf_path, password=None)
    
    assert "password" in str(exc_info.value).lower()
    
    # Now with password
    document = ingest_pdf(pdf_path, password="secret123")
    assert document is not None

def test_corrupted_pdf_handling():
    """
    GIVEN: Malformed PDF
    WHEN: Attempting to parse
    THEN: PDFProcessingError with specific message
    """
    pdf_path = Path("tests/fixtures/corrupted.pdf")
    
    with pytest.raises(PDFProcessingError) as exc_info:
        ingest_pdf(pdf_path)
    
    error_msg = str(exc_info.value)
    assert "corrupted" in error_msg.lower() or "invalid" in error_msg.lower()
    assert pdf_path.name in error_msg

def test_batch_continues_after_error():
    """
    GIVEN: Batch with 1 corrupted and 2 valid PDFs
    WHEN: Processing batch
    THEN: Valid PDFs are indexed, corrupted is skipped
    """
    pdf_paths = [
        Path("tests/fixtures/valid1.pdf"),
        Path("tests/fixtures/corrupted.pdf"),
        Path("tests/fixtures/valid2.pdf"),
    ]
    
    result = batch_ingest(pdf_paths, skip_errors=True)
    
    assert len(result.successful) == 2
    assert len(result.failed) == 1
    assert result.failed[0].path.name == "corrupted.pdf"

def test_ocr_cancellation():
    """
    GIVEN: Long-running OCR operation
    WHEN: User presses Esc
    THEN: Operation is cancelled gracefully
    """
    pdf_path = Path("tests/fixtures/scanned_100pages.pdf")
    
    cancel_event = threading.Event()
    
    # Start OCR in background
    future = executor.submit(
        ingest_pdf, 
        pdf_path, 
        cancel_event=cancel_event
    )
    
    # Simulate user cancellation after 2 seconds
    time.sleep(2)
    cancel_event.set()
    
    with pytest.raises(CancellationError):
        future.result(timeout=5)
```

---

## 🎯 Epic 7: Performance & Benchmarking

### User Story 7.1: Built-in Benchmark

```gherkin
Feature: Run performance benchmarks

  As a developer
  I want to benchmark PLATO on my hardware
  So that I can verify it meets performance targets

  Scenario: Run full benchmark suite
    Given I launch PLATO with `plato benchmark`
    Then the system should:
      | Test                         | Target (M1)  | Target (i3) |
      | PDF ingestion (100 pages)    | < 5s         | < 10s       |
      | Chunk embedding (1000)       | < 15s        | < 30s       |
      | Vector search (10K index)    | < 200ms      | < 500ms     |
      | Template execution           | < 3s         | < 8s        |
    And generate a report showing:
      """
      PLATO Benchmark Report
      =====================
      Hardware: MacBook Air M1 (8GB)
      Date: 2025-02-10 14:30:00
      
      PDF Ingestion:     ✓ 4.2s (target: <5s)
      Embedding:         ✓ 12.1s (target: <15s)
      Vector Search:     ✓ 180ms (target: <200ms)
      Template Exec:     ✓ 2.7s (target: <3s)
      
      Memory:
        Peak:     2.9 GB
        Average:  2.4 GB
      
      Status: ALL TESTS PASSED
      """
    And save the report to `~/.cache/plato/benchmarks/`
```

**Acceptance Criteria:**
- ✅ `plato benchmark` command
- ✅ Tests against predefined targets
- ✅ Generates timestamped report
- ✅ Includes hardware profile info
- ✅ Pass/fail indicators for each test

**Test Cases:**

```python
def test_benchmark_command_exists():
    """
    GIVEN: PLATO CLI
    WHEN: Running `plato benchmark`
    THEN: Benchmark suite executes
    """
    result = subprocess.run(
        ["plato", "benchmark", "--quick"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0
    assert "Benchmark Report" in result.stdout

def test_benchmark_pdf_ingestion_m1():
    """
    GIVEN: M1 hardware profile
    WHEN: Benchmarking PDF ingestion
    THEN: Completes in under 5 seconds
    """
    pdf_path = Path("tests/fixtures/benchmark_100page.pdf")
    
    start = time.time()
    document = ingest_pdf(pdf_path)
    duration = time.time() - start
    
    assert duration < 5.0, f"PDF ingestion took {duration:.2f}s (target: <5s)"

def test_benchmark_report_generation():
    """
    GIVEN: Completed benchmark run
    WHEN: Generating report
    THEN: Report includes all metrics
    """
    results = {
        "pdf_ingestion": 4.2,
        "embedding": 12.1,
        "vector_search": 0.18,
        "template_exec": 2.7,
        "peak_memory_gb": 2.9
    }
    
    report = generate_benchmark_report(results, profile="M1")
    
    assert "4.2s" in report
    assert "MacBook Air M1" in report or "M1" in report
    assert "PASSED" in report or "✓" in report
```

---

## 📊 Test Pyramid Summary

### Unit Tests (Fast, Isolated)
- PDF parsing logic
- Chunking algorithms
- Template rendering
- Embedding caching
- Metadata extraction

**Target:** 200+ unit tests, >90% coverage

### Integration Tests (Medium Speed)
- Ollama API interaction
- Vector DB operations
- Template execution pipeline
- File I/O operations

**Target:** 50+ integration tests

### End-to-End Tests (Slow, Full Workflow)
- TUI navigation
- Complete PDF-to-output pipeline
- Error recovery scenarios
- Cross-platform compatibility

**Target:** 20+ E2E tests

### Performance Tests (Benchmarks)
- Latency targets per hardware profile
- Memory usage limits
- Concurrent operations

**Target:** 10+ performance tests

---

## 🚀 Implementation Priority

### Phase 1: Core (MVP)
1. ✅ PDF ingestion (Story 1.1, 1.2)
2. ✅ Vector indexing (Story 2.1)
3. ✅ Template execution (Story 3.1)
4. ✅ Basic TUI (file browser + output panel)

### Phase 2: Usability
5. ✅ Hardware profiles (Story 4.1)
6. ✅ Clipboard/export (Story 5.1, 5.2)
7. ✅ Custom templates (Story 3.2)

### Phase 3: Polish
8. ✅ Error handling (Story 6.1, 6.2)
9. ✅ Memory monitoring (Story 4.2)
10. ✅ Benchmarking (Story 7.1)

---

## 🧪 Test Execution Commands

```bash
# Run all tests
pytest tests/ -v

# Run specific epic
pytest tests/test_pdf_ingestion.py -v

# Run with coverage
pytest tests/ --cov=plato --cov-report=html

# Run only fast tests (skip E2E)
pytest tests/ -m "not slow"

# Run performance tests
pytest tests/test_performance.py -v --benchmark

# Run in TDD watch mode
pytest-watch tests/
```

---

**Status:** READY FOR TDD IMPLEMENTATION  
**Total User Stories:** 14  
**Total Test Scenarios:** 40+  
**Estimated Test Cases:** 100+