# Plato Project Backlog

## 🔴 Epic: Critical Fixes & Standardization
**Status**: To Do
**Priority**: Critical
**Description**: Addressing the critical architectural and configuration issues identified in the technical review.

### [CRIT-01] Standardize Model Naming & Configuration
**Description**: Fix inconsistent model references across documentation and code.
**Implementation Details**:
- Create `src/plato/core/models.yaml` as the source of truth.
- Update `config.py` to load standard models:
    - `embeddinggemma:latest` (Embeddings)
    - `deepseek-ocr:3b` (OCR)
    - `lfm2.5-thinking:1.2b` (Reasoning - Eco/Balanced)
    - `qwen2.5-coder:3b` (Reasoning - Performance)
- Validate `ollama pull` commands in `README.md`.

### [CRIT-02] Recalculate & Enforce RAM Budgets
**Description**: Fix flawed RAM budget math. M1/8GB needs a 1.5B model to avoid OOM.
**Implementation Details**:
- Update `HardwareProfile` in `src/plato/utils/hardware.py`:
    - **M1 (Performance)**: Switch LLM to `qwen2.5-coder:1.5b` (~1.2GB) or `deepseek-coder:1.3b` to stay within 3GB budget.
    - **Allocations**: OS(2.5GB) + Browser/IDE(2.5GB) + PLATO(3.0GB) = 8GB.
    - **Within PLATO**: LLM(1.2GB) + Embedder(0.2GB) + Vector(0.6GB) + App(0.3GB) + Buffer(0.7GB) = 3.0GB.
- Implement strict sequential execution: unload embedder before loading LLM.

### [CRIT-03] Vector DB Strategy & Benchmarking
**Description**: Chroma vs FAISS decision needs empirical backing.
**Implementation Details**:
- Create benchmark script `tests/benchmark_db.py`.
- Compare insert/query latency and RAM usage for 10k chunks.
- Update `src/plato/storage/vector_db.py` to support user-configurable backend.
- **Decision**: Default to ChromaDB for stability, make FAISS opt-in.

### [CRIT-04] Realistic Performance Targets
**Description**: Existing targets are guesses. We need ranges based on complexity.
**Implementation Details**:
- Update `desc.md` (or new `specs.md`) with revised targets from IDEA.md.
- Add `plato benchmark` command to run standard tests and report against these targets.

---

## 🟡 Epic: Security & Robustness
**Status**: To Do
**Priority**: High

### [SEC-01] Sandbox Template Execution
**Description**: Prevent Jinja2 injection attacks.
**Implementation Details**:
- In `src/plato/core/template.py`:
    - Switch to `jinja2.sandbox.SandboxedEnvironment`.
    - Implement `safe_render()` function that only accepts whitelisted context variables (`context`, `user_query`, etc.).

### [SEC-02] JSON Schema Validation
**Description**: Templates claim to validate JSON but no schemas exist.
**Implementation Details**:
- Create `src/plato/templates/schemas/`.
- Add schemas for `security_audit`, `executive_summary`.
- Add `jsonschema` dependency.
- formatting: Update `TemplateEngine` to validate output against schema if defined in template.

### [ROB-01] Robust Hardware Detection
**Description**: Current detection fails in VMs/Docker.
**Implementation Details**:
- Update `src/plato/utils/hardware.py`.
- Check for `/.dockerenv` or `/run/.containerenv`.
- Default to `BALANCED` profile if virtualization detected.

### [ROB-02] Memory Monitoring & Safety
**Description**: 1GB buffer is too aggressive.
**Implementation Details**:
- Update `MemoryMonitor` in `src/plato/utils/memory.py`:
    - Warning threshold: 1.5GB.
    - Critical threshold: 0.5GB.
- Add hook to `Pipeline` to pause/fail gracefully if critical threshold hit.

---

## 🟠 Epic: Reliability & Error Recovery
**Status**: To Do
**Priority**: Critical
**Description**: Addressing the missing error handling strategy to ensure the app doesn't crash on edge cases.

### [ERR-01] Global Error Handling Strategy
**Description**: Implement graceful degradation for common failure points.
**Implementation Details**:
- Create `plato/core/errors.py` with custom exceptions.
- Wrap main TUI loop and processing pipeline in try/except.
- Handle:
    - Ollama disconnect/crash.
    - Missing models (trigger auto-download or guide user).
    - Corrupted PDFs (PyMuPDF exceptions).
    - Disk full (IOError).
- Output: Show non-blocking notifications in the TUI instead of crashing.

### [CONC-01] Concurrency & Multi-Instance Safety
**Description**: Handle multiple instances of PLATO and CPU/RAM contention.
**Implementation Details**:
- Implement file-based locking for ChromaDB and Cache directories.
- Check if Ollama is already busy before starting generation.
- Add `MAX_CONRSURRENT_PROCESSES` setting (default to 1 for Edge profiles).

---

## 🟡 Epic: Validation & Compliance
**Status**: To Do
**Priority**: High

### [MOD-01] Model Availability Verification
**Description**: Ensure models mentioned in spec actually exist in Ollama registry.
**Implementation Details**:
- Audit `models.yaml` against `ollama library`.
- Change `deepseek-ocr:3b` to a verified tag if it's currently a placeholder.
- Provide fallback paths for when primary models are unavailable.

### [I18N-01] Internationalization & Script Support
**Description**: Basic support for non-Latin scripts and RTL languages.
**Implementation Details**:
- Test with Chinese/Arabic/Japanese PDFs.
- Ensure UTF-8 normalization for all text extraction and prompt injection.
- Document limitations of 1.2B models for non-English reasoning.

---

## 🔵 Epic: Functional Enhancements
**Status**: To Do
**Priority**: Medium

### [FUNC-01] Multi-Column PDF Chunking
**Description**: Standard chunking fails on complex layouts.
**Implementation Details**:
- In `src/plato/core/pdf.py`:
    - Integrate `marker-pdf` (or heuristic detection) for multi-column files.
    - Fallback to `pymupdf` for simple layouts.
- Update `ingest_pdf` to route based on layout complexity.

### [FUNC-02] Embedding Cache Invalidation
**Description**: "Re-embed if changed" is too vague.
**Implementation Details**:
- Implement `get_cache_key()` using:
    - Content Hash (SHA256 of text).
    - Chunking Config Version.
    - Embedding Model Version.
- Store embeddings in path derived from this key.

### [FUNC-03] TUI Usability Fixes
**Description**: Fix keyboard shortcuts and theming.
**Implementation Details**:
- Update `src/plato/tui/app.py`:
    - Add `Ctrl+Shift+T` for theme toggle.
    - Bind `Ctrl+S` with `F7` fallback for Save.
    - Bind `Ctrl+Q` with `F10` fallback for Quit.
- Add `RotatingFileHandler` for logs.

---

## ⚪ Epic: Future & Polish
**Status**: Backlog
**Priority**: Low

### [FUT-01] Template Dry-Run
**Description**: Preview prompt before execution.
**Implementation Details**:
- Add `--dry-run` flag to template command.
- Render prompt and show token cost/time estimate.

### [FUT-02] Template Marketplace Schema
**Description**: Define schema for sharing templates.
**Implementation Details**:
- Create `marketplace_schema.yaml`.

### [FUT-03] PDF Annotation Export
**Description**: Highlight findings back into the PDF.
**Implementation Details**:
- Use `pymupdf` to draw highlights on coordinates of relevant chunks.
