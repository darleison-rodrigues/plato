# Plato Project Backlog

## 🟢 Epic: TUI & UX Overhaul
**Status**: In Progress
**Priority**: Critical

### [TUI-01] Implement Robust Startup & Setup Flow
**Description**: The current application crashes if Ollama is not running. We need a "pre-flight" check system.
**Implementation Details**:
- Create `plato/setup.py` with `check_system()` function.
- Check 1: `ollama` process is running (via HTTP ping).
- Check 2: Hardware profile detection (verify `config.active_profile`).
- Check 3: Storage directory permissions.
- In `main.py`, run this check on startup. If failed, encourage `plato setup` or show helpful error.
- Add `plato setup` command to download missing models (`ollama pull`) based on the active profile.

### [TUI-02] Enhanced `process` Command with Granular Control
**Description**: Users need more control over what to process, not just "the whole directory".
**Implementation Details**:
- Update `main.py`: `process(paths: List[Path], ...)`
- Add `--dry-run`: Show what would happen (file list, sizes) without processing.
- Add `--model`: Allow overriding the reasoning model for this run.
- Add `--skip-existing`: Check if `doc_id` exists in storage before processing.
- Implementation: iterate provided paths, expand globs, filter by existence if requested, then run `pipeline.process_pdf_async`.

### [TUI-03] Knowledge Graph Visualization & Export
**Description**: Users cannot see the graph they built. We need export capabilities.
**Implementation Details**:
- **Visualization**:
    - Implement `visualize.py` with `GraphVisualizer` (Mermaid/GraphViz generators).
    - Add `plato visualize --format mermaid|graphviz|json` command.
    - Output interactive HTML for Mermaid.
- **Export**:
    - Add `plato export --format markdown|json` command.
    - Markdown export should include specific sections: Summary, Entities (by type), Relations, and an embedded Mermaid diagram.
    - JSON export should dump the raw node/edge list.

### [TUI-04] Configuration Management CLI
**Description**: Users currently have to edit python code to change settings.
**Implementation Details**:
- Add `plato config` command group.
- `plato config show`: Print current config (hardware profile, models, directories) using `rich.console`.
- `plato config set <key> <value>`: Update `config.yaml` (need to implement persistent config saving in `ConfigManager`).
- `plato config models`: List available Ollama models `ollama.list()` and highlight current selections.

### [TUI-05] Enhanced Chat with History & Citations
**Description**: Chat is currently stateless and doesn't cite sources.
**Implementation Details**:
- Implement `ChatSession` class to manually manage list of messages.
- Update `plato chat`:
    - Loop input/output.
    - Append user query and system response to history.
    - If `GraphRAG` returns source nodes, formatting them as citations: `[DocName, p.3]`.
    - Allow `/reset` command to clear history.

---

## 🟡 Epic: Graph Quality & Intelligence
**Status**: To Do
**Priority**: High

### [KG-01] Entity Resolution & Deduplication
**Description**: The graph currently contains duplicates (e.g., "AI" vs "Artificial Intelligence").
**Implementation Details**:
- Create `class EntityResolver` in `graph_rag.py`.
- **Normalization**: Lowercase, strip punctuation/symbols.
- **Alias Tracking**: Dict mapping `normalized_alias -> canonical_name`.
- **Insertion**: Modify `insert_triplets` to pass all entities through `resolver.resolve(entity)`.
- **Storage**: Persist the alias map to `storage/aliases.json` so resolution is consistent across runs.

### [KG-02] Few-Shot Extraction Prompting
**Description**: Extraction quality is variable.
**Implementation Details**:
- Update `llm.py` extraction prompts.
- Add `EXAMPLES` section to the system prompt.
- Provide 2-3 high-quality examples of text -> JSON output.
- Explicitly instruct on entity types (PERSON, ORG, CONCEPT) and relation constraints.

### [KG-03] Hallucination Check / Verification
**Description**: GraphRAG can sometimes make up relations.
**Implementation Details**:
- (Optional/Later) Implement a "Verify" step where a second model call checks if the extracted relation is supported by the chunk text.
- For now, maybe just lower `temperature` for extraction tasks (already set to 0.1/0.2 in edge profiles).

---

## 🔵 Epic: GraphRAG Parity (Advanced Features)
**Status**: Backlog
**Priority**: Medium

### [RAG-01] Hierarchical Summarization
**Description**: Replicate Microsoft GraphRAG's ability to summarize at different levels.
**Implementation Details**:
- **Chunk Summary**: We already have this.
- **Document Summary**: Aggregate chunk summaries -> LLM -> Document Summary.
- **Global Summary**: Aggregate all Document summaries -> LLM -> Corpus Summary.
- Store these in a way that `plato export` can include them.

### [RAG-02] Hybrid Query Routing
**Description**: Intelligent switching between local (fact) and global (theme) search.
**Implementation Details**:
- Implement `Classifier` in `llm.py`: `classify_query(query) -> "specific" | "thematic"`.
- If "specific": Use existing `GraphRAG` vector/keyword search.
- If "thematic": Use Map-Reduce over Document Summaries.
- Update `plato chat` to use this routing logic.

---

## 🟣 Epic: Core Architecture & Fixes
**Status**: Immediate
**Priority**: Critical

### [CORE-01] Fix OllamaConfig Compatibility
**Description**: `llm.py` and `graph_rag.py` are accessing deprecated `models_by_task` and `get_model_for_task`.
**Implementation Details**:
- Refactor `src/plato/llm.py`:
    - Replace `self.config.models_by_task.get(...)` with `self.config.get_model_name(...)`.
    - Replace `get_model_for_task(...)` with `get_model_name(...)`.
- Verify `src/plato/graph_rag.py` uses `get_model_name`.
- Ensure all calls match the new `OllamaConfig` schema in `config.py`.

### [CORE-03] Core DX Improvements
**Description**: Enhance developer experience based on critical feedback (lazy init, concurrency control, error handling).
**Implementation Details**:
- **Lazy Initialization**: Delay `OllamaClient` and `GraphRAG` loading in `Pipeline.__init__` until first use.
- **Configurable Concurrency**: Allow `process` command to set `MAX_CONCURRENT_DOCS` (default 2 for edge).
- **Documentation**: Add clear comments explaining async/executor offloading for CPU-bound tasks.
- **Error Handling**: Catch specific exceptions (PDF parsing vs LLM vs Graph) for better user feedback.

---

## ⚪ Epic: Edge Optimization (Validation)
**Status**: Done (Verification Pending)

### [EDGE-01] Verify Memory Usage on M1
**Description**: Ensure the new `EdgeModelManager` effectively prevents OOM.
**Implementation Details**:
- Run `plato process` on a folder with 10+ PDFs.
- Monitor RAM usage.
- Confirm `model_manager` lock is working (logs should show sequential model access if conflicts occur).
