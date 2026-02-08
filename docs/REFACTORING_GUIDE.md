# OllamaClient Refactoring Guide

## Philosophy: Implementing the Contexter Pattern for PLATO

The refactored client shifts from a generic "LLM caller" to a **resource-aware, document-tracking knowledge extractor** aligned with PLATO's vision.

---

## Key Changes

### 1. **DocumentContext: The Core Innovation** 🦫

**Before**: Text was a string. No tracking of source.
```python
extract_entities("some text")  # Where did this come from? Who knows.
```

**After**: Text carries provenance metadata.
```python
@dataclass
class DocumentContext:
    doc_id: str
    source_path: str
    doc_name: str
    chunk_index: int          # Which part of the document?
    total_chunks: int         # Total parts
    text: str
    metadata: Dict[str, Any]  # Custom tracking
```

**Why**: For building knowledge graphs, you *must* know what came from where. Without this, extracted relations are orphaned facts.

**Example flow**:
```
PDF: "research.pdf" (1200 pages)
  → Chunk 1/120 (intro)
    → Extract: Person="Dr. Smith", Org="MIT"
    → Now you know: This relation came from page 10 of research.pdf
```

---

### 2. **ExtractionResult: Full Provenance Tracking**

**Before**: Scattered returns
```python
entities = extract_entities(text)      # Just a dict
relations = extract_relations(text)    # Separate dict
summary = summarize(text)              # String somewhere
```

**After**: Unified result with source tracking
```python
result = ExtractionResult(
    doc_context=chunk,        # Know exactly where this came from
    entities={...},
    relations=[...],
    summary="...",
    embeddings=[...],
    extraction_metadata={...}  # When, how, which model
)

# Convert to JSON for knowledge graph
result.to_dict()  # Full lineage for downstream processing
```

**Why**: TUI needs to show "Found in: research.pdf, chunk 15/120" not just "Found: Smith."

---

### 3. **SmartJSONExtractor: Robust LLM Output Parsing**

**Before**: Fragile regex, admitted it "might not catch edge cases"

**After**: Multi-strategy extraction with fallbacks
```
Strategy 1: Markdown code blocks (```json ... ```)
    ↓ (if fails)
Strategy 2: Raw JSON boundary detection (bracket counting)
    ↓ (if fails)
Strategy 3: Repair common errors (trailing commas, unquoted keys)
    ↓ (if still fails)
→ Log warning, return None gracefully
```

**Why**: 1B models are quirky. They might output:
```
Here's the JSON:
```json
{"entities": {"PERSON": ["Smith",]}}  // ← trailing comma!
```
```

The repaired extractor handles this.

---

### 4. **TextChunker: Semantic-Aware Splitting**

**Before**: Crude truncation
```python
text[:self.max_text_length]  # Cuts mid-sentence 💀
```

**After**: Paragraph-aware chunking
```python
# Input: 10,000-word PDF
# Output: 5 chunks of ~2,000 words each
chunks = text_chunker.chunk(text, doc_id="research.pdf")

# Each chunk knows:
# - doc_id: "research.pdf"
# - chunk_index: 0/4
# - text: full paragraphs (no sentence breaks)
```

**Why**: Entity/relation extraction depends on semantic context. Breaking a sentence kills meaning.

---

### 5. **Async/Streaming: TUI Responsiveness** ⚡

**Before**: Blocking calls
```python
entities = client.extract_entities(text)  # Waits... waits... waits...
# Nothing shows to user until done
```

**After**: Streaming extraction
```python
# TUI shows results progressively
async for entity_type, entity_list in client.extract_entities_streaming(doc):
    print(f"Found {entity_type}: {entity_list}")
    # Shows immediately as extraction completes
```

**Parallel tasks**:
```python
# Extract entities, relations, summary in parallel, not sequential
entities, relations, summary = await asyncio.gather(
    extract_entities(chunk),
    extract_relations(chunk),
    summarize(chunk)
)
```

**Why**: User dumps 100 PDFs. Instead of "Processing... 5 hours", they see:
```
PDF 1: Found 23 entities ✓
PDF 2: Building knowledge graph... 12/50 relations ✓
PDF 3: Summarizing... ✓
```

---

### 6. **OllamaClientConfig: Configuration Validation**

**Before**: Silent failures
```python
self.config = get_config().ollama  # What if it doesn't exist?
prompt = self.prompts.get('entity_extraction', "")  # Empty default
```

**After**: Validated config
```python
class OllamaClientConfig:
    def _validate(self):
        if not self.model:
            raise ValueError("Ollama model not configured")
        if self.max_text_length < 500:
            raise ValueError("max_text_length must be at least 500")

# Fails loudly at startup, not during extraction
```

**Why**: Better to crash on `__init__` than silently return empty results.

---

### 7. **Contexter Pattern: Resource Management**

**Before**: No cleanup
```python
text = read_pdf()
# RAM stays allocated
# File handle stays open
```

**After**: Context manager pattern
```python
async with client.process_document(doc_path, doc_id) as text:
    # Extract from text
    pass
# Automatically cleaned up (file closed, memory freed)
```

**Why**: README promises a "polite librarian" that opens/closes efficiently. This implements it.

---

### 8. **Task-Aware Model Selection**

**Before**: Single model for everything
```python
response = ollama.chat(model="lfm2.5-thinking", ...)
# Used for entity extraction, relations, summarization
```

**After**: Model per task
```python
models_by_task = {
    'entity_extraction': 'lfm2.5-thinking',
    'relation_extraction': 'lfm2.5-thinking',
    'summarization': 'lfm2.5'  # Could be different
}

model = config.get_model_for_task('summarization')
```

**Why**: Future flexibility. Some tasks might need different models.

---

## Usage Example: Full Pipeline

### Old Way
```python
from ollama_client import OllamaClient

client = OllamaClient()
text = open("research.pdf").read()
entities = client.extract_entities(text)
relations = client.extract_relations(text)
summary = client.summarize(text)

# Result: No idea where data came from
# Can't build knowledge graph
# User sees frozen UI while processing
```

### New Way
```python
import asyncio
from ollama_client import OllamaClient

client = OllamaClient()

async def process():
    results = await client.process_document_complete(
        doc_path="/pdfs/research.pdf",
        doc_id="arxiv-2024-001"
    )
    
    for result in results:
        # Each result knows its source
        print(f"Chunk {result.doc_context.chunk_index}:")
        print(f"  Entities: {result.entities}")
        print(f"  Relations: {result.relations}")
        print(f"  Source: {result.doc_context.source_path}")
        
        # Convert to JSON for knowledge graph
        json_data = result.to_dict()
        knowledge_graph.add(json_data)

asyncio.run(process())
```

### TUI Integration
```python
# In your TUI (using Rich, etc.)
async def show_processing():
    async for entity_type, entities in client.extract_entities_streaming(chunk):
        # Update UI in real-time
        table.add_row(entity_type, str(len(entities)))
        refresh_display()
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────┐
│           PLATO TUI (Mac)                   │
│      plato chat --instance=localhost        │
└────────────────────┬────────────────────────┘
                     │ (HTTP or local)
┌────────────────────▼────────────────────────┐
│      OllamaClient (Refactored)              │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │ process_document_complete()          │   │
│  │  → TextChunker                       │   │
│  │  → DocumentContext (each chunk)      │   │
│  │  → Async extraction (parallel)       │   │
│  │  → ExtractionResult (with provenance)│   │
│  └──────────────────────────────────────┘   │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │ extract_entities_streaming()         │   │
│  │ → SmartJSONExtractor                 │   │
│  │ → Yield results progressively        │   │
│  └──────────────────────────────────────┘   │
└──────────┬────────────────────────────────┬──┘
           │ (local files)                  │ (Ollama API)
           │                                │
    ┌──────▼──────┐              ┌──────────▼───┐
    │  PDFs       │              │  Ollama      │
    │  (on Linux) │              │  (inference) │
    └─────────────┘              └──────────────┘
```

---

## Configuration Example

```yaml
# config.yaml
ollama:
  model: "lfm2.5-thinking"
  models_by_task:
    entity_extraction: "lfm2.5-thinking"
    relation_extraction: "lfm2.5-thinking"
    summarization: "lfm2.5"
  max_retries: 3
  timeout: 60
  max_text_length: 4000
  batch_size: 5
  enable_streaming: true

prompts:
  entity_extraction: |
    Extract all named entities from this text.
    Return JSON: {{"entities": {{"PERSON": [...], "ORG": [...], ...}}}}
    
    Text: {text}
  
  relation_extraction: |
    Extract relationships between entities.
    Return JSON array: [{{"subject": "...", "relation": "...", "object": "..."}}]
    
    Text: {text}
  
  summarization: |
    Summarize this text concisely.
    
    Text: {text}

chroma:
  embedding_model: "nomic-embed-text"
```

---

## Migration Path

### Phase 1: Drop-in Replacement
```python
# Existing code still works
old_client = OllamaClient()
entities = old_client.extract_entities_batch(texts)
```

### Phase 2: Adopt New Patterns
```python
# Use async for performance
results = await new_client.process_document_complete(doc_path, doc_id)
```

### Phase 3: Full TUI Integration
```python
# Stream results to UI
async for result in client.extract_entities_streaming(chunk):
    ui.update(result)
```

---

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Provenance** | None | Full (doc_id, chunk, source) |
| **Streaming** | Blocking | Async/progressive |
| **JSON Parsing** | Fragile | Multi-strategy robust |
| **Resource Mgmt** | Leaked | Contexter pattern |
| **Error Handling** | Silent fails | Validated config, early errors |
| **Model Flexibility** | Single model | Task-aware selection |
| **Text Splitting** | Crude truncation | Semantic chunks |
| **UI Feedback** | None | Real-time updates |

---

## Next Steps for PLATO

1. **Integrate into TUI**: Wire streaming extraction to Rich progress bars
2. **Knowledge Graph Builder**: Consume `ExtractionResult` → add to graph DB
3. **Workflow Suggestion**: Analyze extracted entities → recommend next steps ("Build comparison table", etc.)
4. **Multi-file Processing**: Batch documents with progress tracking
5. **Context Export**: Convert knowledge graph → formatted context.md for Claude/ChatGPT

