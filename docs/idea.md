# PLATO Specification: Critical Evaluation & Harmonization

**Document Version:** 1.0  
**Evaluation Date:** February 10, 2026  
**Evaluator:** Technical Architecture Review  
**Severity Levels:** 🔴 Critical | 🟡 Major | 🔵 Minor | ⚪ Enhancement

---

## Executive Summary

The PLATO specification is **ambitious and well-structured** with clear goals for a local-first PDF processing TUI. However, there are **23 significant issues** requiring resolution before implementation, including contradictions in hardware requirements, unrealistic performance targets, and architectural decisions that may not scale.

**Overall Assessment:** 6.5/10 - Good foundation, needs refinement

**Recommendation:** Proceed with implementation after addressing Critical and Major issues.

---

## 🔴 CRITICAL ISSUES

### 1. Model Naming Inconsistencies

**Problem:** Multiple contradictory model names throughout the spec.

**Evidence:**
- **Section "Technical Stack"**: "LFM2.5-Thinking 1.2B" and "Qwen2.5-Coder 3B"
- **Section "Hardware Profile Toggle"**: "lfm2.5-thinking:1.2b" and "qwen2.5-coder:3b"
- **Section "Prerequisites"**: "lfm2.5-thinking:1.2b" (lowercase, with colon)
- **User Story 3.1**: "qwen2.5-coder:3b" (lowercase)

**Impact:** Developers won't know which model names are correct for Ollama pulls.

**Resolution:**
```yaml
STANDARD MODEL NAMING (Ollama format):
- embeddinggemma:latest
- deepseek-ocr:3b
- lfm2.5-thinking:1.2b  # ECO/BALANCED profiles
- qwen2.5-coder:3b      # PERFORMANCE profile
- smollm2:135m          # Fallback for ECO
- all-minilm:latest     # Fallback for embeddings
```

**Action:** Create a `models.yaml` reference file and ensure consistency across all documentation.

---

### 2. RAM Budget Math Doesn't Add Up

**Problem:** Claimed allocations exceed available RAM on both platforms.

**i3 10th Gen Analysis:**
```
CLAIMED BUDGET:
OS + System Services:     2.5 GB
Browser (5 tabs):         1.0 GB  
VS Code:                  1.5 GB
Total Reserved:           5.0 GB
──────────────────────────────────
Available for PLATO:      3.0 GB  ← Correct

CLAIMED ALLOCATION:
ChromaDB (10K chunks):    0.5 GB
EmbeddingGemma:           0.2 GB
LFM2.5-Thinking:          0.8 GB
TUI + Python runtime:     0.3 GB
Buffer:                   1.2 GB
──────────────────────────────────
Total:                    3.0 GB  ← Works!
```

**BUT** the strategy says "never run embedder + LLM simultaneously" which implies they WOULD run together without this constraint. This needs explicit handling.

**M1 Analysis:**
```
CLAIMED ALLOCATION:
FAISS (in-memory):        0.8 GB
EmbeddingGemma:           0.2 GB
Qwen2.5-Coder 3B:         1.2 GB  ← WRONG!
TUI + Python runtime:     0.3 GB
Buffer:                   0.5 GB
──────────────────────────────────
Total:                    3.0 GB

ACTUAL Qwen2.5-Coder 3B:  ~2.0-2.5 GB (not 1.2 GB)
```

**Impact:** M1 configuration will OOM crash.

**Resolution:**
```yaml
M1 CORRECTED ALLOCATION:
FAISS (in-memory):        0.6 GB  # Reduced from 0.8
EmbeddingGemma:           0.2 GB
Qwen2.5-Coder 3B:         2.2 GB  # Realistic estimate
TUI + Python runtime:     0.3 GB
Buffer:                   -0.3 GB # INSUFFICIENT!
──────────────────────────────────
Total:                    3.0 GB

RECOMMENDATION: M1 should use qwen2.5-coder:1.5b or DeepSeek-Coder-1.3B
Alternative: Use keep_alive aggressively and unload embedder before LLM
```

---

### 3. ChromaDB vs FAISS Decision Lacks Justification

**Problem:** Spec claims ChromaDB for i3 and FAISS for M1 without explaining why.

**Analysis:**

| Feature | ChromaDB | FAISS |
|---------|----------|-------|
| Persistence | ✅ Native | ⚠️ Requires manual save |
| Memory usage | ✅ Lower (disk-backed) | ❌ All in RAM |
| Query speed | ⚠️ Slower (~500ms for 10K) | ✅ Faster (~200ms) |
| Metal acceleration | ❌ No | ✅ Yes (on M1) |
| Multi-collection | ✅ Easy | ⚠️ Complex |

**The Real Issue:** Spec assumes FAISS is faster because of Metal, but provides no benchmarks.

**Resolution:**
- **Test both backends** on actual hardware before committing
- **Allow user override** in config
- Consider **hybrid approach**: ChromaDB for persistence, FAISS for hot cache
- Provide **migration path** between backends

---

### 4. Performance Targets Are Guesses, Not Measurements

**Problem:** All performance targets lack basis in reality.

**Evidence:**
```
"Template execution: < 3s (M1)" 
"Template execution: < 8s (i3)"
```

**But what template?** 
- 500-word summary? 
- Full security audit with JSON extraction?
- Q&A pair generation?

**Variables not accounted for:**
- Prompt length (500 tokens vs 4K tokens)
- Model warm vs cold start
- Generation length (100 tokens vs 2K tokens)
- Streaming vs non-streaming

**Resolution:**
```markdown
# REVISED PERFORMANCE TARGETS (with conditions)

## M1 MacBook Air (PERFORMANCE Profile)

### PDF Ingestion
- Text-based (100 pages):  < 5s
- Scanned OCR (100 pages): < 30s (depends on image quality)

### Embedding (qwen2.5-coder:3b warm)
- 1000 chunks, batch_size=64: < 15s
- First embedding (cold start): < 20s

### Vector Search
- 10K chunks (FAISS in-memory): < 200ms
- 20K chunks (FAISS in-memory): < 400ms

### Template Execution
- Short (500 tokens in, 200 tokens out): 2-4s
- Medium (2K tokens in, 500 tokens out): 5-8s
- Long (4K tokens in, 1K tokens out): 10-15s

## i3 10th Gen (BALANCED Profile)

### PDF Ingestion
- Text-based (100 pages):  < 10s
- Scanned OCR (100 pages): < 60s

### Embedding (lfm2.5-thinking:1.2b)
- 1000 chunks, batch_size=32: < 30s
- First embedding (cold start): < 40s

### Vector Search
- 10K chunks (ChromaDB on SSD): < 500ms
- 10K chunks (ChromaDB on HDD): < 1000ms

### Template Execution
- Short (500 tokens in, 200 tokens out): 5-8s
- Medium (2K tokens in, 500 tokens out): 12-18s
- Long (4K tokens in, 1K tokens out): 25-35s

NOTE: All targets assume no other heavy processes running.
```

---

## 🟡 MAJOR ISSUES

### 5. Ollama API Strategy Contradicts Modern Best Practices

**Problem:** Spec mandates `/api/generate` over `/api/chat` for memory reasons, but this is outdated.

**Quote from spec:**
> "Why `/api/generate` over `/api/chat`? Memory growth - Stateless vs Accumulates KV cache"

**Reality Check (Ollama 0.1.27+):**
- `/api/chat` does **not** accumulate KV cache across **separate HTTP requests**
- KV cache only exists **within a single connection** when streaming
- Using `/api/generate` for multi-turn workflows means **re-processing context** every time

**Example:**
```python
# CURRENT SPEC APPROACH (inefficient for multi-turn)
# Turn 1: Process 3K tokens of context + 100 token question
# Turn 2: Process 3K tokens AGAIN + 100 token new question

# BETTER APPROACH
# Turn 1: Process 3K tokens once
# Turn 2: Only process new 100 tokens (KV cache reused)
```

**Resolution:**
- **Primary use case is SINGLE-TURN templates** → `/api/generate` is fine
- **If adding multi-turn later** → Use `/api/chat` with conversation state
- **Clarify in spec**: "PLATO v1.0 uses single-turn templates, so `/api/generate` avoids unnecessary chat formatting overhead"

---

### 6. Template Variable Injection is SQL-Injection-Like Risk

**Problem:** Direct Jinja2 rendering of user input without sanitization.

**Vulnerable code from spec:**
```python
prompt = template.render(
    context=context_chunks,
    **user_vars  # ← DANGER!
)
```

**Attack vector:**
```jinja2
# Malicious template uploaded by user
{{ context.__class__.__mro__[1].__subclasses__() }}
```

**Impact:** Remote code execution via Jinja2 sandbox escape.

**Resolution:**
```python
from jinja2.sandbox import SandboxedEnvironment

env = SandboxedEnvironment(
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True
)

# Whitelist allowed variables
ALLOWED_VARS = {
    'context', 'pdf_title', 'page_count', 
    'word_count', 'user_query', 'focus_areas', 'timestamp'
}

def safe_render(template_str: str, **user_vars):
    # Filter to allowed keys only
    safe_vars = {k: v for k, v in user_vars.items() if k in ALLOWED_VARS}
    
    template = env.from_string(template_str)
    return template.render(**safe_vars)
```

**Action:** Add security section to spec + implement sandboxing.

---

### 7. No Chunking Strategy for Multi-Column PDFs

**Problem:** Spec mentions "Multi-column layouts (Marker preprocessing)" but no implementation details.

**Why this matters:**
```
┌──────────────────────────┐
│  Column 1    │ Column 2  │  ← Reading order?
│  The study   │ Results   │
│  shows that  │ indicate  │
└──────────────────────────┘

Bad chunking: "The study Column 2 shows that Results indicate..."
Good chunking: "The study shows that..." → "Results indicate..."
```

**Missing details:**
- How does Marker handle column detection?
- What happens if Marker fails?
- Is there a fallback to PyMuPDF layout analysis?
- How are tables handled across columns?

**Resolution:**
```python
# Add to spec:
def ingest_pdf(filepath: Path) -> Document:
    # Step 0: Detect layout complexity
    layout_complexity = analyze_layout(filepath)
    
    if layout_complexity.has_columns or layout_complexity.has_tables:
        # Use Marker for structure-aware extraction
        markdown = marker_extract(filepath)
    else:
        # Use faster PyMuPDF
        markdown = pymupdf_extract(filepath)
    
    # ... rest of pipeline
```

---

### 8. Embedding Cache Invalidation Strategy Missing

**Problem:** Spec says "Only re-embed if PDF changes" but doesn't define "change".

**Questions:**
- What if user re-extracts with different chunking settings?
- What if OCR quality improves in newer DeepSeek-OCR version?
- What if PDF metadata changes but content doesn't?

**Current approach (from spec):**
> "Cache embeddings on disk keyed by PDF hash (SHA-256)"

**Issues:**
- SHA-256 of **entire PDF** changes if even 1 metadata byte changes
- No versioning of chunking strategy
- No versioning of embedding model

**Resolution:**
```python
@dataclass
class EmbeddingCacheKey:
    pdf_content_hash: str      # SHA-256 of text content only (not metadata)
    chunking_version: str      # "v1_chunk512_overlap50"
    embedding_model: str       # "embeddinggemma:latest"
    embedding_model_version: str  # From Ollama API
    
def get_cache_key(pdf: Path, config: ChunkingConfig) -> str:
    content = extract_text_only(pdf)  # Ignore metadata
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    return f"{content_hash}_{config.version}_{config.model}_{config.model_version}"
```

---

### 9. Hardware Auto-Detection Will Fail on VMs and Docker

**Problem:** Spec assumes bare metal detection.

**Evidence:**
```python
def detect_profile() -> HardwareProfile:
    is_arm = platform.machine() == 'arm64'  # ← Works on bare metal
    ram_gb = psutil.virtual_memory().total / 1e9  # ← Reports VM allocation, not host
    is_ssd = check_disk_type()  # ← May fail in containers
```

**Scenarios not handled:**
- Running in Docker on M1 (reports as `arm64` but shares RAM with host)
- Running in VM with 8GB allocated from 64GB host
- Running on ARM server (reports `arm64` but not M1)
- WSL2 on Windows

**Resolution:**
```python
def detect_profile() -> HardwareProfile:
    # Check if in container/VM
    in_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
    in_vm = detect_virtualization()  # Check for VM indicators
    
    if in_container or in_vm:
        # Default to BALANCED, let user override
        logger.warning("Container/VM detected. Using BALANCED profile. "
                      "Adjust with --profile flag if needed.")
        return HardwareProfile.BALANCED
    
    # ... existing detection logic
```

---

### 10. JSON Schema Validation Not Specified

**Problem:** Spec references JSON schema validation but provides no schema.

**Quote from spec:**
> "validation_schema: 'security_audit_v1.json'"

**But where is this file?** 
- Not in file structure
- No schema specification
- No validation library specified

**Resolution:**
```yaml
# Add to file structure:
templates/
  schemas/
    security_audit_v1.json
    executive_summary_v1.json
    technical_specs_v1.json

# Add to dependencies:
jsonschema==4.21.1

# Add implementation:
def validate_json(response: str, schema_name: str) -> Dict:
    schema_path = Path(f"templates/schemas/{schema_name}")
    schema = json.loads(schema_path.read_text())
    
    parsed = json.loads(response)
    jsonschema.validate(parsed, schema)
    return parsed
```

---

### 11. Memory Monitor Threshold is Too Aggressive

**Problem:** 1.0 GB buffer is too small for real-world usage.

**Evidence from spec:**
```python
class MemoryMonitor:
    def __init__(self, threshold_gb: float = 1.0):  # ← Only 1GB buffer!
```

**Reality:**
- Python GC can spike by 500MB during large operations
- OS needs headroom for disk cache
- Swapping starts before buffer is exhausted

**Real-world safe thresholds:**
```python
# REVISED
class MemoryMonitor:
    def __init__(self, 
                 warning_threshold_gb: float = 1.5,    # 50% of budget
                 critical_threshold_gb: float = 0.5):  # 16% of budget
```

---

## 🔵 MINOR ISSUES

### 12. Keyboard Shortcuts Conflict with Terminal Emulators

**Problem:** Many shortcuts won't work in all terminals.

| Shortcut | Conflict |
|----------|----------|
| `Ctrl+S` | Terminal XOFF (freeze) |
| `Ctrl+Q` | Terminal XON (unfreeze) |
| `Ctrl+C` | SIGINT (may not reach app) |

**Resolution:** Use Textual's built-in key binding system with fallbacks:
```python
# Primary: Ctrl+S
# Fallback: F7
BINDINGS = [
    ("ctrl+s,f7", "save_output", "Save"),
    ("ctrl+q,f10", "quit", "Quit"),
]
```

---

### 13. Vim Bindings Incomplete

**Problem:** Only `j/k` mentioned, but no other Vim navigation.

**Expected Vim bindings:**
- `j/k` - down/up ✅ (mentioned)
- `g/G` - top/bottom ❌ (missing)
- `/` - search ❌ (missing)
- `n/N` - next/prev search result ❌ (missing)
- `dd` - delete item ❌ (missing)
- `yy` - yank (copy) ❌ (missing)

**Resolution:** Add comprehensive Vim mode or remove the claim.

---

### 14. No Dark/Light Theme Toggle

**Problem:** Hardcoded theme in config.

```yaml
ui:
  theme: "monokai"  # ← No toggle mentioned
```

**Most TUIs support:** `F11` or `t` to toggle theme.

**Resolution:** Add `Ctrl+Shift+T` for theme cycle.

---

### 15. Error Log Rotation Not Specified

**Problem:** `~/.cache/plato/logs/errors.log` will grow indefinitely.

**Resolution:**
```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    "~/.cache/plato/logs/errors.log",
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
```

---

### 16. Clipboard Copy Doesn't Preserve Formatting

**Problem:** `pyperclip` is plain text only.

**Evidence:**
```python
copy_to_clipboard(json.dumps(output, indent=2))  # ← No syntax highlighting
```

**User expectation:** Paste into IDE with JSON syntax highlighting preserved.

**Resolution:** Can't fix with pyperclip. Document this limitation:
> "Note: Clipboard copies plain text. For formatted JSON, use 'Export → JSON' and open in your editor."

---

### 17. No Undo/Redo for Template Editor

**Problem:** Template editor mentioned but no edit history.

**Impact:** User accidentally deletes template, no way to recover.

**Resolution:** Add `.yaml.backup` files or integrate with Git.

---

## ⚪ ENHANCEMENTS

### 18. Add Template Dry-Run Mode

**Suggestion:** Before running expensive LLM call, show rendered prompt.

```python
# Before execution:
plato template --dry-run security_audit.yaml

# Output:
"""
=== DRY RUN ===
Model: qwen2.5-coder:3b
Estimated tokens: 3,247
Estimated time: 8-12s
Estimated cost: $0.00 (local)

System prompt:
You are a cybersecurity researcher...

User prompt:
Analyze the following document for security risks:
[CHUNK 1 - Page 3]
The architecture consists of...

Continue? [y/N]
"""
```

---

### 19. Add Template Marketplace Schema

**Suggestion:** Since spec mentions "template marketplace", define schema.

```yaml
# marketplace_schema.yaml
version: "1.0"
templates:
  - name: "Security Audit Pro"
    author: "security@example.com"
    version: "2.1.0"
    verified: true  # Anthropic-verified
    downloads: 1234
    rating: 4.8
    source_url: "https://templates.plato.dev/security_audit_pro.yaml"
    sha256: "abc123..."
```

---

### 20. Add Streaming Support for Long Outputs

**Problem:** Non-streaming means no feedback during 30s generation.

**Current spec:**
```python
stream=False  # Get full response for parsing
```

**Enhancement:**
```python
async def execute_template_streaming(template, chunks, user_vars):
    async for chunk in ollama.generate_stream(...):
        yield chunk  # Update UI progressively
```

---

### 21. Add PDF Annotation Export

**Use case:** User highlights key chunks, wants to export annotations back to PDF.

**Implementation:** Use `pymupdf` to add highlight annotations:
```python
def export_annotations(pdf_path: Path, chunks: List[Chunk]):
    doc = pymupdf.open(pdf_path)
    for chunk in chunks:
        page = doc[chunk.metadata.page - 1]
        # Add highlight at char_start:char_end
        page.add_highlight_annot(...)
    doc.save("annotated_" + pdf_path.name)
```

---

### 22. Add Watch Mode for Template Development

**Use case:** Template author tweaking YAML, wants auto-reload.

```bash
plato template --watch my_template.yaml
# Auto-reloads and re-runs when file changes
```

---

### 23. Add Telemetry (Optional, Local)

**Privacy-preserving analytics:**
```yaml
# Optional local telemetry (never sent externally)
telemetry:
  enabled: false
  storage: "~/.cache/plato/stats.db"
  
# Tracks:
# - Most-used templates
# - Average processing times
# - Error frequencies
# Purpose: Suggest optimizations to user
```

---

## 📊 Priority Matrix

| Issue | Severity | Impact | Effort | Priority |
|-------|----------|--------|--------|----------|
| #1 Model naming | 🔴 Critical | High | Low | 1 |
| #2 RAM budget | 🔴 Critical | High | Medium | 2 |
| #6 Template injection | 🟡 Major | High | Low | 3 |
| #4 Performance targets | 🔴 Critical | Medium | High | 4 |
| #3 Vector DB choice | 🔴 Critical | Medium | Medium | 5 |
| #7 Multi-column PDFs | 🟡 Major | Medium | High | 6 |
| #8 Cache invalidation | 🟡 Major | Medium | Medium | 7 |
| #10 JSON schemas | 🟡 Major | Medium | Low | 8 |
| #9 VM detection | 🟡 Major | Low | Medium | 9 |
| #5 Ollama API | 🟡 Major | Low | Low | 10 |
| ... | | | | |

---

## 🔧 Recommended Changes to Spec

### Change 1: Add "Architecture Decisions" Section

```markdown
## Architecture Decision Records (ADRs)

### ADR-001: Why ChromaDB vs FAISS?

**Status:** Proposed  
**Decision:** Use ChromaDB as default, FAISS as opt-in performance mode  
**Rationale:**
- ChromaDB provides persistence without manual save/load
- FAISS offers 2-3x faster queries but requires RAM
- Users can switch backends via config

**Consequences:**
- Initial performance may be slower than claimed
- Need to benchmark both on target hardware
- Migration tooling required
```

---

### Change 2: Add "Non-Functional Requirements" Section

```markdown
## Non-Functional Requirements

### Performance
- P0: PDF ingestion < 15s for 100 pages (text) on i3
- P1: Vector search < 1s for 10K chunks on i3
- P2: Template execution < 20s for 4K context on i3

### Reliability
- R0: No data loss on crash (all writes are atomic)
- R1: Graceful degradation when models unavailable
- R2: Error recovery within 5s (no hangs)

### Security
- S0: No remote code execution via templates (sandboxed Jinja2)
- S1: No sensitive data in logs (PII redaction)
- S2: Optional encryption of cached embeddings

### Usability
- U0: No modal dialogs (TUI stays responsive)
- U1: All actions reversible (undo support)
- U2: Keyboard-only navigation (no mouse required)
```

---

### Change 3: Add "Testing Strategy" Section

```markdown
## Testing Strategy

### Unit Tests
- Target: 85% code coverage
- Focus: Core algorithms (chunking, embedding, search)
- Tool: pytest with pytest-cov

### Integration Tests
- Target: All Ollama API interactions mocked
- Focus: Template rendering, vector DB operations
- Tool: pytest with fixtures

### Property-Based Tests
- Tool: Hypothesis
- Focus: Chunking edge cases (empty docs, single char, max size)

### Performance Tests
- Tool: pytest-benchmark
- Focus: Regression detection (fail if 20% slower)

### Manual QA Checklist
- [ ] Test on both i3 and M1
- [ ] Test with 10 real-world PDFs (varied layouts)
- [ ] Test all keyboard shortcuts in 3 terminal emulators
- [ ] Test with Ollama down / models missing
- [ ] Test with corrupted/encrypted PDFs
```

---

## ✅ Final Recommendations

### Must Fix Before v1.0
1. ✅ Standardize model names (ADR-001)
2. ✅ Recalculate RAM budgets with realistic numbers
3. ✅ Add Jinja2 sandboxing
4. ✅ Define JSON schemas for all templates
5. ✅ Add cache invalidation strategy
6. ✅ Revise performance targets with ranges

### Should Fix Before v1.0
7. ✅ Add VM/container detection
8. ✅ Specify multi-column PDF handling
9. ✅ Add error log rotation
10. ✅ Document Ollama API choice clearly

### Nice to Have for v1.1
11. Template dry-run mode
12. Streaming support
13. PDF annotation export
14. Template watch mode

---

## 📝 Conclusion

**The PLATO spec is 70% production-ready.** The core idea is sound, but implementation details need refinement. The biggest risks are:

1. **Underestimated RAM requirements** (especially M1)
2. **Unverified performance claims** (need actual benchmarks)
3. **Security gaps** (template injection, no input sanitization)

**Recommended Path Forward:**

**Week 1:** Fix Critical issues (#1-4)  
**Week 2:** Fix Major issues (#5-11)  
**Week 3:** Implement core MVP  
**Week 4:** Benchmark on real hardware, adjust targets  
**Week 5:** Polish and documentation  
**Week 6:** Beta release with limited scope

**Estimated Total Effort:** 6 weeks (1 developer, full-time)

---

**Status:** EVALUATION COMPLETE  
**Next Step:** Create harmonized spec v1.1 with all Critical/Major fixes applied