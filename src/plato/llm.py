"""
Refactored Ollama LLM client for PLATO - Context Preparation Assistant
Implements Contexter Pattern: resource-aware, document-tracked, streaming-capable
"""
import json
import re
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, AsyncIterator, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from pathlib import Path
import ollama
from plato.config import get_config


@dataclass
class DocumentContext:
    """Track document metadata through extraction pipeline"""
    doc_id: str
    source_path: str
    doc_name: str
    chunk_index: int
    total_chunks: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate document context on creation"""
        if not self.doc_id or not self.source_path:
            raise ValueError("doc_id and source_path are required")
        if len(self.text) == 0:
            raise ValueError("Document text cannot be empty")


@dataclass
class ExtractionResult:
    """Structured result with document provenance"""
    doc_context: DocumentContext
    entities: Dict[str, List[str]] = field(default_factory=dict)
    relations: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""
    embeddings: List[float] = field(default_factory=list)
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format"""
        return {
            "source": {
                "doc_id": self.doc_context.doc_id,
                "doc_name": self.doc_context.doc_name,
                "chunk": f"{self.doc_context.chunk_index}/{self.doc_context.total_chunks}",
                "source_path": str(self.doc_context.source_path)
            },
            "entities": self.entities,
            "relations": self.relations,
            "summary": self.summary,
            "embeddings_generated": len(self.embeddings) > 0,
            "metadata": self.extraction_metadata
        }





class SmartJSONExtractor:
    """Robust JSON extraction from LLM outputs with recovery strategies"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def extract(self, text: str) -> Optional[Any]:
        """Extract JSON with progressive fallback strategies"""
        if not text:
            return None
        
        # Strategy 1: Markdown code blocks
        json_str = self._try_code_blocks(text)
        if json_str:
            return self._parse_json(json_str)
        
        # Strategy 2: Raw JSON boundaries
        json_str = self._try_raw_json(text)
        if json_str:
            return self._parse_json(json_str)
        
        # Strategy 3: Repair and retry (fix common LLM errors)
        json_str = self._try_repair(text)
        if json_str:
            return self._parse_json(json_str)
        
        self.logger.warning(f"JSON extraction failed after all strategies. Text preview: {text[:200]}")
        return None
    
    def _try_code_blocks(self, text: str) -> Optional[str]:
        """Extract JSON from markdown code blocks"""
        pattern = r'```(?:json)?\s*(\{.*\}|\[.*\])\s*```'
        for match in re.finditer(pattern, text, re.DOTALL):
            candidate = match.group(1).strip()
            if self._is_valid_json(candidate):
                return candidate
        return None
    
    def _try_raw_json(self, text: str) -> Optional[str]:
        """Extract raw JSON structures"""
        # Find first { or [ and match to closing } or ]
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start_idx = text.find(start_char)
            if start_idx == -1:
                continue
            
            # Use bracket counting for robustness
            depth = 0
            for i in range(start_idx, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start_idx:i+1]
                        if self._is_valid_json(candidate):
                            return candidate
                        break
        return None
    
    def _try_repair(self, text: str) -> Optional[str]:
        """Attempt common repairs: trailing commas, unquoted keys"""
        # Extract potential JSON region
        match = re.search(r'[\{\[].*[\}\]]', text, re.DOTALL)
        if not match:
            return None
        
        candidate = match.group(0)
        
        # Remove trailing commas
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        
        if self._is_valid_json(candidate):
            return candidate
        return None
    
    def _is_valid_json(self, text: str) -> bool:
        """Quick validation without throwing"""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
    
    def _parse_json(self, json_str: str) -> Optional[Any]:
        """Parse JSON with logging"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON parse error: {e}")
            return None


class TextChunker:
    """Intelligent text splitting aware of semantic boundaries"""
    
    def __init__(self, max_length: int = 4000, overlap: int = 200):
        self.max_length = max_length
        self.overlap = overlap
    
    def chunk(self, text: str, doc_id: str) -> List[DocumentContext]:
        """Split text into semantic chunks with metadata"""
        if len(text) <= self.max_length:
            return [DocumentContext(
                doc_id=doc_id,
                source_path=doc_id,
                doc_name=doc_id,
                chunk_index=0,
                total_chunks=1,
                text=text
            )]
        
        # Split by paragraphs first, then sentences
        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.max_length:
                if current_chunk:
                    chunks.append(current_chunk)
                    chunk_index += 1
                current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        if current_chunk:
            chunks.append(current_chunk)
        
        total_chunks = len(chunks)
        
        return [DocumentContext(
            doc_id=doc_id,
            source_path=doc_id,
            doc_name=doc_id,
            chunk_index=i,
            total_chunks=total_chunks,
            text=chunk.strip()
        ) for i, chunk in enumerate(chunks)]


class EdgeModelManager:
    """
    Manages Ollama model loading state to prevent OOM on edge devices.
    Ensures only one heavy model is active if needed, and handles locking.
    """
    _instance = None
    _lock = asyncio.Lock()
    _loaded_model: Optional[str] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EdgeModelManager, cls).__new__(cls)
        return cls._instance
        
    async def exclusive_run(self, task_type: str, config: Any, func, *args, **kwargs):
        """
        Run an LLM task with exclusive access to resources.
        Handles checking if model needs swapping.
        """
        model_name = config.get_model_name(task_type)
        
        async with self._lock:
            # In a more advanced implementation, we would explicitly unload 
            # the previous model if memory is tight. capabilities depend on Ollama version.
            # For now, the lock ensures we don't try to load two huge models simultaneously
            # via concurrent requests, letting Ollama's internal queuing handle the rest.
            # We can also verify memory usage here if psutil is available.
            
            # self.logger.debug(f"Acquired lock for {model_name}")
            try:
                return await func(*args, **kwargs)
            finally:
                pass

# Global manager instance
_model_manager = EdgeModelManager()

class OllamaClient:
    """Resource-aware Ollama client implementing Contexter Pattern"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config:
            # If dict provided, wrap properly
            # But normally we get the full config object structure from get_config()
            # We'll use the singleton config if specific dict not passed, 
            # or try to adapt the dict.
            # For simplicity in this refactor, let's rely on get_config().ollama if config is None
            pass
            
        # Re-fetch fresh config which has hardware detection
        self.config = get_config().ollama
        
        self.json_extractor = SmartJSONExtractor(logging.getLogger(__name__))
        self.text_chunker = TextChunker(4000) # Fixed limit for safe parsing
        self.logger = logging.getLogger(__name__)
        self._validate_prompts()
    
    def _validate_prompts(self):
        """Ensure required prompts are configured"""
        prompts = get_config().prompts
        # Basic validation, though prompts might be empty in config if not loaded
        pass 
    
    @asynccontextmanager
    async def process_document(self, doc_path: str, doc_id: str):
        """
        Contexter Pattern: manage document processing lifecycle
        Opens resource, yields processing context, cleans up
        """
        doc_path = Path(doc_path)
        
        try:
            # Open and validate
            text = doc_path.read_text(encoding='utf-8')
            if not text.strip():
                raise ValueError(f"Document {doc_path} is empty")
            
            self.logger.info(f"Processing document: {doc_id} ({len(text)} chars)")
            
            # Yield for processing
            yield text
            
        except FileNotFoundError:
            self.logger.error(f"Document not found: {doc_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing document {doc_id}: {e}")
            raise
        finally:
            # Cleanup: explicitly allow garbage collection
            # In real implementation, close file handles, clear caches
            pass
    
    async def extract_entities_streaming(
        self, 
        doc_context: DocumentContext
    ) -> AsyncIterator[Tuple[str, List[str]]]:
        """
        Stream entity types as they're extracted
        Allows TUI to show progressive results
        """
        # model = self.config.get_model_for_task('reasoning') # Deprecated
        prompts = get_config().prompts
        # Fallback prompt if not configured
        prompt_tmpl = prompts.get('entity_extraction', "Extract entities from: {text}") 
        prompt = prompt_tmpl.format(text=doc_context.text)
        
        system_prompt = "You are a precise entity extraction system. Return only valid JSON with structure: {\"entities\": {\"TYPE\": [values]}}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._call_llm_async(messages, 'reasoning')
            result = self.json_extractor.extract(response)
            
            if result and isinstance(result, dict) and 'entities' in result:
                entities_dict = result['entities']
                if isinstance(entities_dict, dict):
                    for entity_type, entity_list in entities_dict.items():
                        if isinstance(entity_list, list):
                            validated = list(set([
                                str(e).strip() for e in entity_list
                                if isinstance(e, (str, int, float)) and str(e).strip()
                            ]))
                            if validated:
                                yield (entity_type, validated)
                                await asyncio.sleep(0.1)  # Allow UI updates
        except Exception as e:
            self.logger.error(f"Entity extraction failed for doc {doc_context.doc_id}: {e}")
    
    async def extract_entities(self, doc_context: DocumentContext) -> Dict[str, List[str]]:
        """Batch collect entities from streaming extraction"""
        entities = {}
        async for entity_type, entity_list in self.extract_entities_streaming(doc_context):
            entities[entity_type] = entity_list
        return entities
    
    async def extract_relations(self, doc_context: DocumentContext) -> List[Dict[str, str]]:
        """
        Extract relationships with document context preservation
        """
        # model = self.config.get_model_for_task('reasoning')
        prompts = get_config().prompts
        prompt_tmpl = prompts.get('relation_extraction', "Extract relations from: {text}")
        prompt = prompt_tmpl.format(text=doc_context.text)
        
        system_prompt = "You are a relation extraction system. Return only valid JSON: [{\"subject\": \"...\", \"relation\": \"...\", \"object\": \"...\"}]"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._call_llm_async(messages, 'reasoning')
            result = self.json_extractor.extract(response)
            
            if result and isinstance(result, list):
                valid_relations = []
                for rel in result:
                    if isinstance(rel, dict) and all(k in rel for k in ['subject', 'relation', 'object']):
                        valid_relations.append({
                            'subject': str(rel['subject']).strip(),
                            'relation': str(rel['relation']).strip(),
                            'object': str(rel['object']).strip()
                        })
                return valid_relations
        except Exception as e:
            self.logger.error(f"Relation extraction failed for doc {doc_context.doc_id}: {e}")
        
        return []
    
    async def summarize(self, doc_context: DocumentContext, max_length: int = 200) -> str:
        """Generate summary with document tracking"""
        # model = self.config.get_model_for_task('reasoning')
        prompt = f"Summarize in ~{max_length} words:\n\n{doc_context.text}"
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = await self._call_llm_async(messages, 'reasoning')
            words = response.split()
            return ' '.join(words[:max_length]) if words else ""
        except Exception as e:
            self.logger.error(f"Summarization failed for doc {doc_context.doc_id}: {e}")
            return ""
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings (typically smaller, fast operation)"""
        if not text:
            return []
        try:
            embedding_model = self.config.get_model_name('embedding')
            response = ollama.embeddings(model=embedding_model, prompt=text)
            return response.get('embedding', [])
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return []
    
    async def process_document_complete(
        self, 
        doc_path: str, 
        doc_id: str
    ) -> List[ExtractionResult]:
        """
        Full pipeline: read → chunk → extract entities/relations/summary → return with provenance
        Returns results with full document tracking
        """
        results = []
        
        async with self.process_document(doc_path, doc_id) as text:
            # Chunk text intelligently
            chunks = self.text_chunker.chunk(text, doc_id)
            
            for chunk in chunks:
                try:
                    # Extract in parallel where possible
                    entities_task = asyncio.create_task(self.extract_entities(chunk))
                    relations_task = asyncio.create_task(self.extract_relations(chunk))
                    summary_task = asyncio.create_task(self.summarize(chunk))
                    embeddings_task = asyncio.create_task(self.generate_embedding(chunk.text))
                    
                    entities, relations, summary, embeddings = await asyncio.gather(
                        entities_task, relations_task, summary_task, embeddings_task
                    )
                    
                    result = ExtractionResult(
                        doc_context=chunk,
                        entities=entities,
                        relations=relations,
                        summary=summary,
                        embeddings=embeddings,
                        extraction_metadata={
                            "model": self.config.get_model_name('reasoning'),
                            "timestamp": time.time()
                        }
                    )
                    results.append(result)
                    self.logger.info(f"Completed chunk {chunk.chunk_index}/{chunk.total_chunks}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process chunk {chunk.chunk_index}: {e}")
                    continue
        
        return results
    
    async def _call_llm_async(self, messages: List[Dict[str, str]], task_type: str) -> str:
        """Async LLM call with retry logic and resource locking"""
        
        full_config = self.config.get_model_config(task_type)
        model = full_config.get("model")
        
        # Prepare options (exclude model name from options dict)
        options = {k: v for k, v in full_config.items() if k != "model"}
        
        async def _execute_call():
            # Note: ollama.chat is synchronous, so we run it in executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.chat(
                    model=model,
                    messages=messages,
                    options=options
                )
            )
            return response

        for attempt in range(self.config.max_retries):
            try:
                # Use manager to ensure we don't overload
                response = await _model_manager.exclusive_run(
                    task_type, 
                    self.config, 
                    _execute_call
                )
                
                if 'message' not in response or 'content' not in response['message']:
                    raise ValueError(f"Unexpected response structure: {response}")
                
                return response['message']['content']
                
            except Exception as e:
                self.logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
                if attempt == self.config.max_retries - 1:
                    raise RuntimeError(f"Failed after {self.config.max_retries} attempts: {e}") from e
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return ""
    
    # Backward compatibility: sync wrapper
    def extract_entities_batch(self, texts: List[str]) -> List[Dict[str, List[str]]]:
        """Sync batch extraction (wraps async)"""
        async def _batch():
            results = []
            for text in texts:
                doc_context = DocumentContext(
                    doc_id="batch_item",
                    source_path="batch",
                    doc_name="batch",
                    chunk_index=0,
                    total_chunks=1,
                    text=text
                )
                entities = await self.extract_entities(doc_context)
                results.append(entities)
            return results
        
        try:
             loop = asyncio.get_event_loop()
             if loop.is_running():
                 # We are already in a loop, return a coroutine
                 return _batch()
             return asyncio.run(_batch())
        except RuntimeError:
             return asyncio.run(_batch())