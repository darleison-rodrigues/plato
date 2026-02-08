"""
Main Pipeline: Orchestrates PDF → Text → Extraction → GraphRAG
"""
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

from llama_index.core import Document

from plato.parser import PDFProcessor, DocumentParser
from plato.llm import OllamaClient, ExtractionResult
from plato.graph_rag import GraphRAGPipeline
from plato.config import get_config


class Pipeline:
    """
    Main pipeline orchestrator.
    - Uses OllamaClient (Contexter) for robust, streaming-capable extraction.
    - Uses GraphRAGPipeline (LlamaIndex) for storage and querying.
    """
    
    def __init__(self):
        self.config = get_config()
        self.output_dir = Path(self.config.pipeline.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Components - Lazy Initialized
        # This prevents long startup times on slower hardware (i3/M1)
        # Components are only loaded when actually needed for processing
        self._pdf_processor = None
        self._llm_client = None
        self._graph_rag = None
        
        self._setup_logging()
        
    @property
    def pdf_processor(self):
        if not self._pdf_processor:
            self._pdf_processor = PDFProcessor()
        return self._pdf_processor

    @property
    def llm_client(self):
        if not self._llm_client:
            self.logger.info("Initializing Ollama Client...")
            self._llm_client = OllamaClient()
        return self._llm_client

    @property
    def graph_rag(self):
        if not self._graph_rag:
            self.logger.info("Initializing GraphRAG Pipeline...")
            self._graph_rag = GraphRAGPipeline(storage_dir=str(self.output_dir / "storage"))
        return self._graph_rag
        
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def process_pdf_async(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Async processing of a single PDF with resource management.
        """
        pdf_path = Path(pdf_path)
        if not doc_id:
            doc_id = pdf_path.stem
            
        self.logger.info(f"Processing: {doc_id}")
        
        result_stats = {
            "doc_id": doc_id,
            "chunks_processed": 0,
            "entities_found": 0,
            "relations_found": 0,
            "success": False,
            "error": None
        }
        
        temp_md_path = None
        
        try:
            # Step 1: Text Conversion (Non-blocking)
            if pdf_path.suffix.lower() == '.pdf':
                self.logger.info("Converting PDF to Markdown...")
                loop = asyncio.get_running_loop()
                # DX: Offload CPU-intensive PDF parsing to a thread executor.
                # This prevents the async event loop from blocking, which is critical
                # for responsiveness on lower-end hardware (i3/M1).
                try:
                    text, _ = await loop.run_in_executor(
                        None, 
                        self.pdf_processor.convert_to_markdown,
                        str(pdf_path)
                    )
                except Exception as e:
                    raise RuntimeError(f"PDF Conversion failed: {e}")
                
                temp_md_path = self.output_dir / f"{doc_id}.md"
                # Write temp file (blocking but fast enough, or use aiofiles if strict)
                temp_md_path.write_text(text, encoding='utf-8')
                source_path = str(temp_md_path)
            else:
                source_path = str(pdf_path)

            # Step 2: Extraction (Contexter Pattern)
            self.logger.info("Extracting knowledge...")
            try:
                extraction_results: List[ExtractionResult] = await self.llm_client.process_document_complete(
                    source_path, 
                    doc_id
                )
            except Exception as e:
                raise RuntimeError(f"LLM Extraction failed: {e}")
            
            # Step 3: Ingestion into GraphRAG
            self.logger.info("Ingesting into GraphRAG...")
            
            llama_docs = []
            
            for res in extraction_results:
                # 3a. Insert Manual Triplets
                if res.entities or res.relations:
                   self.graph_rag.insert_triplets(
                       res.entities, 
                       res.relations, 
                       source_doc=doc_id
                   )
                   result_stats["entities_found"] += sum(len(v) for v in res.entities.values())
                   result_stats["relations_found"] += len(res.relations)
                
                # 3b. Prepare for Vector/Text insertion
                doc = Document(
                    text=res.doc_context.text,
                    metadata={
                        "doc_id": doc_id,
                        "chunk_index": res.doc_context.chunk_index,
                        "source": str(pdf_path),
                        **res.extraction_metadata
                    }
                )
                llama_docs.append(doc)
                result_stats["chunks_processed"] += 1
            
            # Batch insert documents
            if llama_docs:
                try:
                    self.graph_rag.insert_documents(llama_docs)
                except Exception as e:
                    raise RuntimeError(f"Graph Ingestion failed: {e}")
            
            result_stats["success"] = True
            self.logger.info(f"Completed {doc_id}. Stats: {result_stats}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for {doc_id}: {e}")
            result_stats["error"] = str(e)
        finally:
            # Cleanup temp files
            if temp_md_path and temp_md_path.exists():
                try:
                    temp_md_path.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup temp file {temp_md_path}: {e}")
            
        return result_stats

    def process_directory(self, input_dir: str, max_concurrent: int = 2) -> List[Dict]:
        """
        Wrapper for directory processing with configurable concurrency.
        
        Args:
            input_dir: Directory containing PDFs/MDs
            max_concurrent: Max concurrent documents (defaults to 2 for edge safety)
        """
        input_path = Path(input_dir)
        files = list(input_path.glob("*.pdf")) + list(input_path.glob("*.md"))
        
        # DX: Concurrency limit prevents OOM on edge devices.
        # Configurable via CLI/App for power users.
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _process_limited(f):
            async with semaphore:
                return await self.process_pdf_async(str(f))

        results = []
        async def _run_all():
            tasks = [_process_limited(f) for f in files]
            return await asyncio.gather(*tasks)
            
        try:
            results = asyncio.run(_run_all())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(_run_all())
            
        return results

    def query(self, query_text: str) -> str:
        """Query the knowledge graph"""
        return self.graph_rag.query(query_text)
