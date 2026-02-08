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

from .parser import PDFProcessor, DocumentParser
from .llm import OllamaClient, ExtractionResult
from .graph_rag import GraphRAGPipeline
from .config import get_config


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
        
        # Components
        self.pdf_processor = PDFProcessor() # For initial PDF -> MD conversion
        self.llm_client = OllamaClient()
        self.graph_rag = GraphRAGPipeline(storage_dir=str(self.output_dir / "storage"))
        
        self._setup_logging()
        
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
        Async processing of a single PDF.
        1. Convert to Markdown (if PDF)
        2. Extract Entities/Relations/Summary (OllamaClient)
        3. Insert into GraphRAG (LlamaIndex)
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
        
        try:
            # Step 1: Text Conversion
            # Check if it's already text/md or needs PDF conversion
            if pdf_path.suffix.lower() == '.pdf':
                self.logger.info("Converting PDF to Markdown...")
                # Note: pdf_processor is sync, might block loop briefly. 
                # Ideally run in executor if slow.
                text, _ = self.pdf_processor.convert_to_markdown(str(pdf_path))
                # Write to temp file for OllamaClient to read (since it takes path)
                # Or we can refactor OllamaClient to take text. 
                # Current OllamaClient.process_document takes a path.
                temp_md_path = self.output_dir / f"{doc_id}.md"
                temp_md_path.write_text(text, encoding='utf-8')
                source_path = str(temp_md_path)
            else:
                source_path = str(pdf_path)

            # Step 2: Extraction (Contexter Pattern)
            self.logger.info("Extracting knowledge...")
            extraction_results: List[ExtractionResult] = await self.llm_client.process_document_complete(
                source_path, 
                doc_id
            )
            
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
                # Create LlamaIndex Document from chunk
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
                self.graph_rag.insert_documents(llama_docs)
            
            result_stats["success"] = True
            self.logger.info(f"Completed {doc_id}. Stats: {result_stats}")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed for {doc_id}: {e}")
            result_stats["error"] = str(e)
            
        return result_stats

    def process_directory(self, input_dir: str) -> List[Dict]:
        """Synchronous wrapper for directory processing (useful for CLI/Scripts)"""
        input_path = Path(input_dir)
        files = list(input_path.glob("*.pdf")) + list(input_path.glob("*.md"))
        
        results = []
        async def _run_all():
            tasks = [self.process_pdf_async(str(f)) for f in files]
            return await asyncio.gather(*tasks)
            
        try:
            results = asyncio.run(_run_all())
        except RuntimeError:
            # Handle running loop override if needed
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(_run_all())
            
        return results

    def query(self, query_text: str) -> str:
        """Query the knowledge graph"""
        return self.graph_rag.query(query_text)
