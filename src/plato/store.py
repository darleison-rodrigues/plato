"""
ChromaDB vector store for document chunks and retrieval
"""
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import chromadb
from chromadb.config import Settings
from .config import get_config
from .llm import OllamaClient


class VectorStore:
    """Manages document storage and retrieval using ChromaDB with robust error handling"""
    
    def __init__(self):
        self.config = get_config().chroma
        self.pipeline_config = get_config().pipeline
        self.logger = logging.getLogger(__name__)
        
        # Initialize Chroma client
        self._initialize_client()
        
        # Initialize LLM client for embeddings
        self.llm_client = OllamaClient()
        
    def _initialize_client(self):
        """Initialize Chroma client with proper error handling"""
        try:
            persist_dir = Path(self.config.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"description": "PDF document chunks"}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Chroma client: {e}")
            raise RuntimeError(f"Vector store initialization failed: {e}") from e
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks with improved boundary detection"""
        if not text:
            return []
            
        chunks = []
        chunk_size = self.pipeline_config.chunk_size
        overlap = self.pipeline_config.chunk_overlap
        
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            
            # If not at the end, try to find a natural break point
            if end < text_len:
                section = text[start:end]
                # Priority: Paragraph -> Sentence -> Line -> Word
                break_candidates = [
                    section.rfind('\n\n'),
                    section.rfind('. '),
                    section.rfind('\n'),
                    section.rfind(' ')
                ]
                
                # Use the best break point that isn't too early (e.g. < 50% of chunk)
                best_break = -1
                for bp in break_candidates:
                    if bp > chunk_size * 0.5:
                        best_break = bp
                        break
                
                if best_break != -1:
                    end = start + best_break + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Update start for next chunk, ensuring forward progress
            next_start = end - overlap
            if next_start <= start: # Prevent infinite loop if overlap >= chunk_size or logic fails
                next_start = end
            start = next_start
            
        return chunks
    
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None):
        """Add a single document to the vector store"""
        self.add_documents_batch([{"doc_id": doc_id, "text": text, "metadata": metadata}])

    def add_documents_batch(self, documents: List[Dict[str, Any]], batch_size: int = 50) -> Dict[str, Any]:
        """Add multiple documents efficiently using batch processing"""
        if not documents:
            return {"processed": 0, "errors": []}
            
        all_ids = []
        all_chunks = []
        all_metadatas = []
        errors = []
        
        # 1. Prepare chunks
        for doc in documents:
            try:
                doc_id = doc.get('doc_id')
                text = doc.get('text')
                base_meta = doc.get('metadata', {}) or {}
                
                if not doc_id or not text:
                    continue
                    
                chunks = self._chunk_text(text)
                for i, chunk in enumerate(chunks):
                    all_ids.append(f"{doc_id}_chunk_{i}")
                    all_chunks.append(chunk)
                    meta = base_meta.copy()
                    meta.update({
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "total_chunks": len(chunks)
                    })
                    all_metadatas.append(meta)
            except Exception as e:
                errors.append(f"Error preparing doc {doc.get('doc_id')}: {e}")

        # 2. Process in batches
        total_added = 0
        for i in range(0, len(all_chunks), batch_size):
            end_idx = i + batch_size
            batch_chunks = all_chunks[i:end_idx]
            batch_ids = all_ids[i:end_idx]
            batch_metas = all_metadatas[i:end_idx]
            
            try:
                # Generate embeddings
                batch_embeddings = []
                for chunk in batch_chunks:
                     emb = self.llm_client.generate_embedding(chunk)
                     if not emb:
                         emb = [0.0] * 768 # Handle failure gracefully with zero vector or skip
                         self.logger.warning("Generated empty embedding, using zero vector")
                     batch_embeddings.append(emb)
                
                # Add to Chroma
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_chunks,
                    metadatas=batch_metas,
                    embeddings=batch_embeddings
                )
                total_added += len(batch_chunks)
                
            except Exception as e:
                self.logger.error(f"Batch insertion failed: {e}")
                errors.append(f"Batch {i//batch_size} failed: {e}")
                
        self.logger.info(f"Added {total_added} chunks from {len(documents)} documents")
        return {"processed": total_added, "errors": errors}

    def search(self, query: str, n_results: int = None) -> Dict[str, Any]:
        """Search for relevant document chunks with validation"""
        if not query or not query.strip():
             return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        if n_results is None:
            n_results = self.pipeline_config.max_chunks_for_context
            
        try:
            # Generate embedding
            query_embedding = self.llm_client.generate_embedding(query)
            if not query_embedding:
                 self.logger.warning("Failed to generate query embedding")
                 return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        try:
            count = self.collection.count()
            # To get distinct docs, we might need a peek if collection is small,
            # but for large collections, just returning data size is safer/faster.
            return {
                "total_chunks": count,
                "collection_name": self.config.collection_name
            }
        except Exception as e:
             self.logger.error(f"Failed to get stats: {e}")
             return {}

    def list_documents(self) -> List[str]:
         """List all document IDs in the store"""
         try:
             # This can be slow for large collections
             data = self.collection.get()
             if not data or 'metadatas' not in data:
                 return []
             
             doc_ids = set()
             for meta in data['metadatas']:
                 if meta and 'doc_id' in meta:
                     doc_ids.add(meta['doc_id'])
             return sorted(list(doc_ids))
         except Exception as e:
             self.logger.error(f"Failed to list documents: {e}")
             return []

