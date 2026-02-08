"""
ChromaDB vector store for document chunks and retrieval
"""
from typing import List, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from .config import get_config
from .llm import OllamaClient


class VectorStore:
    """Manages document storage and retrieval using ChromaDB"""
    
    def __init__(self):
        self.config = get_config().chroma
        self.pipeline_config = get_config().pipeline
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.config.collection_name,
            metadata={"description": "PDF document chunks"}
        )
        
        # Initialize LLM client for embeddings
        self.llm_client = OllamaClient()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        chunk_size = self.pipeline_config.chunk_size
        overlap = self.pipeline_config.chunk_overlap
        
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size * 0.7:  # At least 70% of chunk
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return chunks
    
    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any] = None):
        """
        Add a document to the vector store
        
        Args:
            doc_id: Unique identifier for the document
            text: Full text content
            metadata: Additional metadata (title, source, etc.)
        """
        chunks = self._chunk_text(text)
        
        if not chunks:
            print(f"Warning: No chunks created for document {doc_id}")
            return
        
        print(f"Adding {len(chunks)} chunks for document {doc_id}...")
        
        # Prepare data for batch insertion
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            if metadata:
                chunk_metadata.update(metadata)
            metadatas.append(chunk_metadata)
        
        # Add to collection (Chroma handles embeddings via embedding function)
        # For Ollama embeddings, we need to generate them manually
        embeddings = []
        for chunk in chunks:
            embedding = self.llm_client.generate_embedding(chunk)
            embeddings.append(embedding)
        
        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
            embeddings=embeddings if embeddings and embeddings[0] else None
        )
        
        print(f"✓ Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, n_results: int = None) -> Dict[str, Any]:
        """
        Search for relevant document chunks
        
        Args:
            query: Search query
            n_results: Number of results to return (default from config)
        
        Returns:
            Dictionary with documents, metadatas, and distances
        """
        if n_results is None:
            n_results = self.pipeline_config.max_chunks_for_context
        
        # Generate embedding for query
        query_embedding = self.llm_client.generate_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding] if query_embedding else None,
            query_texts=[query] if not query_embedding else None,
            n_results=n_results
        )
        
        return results
    
    def get_document_chunks(self, doc_id: str) -> List[str]:
        """Retrieve all chunks for a specific document"""
        results = self.collection.get(
            where={"doc_id": doc_id}
        )
        
        if not results['documents']:
            return []
        
        # Sort by chunk index
        chunks_with_index = zip(
            results['documents'],
            [m['chunk_index'] for m in results['metadatas']]
        )
        sorted_chunks = sorted(chunks_with_index, key=lambda x: x[1])
        
        return [chunk for chunk, _ in sorted_chunks]
    
    def delete_document(self, doc_id: str):
        """Remove all chunks for a document"""
        self.collection.delete(
            where={"doc_id": doc_id}
        )
        print(f"✓ Deleted document {doc_id} from vector store")
    
    def list_documents(self) -> List[str]:
        """Get list of all document IDs in the store"""
        all_metadata = self.collection.get()['metadatas']
        doc_ids = set(m['doc_id'] for m in all_metadata)
        return sorted(list(doc_ids))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        all_data = self.collection.get()
        doc_ids = set(m['doc_id'] for m in all_data['metadatas'])
        
        return {
            "total_chunks": len(all_data['documents']),
            "total_documents": len(doc_ids),
            "collection_name": self.config.collection_name
        }
