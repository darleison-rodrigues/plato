import logging
import threading
import chromadb
import chromadb.errors as chroma_errors
from chromadb.api.types import EmbeddingFunction
from pathlib import Path
from typing import List, Dict, Any, Optional
from plato.core.errors import VectorStorageError
from plato.models.document import SearchResult

logger = logging.getLogger(__name__)

class VectorRetriever:
    """
    ChromaDB-based vector storage for document chunks.
    Safe for multi-threaded access.
    """
    COLLECTION_VERSION = "v1"
    
    _client_cache: Dict[str, chromadb.PersistentClient] = {}
    _cache_lock = threading.Lock()

    def __init__(
        self, 
        persist_dir: str = "~/.local/share/plato/vectors",
        embedding_fn: Optional[EmbeddingFunction] = None,
        collection_name: Optional[str] = None
    ):
        """
        Initialize the retriever.
        
        Args:
            persist_dir: Directory for ChromaDB persistence.
            embedding_fn: ChromaDB EmbeddingFunction. If None, uses default (not recommended for production).
            collection_name: Optional custom collection name.
        """
        try:
            self.persist_dir = str(Path(persist_dir).expanduser())
            
            # Thread-safe client connection pooling
            with self._cache_lock:
                if self.persist_dir not in self._client_cache:
                    self._client_cache[self.persist_dir] = chromadb.PersistentClient(
                        path=self.persist_dir
                    )
                self.client = self._client_cache[self.persist_dir]
            
            self.embedding_fn = embedding_fn
            
            actual_collection_name = collection_name or f"plato_docs_{self.COLLECTION_VERSION}"
            self.collection = self.client.get_or_create_collection(
                name=actual_collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Initialized VectorRetriever with collection: {actual_collection_name}")
            
        except chroma_errors.ChromaError as e:
            logger.error(f"ChromaDB initialization error: {e}")
            raise VectorStorageError(f"Failed to initialize vector storage: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during retriever init: {e}")
            raise

    def index_document(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Indexes or updates a single document chunk (idempotent)."""
        try:
            self.collection.upsert(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except chroma_errors.ChromaError as e:
            logger.error(f"Failed to index document {doc_id}: {e}")
            raise VectorStorageError(f"Vector storage error during indexing: {e}")

    def index_batch(
        self,
        doc_ids: List[str],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 100
    ):
        """Index multiple documents efficiently with batching."""
        if not (len(doc_ids) == len(contents) == len(metadatas)):
            raise ValueError("Length mismatch in batch inputs")
        
        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i:i + batch_size]
            batch_contents = contents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            try:
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_contents,
                    metadatas=batch_metadatas
                )
                logger.debug(f"Indexed batch of {len(batch_ids)} documents")
            except chroma_errors.ChromaError as e:
                logger.error(f"Batch indexing failed at position {i}: {e}")
                raise VectorStorageError(f"Vector storage error during batch indexing: {e}")

    def query(
        self, 
        text: str, 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Queries the vector database for similar chunks with optional filtering."""
        try:
            raw = self.collection.query(
                query_texts=[text],
                n_results=n_results,
                where=where
            )
            
            if not raw['ids'] or len(raw['ids']) == 0:
                return []
                
            ids = raw['ids'][0]
            documents = raw['documents'][0]
            metadatas = raw['metadatas'][0]
            distances = raw['distances'][0]
            
            return [
                SearchResult.from_raw(
                    doc_id=ids[i],
                    content=documents[i],
                    raw_metadata=metadatas[i],
                    distance=distances[i],
                    metric="cosine"  # Default in retriever
                )
                for i in range(len(ids))
            ]
        except chroma_errors.ChromaError as e:
            logger.error(f"Query failed: {e}")
            raise VectorStorageError(f"Vector storage query failed: {e}")

    def delete_pdf(self, pdf_hash: str) -> int:
        """Delete all chunks from a specific PDF."""
        try:
            results = self.collection.get(
                where={"pdf_hash": pdf_hash}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for PDF {pdf_hash}")
                return len(results['ids'])
            return 0
        except chroma_errors.ChromaError as e:
            logger.error(f"Failed to delete PDF {pdf_hash}: {e}")
            raise VectorStorageError(f"Failed to delete document data: {e}")

    def clear(self):
        """
        Wipes the entire collection.
        WARNING: This deletes the collection and recreates it. 
        References to the old collection object in other parts of the app will become stale.
        """
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection.name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.warning(f"Vector storage collection '{self.collection.name}' cleared and recreated")
        except chroma_errors.ChromaError as e:
            logger.error(f"Failed to clear collection: {e}")
            raise VectorStorageError(f"Failed to wipe vector storage: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Returns storage statistics."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection.name,
            "persist_dir": self.persist_dir
        }
