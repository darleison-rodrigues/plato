import logging
import threading
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from plato.core.errors import VectorStorageError
from plato.models.document import SearchResult

logger = logging.getLogger(__name__)

class VectorRetriever:
    """
    Lightweight NumPy-based vector storage for document chunks.
    Compatible with Python 3.14+ and optimized for local scholarship.
    """
    
    def __init__(
        self, 
        persist_dir: str = "~/.local/share/plato/vectors",
        embedding_fn: Any = None,
        collection_name: Optional[str] = None
    ):
        self.persist_dir = str(Path(persist_dir).expanduser())
        self.collection_name = collection_name or "default"
        self.embedding_fn = embedding_fn
        
        self.vectors_path = os.path.join(self.persist_dir, f"{self.collection_name}_vectors.npy")
        self.metadata_path = os.path.join(self.persist_dir, f"{self.collection_name}_metadata.json")
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self._lock = threading.Lock()
        self._vectors: Optional[np.ndarray] = None
        self._metadata: List[Dict[str, Any]] = []
        self._ids: List[str] = []
        
        self._load()

    def _load(self):
        """Load storage from disk."""
        with self._lock:
            if os.path.exists(self.vectors_path) and os.path.exists(self.metadata_path):
                try:
                    self._vectors = np.load(self.vectors_path)
                    with open(self.metadata_path, 'r') as f:
                        data = json.load(f)
                        self._metadata = data.get("metadata", [])
                        self._ids = data.get("ids", [])
                    logger.info(f"Loaded {len(self._ids)} vectors from disk.")
                except Exception as e:
                    logger.error(f"Failed to load vector storage: {e}")
                    self._vectors = None
                    self._metadata = []
                    self._ids = []

    def _save(self):
        """Save storage to disk."""
        if self._vectors is None:
            return
            
        try:
            np.save(self.vectors_path, self._vectors)
            with open(self.metadata_path, 'w') as f:
                json.dump({
                    "ids": self._ids,
                    "metadata": self._metadata
                }, f)
        except Exception as e:
            logger.error(f"Failed to save vector storage: {e}")
            raise VectorStorageError(f"Persistence failure: {e}")

    def index_document(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Indexes a single document. Requires embedding_fn to be set."""
        self.index_batch([doc_id], [content], [metadata])

    def index_batch(
        self,
        doc_ids: List[str],
        contents: List[str],
        metadatas: List[Dict[str, Any]],
        batch_size: int = 100
    ):
        if not self.embedding_fn:
            raise VectorStorageError("Embedding function not configured.")

        # In a real sync call from TUI, we expect embedding_fn to handle the batching/loop
        embeddings = self.embedding_fn(contents)
        new_vectors = np.array(embeddings, dtype=np.float32)

        with self._lock:
            for i, d_id in enumerate(doc_ids):
                if d_id in self._ids:
                    # Update existing
                    idx = self._ids.index(d_id)
                    if self._vectors is not None:
                        self._vectors[idx] = new_vectors[i]
                    self._metadata[idx] = {**metadatas[i], "content": contents[i]}
                else:
                    # Append new
                    self._ids.append(d_id)
                    self._metadata.append({**metadatas[i], "content": contents[i]})
                    if self._vectors is None:
                        self._vectors = new_vectors[i:i+1]
                    else:
                        self._vectors = np.vstack([self._vectors, new_vectors[i:i+1]])
            
            self._save()
            logger.info(f"Indexed {len(doc_ids)} items. Total: {len(self._ids)}")

    def query(
        self, 
        text: str, 
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if self._vectors is None or len(self._ids) == 0:
            return []

        if not self.embedding_fn:
             raise VectorStorageError("Embedding function not configured.")

        embeddings = self.embedding_fn([text])
        try:
            query_vec = np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.error(f"Error creating query vector: {e}")
            raise

        with self._lock:
            # Simple cosine similarity: (A . B) / (||A|| * ||B||)
            # Normalize vectors for fast cosine via dot product
            if self._vectors is None:
                 return []
            
            # Manual norm calculation to avoid potential BLAS deadlocks on M1/Textual
            # norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            norms = np.sqrt(np.sum(self._vectors**2, axis=1, keepdims=True))
            normalized_vectors = self._vectors / (norms + 1e-9)
            
            # q_norm = np.linalg.norm(query_vec)
            q_norm = float(np.sqrt(np.sum(query_vec**2)))
            normalized_q = query_vec / (q_norm + 1e-9)
            
            similarities = np.dot(normalized_vectors, normalized_q)
            
            # Apply filters if any (very basic 'where' support)
            mask = np.ones(len(self._ids), dtype=bool)
            if where:
                for idx, meta in enumerate(self._metadata):
                    for key, val in where.items():
                        if meta.get(key) != val:
                            mask[idx] = False
                            break
            
            # Get top indices
            indices = np.where(mask)[0]
            if len(indices) == 0:
                return []
                
            filtered_sims = similarities[indices]
            top_k_indices = indices[np.argsort(filtered_sims)[::-1][:n_results]]
            
            results = []
            for idx in top_k_indices:
                meta = self._metadata[idx].copy()
                content = meta.pop("content", "")
                results.append(SearchResult.from_raw(
                    doc_id=self._ids[idx],
                    content=content,
                    raw_metadata=meta,
                    distance=float(1.0 - similarities[idx]), # Use distance (1 - sim)
                    metric="cosine"
                ))
            return results

    def delete_pdf(self, pdf_hash: str) -> int:
        with self._lock:
            indices_to_keep = [
                i for i, m in enumerate(self._metadata) 
                if m.get("pdf_hash") != pdf_hash
            ]
            deleted_count = len(self._ids) - len(indices_to_keep)
            
            if deleted_count > 0:
                self._ids = [self._ids[i] for i in indices_to_keep]
                self._metadata = [self._metadata[i] for i in indices_to_keep]
                if self._vectors is not None:
                    self._vectors = self._vectors[indices_to_keep]
                self._save()
                
            return deleted_count

    def clear(self):
        with self._lock:
            self._vectors = None
            self._metadata = []
            self._ids = []
            if os.path.exists(self.vectors_path): os.remove(self.vectors_path)
            if os.path.exists(self.metadata_path): os.remove(self.metadata_path)
            logger.warning("Vector storage cleared.")

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self._ids),
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir
        }
