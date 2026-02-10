import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any
from pathlib import Path

class HybridRetriever:
    """
    Handles document indexing and retrieval using ChromaDB.
    """
    def __init__(self, persist_dir: str = "~/.local/share/plato/vectors"):
        self.persist_dir = str(Path(persist_dir).expanduser())
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(name="plato_docs")

    def index_document(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """
        Indexes a document chunk.
        """
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[doc_id]
        )

    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Searches for relevant document chunks.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
