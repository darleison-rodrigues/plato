import logging
import sys
from pathlib import Path
from typing import List

from llama_index.core import (
    PropertyGraphIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.graph_stores.simple import SimplePropertyGraphStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from plato.config import get_config

logger = logging.getLogger(__name__)

class GraphRAGPipeline:
    """
    A simplified, local-first GraphRAG Pipeline using LlamaIndex.
    This pipeline uses file-based stores and local models, requiring no
    external databases or services.
    """
    
    def __init__(self, storage_dir: str = "./storage"):
        self.config = get_config()
        self.storage_dir = Path(storage_dir)
        self._setup_settings()
        self.index = self._initialize_index()

    def _setup_settings(self):
        """Configure global LLM and embedding models from config."""
        logger.info("Configuring global settings for LLM and embeddings...")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.config.ollama.embedding_model
        )
        # Use the high-quality extraction model for building the graph
        Settings.llm = Ollama(
            model=self.config.ollama.extraction_model, 
            base_url=self.config.ollama.base_url,
            request_timeout=self.config.ollama.timeout,
        )
        Settings.chunk_size = self.config.pipeline.chunk_size
        Settings.chunk_overlap = self.config.pipeline.chunk_overlap
    
    def _initialize_index(self) -> PropertyGraphIndex:
        """
        Initialize the PropertyGraphIndex.
        Loads from disk if available, otherwise creates a new one.
        """
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
            index = load_index_from_storage(storage_context)
            logger.info("Successfully loaded existing index from disk.")
            return index
        except Exception:
            logger.info("Could not load from disk. Creating new index.")
            storage_context = StorageContext.from_defaults(
                graph_store=SimplePropertyGraphStore(),
                vector_store=SimpleVectorStore(),
            )
            index = PropertyGraphIndex.from_documents(
                documents=[],
                storage_context=storage_context,
                show_progress=True
            )
            index.storage_context.persist(persist_dir=str(self.storage_dir))
            return index

    def insert_documents(self, documents: List):
        """Insert new documents into the graph index."""
        if not documents:
            logger.warning("No documents provided to insert.")
            return
            
        logger.info(f"Inserting {len(documents)} documents into the index...")
        for doc in documents:
            self.index.insert(doc)
        
        logger.info("Persisting index updates to disk...")
        self.index.storage_context.persist(persist_dir=str(self.storage_dir))

    def query(self, query_text: str) -> str:
        """
        Execute a GraphRAG query against the index.
        """
        logger.info(f"Querying: {query_text}")
        
        # For querying, we can switch to the faster chat model
        query_llm = Ollama(
            model=self.config.ollama.chat_model,
            base_url=self.config.ollama.base_url,
            request_timeout=self.config.ollama.timeout
        )

        query_engine = self.index.as_query_engine(
            llm=query_llm,
            include_text=True,
            similarity_top_k=3,
        )
        response = query_engine.query(query_text)
        return str(response)

# Testable Main Block
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    print("--- Test: Initializing GraphRAG Pipeline ---")
    try:
        pipeline = GraphRAGPipeline(storage_dir="./test_storage")
        print("✅ Pipeline Initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        raise
