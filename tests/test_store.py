import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Improve path handling to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies before importing store
sys.modules['chromadb'] = MagicMock()
sys.modules['chromadb.config'] = MagicMock()
sys.modules['ollama'] = MagicMock()

# Now we can import
from plato.store import VectorStore

class TestVectorStore(unittest.TestCase):
    def setUp(self):
        self.mock_config_get = patch('plato.store.get_config').start()
        self.mock_ollama = patch('plato.store.OllamaClient').start()
        
        # Setup mock config
        self.mock_config = MagicMock()
        self.mock_config.chroma.persist_directory = "./test_db"
        self.mock_config.chroma.collection_name = "test_collection"
        self.mock_config.pipeline.chunk_size = 100
        self.mock_config.pipeline.chunk_overlap = 10
        self.mock_config.pipeline.max_chunks_for_context = 5
        self.mock_config_get.return_value = self.mock_config
        
        # Initialize store
        with patch('plato.store.chromadb.PersistentClient') as mock_client:
            self.store = VectorStore()
            self.mock_collection = mock_client.return_value.get_or_create_collection.return_value

    def tearDown(self):
        patch.stopall()

    def test_chunk_text(self):
        """Test text chunking logic"""
        text = "Hello world. " * 20 # 260 chars
        # chunk_size is 100
        chunks = self.store._chunk_text(text)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(c) <= 100 for c in chunks))

    def test_add_documents_batch(self):
        """Test batch addition"""
        docs = [
            {"doc_id": "doc1", "text": "Content 1", "metadata": {"source": "test"}},
            {"doc_id": "doc2", "text": "Content 2", "metadata": {"source": "test"}}
        ]
        
        # Mock embedding generation
        self.store.llm_client.generate_embedding.return_value = [0.1] * 768
        
        result = self.store.add_documents_batch(docs, batch_size=1)
        
        self.assertEqual(result['processed'], 2)
        # Should have called add twice (since batch_size=1 and 2 docs)
        self.assertEqual(self.mock_collection.add.call_count, 2)

    def test_search_robustness(self):
        """Test search with empty query and failed embedding"""
        # Empty query
        res = self.store.search("")
        self.assertEqual(res['documents'], [[]])
        
        # Failed embedding
        self.store.llm_client.generate_embedding.return_value = None
        res = self.store.search("valid query")
        self.assertEqual(res['documents'], [[]])

if __name__ == '__main__':
    unittest.main()
