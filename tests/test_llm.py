import unittest
from unittest.mock import patch, MagicMock
from plato.llm import OllamaClient
from plato.config import get_config, Config, OllamaConfig, PipelineConfig, KnowledgeGraphConfig

class TestOllamaClient(unittest.TestCase):

    @patch('plato.llm.get_config')
    def test_init_success(self, mock_get_config):
        """Test successful initialization with valid config."""
        mock_config = Config(
            ollama=OllamaConfig(),
            pipeline=PipelineConfig(),
            knowledge_graph=KnowledgeGraphConfig(),
            prompts={'entity_extraction': 'prompt1', 'relation_extraction': 'prompt2'}
        )
        mock_get_config.return_value = mock_config
        
        try:
            client = OllamaClient()
            self.assertIsNotNone(client)
        except Exception as e:
            self.fail(f"Initialization failed with valid config: {e}")

    @patch('plato.llm.get_config')
    def test_init_missing_prompt(self, mock_get_config):
        """Test that initialization fails if a required prompt is missing."""
        mock_config = Config(
            ollama=OllamaConfig(),
            pipeline=PipelineConfig(),
            knowledge_graph=KnowledgeGraphConfig(),
            prompts={'entity_extraction': 'prompt1'} # Missing relation_extraction
        )
        mock_get_config.return_value = mock_config
        
        with self.assertRaises(ValueError):
            OllamaClient()

    def test_context_manager(self):
        """Test that the client can be used as a context manager."""
        with patch('plato.llm.get_config') as mock_get_config:
            mock_get_config.return_value = Config(
                ollama=OllamaConfig(),
                pipeline=PipelineConfig(),
                knowledge_graph=KnowledgeGraphConfig(),
                prompts={'entity_extraction': 'p1', 'relation_extraction': 'p2'}
            )
            try:
                with OllamaClient() as client:
                    self.assertIsInstance(client, OllamaClient)
                # Should exit without error
            except Exception as e:
                self.fail(f"Context manager failed: {e}")

    @patch('plato.llm.ollama.chat')
    @patch('plato.llm.get_config')
    def test_extract_entities_includes_doc_id(self, mock_get_config, mock_chat):
        """Test that extract_entities returns a dict with the document_id."""
        mock_chat.return_value = {'message': {'content': '```json\n{"entities": {"PERSON": ["Alice"]}}\n```'}}
        
        mock_get_config.return_value = Config(
            ollama=OllamaConfig(),
            pipeline=PipelineConfig(),
            knowledge_graph=KnowledgeGraphConfig(),
            prompts={'entity_extraction': 'p1', 'relation_extraction': 'p2'}
        )
        client = OllamaClient()
        result = client.extract_entities("some text", document_id="doc123")
        self.assertIn('document_id', result)
        self.assertEqual(result['document_id'], "doc123")
        self.assertIn('entities', result)

if __name__ == '__main__':
    unittest.main()
