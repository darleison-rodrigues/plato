import os
import unittest
from contexter.config import ConfigManager, Config

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Reset singleton for tests
        ConfigManager._instance = None
        self.manager = ConfigManager()

    def test_default_config(self):
        """Test loading defaults when no file is present"""
        # We assume config.yaml exists in repo, so this might load it.
        # But let's check basic structure
        config = self.manager.config
        self.assertIsNotNone(config.ollama.model)
        self.assertIsNotNone(config.pipeline.output_dir)

    def test_env_override(self):
        """Test environment variable overrides"""
        os.environ["OLLAMA_MODEL"] = "test-model-env"
        os.environ["PIPELINE_OUTPUT_DIR"] = "env-output"
        
        # Reload config
        config = self.manager.load_config()
        
        self.assertEqual(config.ollama.model, "test-model-env")
        self.assertEqual(config.pipeline.output_dir, "env-output")
        
        # Cleanup
        del os.environ["OLLAMA_MODEL"]
        del os.environ["PIPELINE_OUTPUT_DIR"]

    def test_validation(self):
        """Test URL validation"""
        os.environ["OLLAMA_BASE_URL"] = "ftp://invalid-url"
        with self.assertRaises(ValueError):
            self.manager.load_config()
        del os.environ["OLLAMA_BASE_URL"]

if __name__ == '__main__':
    unittest.main()
