import os
import unittest
from contexter.config import ConfigManager, Config

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Reset singleton for tests
        ConfigManager._instance = None
        self.manager = ConfigManager()

    def test_default_development_config(self):
        """Test loading default (development) environment"""
        if "ENVIRONMENT" in os.environ:
            del os.environ["ENVIRONMENT"]
            
        config = self.manager.load_config()
        self.assertEqual(config.environment, "development")
        # pipeline value should be merged from development section
        # We manually set max_concurrent_pdfs to 2 in development in config.yaml
        self.assertEqual(config.pipeline.max_concurrent_pdfs, 2)

    def test_production_inheritance(self):
        """Test production environment inheritance"""
        os.environ["ENVIRONMENT"] = "production"
        
        # Clear cache/instance to force reload
        ConfigManager._instance = None
        self.manager = ConfigManager()
        
        config = self.manager.load_config()
        
        self.assertEqual(config.environment, "production")
        # Should override base default (3) and dev (2) with prod value (6)
        self.assertEqual(config.pipeline.max_concurrent_pdfs, 6)
        # Should enable_visualization=False in prod
        self.assertFalse(config.pipeline.enable_visualization)
        
        del os.environ["ENVIRONMENT"]

    def test_env_override_priority(self):
        """Test env var overrides YAML settings"""
        os.environ["ENVIRONMENT"] = "production"
        os.environ["PIPELINE_OUTPUT_DIR"] = "env-override-output"
        
        ConfigManager._instance = None
        self.manager = ConfigManager()
        
        config = self.manager.load_config()
        
        self.assertEqual(config.pipeline.output_dir, "env-override-output")
        
        del os.environ["ENVIRONMENT"]
        del os.environ["PIPELINE_OUTPUT_DIR"]

if __name__ == '__main__':
    unittest.main()
