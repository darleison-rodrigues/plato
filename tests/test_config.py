import os
import unittest
from plato.config import ConfigManager, Config

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

    def test_production_inheritance(self):
        """Test production environment inheritance"""
        os.environ["ENVIRONMENT"] = "production"
        
        # Clear cache/instance to force reload
        ConfigManager._instance = None
        self.manager = ConfigManager()
        
        config = self.manager.load_config()
        
        self.assertEqual(config.environment, "production")
        
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
