import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator

# Setup logging
logger = logging.getLogger(__name__)

class OllamaConfig(BaseModel):
    model: str = Field(default="llama3.2:latest")
    base_url: str = Field(default="http://localhost:11434")
    timeout: int = Field(default=60, ge=1, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)

    @validator('base_url')
    def validate_base_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('base_url must start with http:// or https://')
        return v.rstrip('/')

class ChromaConfig(BaseModel):
    persist_directory: str = Field(default="./chroma_db")
    collection_name: str = Field(default="pdf_documents")
    embedding_model: str = Field(default="nomic-embed-text:latest")

class PipelineConfig(BaseModel):
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    max_chunks_for_context: int = Field(default=5, ge=1, le=20)
    output_dir: str = Field(default="output")
    max_concurrent_pdfs: int = Field(default=3, ge=1, le=10)
    enable_visualization: bool = Field(default=True)

    @validator('chunk_overlap')
    def validate_overlap(cls, v, values):
        if 'chunk_size' in values and v >= values['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v

class Config(BaseModel):
    ollama: OllamaConfig
    chroma: ChromaConfig
    pipeline: PipelineConfig
    prompts: Dict[str, str] = Field(default_factory=dict)
    environment: str = Field(default="development")

class ConfigManager:
    """Thread-safe configuration manager with environment support"""
    _instance = None
    _config: Optional[Config] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def _get_env_overrides(self) -> Dict[str, Any]:
        """Extract configuration overrides from environment variables"""
        overrides = {}
        
        # Map env vars to config paths
        mappings = {
            "OLLAMA_MODEL": ["ollama", "model"],
            "OLLAMA_BASE_URL": ["ollama", "base_url"],
            "CHROMA_PERSIST_DIR": ["chroma", "persist_directory"],
            "PIPELINE_OUTPUT_DIR": ["pipeline", "output_dir"],
            "ENVIRONMENT": ["environment"]
        }
        
        for env_var, path in mappings.items():
            val = os.getenv(env_var)
            if val:
                # Nested dict update
                current = overrides
                for key in path[:-1]:
                    current = current.setdefault(key, {})
                current[path[-1]] = val
        return overrides

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                self._deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    def load_config(self, config_path: str = "config.yaml") -> Config:
        """Load configuration from YAML and environment"""
        # Search paths
        paths = [
            Path(config_path),
            Path(__file__).parent / config_path,
            Path("context") / config_path,
            Path.cwd() / "config.yaml"
        ]
        
        yaml_data = {}
        found_path = None
        
        for p in paths:
            if p.exists():
                found_path = p
                break
        
        if found_path:
            try:
                with open(found_path, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Failed to load config from {found_path}: {e}")
                raise
        else:
            logger.warning(f"Config file not found in {paths}. Using defaults/environment only.")

        # Apply environment overrides
        env_overrides = self._get_env_overrides()
        final_data = self._deep_merge(yaml_data, env_overrides)
        
        try:
            self._config = Config(**final_data)
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ValueError(f"Invalid configuration: {e}") from e
            
        return self._config

    @property
    def config(self) -> Config:
        if self._config is None:
            return self.load_config()
        return self._config

# Global accessor
_manager = ConfigManager()

def get_config(reload: bool = False) -> Config:
    if reload:
        return _manager.load_config()
    return _manager.config
