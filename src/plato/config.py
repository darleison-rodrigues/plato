import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

# Setup logging
logger = logging.getLogger(__name__)

import platform
import psutil

# Load models.yaml as the source of truth for model naming
MODELS_YAML_PATH = Path(__file__).parent / "core" / "models.yaml"

def load_models_config() -> Dict[str, Any]:
    """Loads standardized model definitions from YAML."""
    if MODELS_YAML_PATH.exists():
        with open(MODELS_YAML_PATH, 'r') as f:
            return yaml.safe_load(f)
    logger.warning(f"models.yaml not found at {MODELS_YAML_PATH}. Using fallback defaults.")
    return {}

def detect_hardware_profile() -> str:
    """Auto-detect and return appropriate hardware profile."""
    try:
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu = platform.processor() or ""
        
        # Mac M1/M2/M3 detection
        if platform.system() == "Darwin" and ("Apple" in cpu or platform.machine() == 'arm64'):
            # On M1/8GB, we must be conservative to avoid swap death
            return "m1_8gb"
                
        # Intel/AMD or high-RAM Macs
        if ram_gb >= 15:
            return "performance"
        return "balanced"
    except Exception as e:
        logger.warning(f"Hardware detection failed: {e}. Defaulting to 'balanced' profile.")
        return "balanced"

class OllamaConfig(BaseModel):
    """
    Configuration for Ollama interacting, driven by hardware profiles.
    """
    base_url: str = Field(default_factory=lambda: os.getenv('OLLAMA_HOST', "http://localhost:11434"))
    active_profile: str = Field(default_factory=detect_hardware_profile)
    
    # Model selections based on profile
    embedding_model: str = Field(default="embeddinggemma:latest")
    ocr_model: str = Field(default="deepseek-ocr:3b")
    reasoning_model: str = Field(default="qwen2.5-coder:1.5b")
    
    timeout: int = Field(default=120, ge=30)
    max_retries: int = Field(default=3, ge=1, le=5)

    def __init__(self, **data):
        super().__init__(**data)
        # Load from models.yaml to override defaults
        m_cfg = load_models_config()
        if m_cfg and "profiles" in m_cfg:
            profile_data = m_cfg["profiles"].get(self.active_profile, m_cfg["profiles"].get("balanced", {}))
            if profile_data:
                self.embedding_model = profile_data.get("embedding", self.embedding_model)
                self.ocr_model = profile_data.get("ocr", self.ocr_model)
                self.reasoning_model = profile_data.get("reasoning", self.reasoning_model)
                logger.info(f"Loaded models for profile '{self.active_profile}': {self.reasoning_model}, {self.ocr_model}")

    def get_model_name(self, task: str) -> str:
        """Helper to get standardized model name for a task."""
        if task == "embedding": return self.embedding_model
        if task == "ocr": return self.ocr_model
        return self.reasoning_model

class PipelineConfig(BaseModel):
    chunk_size: int = Field(default=512, ge=100, le=4096)
    chunk_overlap: int = Field(default=50, ge=0, le=1024)
    output_dir: str = Field(default="output")

    @field_validator('chunk_overlap')
    def validate_overlap(cls, v, info):
        if 'chunk_size' in info.data and v >= info.data['chunk_size']:
            raise ValueError('chunk_overlap must be less than chunk_size')
        return v

class KnowledgeGraphConfig(BaseModel):
    max_visualization_nodes: int = Field(default=100, ge=10, le=1000)

class Config(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    knowledge_graph: KnowledgeGraphConfig = Field(default_factory=KnowledgeGraphConfig)
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
                current = overrides
                for key in path[:-1]:
                    current = current.setdefault(key, {})
                current[path[-1]] = val
        return overrides

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        if not isinstance(base, dict) or not isinstance(override, dict):
            return override
            
        merged = base.copy()
        for k, v in override.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = self._deep_merge(merged[k], v)
            else:
                merged[k] = v
        return merged

    def load_config(self, config_path: str = "config.yaml") -> Config:
        """Load configuration from YAML (base + env) and environment variables"""
        # Determine environment
        env = os.getenv("ENVIRONMENT", "development")
        
        # Load YAML
        paths = [
            Path(config_path),
            Path(__file__).parent / config_path,
            Path(__file__).parents[2] / config_path, # Project root
            Path("context") / config_path,
            Path.cwd() / "config.yaml"
        ]
        
        yaml_data = {}
        found_path = None
        for p in paths:
            if p.exists():
                found_path = p
                break
        
        base_config = {}
        env_config = {}
        
        if found_path:
            try:
                with open(found_path, 'r', encoding='utf-8') as f:
                    full_yaml = yaml.safe_load(f) or {}
                    base_config = full_yaml.get('_base', {})
                    env_config = full_yaml.get(env, {})
            except Exception as e:
                logger.error(f"Failed to load config from {found_path}: {e}")
                raise
        else:
            logger.warning("Config file not found. Using API defaults.")

        # Merge sequence: Base -> Env Section -> Env Vars
        config_data = self._deep_merge(base_config, env_config)
        env_overrides = self._get_env_overrides()
        final_data = self._deep_merge(config_data, env_overrides)
        
        # Ensure 'environment' is set in the config object
        final_data['environment'] = env

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
