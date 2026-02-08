import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import Dict, Any, Optional

class OllamaConfig(BaseModel):
    model: str = "llama3.2:latest"
    base_url: str = "http://localhost:11434"

class ChromaConfig(BaseModel):
    persist_directory: str = "./chroma_db"
    collection_name: str = "pdf_documents"

class PipelineConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_for_context: int = 5
    output_dir: str = "output"

class Config(BaseModel):
    ollama: OllamaConfig
    chroma: ChromaConfig
    pipeline: PipelineConfig
    prompts: Dict[str, str]

_config = None

def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file"""
    global _config
    
    # Try local path first, then parent if needed (for scripts in subdirs)
    paths_to_try = [
        Path(config_path),
        Path(__file__).parent / config_path,
        Path("context") / config_path
    ]
    
    final_path = None
    for p in paths_to_try:
        if p.exists():
            final_path = p
            break
            
    if not final_path:
        raise FileNotFoundError(f"Configuration file {config_path} not found")
        
    with open(final_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        
    _config = Config(**data)
    return _config

def get_config() -> Config:
    """Get the singleton configuration object"""
    global _config
    if _config is None:
        return load_config()
    return _config
