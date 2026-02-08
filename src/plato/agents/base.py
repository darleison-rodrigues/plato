import json
import logging
import ollama
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from plato.config import get_config

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all Platograph agents implementing the Contexter Pattern.
    Each agent is a context manager that acquires resources (LLM connection) 
    on entry and releases them on exit.
    """
    
    def __init__(self, model_override: Optional[str] = None):
        self.config = get_config()
        # Allow agents to use lighter/heavier models as needed
        self.model = model_override or self.config.ollama.model
        self.client = None
        self._host = self.config.ollama.base_url
        
    def __enter__(self):
        """Acquire resources (LLM connection)"""
        if not self.client:
            # We use the python library's synchronous client
            self.client = ollama.Client(host=self._host)
            # Verify connection (lightweight ping)
            try:
                self.client.list() 
            except Exception as e:
                logger.error(f"Failed to connect to Ollama at {self._host}: {e}")
                raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resources"""
        # Python's Ollama client is stateless/HTTP-based, so 'closing' 
        # just means nullifying the client ref to ensure GC.
        # In a C-based binding, we would explicitly free memory here.
        self.client = None
        if exc_type:
            logger.error(f"Agent error: {exc_val}")
        return False # Propagate exceptions

    def generate_json(self, prompt: str, system: str = "") -> Dict[str, Any]:
        """Helper to safely generate JSON from LLM"""
        if not self.client:
            raise RuntimeError("Agent must be used within a 'with' block.")
            
        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system + "\nRespond ONLY with valid JSON."},
                    {'role': 'user', 'content': prompt}
                ],
                format='json', # Force JSON mode (Ollama feature)
                options={'temperature': 0} # Deterministic
            )
            return json.loads(response['message']['content'])
        except json.JSONDecodeError:
            logger.error("LLM failed to produce valid JSON")
            return {}
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {}

    @abstractmethod
    def run(self, *args, **kwargs):
        """Main logic to be implemented by subclasses"""
        pass
