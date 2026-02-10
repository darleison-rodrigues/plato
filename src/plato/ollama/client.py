import httpx
from typing import Dict, Any, AsyncGenerator

class OllamaClient:
    """
    Client for interacting with the Ollama API.
    """
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    async def generate(self, model: str, prompt: str, system: str = "", options: Dict[str, Any] = None) -> str:
        """
        Generates a completion from Ollama.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": options or {}
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()["response"]

    async def embeddings(self, model: str, prompt: str) -> list[float]:
        """
        Generates embeddings for a given prompt.
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": prompt
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()["embedding"]
