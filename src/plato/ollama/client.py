import asyncio
import json
import logging
import time
import httpx
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from dataclasses import dataclass, field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from plato.core.errors import OllamaError, ModelNotFoundError

logger = logging.getLogger(__name__)

@dataclass
class GenerationMetrics:
    """Performance metrics for a generation."""
    model: str
    total_duration_ns: int = 0
    load_duration_ns: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration_ns: int = 0
    eval_count: int = 0
    eval_duration_ns: int = 0
    
    @property
    def total_duration_seconds(self) -> float:
        return self.total_duration_ns / 1e9
    
    @property
    def tokens_per_second(self) -> float:
        if self.eval_duration_ns == 0:
            return 0
        return self.eval_count / (self.eval_duration_ns / 1e9)

@dataclass
class ModelInfo:
    """Ollama model metadata."""
    name: str
    size: int
    digest: str
    modified_at: str
    details: Dict[str, Any] = field(default_factory=dict)

class OllamaClient:
    """
    Production-grade Ollama API client.
    Implements connection pooling, streaming, batching, and robust error handling.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_generate: float = 120.0,
        timeout_embed: float = 30.0,
        timeout_connect: float = 5.0,
        max_concurrent: int = 3
    ):
        self.base_url = base_url
        self.timeout_generate = timeout_generate
        self.timeout_embed = timeout_embed
        self.timeout_connect = timeout_connect
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def __aenter__(self):
        """Context manager support (preferred usage)."""
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_generate, connect=self.timeout_connect),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the internal client, creating it if it doesn't exist."""
        if self._client is None:
            # Auto-init if not using context manager, though discouraged
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_generate, connect=self.timeout_connect),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
            )
        return self._client

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _post(self, endpoint: str, payload: Dict[str, Any], timeout: httpx.Timeout) -> httpx.Response:
        """Internal POST method with retries for transient failures."""
        url = f"{self.base_url}/api/{endpoint}"
        try:
            return await self.client.post(url, json=payload, timeout=timeout)
        except httpx.ConnectError:
            raise OllamaError(f"Could not connect to Ollama at {self.base_url}. Is it running?")

    async def generate_stream(
        self,
        model: str,
        prompt: str,
        system: str = "",
        options: Dict[str, Any] = None,
        keep_alive: str = "5m"
    ) -> AsyncGenerator[Union[str, GenerationMetrics], None]:
        """Stream tokens as they're generated."""
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system,
            "stream": True,
            "keep_alive": keep_alive,
            "options": options or {}
        }
        
        timeout = httpx.Timeout(connect=self.timeout_connect, read=None, write=5.0, pool=5.0)
        
        async with self._semaphore:
            async with self.client.stream("POST", f"{self.base_url}/api/generate", json=payload, timeout=timeout) as response:
                if response.status_code == 404:
                    raise ModelNotFoundError(f"Model '{model}' not found.")
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        if "error" in chunk:
                            raise OllamaError(f"Ollama error: {chunk['error']}")
                        
                        if token := chunk.get("response"):
                            yield token
                            
                        if chunk.get("done"):
                            metrics = GenerationMetrics(
                                model=model,
                                total_duration_ns=chunk.get("total_duration", 0),
                                load_duration_ns=chunk.get("load_duration", 0),
                                prompt_eval_count=chunk.get("prompt_eval_count", 0),
                                prompt_eval_duration_ns=chunk.get("prompt_eval_duration", 0),
                                eval_count=chunk.get("eval_count", 0),
                                eval_duration_ns=chunk.get("eval_duration", 0)
                            )
                            yield metrics
                            break
                    except json.JSONDecodeError:
                        continue

    async def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        options: Dict[str, Any] = None,
        keep_alive: str = "5m"
    ) -> str:
        """Non-streaming generation. Blocks until complete."""
        text = ""
        async for chunk in self.generate_stream(model, prompt, system, options, keep_alive):
            if isinstance(chunk, str):
                text += chunk
        return text

    async def embeddings(self, model: str, prompt: str) -> List[float]:
        """Generate embedding for a single text."""
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": model, "prompt": prompt}
        timeout = httpx.Timeout(connect=self.timeout_connect, read=self.timeout_embed)
        
        async with self._semaphore:
            try:
                response = await self._post("embeddings", payload, timeout)
                if response.status_code == 404:
                    raise ModelNotFoundError(f"Model '{model}' not found.")
                response.raise_for_status()
                data = response.json()
                if "embedding" not in data:
                    raise OllamaError(f"Unexpected embedding response: {data}")
                return data["embedding"]
            except httpx.HTTPStatusError as e:
                raise OllamaError(f"Ollama embedding API error: {e}")

    async def embeddings_batch(
        self,
        model: str,
        prompts: List[str],
        batch_size: int = 10
    ) -> List[List[float]]:
        """Process multiple embedding requests efficiently."""
        # Ollama currently doesn't have a single-call batch embedding API that is standard,
        # so we gather multiple individual calls while respecting the semaphore.
        tasks = [self.embeddings(model, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def list_models(self) -> List[ModelInfo]:
        """List all downloaded models."""
        async with self._semaphore:
            try:
                response = await self.client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [
                    ModelInfo(
                        name=m["name"],
                        size=m["size"],
                        digest=m["digest"],
                        modified_at=m["modified_at"],
                        details=m.get("details", {})
                    )
                    for m in data.get("models", [])
                ]
            except Exception as e:
                raise OllamaError(f"Failed to list models: {e}")

    async def health_check(self) -> bool:
        """Quick check if Ollama is responsive."""
        try:
            response = await self.client.get(f"{self.base_url}/", timeout=2.0)
            return response.status_code == 200
        except:
            return False
