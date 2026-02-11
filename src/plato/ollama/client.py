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
        timeout_generate: float = 300.0,  # Increased to 5 minutes
        timeout_embed: float = 300.0,     # Increased to 5 minutes
        timeout_connect: float = 10.0,
        max_concurrent: int = 3
    ):
        self.base_url = base_url
        self.timeout_generate = timeout_generate
        self.timeout_embed = timeout_embed
        self.timeout_connect = timeout_connect
        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)

    def _create_http_client(self) -> httpx.AsyncClient:
        """Helper to create a configured HTTP client."""
        return httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout_generate, connect=self.timeout_connect, read=self.timeout_generate, write=self.timeout_generate, pool=self.timeout_connect),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            headers={"Content-Type": "application/json"}
        )
    
    async def __aenter__(self):
        """Standard usage via async context manager."""
        if not self._client:
            self._client = self._create_http_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _post(self, endpoint: str, payload: Dict[str, Any], timeout: httpx.Timeout) -> httpx.Response:
        """Internal POST method with retries for transient failures."""
        if not self._client:
            raise RuntimeError("OllamaClient must be used as an async context manager (async with).")
            
        url = f"/api/{endpoint}" # base_url is in the client
        try:
            return await self._client.post(url, json=payload, timeout=timeout)
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
        
        # Explicitly set all timeout parameters to avoid httpx error
        timeout = httpx.Timeout(
            self.timeout_generate, 
            connect=self.timeout_connect, 
            read=self.timeout_generate, 
            write=5.0, 
            pool=5.0
        )
        
        if not self._client:
            raise RuntimeError("OllamaClient must be used as an async context manager (async with).")
            
        async with self._semaphore:
            async with self._client.stream("POST", "/api/generate", json=payload, timeout=timeout) as response:
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
        payload = {"model": model, "prompt": prompt}
        timeout = httpx.Timeout(self.timeout_embed, connect=self.timeout_connect, read=self.timeout_embed, write=5.0, pool=5.0)
        
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
        tasks = [self.embeddings(model, prompt) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def check_models(self, required_models: List[str]) -> Dict[str, bool]:
        """
        Checks which of the required models are available on the Ollama host.
        """
        try:
            available = await self.list_models()
            available_names = {m.name for m in available}
            available_base = {m.name.split(":")[0] for m in available}
            
            results = {}
            for model in required_models:
                base = model.split(":")[0]
                results[model] = (model in available_names) or (base in available_base)
            
            return results
        except OllamaError as e:
            logger.error(f"Failed to check model availability: {e}")
            return {model: False for model in required_models}

    async def list_models(self) -> List[ModelInfo]:
        """List all downloaded models."""
        if not self._client:
            raise RuntimeError("OllamaClient must be used as an async context manager (async with).")
            
        async with self._semaphore:
            try:
                response = await self._client.get("/api/tags")
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
            except httpx.HTTPError as e:
                raise OllamaError(f"Ollama connection failed: {e}")
            except (json.JSONDecodeError, KeyError) as e:
                raise OllamaError(f"Failed to parse Ollama response: {e}")

    async def health_check(self) -> bool:
        """Quick check if Ollama is responsive."""
        if not self._client:
            return False
            
        try:
            response = await self._client.get("/", timeout=2.0)
            return response.status_code == 200
        except (httpx.HTTPError, asyncio.TimeoutError):
            return False

class OllamaEmbeddingFunction:
    """
    Adapter for ChromaDB to use our OllamaClient.
    """
    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: float = 300.0):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Synchronous wrapper for ChromaDB compatibility."""
        async def _get_embeddings():
            async with OllamaClient(base_url=self.base_url, timeout_embed=self.timeout) as client:
                return await client.embeddings_batch(self.model, input)
        
        try:
            # If we are already in an async loop, we can't use run_until_complete
            # or asyncio.run. We need a different approach for true async usage.
            # However, for ChromaDB's local indexing, it's often called from sync code.
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # In a TUI (running loop), we should ideally call embedding indexing asynchronously.
                # If we MUST do it sync, we use a thread.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    return executor.submit(lambda: asyncio.run(_get_embeddings())).result()
            
            # No running loop (CLI case), so we can use asyncio.run
            return asyncio.run(_get_embeddings())
        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"OllamaEmbeddingFunction failed: {e}")
            raise OllamaError(f"Embedding generation failed: {repr(e)}")
