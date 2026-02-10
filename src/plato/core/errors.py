"""
PLATO Custom Exceptions
"""

class PlatoError(Exception):
    """Base class for all Plato exceptions."""
    pass

class OllamaError(PlatoError):
    """Raised when Ollama API fails or is unreachable."""
    pass

class ModelNotFoundError(OllamaError):
    """Raised when the requested model is not found in Ollama."""
    pass

class PDFProcessingError(PlatoError):
    """Raised when PDF parsing or text extraction fails."""
    pass

class ChunkingError(PlatoError):
    """Raised when content chunking fails due to size or logic errors."""
    pass

class VectorStorageError(PlatoError):
    """Raised when ChromaDB or vector store operations fail."""
    pass

class TemplateRenderError(PlatoError):
    """Raised when Jinja2 rendering fails or triggers sandbox violations."""
    pass

class ValidationError(PlatoError):
    """Raised when LLM output fails JSON schema validation."""
    pass
