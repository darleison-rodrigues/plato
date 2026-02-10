from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
import hashlib

def generate_chunk_id(content: str, metadata: "ChunkMetadata") -> str:
    """
    Generate deterministic chunk ID from content and metadata.
    Stable identifiers are critical for caching and idempotency.
    """
    # Use content and position to ensure stability across re-runs
    components = f"{metadata.pdf_hash}:{metadata.page}:{metadata.chunk_index}"
    return hashlib.sha256(components.encode()).hexdigest()[:16]

@dataclass(frozen=True)
class ChunkMetadata:
    """
    Immutable standardized metadata for document chunks.
    Frozen for safety as a value object.
    """
    pdf_hash: str
    page: int  # 1-indexed
    chunk_index: int  # 0-indexed within document
    pdf_title: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Validate metadata fields."""
        if not self.pdf_hash or len(self.pdf_hash) < 8:
            raise ValueError(f"Invalid pdf_hash: {self.pdf_hash}")
        if self.page < 1:
            raise ValueError(f"Page must be >= 1, got {self.page}")
        if self.chunk_index < 0:
            raise ValueError(f"chunk_index must be >= 0, got {self.chunk_index}")
        if self.char_start is not None and self.char_start < 0:
            raise ValueError("char_start must be >= 0")
        if self.char_end is not None and self.char_end < 0:
            raise ValueError("char_end must be >= 0")
        if (self.char_start is not None and self.char_end is not None 
            and self.char_end <= self.char_start):
            raise ValueError("char_end must be > char_start")
    
    @property
    def span_length(self) -> Optional[int]:
        """Character span length."""
        if self.char_start is not None and self.char_end is not None:
            return self.char_end - self.char_start
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary suitable for JSON/Storage."""
        return {
            "pdf_hash": self.pdf_hash,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "pdf_title": self.pdf_title,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkMetadata":
        """Deserialize from dictionary."""
        created_at_val = data.get("created_at")
        if isinstance(created_at_val, str):
            created_at = datetime.fromisoformat(created_at_val)
        else:
            created_at = datetime.now(timezone.utc)
            
        return cls(
            pdf_hash=data["pdf_hash"],
            page=int(data["page"]),
            chunk_index=int(data["chunk_index"]),
            pdf_title=data.get("pdf_title"),
            char_start=data.get("char_start"),
            char_end=data.get("char_end"),
            created_at=created_at
        )

@dataclass
class Chunk:
    """Document chunk with deterministic ID and embedding support."""
    content: str
    metadata: ChunkMetadata
    id: str = field(init=False)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Generate deterministic ID and validate content."""
        if not self.content or not self.content.strip():
            raise ValueError("Chunk content cannot be empty")
        self.id = generate_chunk_id(self.content, self.metadata)
    
    @property
    def char_count(self) -> int:
        return len(self.content)
    
    @property
    def embedding_dim(self) -> Optional[int]:
        return len(self.embedding) if self.embedding else None
    
    def to_dict(self, include_embedding: bool = True) -> Dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata.to_dict()
        }
        if include_embedding and self.embedding:
            result["embedding"] = self.embedding
        return result
    
    def __repr__(self) -> str:
        """Concise representation for logs (hiding massive binary data)."""
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        emb_info = f"[{self.embedding_dim} dims]" if self.embedding else "None"
        return f"Chunk(id={self.id!r}, page={self.metadata.page}, content={preview!r}, embedding={emb_info})"

class DistanceMetric(Enum):
    """Supported distance metrics for vector search."""
    COSINE = "cosine"
    EUCLIDEAN = "l2"  # ChromaDB uses 'l2'
    DOT_PRODUCT = "ip"

@dataclass
class SearchResult:
    """Unified search result object with multi-metric similarity calculation."""
    doc_id: str
    content: str
    metadata: ChunkMetadata
    distance: float
    metric: DistanceMetric = DistanceMetric.COSINE
    
    @property
    def similarity(self) -> float:
        """Convert distance to similarity score in range [0, 1]."""
        if self.metric == DistanceMetric.COSINE:
            # Cosine distance: [0, 2] -> similarity [1, 0]
            return max(0.0, min(1.0, 1 - (self.distance / 2)))
        elif self.metric == DistanceMetric.EUCLIDEAN:
            # Euclidean distance: [0, inf] -> similarity [1, 0]
            return 1 / (1 + self.distance)
        elif self.metric == DistanceMetric.DOT_PRODUCT:
            # Dot product similarity
            return max(0.0, min(1.0, self.distance))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
    
    @classmethod
    def from_raw(cls, doc_id: str, content: str, raw_metadata: Dict[str, Any], distance: float, metric: str = "cosine") -> "SearchResult":
        """Factory to build from Raw ChromaDB output."""
        # Map string metric to Enum
        m_enum = DistanceMetric.COSINE
        if metric == "l2": m_enum = DistanceMetric.EUCLIDEAN
        elif metric == "ip": m_enum = DistanceMetric.DOT_PRODUCT
        
        return cls(
            doc_id=doc_id,
            content=content,
            metadata=ChunkMetadata.from_dict(raw_metadata),
            distance=distance,
            metric=m_enum
        )

    def __lt__(self, other: "SearchResult") -> bool:
        """Comparison by similarity (higher similarity is 'less than' for reverse sorting)."""
        return self.similarity > other.similarity

@dataclass
class DocumentMetadata:
    """Structured document-level metadata."""
    title: str = "Untitled"
    author: Optional[str] = None
    page_count: int = 0
    file_size_bytes: int = 0
    created_date: Optional[datetime] = None
    language: str = "en"
    custom: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Document:
    """Root document object holding metadata and chunks."""
    filepath: Path
    pdf_hash: str
    chunks: List[Chunk] = field(default_factory=list)
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    
    def __post_init__(self):
        """Normalize filepath and validate."""
        if isinstance(self.filepath, str):
            self.filepath = Path(self.filepath)
        self.filepath = self.filepath.expanduser()
        # Existence check is handled during processing, 
        # but normalize for stability.
    
    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
    
    def add_chunk(self, chunk: Chunk):
        """Add chunk with validation."""
        if chunk.metadata.pdf_hash != self.pdf_hash:
            raise ValueError("Chunk does not belong to this document (hash mismatch)")
        self.chunks.append(chunk)
