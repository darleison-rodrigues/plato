from typing import List, Optional, Dict, Any, ClassVar
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
import hashlib

from pydantic import BaseModel, Field, computed_field, field_validator, ConfigDict

# Shared optimization: fast datetime default
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)

class ChunkMetadata(BaseModel):
    """Immutable standardized metadata for document chunks."""
    model_config = ConfigDict(frozen=True)
    
    pdf_hash: str = Field(min_length=8)
    page: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    pdf_title: Optional[str] = None
    char_start: Optional[int] = Field(None, ge=0)
    char_end: Optional[int] = Field(None, ge=0)
    created_at: datetime = Field(default_factory=_utc_now)
    
    @field_validator('char_end')
    @classmethod
    def check_char_range(cls, v, info):
        if v is not None and info.data.get('char_start') is not None:
            if v <= info.data['char_start']:
                raise ValueError('char_end must be > char_start')
        return v
    
    @computed_field
    @property
    def span_length(self) -> Optional[int]:
        if self.char_start is not None and self.char_end is not None:
            return self.char_end - self.char_start
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Compatibility alias for model_dump (json)."""
        return self.model_dump(mode='json')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], validate: bool = True) -> "ChunkMetadata":
        """Compatibility alias for model_validate. Handles legacy fields."""
        # Clean data for Pydantic if needed
        d = data.copy()
        if "created_at" in d and isinstance(d["created_at"], str):
             try:
                 d["created_at"] = datetime.fromisoformat(d["created_at"])
             except ValueError:
                 d["created_at"] = _utc_now()
        
        # Handle defaults for legacy data missing required fields
        if "pdf_hash" not in d: d["pdf_hash"] = "legacy_no_hash"
        if "page" not in d: d["page"] = 1
        if "chunk_index" not in d: d["chunk_index"] = 0

        if validate:
            return cls.model_validate(d)
        else:
            return cls.model_construct(**d)

class Chunk(BaseModel):
    """Document chunk with deterministic ID and embedding support."""
    content: str = Field(min_length=1)
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None
    # ID is computed, so not a field in __init__? 
    # Pydantic computed fields are properties. If we want it in model_dump, computed_field works.
    
    @computed_field
    @property
    def id(self) -> str:
        """Fast stable deterministic ID using MD5."""
        # MD5 is faster than SHA256 and stable across runs/platforms (unlike python hash())
        components = f"{self.metadata.pdf_hash}:{self.metadata.page}:{self.metadata.chunk_index}"
        return hashlib.md5(components.encode('ascii')).hexdigest()[:16]
    
    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.content)
    
    @computed_field
    @property
    def embedding_dim(self) -> Optional[int]:
        return len(self.embedding) if self.embedding else None
    
    def to_dict(self, include_embedding: bool = True) -> Dict[str, Any]:
        params = {"exclude": {"embedding"} if not include_embedding else None}
        return self.model_dump(mode='json', **params)

class DistanceMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "l2"
    DOT_PRODUCT = "ip"

class SearchResult(BaseModel):
    """Unified search result with similarity calculation."""
    doc_id: str
    content: str
    metadata: ChunkMetadata
    distance: float
    metric: DistanceMetric = DistanceMetric.COSINE
    
    @computed_field
    @property
    def similarity(self) -> float:
        """Cached similarity score."""
        if self.metric == DistanceMetric.COSINE:
            return max(0.0, min(1.0, 1 - (self.distance / 2)))
        elif self.metric == DistanceMetric.EUCLIDEAN:
            return 1 / (1 + self.distance)
        elif self.metric == DistanceMetric.DOT_PRODUCT:
            return max(0.0, min(1.0, self.distance))
        return 0.0
    
    @classmethod
    def from_raw(cls, doc_id: str, content: str, raw_metadata: Dict[str, Any], distance: float, metric: str = "cosine") -> "SearchResult":
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
        return self.similarity > other.similarity

class DocumentMetadata(BaseModel):
    """Structured document-level metadata."""
    title: str = "Untitled"
    author: Optional[str] = None
    page_count: int = 0
    file_size_bytes: int = 0
    created_date: Optional[datetime] = None
    language: str = "en"
    custom: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    """Root document object holding metadata and chunks."""
    filepath: Path
    pdf_hash: str
    chunks: List[Chunk] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    
    @field_validator('filepath', mode='before')
    @classmethod
    def validate_filepath(cls, v):
        if isinstance(v, str):
            v = Path(v)
        return v.expanduser()

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
    
    def add_chunk(self, chunk: Chunk):
        # Optional validation
        # if chunk.metadata.pdf_hash != self.pdf_hash: raise ValueError(...)
        self.chunks.append(chunk)
