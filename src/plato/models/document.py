from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid

@dataclass
class ChunkMetadata:
    source: str
    page: int
    char_start: int
    char_end: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    
@dataclass
class Chunk:
    content: str
    metadata: ChunkMetadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    embedding: Optional[List[float]] = None

@dataclass
class Document:
    filepath: str
    chunks: List[Chunk] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
