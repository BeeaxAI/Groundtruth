"""
Phase 2: Data models for documents, chunks, queries, and citations.
Pure dataclasses — no framework dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
from enum import Enum


class GroundingStatus(str, Enum):
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"
    NO_CONTEXT = "no_context"


@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    content: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0
    token_estimate: int = 0

    def __post_init__(self):
        self.token_estimate = len(self.content) // 4


@dataclass
class Document:
    doc_id: str
    name: str
    content: str
    chunks: list[DocumentChunk] = field(default_factory=list)
    page_count: int = 1
    file_type: str = ""
    file_size_bytes: int = 0
    ingested_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "name": self.name,
            "chunks": len(self.chunks),
            "content_length": len(self.content),
            "file_type": self.file_type,
            "file_size_bytes": self.file_size_bytes,
            "page_count": self.page_count,
            "ingested_at": self.ingested_at,
        }


@dataclass
class Citation:
    index: int
    doc_name: str
    doc_id: str
    chunk_id: str
    excerpt: str
    relevance_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "doc_name": self.doc_name,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "excerpt": self.excerpt,
            "relevance_score": round(self.relevance_score, 3),
        }


@dataclass
class GroundingResult:
    status: GroundingStatus
    valid: bool
    cited_sources: list[int] = field(default_factory=list)
    total_available: int = 0
    hallucinated_refs: list[int] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "valid": self.valid,
            "cited_sources": self.cited_sources,
            "total_available": self.total_available,
            "hallucinated_refs": self.hallucinated_refs,
            "reason": self.reason,
        }


@dataclass
class QueryRecord:
    query: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat())
    num_citations: int = 0
    has_context: bool = False
    grounding_result: Optional[GroundingResult] = None
    response_preview: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query[:100],
            "timestamp": self.timestamp,
            "num_citations": self.num_citations,
            "has_context": self.has_context,
            "grounding": self.grounding_result.to_dict() if self.grounding_result else None,
        }
