"""
GroundTruth Document Store
Handles document ingestion, chunking, and retrieval for grounded responses.
Zero-hallucination architecture: every response must trace back to source chunks.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata for citation."""
    chunk_id: str
    doc_id: str
    doc_name: str
    content: str
    page_number: Optional[int] = None
    chunk_index: int = 0
    char_start: int = 0
    char_end: int = 0


@dataclass
class Document:
    """An ingested document."""
    doc_id: str
    name: str
    content: str
    chunks: list[DocumentChunk] = field(default_factory=list)
    page_count: int = 1


class DocumentStore:
    """
    In-memory document store with chunking and retrieval.

    Design: Simple keyword/TF-IDF style retrieval instead of embeddings
    to keep the hackathon scope manageable while still providing grounded responses.
    For production, replace with Vertex AI Search or a vector DB.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.documents: dict[str, Document] = {}
        self.chunks: list[DocumentChunk] = []
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _generate_id(self, content: str) -> str:
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _chunk_text(self, text: str, doc_id: str, doc_name: str) -> list[DocumentChunk]:
        """Split text into overlapping chunks for retrieval."""
        chunks = []
        # Split by paragraphs first, then by size
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk = ""
        current_start = 0
        chunk_idx = 0
        char_pos = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                char_pos += 2  # account for \n\n
                continue

            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunk_id = f"{doc_id}_c{chunk_idx}"
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    content=current_chunk.strip(),
                    chunk_index=chunk_idx,
                    char_start=current_start,
                    char_end=char_pos,
                ))
                chunk_idx += 1
                # Overlap: keep last portion
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text + "\n\n" + para
                current_start = max(0, char_pos - self.chunk_overlap)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = char_pos

            char_pos += len(para) + 2

        # Final chunk
        if current_chunk.strip():
            chunk_id = f"{doc_id}_c{chunk_idx}"
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_name=doc_name,
                content=current_chunk.strip(),
                chunk_index=chunk_idx,
                char_start=current_start,
                char_end=char_pos,
            ))

        return chunks

    def add_document(self, name: str, content: str) -> Document:
        """Ingest a document, chunk it, and store."""
        doc_id = self._generate_id(name + content[:200])

        if doc_id in self.documents:
            logger.info(f"Document '{name}' already exists, skipping.")
            return self.documents[doc_id]

        chunks = self._chunk_text(content, doc_id, name)
        doc = Document(
            doc_id=doc_id,
            name=name,
            content=content,
            chunks=chunks,
        )
        self.documents[doc_id] = doc
        self.chunks.extend(chunks)
        logger.info(f"Ingested '{name}': {len(chunks)} chunks")
        return doc

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and its chunks."""
        if doc_id not in self.documents:
            return False
        self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
        del self.documents[doc_id]
        return True

    def search(self, query: str, top_k: int = 5) -> list[DocumentChunk]:
        """
        Simple keyword-based retrieval with BM25-style scoring.
        Returns the most relevant chunks for a given query.
        """
        if not self.chunks:
            return []

        query_terms = set(re.findall(r'\w+', query.lower()))
        if not query_terms:
            return []

        scored_chunks = []
        for chunk in self.chunks:
            chunk_terms = re.findall(r'\w+', chunk.content.lower())
            chunk_term_set = set(chunk_terms)

            # Term frequency scoring
            matching_terms = query_terms & chunk_term_set
            if not matching_terms:
                continue

            score = 0.0
            for term in matching_terms:
                tf = chunk_terms.count(term) / max(len(chunk_terms), 1)
                # Boost exact phrase matches
                score += tf

            # Boost for more matching terms (coverage)
            coverage = len(matching_terms) / len(query_terms)
            score *= (1.0 + coverage)

            scored_chunks.append((score, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:top_k]]

    def get_context_for_query(self, query: str, max_chars: int = 3000) -> tuple[str, list[dict]]:
        """
        Retrieve relevant context and format it for the LLM prompt.
        Returns (context_string, citations_metadata).
        """
        relevant_chunks = self.search(query, top_k=5)

        if not relevant_chunks:
            return "", []

        context_parts = []
        citations = []
        total_chars = 0

        for i, chunk in enumerate(relevant_chunks):
            if total_chars + len(chunk.content) > max_chars:
                break

            citation_tag = f"[Source {i+1}: {chunk.doc_name}]"
            context_parts.append(f"{citation_tag}\n{chunk.content}")
            citations.append({
                "index": i + 1,
                "doc_name": chunk.doc_name,
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "excerpt": chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content,
            })
            total_chars += len(chunk.content)

        context_str = "\n\n---\n\n".join(context_parts)
        return context_str, citations

    def get_all_documents(self) -> list[dict]:
        """Return metadata for all stored documents."""
        return [
            {
                "doc_id": doc.doc_id,
                "name": doc.name,
                "chunks": len(doc.chunks),
                "content_length": len(doc.content),
            }
            for doc in self.documents.values()
        ]
