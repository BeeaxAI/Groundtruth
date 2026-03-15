"""
Phase 8: Document service — orchestrates extraction, chunking, indexing, and retrieval.
Single entry point for all document operations.
Now powered by Super Memory for compressed, semantic-aware retrieval.
"""

import asyncio
import logging
from typing import Optional
from core.models import Document, DocumentChunk, Citation
from core.chunker import DocumentChunker, generate_doc_id
from core.extractor import TextExtractor, ExtractionResult
from core.retriever import BM25Retriever
from core.super_memory import SuperMemory
from utils.security import InputSanitizer

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Manages the document lifecycle:
    upload → extract → sanitize → chunk → index → retrieve.

    Retrieval pipeline:
      1. Super Memory hybrid search (BM25 + semantic + hierarchical)
      2. Fallback to BM25-only if Super Memory unavailable
      3. Fallback to first chunks if no keyword matches
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, max_documents: int = 50):
        self.extractor = TextExtractor()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = BM25Retriever()
        self.sanitizer = InputSanitizer()
        self.max_documents = max_documents
        self.super_memory: Optional[SuperMemory] = None

        self._documents: dict[str, Document] = {}

    def init_super_memory(self, gemini_client=None, embedding_model: str = "text-embedding-004"):
        """Initialize Super Memory with optional Gemini client for embeddings."""
        self.super_memory = SuperMemory(
            gemini_client=gemini_client,
            embedding_model=embedding_model,
        )
        logger.info(
            f"Super Memory initialized "
            f"(embeddings={'enabled' if gemini_client else 'disabled'})"
        )

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def total_chunks(self) -> int:
        return sum(len(d.chunks) for d in self._documents.values())

    def ingest(self, filename: str, content_bytes: bytes) -> dict:
        if len(self._documents) >= self.max_documents:
            raise ValueError(f"Maximum {self.max_documents} documents allowed. Remove some first.")

        # Phase 3: Extract text
        result: ExtractionResult = self.extractor.extract(content_bytes, filename)

        # Phase 7: Sanitize content
        sanitized_text, security_warnings = self.sanitizer.sanitize_document_content(result.text)

        # Generate ID
        doc_id = generate_doc_id(filename, sanitized_text)
        if doc_id in self._documents:
            existing = self._documents[doc_id]
            return {
                "status": "exists",
                "doc_id": existing.doc_id,
                "name": existing.name,
                "chunks": len(existing.chunks),
                "content_length": len(existing.content),
            }

        # Phase 2: Chunk text
        chunks = self.chunker.chunk_document(
            sanitized_text, doc_id, filename, result.page_boundaries
        )

        # Create document
        doc = Document(
            doc_id=doc_id,
            name=filename,
            content=sanitized_text,
            chunks=chunks,
            page_count=result.page_count,
            file_type=result.file_type,
            file_size_bytes=len(content_bytes),
        )
        self._documents[doc_id] = doc

        # Phase 4: Index for BM25 retrieval
        self.retriever.add_chunks(chunks)

        # Super Memory: index into compressed memory (async)
        if self.super_memory:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're inside an async context — schedule as task
                    asyncio.create_task(
                        self.super_memory.index_document(doc_id, filename, chunks)
                    )
                else:
                    loop.run_until_complete(
                        self.super_memory.index_document(doc_id, filename, chunks)
                    )
            except Exception as e:
                logger.warning(f"Super Memory indexing failed (BM25 still active): {e}")

        logger.info(f"Ingested '{filename}': {len(chunks)} chunks, {len(sanitized_text)} chars")

        response = {
            "status": "success",
            "doc_id": doc.doc_id,
            "name": doc.name,
            "chunks": len(doc.chunks),
            "content_length": len(doc.content),
            "page_count": doc.page_count,
            "file_type": doc.file_type,
        }

        if security_warnings:
            response["security_warnings"] = security_warnings
        if result.warnings:
            response["extraction_warnings"] = result.warnings

        return response

    def remove(self, doc_id: str) -> bool:
        if doc_id not in self._documents:
            return False

        self.retriever.remove_doc_chunks(doc_id)
        if self.super_memory:
            self.super_memory.remove_document(doc_id)
        del self._documents[doc_id]
        logger.info(f"Removed document: {doc_id}")
        return True

    def search(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """
        Search using BM25 with fallback.
        For async hybrid search, use search_hybrid() instead.
        """
        results = self.retriever.search(query, top_k=top_k)
        if not results and self._documents:
            # BM25 found no keyword matches — fallback to first chunks
            # so Gemini always has document context to work with
            return self._get_fallback_chunks(top_k)
        return results

    async def search_hybrid(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        """
        Hybrid search using Super Memory (BM25 + semantic + hierarchical).
        Falls back to BM25-only if Super Memory unavailable.
        """
        if self.super_memory:
            results = await self.super_memory.search(
                query, top_k=top_k, bm25_retriever=self.retriever,
            )
            if results:
                return results

        # Fallback to BM25
        results = self.retriever.search(query, top_k=top_k)
        if not results and self._documents:
            return self._get_fallback_chunks(top_k)
        return results

    def _get_fallback_chunks(self, top_k: int) -> list[tuple[DocumentChunk, float]]:
        """Return first chunks from each document as fallback."""
        fallback = []
        for doc in self._documents.values():
            for chunk in doc.chunks[:2]:
                fallback.append((chunk, 0.0))
                if len(fallback) >= top_k:
                    break
            if len(fallback) >= top_k:
                break
        return fallback

    def get_all_documents(self) -> list[dict]:
        return [doc.to_dict() for doc in self._documents.values()]

    def get_document_summary(self) -> str:
        """Return document names + actual content for system instruction."""
        if not self._documents:
            return "No documents loaded. The user should upload documents first."

        parts = []
        total_chars = 0
        max_chars = 8000

        for doc in self._documents.values():
            parts.append(f"\n### Document: {doc.name} ({doc.page_count} pages, {doc.file_type.upper()})")
            for i, chunk in enumerate(doc.chunks):
                if total_chars + len(chunk.content) > max_chars:
                    parts.append(f"[... {len(doc.chunks) - i} more sections truncated ...]")
                    break
                parts.append(f"[Source: {doc.name}, Section {i + 1}]\n{chunk.content}")
                total_chars += len(chunk.content)

        return "\n\n".join(parts)

    def has_documents(self) -> bool:
        return len(self._documents) > 0

    def get_context_preview(self, max_chunks: int = 10) -> list[DocumentChunk]:
        """Return first chunks from each document for session preloading."""
        chunks = []
        for doc in self._documents.values():
            for chunk in doc.chunks[:3]:
                chunks.append(chunk)
                if len(chunks) >= max_chunks:
                    return chunks
        return chunks

    def get_memory_stats(self) -> dict:
        """Get Super Memory compression statistics."""
        if self.super_memory:
            return self.super_memory.stats
        return {"status": "Super Memory not initialized"}

    def find_duplicates(self) -> list[dict]:
        """Detect near-duplicate documents using SimHash."""
        if self.super_memory:
            return self.super_memory.find_duplicates()
        return []

    def get_document_insights(self, doc_id: str) -> dict:
        """Get AI-extracted insights for a document."""
        if self.super_memory:
            return self.super_memory.get_document_insights(doc_id)
        return {}
