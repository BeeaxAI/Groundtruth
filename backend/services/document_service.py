"""
Phase 8: Document service — orchestrates extraction, chunking, indexing, and retrieval.
Single entry point for all document operations.
"""

import logging
from typing import Optional
from core.models import Document, DocumentChunk, Citation
from core.chunker import DocumentChunker, generate_doc_id
from core.extractor import TextExtractor, ExtractionResult
from core.retriever import BM25Retriever
from utils.security import InputSanitizer

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Manages the document lifecycle:
    upload → extract → sanitize → chunk → index → retrieve.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, max_documents: int = 50):
        self.extractor = TextExtractor()
        self.chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = BM25Retriever()
        self.sanitizer = InputSanitizer()
        self.max_documents = max_documents

        self._documents: dict[str, Document] = {}

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

        # Phase 4: Index for retrieval
        self.retriever.add_chunks(chunks)

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
        del self._documents[doc_id]
        logger.info(f"Removed document: {doc_id}")
        return True

    def search(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        results = self.retriever.search(query, top_k=top_k)
        if not results and self._documents:
            # BM25 found no keyword matches — fallback to first chunks
            # so Gemini always has document context to work with
            fallback = []
            for doc in self._documents.values():
                for chunk in doc.chunks[:2]:
                    fallback.append((chunk, 0.0))
                    if len(fallback) >= top_k:
                        break
                if len(fallback) >= top_k:
                    break
            return fallback
        return results

    def get_all_documents(self) -> list[dict]:
        return [doc.to_dict() for doc in self._documents.values()]

    def get_document_summary(self) -> str:
        if not self._documents:
            return "No documents loaded. The user should upload documents first."

        lines = []
        for doc in self._documents.values():
            lines.append(f"- {doc.name} ({len(doc.chunks)} sections, {doc.page_count} pages, {doc.file_type.upper()})")
        return "\n".join(lines)

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
