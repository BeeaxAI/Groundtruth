"""
Phase 2: Advanced document chunking engine.
Splits documents into semantically meaningful, overlapping chunks.
Supports paragraph-aware splitting with configurable size/overlap.
"""

import re
import hashlib
import logging
from typing import Optional
from core.models import DocumentChunk

logger = logging.getLogger(__name__)

# Sentence boundary pattern
SENTENCE_BOUNDARY = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
PARAGRAPH_BOUNDARY = re.compile(r'\n\s*\n')


class DocumentChunker:
    """
    Splits text into overlapping chunks optimized for retrieval.

    Strategy:
    1. Split by paragraphs first (semantic boundaries)
    2. Merge small paragraphs into chunks up to chunk_size
    3. Split oversized paragraphs at sentence boundaries
    4. Add overlap from previous chunk for context continuity
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(
        self, text: str, doc_id: str, doc_name: str, page_boundaries: Optional[list[int]] = None
    ) -> list[DocumentChunk]:
        if not text.strip():
            return []

        paragraphs = self._split_paragraphs(text)
        chunks = self._merge_into_chunks(paragraphs, doc_id, doc_name)

        if page_boundaries:
            self._assign_page_numbers(chunks, page_boundaries)

        logger.info(f"Chunked '{doc_name}' → {len(chunks)} chunks (avg {sum(len(c.content) for c in chunks) // max(len(chunks), 1)} chars)")
        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        raw = PARAGRAPH_BOUNDARY.split(text)
        paragraphs = []
        for p in raw:
            p = p.strip()
            if not p:
                continue
            if len(p) > self.chunk_size * 2:
                paragraphs.extend(self._split_long_paragraph(p))
            else:
                paragraphs.append(p)
        return paragraphs

    def _split_long_paragraph(self, text: str) -> list[str]:
        sentences = SENTENCE_BOUNDARY.split(text)
        if len(sentences) <= 1:
            result = []
            for i in range(0, len(text), self.chunk_size):
                result.append(text[i:i + self.chunk_size])
            return result

        segments = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > self.chunk_size and current:
                segments.append(current.strip())
                current = sent
            else:
                current = (current + " " + sent).strip() if current else sent
        if current.strip():
            segments.append(current.strip())
        return segments

    def _merge_into_chunks(self, paragraphs: list[str], doc_id: str, doc_name: str) -> list[DocumentChunk]:
        chunks = []
        current_text = ""
        current_start = 0
        char_pos = 0
        prev_overlap = ""

        for para in paragraphs:
            new_text = (current_text + "\n\n" + para).strip() if current_text else para

            if len(new_text) > self.chunk_size and current_text:
                chunk = self._create_chunk(
                    prev_overlap + current_text if prev_overlap and not chunks else current_text,
                    doc_id, doc_name, len(chunks), current_start, char_pos,
                )
                chunks.append(chunk)

                prev_overlap = current_text[-self.chunk_overlap:] + "\n\n" if len(current_text) > self.chunk_overlap else ""
                current_text = para
                current_start = char_pos
            else:
                current_text = new_text
            char_pos += len(para) + 2

        if current_text.strip():
            final_text = prev_overlap + current_text if prev_overlap and chunks else current_text
            chunks.append(self._create_chunk(
                final_text, doc_id, doc_name, len(chunks), current_start, char_pos,
            ))

        return chunks

    def _create_chunk(self, content: str, doc_id: str, doc_name: str, idx: int, char_start: int, char_end: int) -> DocumentChunk:
        chunk_id = f"{doc_id}_c{idx}"
        return DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            doc_name=doc_name,
            content=content.strip(),
            chunk_index=idx,
            char_start=char_start,
            char_end=char_end,
        )

    def _assign_page_numbers(self, chunks: list[DocumentChunk], page_boundaries: list[int]):
        for chunk in chunks:
            for page_num, boundary in enumerate(page_boundaries, start=1):
                if chunk.char_start < boundary:
                    chunk.page_number = page_num
                    break
            else:
                chunk.page_number = len(page_boundaries)


def generate_doc_id(name: str, content_prefix: str) -> str:
    return hashlib.sha256((name + content_prefix[:200]).encode()).hexdigest()[:12]
