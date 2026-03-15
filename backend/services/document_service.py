"""
Phase 8: Document service — orchestrates extraction, chunking, indexing, and retrieval.
Single entry point for all document operations.
Now powered by Super Memory for compressed, semantic-aware retrieval.
"""

import asyncio
import re
import logging
from datetime import datetime
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
        self.chunker = DocumentChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.retriever = BM25Retriever()
        self.sanitizer = InputSanitizer()
        self.max_documents = max_documents
        self.super_memory: Optional[SuperMemory] = None

        self._documents: dict[str, Document] = {}
        self._citation_heatmap: dict[str, int] = {}  # chunk_id → citation_count
        self._knowledge_gaps: list[dict] = []

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
            raise ValueError(
                f"Maximum {self.max_documents} documents allowed. Remove some first.")

        # Phase 3: Extract text
        result: ExtractionResult = self.extractor.extract(
            content_bytes, filename)

        # Phase 7: Sanitize content
        sanitized_text, security_warnings = self.sanitizer.sanitize_document_content(
            result.text)

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
                        self.super_memory.index_document(
                            doc_id, filename, chunks)
                    )
                else:
                    loop.run_until_complete(
                        self.super_memory.index_document(
                            doc_id, filename, chunks)
                    )
            except Exception as e:
                logger.warning(
                    f"Super Memory indexing failed (BM25 still active): {e}")

        logger.info(
            f"Ingested '{filename}': {len(chunks)} chunks, {len(sanitized_text)} chars")

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
            parts.append(
                f"\n### Document: {doc.name} ({doc.page_count} pages, {doc.file_type.upper()})")
            for i, chunk in enumerate(doc.chunks):
                if total_chars + len(chunk.content) > max_chars:
                    parts.append(
                        f"[... {len(doc.chunks) - i} more sections truncated ...]")
                    break
                parts.append(
                    f"[Source: {doc.name}, Section {i + 1}]\n{chunk.content}")
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

    # ================================================================
    # Feature: Smart Follow-up Suggestions
    # ================================================================

    def generate_follow_ups(self, query: str, citations: list, top_n: int = 3) -> list[str]:
        """
        Generate smart follow-up questions from neighboring document content.

        Algorithm:
          1. Find chunks adjacent to cited chunks (unexplored territory)
          2. Extract key topics from those neighbors
          3. Pull uncovered keywords from Super Memory hierarchy
          4. Extract key phrases from uncited chunks (fallback)

        This guides users to explore MORE of their documents,
        making the app feel intelligent and proactive.
        """
        if not citations or not self._documents:
            return []

        cited_ids = {c.chunk_id for c in citations}
        cited_docs = {c.doc_id for c in citations}
        query_words = set(re.findall(r'\w+', query.lower()))
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
                      'could', 'should', 'may', 'might', 'can', 'shall', 'to', 'of',
                      'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'and', 'or', 'but', 'not', 'no', 'if', 'then', 'than', 'that',
                      'this', 'what', 'which', 'who', 'how', 'his', 'her', 'its',
                      'he', 'she', 'it', 'they', 'them', 'we', 'you', 'your'}

        # Strategy 1: Topics from adjacent (uncited) chunks
        adjacent_topics = []
        for doc_id in cited_docs:
            doc = self._documents.get(doc_id)
            if not doc:
                continue
            for i, chunk in enumerate(doc.chunks):
                if chunk.chunk_id not in cited_ids:
                    continue
                # Look at the next 1-3 chunks (unexplored content)
                for offset in [1, 2, 3]:
                    idx = i + offset
                    if idx < len(doc.chunks) and doc.chunks[idx].chunk_id not in cited_ids:
                        content = doc.chunks[idx].content
                        # Extract meaningful sentences/phrases (relaxed filter)
                        sentences = [
                            s.strip() for s in re.split(r'[.!?\n]+', content)
                            if 10 < len(s.strip()) < 200
                        ]
                        if sentences:
                            raw = sentences[0]
                            # Strip leading numbering like "3." or "10."
                            raw = re.sub(r'^\d+[\.\)]\s*', '', raw).strip()
                            if len(raw) < 8:
                                continue
                            topic_words = set(re.findall(r'\w+', raw.lower()))
                            # Ensure low overlap with query (= new topic)
                            overlap = len(topic_words & query_words) / max(len(query_words), 1)
                            if overlap < 0.5:
                                # Trim to a reasonable display length
                                display = raw.rstrip('.,;:')
                                if len(display) > 80:
                                    display = display[:77].rsplit(' ', 1)[0] + '...'
                                adjacent_topics.append(display)

        # Strategy 2: Uncovered keywords from Super Memory hierarchy
        uncovered_keywords = []
        if self.super_memory:
            for doc_id in cited_docs:
                kws = self.super_memory.hierarchy.get_keywords(doc_id)
                for kw, weight in sorted(kws.items(), key=lambda x: -x[1]):
                    if kw not in query_words and len(kw) > 3 and weight > 0.1:
                        uncovered_keywords.append(kw)

        # Strategy 3: Key phrases from uncited chunks in cited documents
        uncited_phrases = []
        if not adjacent_topics and not uncovered_keywords:
            seen_phrases = set()
            for doc_id in cited_docs:
                doc = self._documents.get(doc_id)
                if not doc:
                    continue
                for chunk in doc.chunks:
                    if chunk.chunk_id in cited_ids:
                        continue
                    # Extract meaningful multi-word phrases
                    words = [w for w in re.findall(r'\w+', chunk.content.lower())
                             if w not in stop_words and len(w) > 3]
                    # Find bigrams as topics
                    for j in range(len(words) - 1):
                        phrase = f"{words[j]} {words[j+1]}"
                        if phrase not in query_words and phrase not in seen_phrases:
                            uncited_phrases.append(phrase)
                            seen_phrases.add(phrase)
                            break

        # Build suggestions with deduplication
        templates = [
            "What do the documents say about {}?",
            "Can you explain more about {}?",
            "Tell me about {} from the sources.",
        ]
        suggestions = []
        seen = set()

        # Adjacent chunk topics first (most contextually relevant)
        for topic in adjacent_topics:
            if len(suggestions) >= top_n:
                break
            key = topic[:25].lower()
            if key not in seen:
                # If it already reads like a question, use it directly
                is_question = bool(re.match(
                    r'^(What|How|Why|When|Where|Who|Is|Are|Was|Were|Do|Does|Did|Can|Could|Tell)\s',
                    topic, re.IGNORECASE
                ))
                if is_question:
                    suggestions.append(topic.rstrip('?') + '?')
                else:
                    suggestions.append(f"Tell me about {topic[0].lower() + topic[1:]}")
                seen.add(key)

        # Fill with keyword-based questions
        for kw in uncovered_keywords:
            if len(suggestions) >= top_n:
                break
            if kw not in seen:
                tmpl = templates[len(suggestions) % len(templates)]
                suggestions.append(tmpl.format(kw))
                seen.add(kw)

        # Fill with uncited phrase questions
        for phrase in uncited_phrases:
            if len(suggestions) >= top_n:
                break
            if phrase not in seen:
                tmpl = templates[len(suggestions) % len(templates)]
                suggestions.append(tmpl.format(phrase))
                seen.add(phrase)

        return suggestions[:top_n]

    # ================================================================
    # Feature: Document Health Score
    # ================================================================

    def get_document_health(self, doc_id: str = None) -> dict:
        """
        Compute document health scores (0-100) for quality assessment.

        Health Score Formula:
          H = 0.25·R + 0.25·D + 0.25·E + 0.25·S

        Where:
          R = Content Richness  (content length + chunk count)
          D = Keyword Diversity (unique meaningful terms)
          E = Embedding Coverage (chunks with semantic embeddings)
          S = Structure Quality (page count + file type bonus)
        """
        if doc_id:
            doc = self._documents.get(doc_id)
            if not doc:
                return {}
            return self._compute_doc_health(doc)
        return {did: self._compute_doc_health(d) for did, d in self._documents.items()}

    def _compute_doc_health(self, doc: Document) -> dict:
        """Compute health score for a single document."""
        scores = {}

        # 1. Content Richness (25%) — content length + chunk count
        content_score = min(len(doc.content) / 1000, 70)  # 1000+ chars = 70
        chunk_bonus = min(len(doc.chunks) * 6, 30)  # up to 30 for chunk count
        scores["content_richness"] = min(content_score + chunk_bonus, 100)

        # 2. Keyword Diversity (25%) — unique meaningful terms
        if self.super_memory:
            kws = self.super_memory.hierarchy.get_keywords(doc.doc_id)
            scores["keyword_diversity"] = min(len(kws) * 2, 100)
        else:
            words = doc.content.lower().split()
            unique = len(set(w for w in words if len(w) > 3))
            scores["keyword_diversity"] = min(
                (unique / max(len(words), 1)) * 300, 100
            )

        # 3. Embedding Coverage (25%) — chunks with semantic embeddings
        if self.super_memory and self.super_memory._embedding_enabled:
            embedded = sum(
                1 for c in doc.chunks
                if c.chunk_id in self.super_memory.embeddings._embeddings
            )
            scores["embedding_coverage"] = (
                embedded / max(len(doc.chunks), 1)
            ) * 100
        else:
            scores["embedding_coverage"] = 0

        # 4. Structure Quality (25%) — page count, file type bonus
        struct_score = 40  # base
        if doc.page_count > 1:
            struct_score += min(doc.page_count * 5, 30)
        if doc.file_type in ('pdf', 'docx'):
            struct_score += 20
        elif doc.file_type in ('txt', 'md'):
            struct_score += 10
        scores["structure_quality"] = min(struct_score, 100)

        # Overall health
        overall = sum(scores.values()) / len(scores)

        if overall >= 80:
            level = "excellent"
        elif overall >= 60:
            level = "good"
        elif overall >= 40:
            level = "fair"
        else:
            level = "poor"

        return {
            "doc_id": doc.doc_id,
            "name": doc.name,
            "overall_score": round(overall, 1),
            "level": level,
            "breakdown": {k: round(v, 1) for k, v in scores.items()},
        }

    # ================================================================
    # Feature: Hallucination Heatmap
    # ================================================================

    def record_citation(self, chunk_id: str) -> None:
        """Increment the citation count for a chunk."""
        self._citation_heatmap[chunk_id] = self._citation_heatmap.get(chunk_id, 0) + 1

    def get_heatmap(self, doc_id: str = None) -> dict:
        """
        Return per-chunk citation frequency with heat levels.

        Heat levels:
          frozen = 0 citations  (never referenced)
          cold   = 1-2 citations
          warm   = 3-5 citations
          hot    = 6+  citations

        If doc_id is provided, returns heatmap for that document only.
        Otherwise returns heatmaps for all documents.
        """
        if doc_id:
            doc = self._documents.get(doc_id)
            if not doc:
                return {}
            return self._build_doc_heatmap(doc)

        return {did: self._build_doc_heatmap(d) for did, d in self._documents.items()}

    def _build_doc_heatmap(self, doc: Document) -> dict:
        """Build heatmap data for a single document."""
        chunks_data = []
        heat_counts = {"hot": 0, "warm": 0, "cold": 0, "frozen": 0}

        for chunk in doc.chunks:
            count = self._citation_heatmap.get(chunk.chunk_id, 0)
            level = self._heat_level(count)
            heat_counts[level] += 1
            chunks_data.append({
                "chunk_id": chunk.chunk_id,
                "index": chunk.index,
                "citation_count": count,
                "heat_level": level,
                "preview": chunk.content[:120] + ("..." if len(chunk.content) > 120 else ""),
            })

        total = len(doc.chunks)
        coverage = ((heat_counts["hot"] + heat_counts["warm"] + heat_counts["cold"])
                    / max(total, 1)) * 100

        return {
            "doc_id": doc.doc_id,
            "name": doc.name,
            "total_chunks": total,
            "coverage_pct": round(coverage, 1),
            "heat_summary": heat_counts,
            "chunks": chunks_data,
        }

    @staticmethod
    def _heat_level(count: int) -> str:
        """Classify a citation count into a heat level."""
        if count == 0:
            return "frozen"
        elif count <= 2:
            return "cold"
        elif count <= 5:
            return "warm"
        else:
            return "hot"

    # ================================================================
    # Feature: Knowledge Gap Detector
    # ================================================================

    _GAP_STOP_WORDS = frozenset({
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
        'on', 'with', 'at', 'by', 'from', 'as', 'into', 'and', 'or', 'but',
        'not', 'no', 'if', 'then', 'than', 'that', 'this', 'what', 'which',
        'who', 'how', 'his', 'her', 'its', 'he', 'she', 'it', 'they', 'them',
        'we', 'you', 'your', 'my', 'me', 'about', 'tell', 'know', 'please',
        'want', 'need', 'like', 'just', 'get', 'got', 'also', 'some', 'any',
        'more', 'much', 'many', 'very', 'too', 'so', 'up', 'out', 'there',
    })

    def record_gap(self, query: str, confidence_score: float, status: str) -> None:
        """
        Record a knowledge gap when a query could not be answered well.

        Only records if confidence_score < 40 or status is one of
        'no_match', 'no_context', or 'ungrounded'.
        """
        should_record = (
            confidence_score < 40
            or status in ('no_match', 'no_context', 'ungrounded')
        )
        if not should_record:
            return

        self._knowledge_gaps.append({
            "query": query.strip(),
            "confidence_score": confidence_score,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        })
        logger.info(
            f"Knowledge gap recorded: status={status}, "
            f"confidence={confidence_score}, query='{query[:80]}'"
        )

    def _extract_phrases(self, text: str) -> list[str]:
        """
        Extract 2-3 word key phrases from text using simple word filtering.

        Returns lowercased phrases built from consecutive meaningful words.
        """
        words = re.findall(r'[a-zA-Z]{3,}', text.lower())
        meaningful = [w for w in words if w not in self._GAP_STOP_WORDS]

        phrases = []
        # Build bigrams and trigrams from consecutive meaningful words
        for i in range(len(meaningful)):
            if i + 1 < len(meaningful):
                phrases.append(f"{meaningful[i]} {meaningful[i + 1]}")
            if i + 2 < len(meaningful):
                phrases.append(
                    f"{meaningful[i]} {meaningful[i + 1]} {meaningful[i + 2]}"
                )
        # If only one meaningful word, use it as a single-word phrase
        if len(meaningful) == 1:
            phrases.append(meaningful[0])

        return phrases

    def get_knowledge_gaps(self) -> dict:
        """
        Analyse recorded knowledge gaps by clustering similar queries.

        Returns:
            {
                "gaps": [
                    {
                        "topic": "key phrase",
                        "frequency": int,
                        "last_asked": "ISO timestamp",
                        "sample_queries": ["..."],
                    },
                    ...
                ],
                "suggestions": ["Recommended document topic to upload", ...],
                "total_gap_queries": int,
            }
        """
        if not self._knowledge_gaps:
            return {"gaps": [], "suggestions": [], "total_gap_queries": 0}

        # --- Step 1: Extract phrases from every gap query and tally ---
        # phrase -> { count, last_timestamp, sample_queries }
        phrase_data: dict[str, dict] = {}

        for gap in self._knowledge_gaps:
            query = gap["query"]
            timestamp = gap["timestamp"]
            phrases = self._extract_phrases(query)

            for phrase in phrases:
                if phrase not in phrase_data:
                    phrase_data[phrase] = {
                        "count": 0,
                        "last_timestamp": timestamp,
                        "sample_queries": [],
                    }
                entry = phrase_data[phrase]
                entry["count"] += 1
                if timestamp > entry["last_timestamp"]:
                    entry["last_timestamp"] = timestamp
                # Keep up to 5 unique sample queries per phrase
                if (query not in entry["sample_queries"]
                        and len(entry["sample_queries"]) < 5):
                    entry["sample_queries"].append(query)

        # --- Step 2: Deduplicate overlapping phrases ---
        # If a bigram is a subset of a trigram with similar count, drop it.
        sorted_phrases = sorted(
            phrase_data.items(), key=lambda x: (-x[1]["count"], x[0])
        )

        # Build a set of phrases to suppress (subsumed by longer phrases)
        suppressed: set[str] = set()
        phrase_words_map = {p: set(p.split()) for p, _ in sorted_phrases}
        for i, (p1, d1) in enumerate(sorted_phrases):
            w1 = phrase_words_map[p1]
            for j, (p2, d2) in enumerate(sorted_phrases):
                if i == j or p2 in suppressed:
                    continue
                w2 = phrase_words_map[p2]
                # If p1 is a strict subset of p2 and p2 has at least half the count
                if w1 < w2 and d2["count"] >= d1["count"] * 0.5:
                    suppressed.add(p1)
                    # Merge sample queries into the longer phrase
                    for q in d1["sample_queries"]:
                        if (q not in d2["sample_queries"]
                                and len(d2["sample_queries"]) < 5):
                            d2["sample_queries"].append(q)
                    break

        # --- Step 3: Build ranked gap list ---
        gaps = []
        seen_topics: set[str] = set()
        for phrase, data in sorted_phrases:
            if phrase in suppressed:
                continue
            if data["count"] < 1:
                continue
            # Avoid near-duplicate topics
            if phrase in seen_topics:
                continue
            seen_topics.add(phrase)
            gaps.append({
                "topic": phrase,
                "frequency": data["count"],
                "last_asked": data["last_timestamp"],
                "sample_queries": data["sample_queries"],
            })

        # Limit to top 20 gaps
        gaps = gaps[:20]

        # --- Step 4: Generate upload suggestions ---
        suggestions = []
        for gap in gaps[:10]:
            topic = gap["topic"]
            # Capitalise for display
            display_topic = topic.title()
            suggestions.append(
                f"Upload documents about: {display_topic}"
            )

        return {
            "gaps": gaps,
            "suggestions": suggestions,
            "total_gap_queries": len(self._knowledge_gaps),
        }
