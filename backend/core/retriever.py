"""
Phase 4: BM25 retrieval engine.
Implements Okapi BM25 ranking for document chunk retrieval.
Significantly better than naive keyword matching for relevance.
"""

import math
import re
import logging
from collections import Counter
from typing import Optional
from core.models import DocumentChunk

logger = logging.getLogger(__name__)

# Stopwords to filter (common English)
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "just", "because", "but", "and", "or", "if", "while", "that", "this",
    "it", "its", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "about", "up",
})

WORD_PATTERN = re.compile(r'\b[a-zA-Z0-9]+\b')


def tokenize(text: str) -> list[str]:
    tokens = WORD_PATTERN.findall(text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Retriever:
    """
    Okapi BM25 ranking algorithm for document chunk retrieval.

    Parameters:
        k1: Term frequency saturation parameter (1.2–2.0 typical)
        b:  Length normalization (0 = no normalization, 1 = full)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[DocumentChunk] = []
        self._tokenized: list[list[str]] = []
        self._doc_freqs: Counter = Counter()
        self._avg_dl: float = 0.0
        self._n_docs: int = 0

    def index(self, chunks: list[DocumentChunk]):
        self._chunks = chunks
        self._tokenized = [tokenize(c.content) for c in chunks]
        self._n_docs = len(chunks)

        # Compute document frequencies
        self._doc_freqs = Counter()
        for tokens in self._tokenized:
            unique_terms = set(tokens)
            for term in unique_terms:
                self._doc_freqs[term] += 1

        # Average document length
        total = sum(len(t) for t in self._tokenized)
        self._avg_dl = total / max(self._n_docs, 1)

    def add_chunks(self, new_chunks: list[DocumentChunk]):
        start_idx = len(self._chunks)
        self._chunks.extend(new_chunks)

        for chunk in new_chunks:
            tokens = tokenize(chunk.content)
            self._tokenized.append(tokens)
            for term in set(tokens):
                self._doc_freqs[term] += 1

        self._n_docs = len(self._chunks)
        total = sum(len(t) for t in self._tokenized)
        self._avg_dl = total / max(self._n_docs, 1)

    def remove_doc_chunks(self, doc_id: str):
        indices_to_remove = {i for i, c in enumerate(self._chunks) if c.doc_id == doc_id}
        if not indices_to_remove:
            return

        # Decrement doc freqs
        for idx in indices_to_remove:
            for term in set(self._tokenized[idx]):
                self._doc_freqs[term] -= 1
                if self._doc_freqs[term] <= 0:
                    del self._doc_freqs[term]

        self._chunks = [c for i, c in enumerate(self._chunks) if i not in indices_to_remove]
        self._tokenized = [t for i, t in enumerate(self._tokenized) if i not in indices_to_remove]
        self._n_docs = len(self._chunks)
        total = sum(len(t) for t in self._tokenized)
        self._avg_dl = total / max(self._n_docs, 1)

    def search(self, query: str, top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
        if not self._chunks:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for idx, (chunk, doc_tokens) in enumerate(zip(self._chunks, self._tokenized)):
            score = self._score_document(query_tokens, doc_tokens)
            if score > 0:
                scores.append((chunk, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _score_document(self, query_tokens: list[str], doc_tokens: list[str]) -> float:
        dl = len(doc_tokens)
        tf_map = Counter(doc_tokens)
        score = 0.0

        for term in query_tokens:
            if term not in tf_map:
                continue

            tf = tf_map[term]
            df = self._doc_freqs.get(term, 0)

            # IDF with smoothing
            idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)

            # BM25 term score
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
            score += idf * (numerator / denominator)

        return score
