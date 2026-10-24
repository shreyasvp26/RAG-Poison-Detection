"""
Sparse retrieval using BM25 for lexical matching and hybrid disagreement scoring.

Supports:
1. rank_bm25.BM25Okapi if installed (loaded dynamically via importlib).
2. Pure Python / NumPy Okapi BM25 implementation (k1=1.5, b=0.75) fallback with inverted index.
3. 1-indexed rank ordering where rank 1 corresponds to the highest BM25 score.
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter
import importlib
import math
import re
import numpy as np

from src.schemas import CorpusDocument, QuerySample


def _get_rank_bm25():
    """Dynamically load rank_bm25 if installed."""
    try:
        return importlib.import_module("rank_bm25")
    except Exception:
        return None


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into lowercased alphanumeric tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class _PureOkapiBM25:
    """Fast pure-Python/NumPy BM25 Okapi inverted index."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_lens = np.array([len(doc) for doc in corpus_tokens], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_lens)) if self.corpus_size > 0 else 0.0

        # Inverted index: term -> list of (doc_idx, term_freq)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}
        self.idf: Dict[str, float] = {}

        for doc_idx, doc in enumerate(corpus_tokens):
            term_freqs = Counter(doc)
            for term, tf in term_freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, tf))

        for term, postings in self.inverted_index.items():
            df = len(postings)
            # Okapi BM25 idf with standard Lucene/rank_bm25 smoothing
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)

    def get_scores(self, query_tokens: List[str]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)
        if self.corpus_size == 0 or self.avgdl == 0:
            return scores

        for term in query_tokens:
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            postings = self.inverted_index[term]
            for doc_idx, tf in postings:
                dl = self.doc_lens[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[doc_idx] += idf_val * (tf * (self.k1 + 1.0)) / denom

        return scores


class SparseBM25Retriever:
    """Tokenizes and indexes documents with BM25Okapi to compute sparse retrieval scores."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """Initialize the SparseBM25Retriever.

        Args:
            k1: BM25 term frequency saturation parameter (default: 1.5).
            b: BM25 document length normalization parameter (default: 0.75).
        """
        self.k1 = k1
        self.b = b
        self.corpus_docs: List[CorpusDocument] = []
        self.doc_id_to_idx: Dict[str, int] = {}
        self.corpus_tokens: List[List[str]] = []
        self.bm25 = None
        self._pure_bm25: Optional[_PureOkapiBM25] = None

    def index_corpus(self, docs: List[CorpusDocument]) -> None:
        """Tokenize documents and build BM25Okapi index.

        Args:
            docs: List of CorpusDocument instances to index.
        """
        self.corpus_docs = list(docs)
        self.doc_id_to_idx = {doc.doc_id: i for i, doc in enumerate(docs)}
        self.corpus_tokens = [
            tokenize_text(f"{doc.title} {doc.text}") for doc in docs
        ]

        if not docs:
            self.bm25 = None
            self._pure_bm25 = None
            return

        rank_bm25_mod = _get_rank_bm25()
        if rank_bm25_mod is not None:
            try:
                self.bm25 = rank_bm25_mod.BM25Okapi(
                    self.corpus_tokens, k1=self.k1, b=self.b
                )
            except Exception:
                self.bm25 = None

        # Always prepare pure-Python fallback for maximum reliability
        self._pure_bm25 = _PureOkapiBM25(self.corpus_tokens, k1=self.k1, b=self.b)

    def build_index(self, corpus: List[CorpusDocument]) -> None:
        """Alias for index_corpus for backward compatibility."""
        self.index_corpus(corpus)

    def score_all(self, query_text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute BM25 scores and 1-indexed ranks for all indexed corpus documents.

        Args:
            query_text: Input query string.

        Returns:
            Tuple of (bm25_scores, bm25_ranks):
                bm25_scores: np.ndarray of shape (len(corpus),) with dtype float32.
                bm25_ranks: np.ndarray of shape (len(corpus),) with dtype int32 (1 = highest score).
        """
        n_docs = len(self.corpus_docs)
        if n_docs == 0:
            return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int32)

        q_tokens = tokenize_text(query_text)

        if self.bm25 is not None:
            try:
                scores = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)
            except Exception:
                scores = self._pure_bm25.get_scores(q_tokens)
        elif self._pure_bm25 is not None:
            scores = self._pure_bm25.get_scores(q_tokens)
        else:
            scores = np.zeros(n_docs, dtype=np.float32)

        # Compute 1-indexed ranks (rank 1 = highest score)
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(n_docs, dtype=np.int32)
        ranks[order] = np.arange(1, n_docs + 1, dtype=np.int32)

        return scores, ranks

    def score_document(self, query: QuerySample, doc: CorpusDocument) -> float:
        """Compute BM25 score for a specific document against query."""
        if doc.doc_id in self.doc_id_to_idx:
            idx = self.doc_id_to_idx[doc.doc_id]
            scores, _ = self.score_all(query.question)
            return float(scores[idx])

        # If document not in indexed corpus, score it ad-hoc against current IDF stats
        q_tokens = tokenize_text(query.question)
        doc_tokens = tokenize_text(f"{doc.title} {doc.text}")
        if not self._pure_bm25 or self._pure_bm25.avgdl == 0:
            return 0.0

        tf_dict = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0
        for term in q_tokens:
            if term not in self._pure_bm25.idf:
                continue
            idf_val = self._pure_bm25.idf[term]
            tf = tf_dict.get(term, 0)
            if tf > 0:
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self._pure_bm25.avgdl))
                score += idf_val * (tf * (self.k1 + 1.0)) / denom

        return float(score)

    def retrieve(
        self,
        query: QuerySample,
        top_n: int = 15,
    ) -> List[Tuple[CorpusDocument, float, int]]:
        """Retrieve top-N candidates ranked by BM25.

        Args:
            query: Input QuerySample.
            top_n: Number of candidates to retrieve.

        Returns:
            List of tuples: (CorpusDocument, bm25_score, bm25_rank)
        """
        scores, ranks = self.score_all(query.question)
        k = min(top_n, len(self.corpus_docs))
        if k <= 0:
            return []

        # Sort corpus by BM25 score descending
        top_indices = np.argsort(-scores, kind="stable")[:k]
        return [
            (self.corpus_docs[idx], float(scores[idx]), int(ranks[idx]))
            for idx in top_indices
        ]
