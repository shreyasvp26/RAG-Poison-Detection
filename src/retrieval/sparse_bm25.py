"""
Sparse retrieval using BM25 for lexical matching and hybrid disagreement scoring.
"""

from typing import List, Tuple
from rank_bm25 import BM25Okapi  # type: ignore

from src.schemas import CorpusDocument, QuerySample


class SparseBM25Retriever:
    """Tokenizes and indexes documents with BM25Okapi to compute sparse retrieval scores."""

    def __init__(self):
        self.corpus_docs: List[CorpusDocument] = []
        self.bm25: BM25Okapi = None

    def build_index(self, corpus: List[CorpusDocument]) -> None:
        """Tokenize documents and build BM25Okapi index."""
        self.corpus_docs = corpus
        # Stub: to be populated with proper tokenization

    def score_document(self, query: QuerySample, doc: CorpusDocument) -> float:
        """Compute the BM25 score for a specific document against query."""
        # Stub implementation
        return 0.0

    def retrieve(
        self,
        query: QuerySample,
        top_n: int = 15,
    ) -> List[Tuple[CorpusDocument, float, int]]:
        """Retrieve top-N candidates ranked by BM25.

        Returns:
            List of tuples: (CorpusDocument, bm25_score, bm25_rank)
        """
        # Stub implementation
        return []
