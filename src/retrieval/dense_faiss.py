"""
Dense retrieval using Contriever and FAISS inner-product (cosine similarity) index.
"""

from typing import List, Tuple
import numpy as np

from src.schemas import CorpusDocument, QuerySample


class DenseFaissRetriever:
    """Encodes documents and queries into 768-dim L2-normalized embeddings and retrieves via FAISS."""

    def __init__(
        self,
        model_name: str = "facebook/contriever",
        embedding_dim: int = 768,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.device = device
        self.corpus_docs: List[CorpusDocument] = []
        self.index = None

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts into L2-normalized dense embeddings.

        Returns:
            np.ndarray of shape (len(texts), embedding_dim) with unit L2 norm.
        """
        # Stub implementation - to be completed with HuggingFace/SentenceTransformers
        vectors = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)

    def build_index(self, corpus: List[CorpusDocument]) -> None:
        """Build FAISS IndexFlatIP over L2-normalized document embeddings."""
        self.corpus_docs = corpus
        # Stub implementation for index creation

    def retrieve(
        self,
        query: QuerySample,
        top_n: int = 15,
    ) -> List[Tuple[CorpusDocument, float, int]]:
        """Retrieve top-N documents for query.

        Returns:
            List of tuples: (CorpusDocument, dense_score, dense_rank)
        """
        # Stub implementation
        return []
