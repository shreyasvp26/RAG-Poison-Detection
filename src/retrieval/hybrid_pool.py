"""
Hybrid pool builder combining dense and sparse retrievers into a RetrievalPool.
"""

from typing import Optional
import numpy as np

from src.schemas import QuerySample, RetrievalPool
from src.retrieval.dense_faiss import DenseFaissRetriever
from src.retrieval.sparse_bm25 import SparseBM25Retriever


class HybridPoolBuilder:
    """Builds expanded candidate pools (top-N) partitioned into active set (top-K) and reserve pool."""

    def __init__(
        self,
        dense_retriever: DenseFaissRetriever,
        sparse_retriever: SparseBM25Retriever,
        top_k_active: int = 5,
        top_n_expanded: int = 15,
    ):
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.top_k_active = top_k_active
        self.top_n_expanded = top_n_expanded

    def build_pool(
        self,
        query: QuerySample,
        query_embedding: Optional[np.ndarray] = None,
    ) -> RetrievalPool:
        """Retrieve candidates and construct RetrievalPool with active set and reserve pool.

        Args:
            query: Input QuerySample.
            query_embedding: Optional precomputed L2-normalized 768-dim query embedding.

        Returns:
            Fully populated RetrievalPool.
        """
        # Stub implementation - to be implemented by Member 2
        from src.mock_factory import create_mock_retrieval_pool
        return create_mock_retrieval_pool(
            k=self.top_k_active,
            n=self.top_n_expanded,
        )
