"""
Retrieval module for EviGuard-RAG.

Provides dense FAISS indexation, sparse BM25 retrieval, and expanded hybrid pool generation.
"""

from src.retrieval.dense_faiss import DenseFaissRetriever
from src.retrieval.sparse_bm25 import SparseBM25Retriever
from src.retrieval.hybrid_pool import HybridPoolBuilder

__all__ = ["DenseFaissRetriever", "SparseBM25Retriever", "HybridPoolBuilder"]
