"""
Hybrid pool builder combining dense and sparse retrievers into a RetrievalPool.

Partitioning:
- Active set: Top-K dense candidates (is_in_active_set=True)
- Reserve pool: Remaining (N - K) dense candidates (is_in_active_set=False)
- Tracks latency_ms["retrieval"] for operational observability.
"""

from typing import List, Optional
import os
import time
import numpy as np
import yaml

from src.schemas import (
    QuerySample,
    RetrievalPool,
    RetrievedCandidate,
    TrustStatus,
)
from src.retrieval.dense_faiss import DenseFaissRetriever
from src.retrieval.sparse_bm25 import SparseBM25Retriever


class HybridPoolBuilder:
    """Builds expanded candidate pools (top-N) partitioned into active set (top-K) and reserve pool."""

    def __init__(
        self,
        dense_retriever: DenseFaissRetriever,
        sparse_retriever: SparseBM25Retriever,
        top_k_active: Optional[int] = None,
        top_n_expanded: Optional[int] = None,
        config_path: str = "config.yaml",
    ):
        """Initialize HybridPoolBuilder.

        Args:
            dense_retriever: Initialized DenseFaissRetriever instance.
            sparse_retriever: Initialized SparseBM25Retriever instance.
            top_k_active: Size of active set (read from config.yaml if None, default: 5).
            top_n_expanded: Total size of expanded pool (read from config.yaml if None, default: 15).
            config_path: Path to configuration YAML file.
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.config_path = config_path

        # Read defaults from config.yaml if available
        cfg_k = None
        cfg_n = None
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    if isinstance(cfg, dict):
                        retrieval_cfg = cfg.get("retrieval", {})
                        cfg_k = retrieval_cfg.get("top_k_active")
                        cfg_n = retrieval_cfg.get("top_n_expanded")
            except Exception:
                pass

        self.top_k_active: int = (
            top_k_active if top_k_active is not None else (cfg_k if cfg_k is not None else 5)
        )
        self.top_n_expanded: int = (
            top_n_expanded if top_n_expanded is not None else (cfg_n if cfg_n is not None else 15)
        )

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
        start_time = time.perf_counter()

        # 1. Query embedding (guaranteed unit L2 norm, float32, (768,))
        if query_embedding is None:
            q_emb = self.dense_retriever.encode_query(query.question)
        else:
            q_emb = query_embedding.astype(np.float32).flatten()
            norm = np.linalg.norm(q_emb)
            if norm > 0:
                q_emb = (q_emb / norm).astype(np.float32)

        # 2. Dense retrieval for top-N candidates
        dense_hits = self.dense_retriever.search(q_emb, top_n=self.top_n_expanded)

        # 3. Sparse BM25 scores and ranks for all indexed corpus documents
        bm25_scores, bm25_ranks = self.sparse_retriever.score_all(query.question)

        candidates: List[RetrievedCandidate] = []
        for rank_idx, (corpus_idx, dense_score) in enumerate(dense_hits):
            dense_rank = rank_idx + 1
            doc = self.dense_retriever.corpus_docs[corpus_idx]

            # Fetch or generate document embedding
            if (
                self.dense_retriever.doc_embeddings is not None
                and corpus_idx < len(self.dense_retriever.doc_embeddings)
            ):
                cand_emb = self.dense_retriever.doc_embeddings[corpus_idx].copy()
            else:
                cand_emb = self.dense_retriever.encode_texts([f"{doc.title} {doc.text}"])[0]

            # Fetch BM25 score and rank
            bm_idx = None
            if hasattr(self.sparse_retriever, "doc_id_to_idx") and doc.doc_id in self.sparse_retriever.doc_id_to_idx:
                bm_idx = self.sparse_retriever.doc_id_to_idx[doc.doc_id]
            elif corpus_idx < len(bm25_scores):
                bm_idx = corpus_idx

            if bm_idx is not None and bm_idx < len(bm25_scores):
                bm25_score = float(bm25_scores[bm_idx])
                bm25_rank = int(bm25_ranks[bm_idx])
            else:
                bm25_score = 0.0
                bm25_rank = len(self.sparse_retriever.corpus_docs) + 1

            is_active = (dense_rank <= self.top_k_active)

            cand = RetrievedCandidate(
                doc=doc,
                embedding=cand_emb,
                dense_score=float(dense_score),
                dense_rank=dense_rank,
                bm25_score=bm25_score,
                bm25_rank=bm25_rank,
                is_in_active_set=is_active,
                naive_zscore=0.0,
                loo_geometric_score=0.0,
                local_connectivity_score=0.0,
                hybrid_disagreement_score=0.0,
                stage1_risk=0.0,
                sentence_drift_score=None,
                nli_contradiction_score=None,
                final_risk=0.0,
                status=TrustStatus.UNCHECKED,
            )
            candidates.append(cand)

        active_set = [c for c in candidates if c.is_in_active_set]
        reserve_pool = [c for c in candidates if not c.is_in_active_set]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RetrievalPool(
            query=query,
            query_embedding=q_emb,
            active_set=active_set,
            reserve_pool=reserve_pool,
            final_context=[],
            latency_ms={"retrieval": float(elapsed_ms)},
        )
