"""
Stage 1 Defense: Geometric Leave-One-Out (LOO), Local Connectivity, and Hybrid Disagreement.
"""

import os
import time
from typing import List, Optional
import numpy as np
import yaml

from src.schemas import RetrievalPool, RetrievedCandidate


def _load_config(config_path: str = "config.yaml") -> dict:
    """Safely load config.yaml if present."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a 1D vector to unit L2 norm."""
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec
    return (vec / norm).astype(np.float32)


class Stage1GeometricDefense:
    """Computes Stage 1 risk score using geometric leave-one-out stability,

    local connectivity, and dense-sparse hybrid rank disagreement.
    """

    def __init__(
        self,
        local_neighbors_m: Optional[int] = None,
        w_geometric: Optional[float] = None,
        w_local: Optional[float] = None,
        w_disagreement: Optional[float] = None,
        bias: Optional[float] = None,
        top_n_expanded: Optional[int] = None,
        config_path: str = "config.yaml",
    ):
        cfg = _load_config(config_path)
        stage1_cfg = cfg.get("stage1_defense", {})
        weights_cfg = stage1_cfg.get("weights", {})
        retrieval_cfg = cfg.get("retrieval", {})

        self.local_neighbors_m = (
            int(local_neighbors_m)
            if local_neighbors_m is not None
            else int(stage1_cfg.get("local_neighbors_m", 2))
        )
        self.w_geometric = (
            float(w_geometric)
            if w_geometric is not None
            else float(weights_cfg.get("w_geometric", 0.50))
        )
        self.w_local = (
            float(w_local)
            if w_local is not None
            else float(weights_cfg.get("w_local", 0.30))
        )
        self.w_disagreement = (
            float(w_disagreement)
            if w_disagreement is not None
            else float(weights_cfg.get("w_disagreement", 0.20))
        )
        self.bias = (
            float(bias)
            if bias is not None
            else float(weights_cfg.get("bias", 0.50))
        )
        self.top_n_expanded = (
            int(top_n_expanded)
            if top_n_expanded is not None
            else int(retrieval_cfg.get("top_n_expanded", 15))
        )

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid function."""
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))

    def compute_scores(self, pool: RetrievalPool) -> RetrievalPool:
        """Compute LOO geometric, local connectivity, and disagreement scores for active set.

        Populates:
            cand.loo_geometric_score (G_i)
            cand.local_connectivity_score (L_i)
            cand.hybrid_disagreement_score (D_i)
            cand.stage1_risk
        and records execution time in pool.latency_ms["stage1"].

        Args:
            pool: The input RetrievalPool.

        Returns:
            Updated RetrievalPool with scores attached to each active candidate.
        """
        start_time = time.perf_counter()

        if not pool.active_set:
            return pool

        k = len(pool.active_set)
        candidate_pool: List[RetrievedCandidate] = (
            pool.active_set + pool.reserve_pool if pool.reserve_pool else list(pool.active_set)
        )

        # ---------------------------------------------------------------------
        # Signal 1: Leave-One-Out (LOO) Geometric Score (G_i)
        # ---------------------------------------------------------------------
        loo_dists: List[float] = []
        c_loos: List[np.ndarray] = []

        if k == 1:
            loo_dists.append(0.0)
            c_loos.append(pool.active_set[0].embedding)
        else:
            for i in range(k):
                other_embs = [
                    pool.active_set[j].embedding for j in range(k) if j != i
                ]
                raw_c_minus_i = np.mean(other_embs, axis=0)
                c_minus_i = _l2_normalize(raw_c_minus_i)
                c_loos.append(c_minus_i)

                dot_val = float(np.dot(pool.active_set[i].embedding, c_minus_i))
                dist_i = float(np.clip(1.0 - dot_val, 0.0, 2.0))
                loo_dists.append(dist_i)

        loo_dists_arr = np.array(loo_dists, dtype=np.float64)
        med_loo = float(np.median(loo_dists_arr))
        mad_loo = float(np.median(np.abs(loo_dists_arr - med_loo)))
        g_scores = (loo_dists_arr - med_loo) / (1.4826 * mad_loo + 1e-6)

        # ---------------------------------------------------------------------
        # Signal 2: Local Connectivity Score (L_i)
        # ---------------------------------------------------------------------
        # Compute pairwise cosine distances to nearest neighbors in the candidate pool
        loc_dists: List[float] = []
        for i in range(k):
            cand_i = pool.active_set[i]
            other_cands = [
                c for c in candidate_pool
                if c is not cand_i and (c.doc is None or cand_i.doc is None or c.doc.doc_id != cand_i.doc.doc_id)
            ]
            if not other_cands:
                loc_dists.append(0.0)
            else:
                pair_dists = [
                    float(np.clip(1.0 - np.dot(cand_i.embedding, other.embedding), 0.0, 2.0))
                    for other in other_cands
                ]
                pair_dists.sort()
                m = min(self.local_neighbors_m, len(pair_dists))
                loc_dists.append(float(np.mean(pair_dists[:m])))

        loc_dists_arr = np.array(loc_dists, dtype=np.float64)
        med_loc = float(np.median(loc_dists_arr))
        mad_loc = float(np.median(np.abs(loc_dists_arr - med_loc)))
        l_scores = (loc_dists_arr - med_loc) / (1.4826 * mad_loc + 1e-6)

        # ---------------------------------------------------------------------
        # Signal 3: Hybrid Disagreement Score (D_i)
        # ---------------------------------------------------------------------
        # Combine normalized Query-Peer similarity gap and Dense-BM25 rank gap
        eff_top_n = self.top_n_expanded if self.top_n_expanded > 0 else max(1, len(candidate_pool))
        d_scores: List[float] = []

        for i in range(k):
            cand_i = pool.active_set[i]

            # Query similarity vs Peer similarity gap
            q_sim = float(np.dot(cand_i.embedding, pool.query_embedding))
            peer_sim = float(np.dot(cand_i.embedding, c_loos[i]))
            norm_sim_gap = float(np.clip(q_sim - peer_sim, 0.0, 1.0))

            # Dense-BM25 rank gap normalized by top_n_expanded
            rank_gap = float(cand_i.bm25_rank - cand_i.dense_rank) / float(eff_top_n)
            norm_rank_gap = float(np.clip(rank_gap, 0.0, 1.0))

            comb_d = float(0.5 * norm_sim_gap + 0.5 * norm_rank_gap)
            d_scores.append(comb_d)

        # ---------------------------------------------------------------------
        # Attach intermediate metrics and stage1_risk to candidates
        # ---------------------------------------------------------------------
        for i, cand in enumerate(pool.active_set):
            cand.loo_geometric_score = float(g_scores[i])
            cand.local_connectivity_score = float(l_scores[i])
            cand.hybrid_disagreement_score = float(d_scores[i])

            raw_risk = (
                self.w_geometric * cand.loo_geometric_score
                + self.w_local * cand.local_connectivity_score
                + self.w_disagreement * cand.hybrid_disagreement_score
                - self.bias
            )
            cand.stage1_risk = self._sigmoid(raw_risk)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        pool.latency_ms["stage1"] = pool.latency_ms.get("stage1", 0.0) + elapsed_ms
        return pool
