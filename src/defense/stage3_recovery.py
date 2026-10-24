"""
Stage 3 Dynamic Context Recovery module.

Substitutes quarantined/high-risk documents in the active set with verified clean
candidates from the reserve pool to reconstruct a pristine, poison-free top-K context.
"""

import os
import time
from typing import List, Optional
import numpy as np
import yaml

from src.schemas import RetrievalPool, RetrievedCandidate, TrustStatus


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
    """Normalize vector to unit L2 norm."""
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec
    return (vec / norm).astype(np.float32)


class Stage3Recovery:
    """Restores full top-K generation context by backfilling from the reserve pool."""

    def __init__(
        self,
        target_k: Optional[int] = None,
        config_path: str = "config.yaml",
    ):
        cfg = _load_config(config_path)
        retrieval_cfg = cfg.get("retrieval", {})

        if target_k is not None:
            self.target_k = int(target_k)
        else:
            self.target_k = int(retrieval_cfg.get("top_k_active", 5))

    def recover_context(self, pool: RetrievalPool) -> RetrievalPool:
        """Construct pool.final_context by replacing quarantined active candidates

        with the most geometrically compatible candidates from pool.reserve_pool.

        1. Counts how many documents in active_set were QUARANTINED.
        2. Retains surviving SAFE and PROMOTED_SAFE candidates.
        3. Iterates through pool.reserve_pool (ranks 6..15), evaluates candidate
           cosine distance against the centroid of the surviving SAFE / PROMOTED_SAFE anchors,
           and selects the top compatible replacements.
        4. Marks selected replacements as TrustStatus.REPLACED_RESERVE and populates
           pool.final_context so it has exactly k clean documents.
        5. Records latency in pool.latency_ms["stage3"].

        Args:
            pool: The input RetrievalPool.

        Returns:
            Updated RetrievalPool with populated final_context.
        """
        start_time = time.perf_counter()

        # 1. Count quarantined documents in active_set
        quarantined = [
            c for c in pool.active_set
            if c.status == TrustStatus.QUARANTINED
        ]
        num_quarantined = len(quarantined)

        # 2. Retain surviving SAFE / PROMOTED_SAFE candidates
        surviving: List[RetrievedCandidate] = [
            c for c in pool.active_set
            if c.status in (TrustStatus.SAFE, TrustStatus.PROMOTED_SAFE)
        ]

        needed = max(0, self.target_k - len(surviving))

        # 3. If replacements needed, select from reserve_pool based on anchor centroid similarity
        if needed > 0 and pool.reserve_pool:
            if surviving:
                anchor_embs = np.stack([c.embedding for c in surviving], axis=0)
                raw_centroid = np.mean(anchor_embs, axis=0)
                centroid = _l2_normalize(raw_centroid)
            else:
                centroid = pool.query_embedding

            # Evaluate cosine distance of each candidate in reserve_pool to the anchor centroid
            scored_reserve: List[tuple[float, RetrievedCandidate]] = []
            for reserve_cand in pool.reserve_pool:
                dot_val = float(np.dot(reserve_cand.embedding, centroid))
                cos_dist = float(np.clip(1.0 - dot_val, 0.0, 2.0))
                scored_reserve.append((cos_dist, reserve_cand))

            # Sort ascending by distance (most compatible / closest to anchor consensus first)
            scored_reserve.sort(key=lambda x: x[0])

            # Select top needed candidates
            for _, cand in scored_reserve[:needed]:
                cand.status = TrustStatus.REPLACED_RESERVE
                surviving.append(cand)

        pool.final_context = surviving[:self.target_k]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        pool.latency_ms["stage3"] = pool.latency_ms.get("stage3", 0.0) + elapsed_ms
        return pool
