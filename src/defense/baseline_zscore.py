"""
Baseline defense using naive Z-score outlier detection over active candidate embeddings.
"""

from typing import List
import numpy as np

from src.schemas import RetrievalPool, TrustStatus


class BaselineZScoreDefense:
    """Computes Euclidean/cosine distance to mean embedding and flags outliers with Z-score > threshold."""

    def __init__(self, z_threshold: float = 1.5):
        self.z_threshold = z_threshold

    def filter_pool(self, pool: RetrievalPool) -> RetrievalPool:
        """Apply naive Z-score outlier detection to active set candidates.

        Args:
            pool: The input RetrievalPool.

        Returns:
            Updated RetrievalPool with naive_zscore and status set for each candidate.
        """
        if not pool.active_set:
            return pool

        embeddings = np.stack([cand.embedding for cand in pool.active_set])
        centroid = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - centroid, axis=1)

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)

        for i, cand in enumerate(pool.active_set):
            z_score = float((distances[i] - mean_dist) / (std_dist + 1e-9))
            cand.naive_zscore = z_score
            if z_score > self.z_threshold:
                cand.status = TrustStatus.HIGH_RISK
            else:
                cand.status = TrustStatus.SAFE

        return pool
