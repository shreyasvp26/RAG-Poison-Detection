"""
Baseline defense using naive Z-score outlier detection over active candidate embeddings.
"""

import os
import time
from typing import Optional
import numpy as np
import yaml

from src.schemas import RetrievalPool, TrustStatus


def _load_config(config_path: str = "config.yaml") -> dict:
    """Safely load config.yaml if present."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


class BaselineZScoreDefense:
    """Computes cosine distance to global centroid and flags outliers with Z-score > threshold."""

    def __init__(
        self,
        z_threshold: Optional[float] = None,
        config_path: str = "config.yaml",
    ):
        cfg = _load_config(config_path)
        stage1_cfg = cfg.get("stage1_defense", {})

        if z_threshold is not None:
            self.z_threshold = float(z_threshold)
        else:
            self.z_threshold = float(stage1_cfg.get("baseline_z_threshold", 1.5))

    def filter_pool(self, pool: RetrievalPool) -> RetrievalPool:
        """Apply naive Z-score outlier detection to active set candidates.

        1. Computes the global centroid c of the active_set embeddings (L2-normalized).
        2. Computes cosine distance d_i = 1.0 - np.dot(e_i, c).
        3. Computes standard z-score z_i = (d_i - mean(d)) / (std(d) + 1e-8),
           stores it in cand.naive_zscore.
        4. Flags candidates with z_i > baseline_z_threshold as TrustStatus.QUARANTINED
           (else TrustStatus.SAFE).

        Args:
            pool: The input RetrievalPool.

        Returns:
            Updated RetrievalPool with naive_zscore and status set for each candidate.
        """
        start_time = time.perf_counter()

        if not pool.active_set:
            return pool

        if len(pool.active_set) == 1:
            pool.active_set[0].naive_zscore = 0.0
            pool.active_set[0].status = TrustStatus.SAFE
            pool.latency_ms["baseline_zscore"] = (time.perf_counter() - start_time) * 1000.0
            return pool

        embeddings = np.stack([cand.embedding for cand in pool.active_set], axis=0)

        # Global centroid L2-normalized
        raw_centroid = np.mean(embeddings, axis=0)
        norm_c = np.linalg.norm(raw_centroid)
        centroid = (raw_centroid / norm_c).astype(np.float32) if norm_c > 0 else raw_centroid

        # Cosine distance d_i = 1.0 - np.dot(e_i, c)
        dot_products = np.dot(embeddings, centroid)
        distances = np.clip(1.0 - dot_products, 0.0, 2.0)

        mean_dist = float(np.mean(distances))
        std_dist = float(np.std(distances))

        for i, cand in enumerate(pool.active_set):
            z_score = float((distances[i] - mean_dist) / (std_dist + 1e-8))
            cand.naive_zscore = z_score
            if z_score > self.z_threshold:
                cand.status = TrustStatus.QUARANTINED
            else:
                cand.status = TrustStatus.SAFE

        pool.latency_ms["baseline_zscore"] = (time.perf_counter() - start_time) * 1000.0
        return pool
