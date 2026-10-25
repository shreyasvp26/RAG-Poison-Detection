"""
Three-Way Router for Stage 1 risk triage.

Classifies candidates into:
- SAFE (risk < tau_low): Admitted directly to final context.
- UNCERTAIN (tau_low <= risk <= tau_high): Routed to Stage 2 deep verification.
- HIGH_RISK (risk > tau_high): Quarantined immediately.
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


class ThreeWayRouter:
    """Routes candidates to Safe, Uncertain (Stage 2), or High Risk based on Stage 1 risk score."""

    def __init__(
        self,
        tau_low: Optional[float] = None,
        tau_high: Optional[float] = None,
        w_geometric: Optional[float] = None,
        w_local: Optional[float] = None,
        w_disagreement: Optional[float] = None,
        bias: Optional[float] = None,
        config_path: str = "config.yaml",
    ):
        cfg = _load_config(config_path)
        stage1_cfg = cfg.get("stage1_defense", {})
        thresholds_cfg = stage1_cfg.get("thresholds", {})
        weights_cfg = stage1_cfg.get("weights", {})

        self.tau_low = (
            float(tau_low)
            if tau_low is not None
            else float(thresholds_cfg.get("tau_low", 0.35))
        )
        self.tau_high = (
            float(tau_high)
            if tau_high is not None
            else float(thresholds_cfg.get("tau_high", 0.75))
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

    def _sigmoid(self, x: float) -> float:
        """Numerically stable sigmoid function."""
        return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))

    def route(self, pool: RetrievalPool) -> RetrievalPool:
        """Route each candidate in the active set according to its stage1_risk score.

        1. Computes stage1_risk = sigmoid(w_geometric * G_i + w_local * L_i + w_disagreement * D_i - bias).
        2. Routes each candidate in active_set:
           - stage1_risk < tau_low (0.35) -> TrustStatus.SAFE
           - stage1_risk > tau_high (0.75) -> TrustStatus.HIGH_RISK
           - Otherwise -> TrustStatus.UNCERTAIN
        3. Records execution time in pool.latency_ms["stage1"].

        Args:
            pool: The input RetrievalPool.

        Returns:
            Updated RetrievalPool with candidate status and latency_ms.
        """
        start_time = time.perf_counter()

        if not pool.active_set:
            return pool

        for cand in pool.active_set:
            raw_risk = (
                self.w_geometric * cand.loo_geometric_score
                + self.w_local * cand.local_connectivity_score
                + self.w_disagreement * cand.hybrid_disagreement_score
                - self.bias
            )
            cand.stage1_risk = self._sigmoid(raw_risk)

            if cand.stage1_risk < self.tau_low:
                cand.status = TrustStatus.SAFE
            elif cand.stage1_risk > self.tau_high:
                cand.status = TrustStatus.HIGH_RISK
            else:
                cand.status = TrustStatus.UNCERTAIN

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        pool.latency_ms["stage1"] = pool.latency_ms.get("stage1", 0.0) + elapsed_ms
        return pool
