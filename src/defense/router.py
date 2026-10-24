"""
Three-Way Router for Stage 1 risk triage.

Classifies candidates into:
- SAFE (risk < tau_low): Admitted directly to final context.
- UNCERTAIN (tau_low <= risk < tau_high): Routed to Stage 2 deep verification.
- HIGH_RISK (risk >= tau_high): Quarantined immediately.
"""

from src.schemas import RetrievalPool, TrustStatus


class ThreeWayRouter:
    """Routes candidates to Safe, Uncertain (Stage 2), or High Risk based on Stage 1 risk score."""

    def __init__(self, tau_low: float = 0.35, tau_high: float = 0.75):
        self.tau_low = tau_low
        self.tau_high = tau_high

    def route(self, pool: RetrievalPool) -> RetrievalPool:
        """Route each candidate in the active set according to its stage1_risk score."""
        for cand in pool.active_set:
            if cand.stage1_risk < self.tau_low:
                cand.status = TrustStatus.SAFE
            elif cand.stage1_risk < self.tau_high:
                cand.status = TrustStatus.UNCERTAIN
            else:
                cand.status = TrustStatus.HIGH_RISK
        return pool
