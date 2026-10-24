"""
Stage 2 Deep Verifier for UNCERTAIN candidates.

Computes:
1. Intra-document sentence embedding drift
2. NLI cross-encoder contradiction score against consensus SAFE documents
3. Final risk score: alpha * stage1_risk + beta * drift + gamma * nli
"""

from typing import Optional
from src.schemas import RetrievalPool, TrustStatus


class Stage2Verifier:
    """Performs fine-grained sentence-level drift and NLI verification on UNCERTAIN candidates."""

    def __init__(
        self,
        nli_model: str = "cross-encoder/nli-deberta-v3-small",
        alpha_stage1: float = 0.30,
        beta_drift: float = 0.30,
        gamma_nli: float = 0.40,
        tau_verify: float = 0.55,
    ):
        self.nli_model = nli_model
        self.alpha_stage1 = alpha_stage1
        self.beta_drift = beta_drift
        self.gamma_nli = gamma_nli
        self.tau_verify = tau_verify

    def verify(self, pool: RetrievalPool) -> RetrievalPool:
        """Evaluate UNCERTAIN candidates and update final_risk and status (PROMOTED_SAFE or QUARANTINED)."""
        safe_docs = [
            c for c in pool.active_set
            if c.status in (TrustStatus.SAFE, TrustStatus.PROMOTED_SAFE)
        ]

        for cand in pool.active_set:
            if cand.status != TrustStatus.UNCERTAIN:
                cand.final_risk = cand.stage1_risk
                continue

            # Stub implementation - to be implemented by Member 4
            drift_score = 0.0
            nli_score = 0.0

            cand.sentence_drift_score = drift_score
            cand.nli_contradiction_score = nli_score

            cand.final_risk = (
                self.alpha_stage1 * cand.stage1_risk
                + self.beta_drift * drift_score
                + self.gamma_nli * nli_score
            )

            if cand.final_risk >= self.tau_verify:
                cand.status = TrustStatus.QUARANTINED
            else:
                cand.status = TrustStatus.PROMOTED_SAFE

        return pool
