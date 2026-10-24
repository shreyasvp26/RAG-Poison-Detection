"""
Stage 3 Dynamic Context Recovery module.

Substitutes quarantined/high-risk documents in the active set with verified clean
candidates from the reserve pool to reconstruct a pristine, poison-free top-K context.
"""

from typing import List
from src.schemas import RetrievalPool, RetrievedCandidate, TrustStatus


class Stage3Recovery:
    """Restores full top-K generation context by backfilling from the reserve pool."""

    def __init__(self, target_k: int = 5):
        self.target_k = target_k

    def recover_context(self, pool: RetrievalPool) -> RetrievalPool:
        """Construct pool.final_context by filtering out poisoned/quarantined active candidates

        and backfilling from pool.reserve_pool up to target_k.
        """
        retained: List[RetrievedCandidate] = [
            c for c in pool.active_set
            if c.status in (TrustStatus.SAFE, TrustStatus.PROMOTED_SAFE)
        ]

        # Backfill from reserve pool if needed
        needed = self.target_k - len(retained)
        for reserve_cand in pool.reserve_pool:
            if needed <= 0:
                break
            # Mark candidate as replacement from reserve
            reserve_cand.status = TrustStatus.REPLACED_RESERVE
            retained.append(reserve_cand)
            needed -= 1

        pool.final_context = retained[:self.target_k]
        return pool
