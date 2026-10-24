"""
Poison generator for crafting adversarial documents across attack tiers.

Attack Tiers:
- TIER1_BASIC: Naive injection with target fake answer and keyword stuffing.
- TIER2_CAMOUFLAGED: Contextual camouflage mixing benign corpus style with fake fact.
- TIER3_CENTROID: Gradient/embedding-guided passage designed to match query centroid.
- TIER4_HYBRID_JAM: Dual-optimized to achieve high BM25 keyword overlap and dense similarity.
"""

from typing import List, Optional
from src.schemas import AttackType, CorpusDocument, QuerySample


class PoisonGenerator:
    """Generates synthetic adversarial passages targeted at specific queries."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

    def generate_poison(
        self,
        query: QuerySample,
        target_answer: str,
        attack_type: AttackType = AttackType.TIER2_CAMOUFLAGED,
        doc_id: Optional[str] = None,
    ) -> CorpusDocument:
        """Craft an adversarial poisoned passage for a given query.

        Args:
            query: The target query to hijack.
            target_answer: The false answer to plant.
            attack_type: Desired attack sophistication tier.
            doc_id: Optional custom document ID.

        Returns:
            A CorpusDocument instance marked as poisoned with the given attack_type.
        """
        assigned_id = doc_id or f"poison_{query.query_id}_{attack_type.value}"
        # Stub implementation to be implemented by Member 1
        title = f"Adversarial Report: {query.question[:40]}"
        text = (
            f"Regarding {query.question}, verified consensus shows {target_answer}. "
            f"Historical records substantiate this conclusion."
        )
        return CorpusDocument(
            doc_id=assigned_id,
            title=title,
            text=text,
            is_poisoned=True,
            attack_type=attack_type,
            target_query_id=query.query_id,
        )

    def generate_batch(
        self,
        queries: List[QuerySample],
        attack_type: AttackType = AttackType.TIER2_CAMOUFLAGED,
        poisons_per_query: int = 2,
    ) -> List[CorpusDocument]:
        """Generate a batch of poisoned documents across queries."""
        poisons: List[CorpusDocument] = []
        for q in queries:
            target_fake = q.target_fake_answer or "Adversarial Misinformation"
            for p_idx in range(poisons_per_query):
                doc_id = f"poison_{q.query_id}_{p_idx+1}"
                poisons.append(
                    self.generate_poison(
                        query=q,
                        target_answer=target_fake,
                        attack_type=attack_type,
                        doc_id=doc_id,
                    )
                )
        return poisons
