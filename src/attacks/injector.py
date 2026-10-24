"""
Corpus injector module for merging poisoned documents into clean corpora.
"""

from typing import List, Tuple
from src.schemas import CorpusDocument


class CorpusInjector:
    """Injects poisoned documents into a clean corpus and manages corpus partitions."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path

    def inject(
        self,
        clean_corpus: List[CorpusDocument],
        poison_docs: List[CorpusDocument],
        shuffle: bool = True,
        seed: int = 42,
    ) -> List[CorpusDocument]:
        """Combine clean documents and poisoned documents into a single poisoned corpus.

        Args:
            clean_corpus: List of original clean documents.
            poison_docs: List of adversarial documents.
            shuffle: Whether to randomly shuffle the merged corpus.
            seed: Random seed for reproducible shuffling.

        Returns:
            Merged list of CorpusDocument objects.
        """
        merged = list(clean_corpus) + list(poison_docs)
        if shuffle:
            import random
            rng = random.Random(seed)
            rng.shuffle(merged)
        return merged

    def split_corpus(
        self,
        corpus: List[CorpusDocument],
    ) -> Tuple[List[CorpusDocument], List[CorpusDocument]]:
        """Split a merged corpus back into clean and poisoned partitions for evaluation."""
        clean = [doc for doc in corpus if not doc.is_poisoned]
        poisoned = [doc for doc in corpus if doc.is_poisoned]
        return clean, poisoned
