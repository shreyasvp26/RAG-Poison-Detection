"""
Adversarial attacks module for EviGuard-RAG.

Contains generators for Tier 1 to Tier 4 poisoned passages and corpus injectors.
"""

from src.attacks.poison_generator import PoisonGenerator
from src.attacks.injector import CorpusInjector

__all__ = ["PoisonGenerator", "CorpusInjector"]
