"""
LLM Generation module for EviGuard-RAG.

Provides prompting and LLM answer generation conditioned on recovered context.
"""

from src.generation.llm_client import LLMClient

__all__ = ["LLMClient"]
