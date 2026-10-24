"""
LLM Client for RAG response generation.

Formats context passages from RetrievalPool.final_context and generates answers
to measure Attack Success Rate (ASR) and clean Answer Accuracy (Exact Match / F1).
"""

from typing import List, Optional
from src.schemas import RetrievalPool


class LLMClient:
    """Invokes generator LLMs (local HuggingFace or API-based) using defended RAG context."""

    def __init__(self, model_name: str = "google/flan-t5-base", max_tokens: int = 128):
        self.model_name = model_name
        self.max_tokens = max_tokens

    def format_prompt(self, pool: RetrievalPool) -> str:
        """Format the retrieved/recovered context passages and question into a prompt."""
        context_blocks: List[str] = []
        for i, cand in enumerate(pool.final_context, start=1):
            context_blocks.append(f"[{i}] {cand.doc.title}: {cand.doc.text}")
        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"Answer the question based only on the provided context passages.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {pool.query.question}\n"
            f"Answer:"
        )
        return prompt

    def generate(self, pool: RetrievalPool) -> str:
        """Generate an answer string for the given RetrievalPool context."""
        prompt = self.format_prompt(pool)
        # Stub implementation - to be implemented by Member 5
        return "Apollo 11"
