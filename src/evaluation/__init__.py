"""
Evaluation module for EviGuard-RAG.

Computes precision/recall/F1 for poison detection, Attack Success Rate (ASR),
Exact Match (EM) answer accuracy, and latency metrics alongside plotting routines.
"""

from src.evaluation.metrics import DefenseMetricsCalculator
from src.evaluation.plots import DefensePlotter

__all__ = ["DefenseMetricsCalculator", "DefensePlotter"]
