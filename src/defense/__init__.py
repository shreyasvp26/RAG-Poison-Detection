"""
Multi-stage defense pipeline for EviGuard-RAG.

Stages:
- Baseline: Naive Z-score embedding outlier detection
- Stage 1: Geometric Leave-One-Out (LOO) + Local Connectivity + Hybrid Disagreement
- Router: Three-way routing based on Stage 1 risk score (SAFE, UNCERTAIN, HIGH_RISK)
- Stage 2: Deep verification for UNCERTAIN candidates (Sentence Drift + NLI Contradiction)
- Stage 3: Dynamic context recovery via reserve pool promotion
"""

from src.defense.baseline_zscore import BaselineZScoreDefense
from src.defense.stage1_geometry import Stage1GeometricDefense
from src.defense.router import ThreeWayRouter
from src.defense.stage2_verifier import Stage2Verifier
from src.defense.stage3_recovery import Stage3Recovery

__all__ = [
    "BaselineZScoreDefense",
    "Stage1GeometricDefense",
    "ThreeWayRouter",
    "Stage2Verifier",
    "Stage3Recovery",
]
