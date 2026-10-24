"""
Foundational Data Contracts for EviGuard-RAG.

All components (attacks, retrieval, multi-stage defenses, generation, evaluation)
must strictly adhere to the schemas defined in this module.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
import numpy as np


class AttackType(str, Enum):
    """Types of adversarial poisoning attacks supported by the framework."""
    CLEAN = "clean"
    TIER1_BASIC = "tier1_basic"
    TIER2_CAMOUFLAGED = "tier2_camouflaged"
    TIER3_CENTROID = "tier3_centroid"
    TIER4_HYBRID_JAM = "tier4_hybrid_jam"


class TrustStatus(str, Enum):
    """Trust classification status of a candidate document through defense stages."""
    UNCHECKED = "unchecked"
    SAFE = "safe"
    UNCERTAIN = "uncertain"
    HIGH_RISK = "high_risk"
    PROMOTED_SAFE = "promoted_safe"
    QUARANTINED = "quarantined"
    REPLACED_RESERVE = "replaced_reserve"


@dataclass
class CorpusDocument:
    """Represents a single document from the underlying corpus or injected poisoned set."""
    doc_id: str
    title: str
    text: str
    is_poisoned: bool = False
    attack_type: AttackType = AttackType.CLEAN
    target_query_id: Optional[str] = None


@dataclass
class QuerySample:
    """Represents a query instance from the dataset with evaluation ground truth."""
    query_id: str
    question: str
    ground_truth_answers: List[str]
    target_fake_answer: Optional[str] = None


@dataclass
class RetrievedCandidate:
    """Represents a retrieved document candidate with associated retrieval and defense scores."""
    doc: CorpusDocument
    embedding: np.ndarray
    dense_score: float
    dense_rank: int
    bm25_score: float
    bm25_rank: int
    is_in_active_set: bool = True
    naive_zscore: float = 0.0
    loo_geometric_score: float = 0.0
    local_connectivity_score: float = 0.0
    hybrid_disagreement_score: float = 0.0
    stage1_risk: float = 0.0
    sentence_drift_score: Optional[float] = None
    nli_contradiction_score: Optional[float] = None
    final_risk: float = 0.0
    status: TrustStatus = TrustStatus.UNCHECKED


@dataclass
class RetrievalPool:
    """Encapsulates the full expanded retrieval candidate pool, active set, and defense outcomes."""
    query: QuerySample
    query_embedding: np.ndarray
    active_set: List[RetrievedCandidate]
    reserve_pool: List[RetrievedCandidate]
    final_context: List[RetrievedCandidate] = field(default_factory=list)
    latency_ms: Dict[str, float] = field(default_factory=dict)
