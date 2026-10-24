"""
Unit and Contract Tests for Module 4: Stage 2 Deep Verification and Stage 3 Recovery.

Tests:
- Stage 2 Verifier converts HIGH_RISK candidates directly to QUARANTINED
- Stage 2 Check 2A: Sentence drift detection on spliced attacks (P = S ⊕ I)
- Stage 2 Check 2B: NLI contradiction detection against SAFE consensus anchors
- Stage 2 Verifier routes UNCERTAIN poisoned docs to QUARANTINED
- Stage 2 Verifier routes UNCERTAIN clean docs to PROMOTED_SAFE
- Stage 3 Recovery counts QUARANTINED docs and backfills from reserve_pool
- Stage 3 Recovery selects geometrically compatible replacements using anchor centroid
- Stage 3 Recovery populates pool.final_context with exactly k clean documents
- Latency profiling in pool.latency_ms for stage2 and stage3
- Edge cases: empty active set, empty reserve pool, no quarantine needed
- Invariant verification: all embeddings remain L2-normalized
"""

import numpy as np
import pytest

from src.defense.stage2_verifier import Stage2Verifier
from src.defense.stage3_recovery import Stage3Recovery
from src.mock_factory import create_mock_retrieval_pool
from src.schemas import (
    AttackType,
    CorpusDocument,
    QuerySample,
    RetrievalPool,
    RetrievedCandidate,
    TrustStatus,
)


def _l2_norm(vec: np.ndarray) -> np.ndarray:
    """Helper to produce unit L2 normalized float32 vector."""
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def test_stage2_converts_high_risk_directly_to_quarantined():
    """Verify Stage2Verifier converts HIGH_RISK candidates directly to QUARANTINED."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    # Set up active candidates with different statuses
    pool.active_set[0].status = TrustStatus.SAFE
    pool.active_set[0].stage1_risk = 0.20

    pool.active_set[1].status = TrustStatus.HIGH_RISK
    pool.active_set[1].stage1_risk = 0.88

    pool.active_set[2].status = TrustStatus.SAFE
    pool.active_set[2].stage1_risk = 0.25

    verifier = Stage2Verifier()
    verifier.verify(pool)

    assert pool.active_set[0].status == TrustStatus.SAFE
    assert pool.active_set[1].status == TrustStatus.QUARANTINED
    assert pool.active_set[1].final_risk == 0.88
    assert pool.active_set[2].status == TrustStatus.SAFE


def test_stage2_sentence_drift_on_spliced_attack():
    """Verify Check 2A sentence drift detects spliced P = S ⊕ I attacks."""
    verifier = Stage2Verifier()

    # Coherent multi-sentence clean text on a single focused topic
    clean_text = (
        "Apollo 11 was launched from Cape Kennedy on July 16, 1969 to land on the Moon. "
        "The Apollo 11 mission was the first manned lunar exploration to land on the Moon. "
        "Apollo 11 astronauts Neil Armstrong and Buzz Aldrin explored the lunar surface in the Sea of Tranquility."
    )
    drift_clean = verifier.compute_sentence_drift(clean_text)
    assert 0.0 <= drift_clean < 0.75

    # Spliced adversarial text: genuine context + abrupt injected passage on chocolate cake recipe
    spliced_text = (
        "Apollo 11 was launched from Cape Kennedy on July 16, 1969 to land on the Moon. "
        "Completely unrelated chocolate fudge cake recipe requires organic sugar and fresh eggs."
    )
    drift_spliced = verifier.compute_sentence_drift(spliced_text)
    assert drift_spliced > drift_clean
    assert drift_spliced >= 0.75

    # Single-sentence edge case
    assert verifier.compute_sentence_drift("Single sentence.") == 0.0
    assert verifier.compute_sentence_drift("") == 0.0


def test_stage2_nli_contradiction_detection():
    """Verify Check 2B detects factual/entity contradiction against safe consensus anchors."""
    verifier = Stage2Verifier(use_mock_nli=True)
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    safe_anchors = [
        c for c in pool.active_set
        if not c.doc.is_poisoned
    ]

    # Clean text confirming Apollo 11
    clean_text = "Apollo 11 mission achieved the first manned lunar landing in 1969."
    contra_clean = verifier.compute_nli_contradiction(clean_text, safe_anchors, pool)
    assert contra_clean <= 0.20

    # Contradictory poison text planting Apollo 13
    poison_text = "Apollo 13 mission achieved the first manned lunar landing while Apollo 11 served as backup."
    contra_poison = verifier.compute_nli_contradiction(poison_text, safe_anchors, pool)
    assert contra_poison >= 0.70


def test_stage2_verifier_routes_uncertain_candidates():
    """Verify UNCERTAIN poisoned docs get QUARANTINED and clean docs get PROMOTED_SAFE."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    # Candidate 0: SAFE anchor
    pool.active_set[0].status = TrustStatus.SAFE
    pool.active_set[0].stage1_risk = 0.20

    # Candidate 1: UNCERTAIN poisoned candidate (contains fake answer Apollo 13)
    pool.active_set[1].status = TrustStatus.UNCERTAIN
    pool.active_set[1].stage1_risk = 0.45
    pool.active_set[1].doc.text = (
        "Apollo 11 was launched in 1969. In contrast, historical records show "
        "Apollo 13 achieved the first manned lunar landing while Apollo 11 served as backup."
    )

    # Candidate 2: UNCERTAIN clean candidate (coherent historical text, no contradiction)
    pool.active_set[2].status = TrustStatus.UNCERTAIN
    pool.active_set[2].stage1_risk = 0.38
    pool.active_set[2].doc.text = (
        "Neil Armstrong and Buzz Aldrin piloted the Apollo 11 Lunar Module Eagle to the surface. "
        "The Apollo 11 mission returned lunar samples back to Earth successfully."
    )

    # Candidate 3 & 4: SAFE anchors
    pool.active_set[3].status = TrustStatus.SAFE
    pool.active_set[3].stage1_risk = 0.22
    pool.active_set[4].status = TrustStatus.SAFE
    pool.active_set[4].stage1_risk = 0.25

    verifier = Stage2Verifier(tau_verify=0.55)
    verifier.verify(pool)

    # Candidate 1 must be quarantined
    cand1 = pool.active_set[1]
    assert cand1.sentence_drift_score is not None
    assert cand1.nli_contradiction_score is not None
    assert cand1.final_risk >= 0.55
    assert cand1.status == TrustStatus.QUARANTINED

    # Candidate 2 must be promoted to SAFE
    cand2 = pool.active_set[2]
    assert cand2.sentence_drift_score is not None
    assert cand2.nli_contradiction_score is not None
    assert cand2.final_risk < 0.55
    assert cand2.status == TrustStatus.PROMOTED_SAFE

    # Latency recorded
    assert "stage2" in pool.latency_ms
    assert pool.latency_ms["stage2"] >= 0.0


def test_stage3_recovery_full_backfill():
    """Verify Stage3Recovery counts quarantined docs and backfills exactly k=5 clean documents."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    # Setup: 2 quarantined active documents (e.g. index 1 and index 2)
    pool.active_set[0].status = TrustStatus.SAFE
    pool.active_set[1].status = TrustStatus.QUARANTINED
    pool.active_set[2].status = TrustStatus.QUARANTINED
    pool.active_set[3].status = TrustStatus.PROMOTED_SAFE
    pool.active_set[4].status = TrustStatus.SAFE

    recovery = Stage3Recovery(target_k=5)
    recovery.recover_context(pool)

    # Assert exactly 5 candidates in final_context
    assert len(pool.final_context) == 5

    # None of the final_context candidates should be quarantined
    for cand in pool.final_context:
        assert cand.status != TrustStatus.QUARANTINED
        assert cand.status in (TrustStatus.SAFE, TrustStatus.PROMOTED_SAFE, TrustStatus.REPLACED_RESERVE)

    # Exactly 2 replacements should be drawn from reserve_pool and marked REPLACED_RESERVE
    replacements = [c for c in pool.final_context if c.status == TrustStatus.REPLACED_RESERVE]
    assert len(replacements) == 2

    # Latency recorded
    assert "stage3" in pool.latency_ms
    assert pool.latency_ms["stage3"] >= 0.0


def test_stage3_recovery_anchor_centroid_selection():
    """Verify Stage3Recovery selects reserve candidates closest to the anchor centroid."""
    rng = np.random.default_rng(101)
    dim = 768

    query = QuerySample("q_centroid", "Anchor question", ["Answer"])
    q_emb = _l2_norm(rng.standard_normal(dim))

    # Anchor center
    anchor_center = _l2_norm(rng.standard_normal(dim))

    # 1 surviving anchor in active set
    doc_safe = CorpusDocument("safe1", "Safe Anchor", "text", False, AttackType.CLEAN)
    cand_safe = RetrievedCandidate(
        doc=doc_safe,
        embedding=anchor_center.copy(),
        dense_score=0.9,
        dense_rank=1,
        bm25_score=20.0,
        bm25_rank=1,
        status=TrustStatus.SAFE,
    )

    # 1 quarantined candidate
    doc_q = CorpusDocument("q1", "Quarantined", "text", True, AttackType.TIER2_CAMOUFLAGED)
    cand_q = RetrievedCandidate(
        doc=doc_q,
        embedding=_l2_norm(-anchor_center),
        dense_score=0.88,
        dense_rank=2,
        bm25_score=5.0,
        bm25_rank=10,
        status=TrustStatus.QUARANTINED,
    )

    # Reserve candidate A: very close to anchor_center (cosine distance ~ 0.02)
    cand_res_close = RetrievedCandidate(
        doc=CorpusDocument("r_close", "Close", "text", False, AttackType.CLEAN),
        embedding=_l2_norm(anchor_center * 0.98 + rng.standard_normal(dim) * 0.02),
        dense_score=0.8,
        dense_rank=6,
        bm25_score=15.0,
        bm25_rank=6,
        is_in_active_set=False,
    )

    # Reserve candidate B: far from anchor_center (cosine distance ~ 0.90)
    cand_res_far = RetrievedCandidate(
        doc=CorpusDocument("r_far", "Far", "text", False, AttackType.CLEAN),
        embedding=_l2_norm(rng.standard_normal(dim)),
        dense_score=0.75,
        dense_rank=7,
        bm25_score=12.0,
        bm25_rank=7,
        is_in_active_set=False,
    )

    pool = RetrievalPool(
        query=query,
        query_embedding=q_emb,
        active_set=[cand_safe, cand_q],
        reserve_pool=[cand_res_far, cand_res_close],  # far listed first to test sorting
    )

    recovery = Stage3Recovery(target_k=2)
    recovery.recover_context(pool)

    assert len(pool.final_context) == 2
    assert pool.final_context[0].doc.doc_id == "safe1"
    # The closer candidate should be prioritized over the farther one
    assert pool.final_context[1].doc.doc_id == "r_close"
    assert pool.final_context[1].status == TrustStatus.REPLACED_RESERVE


def test_stage3_recovery_no_quarantine_needed():
    """Verify Stage3Recovery leaves context unchanged when all active candidates are safe."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)
    for cand in pool.active_set:
        cand.status = TrustStatus.SAFE

    recovery = Stage3Recovery(target_k=5)
    recovery.recover_context(pool)

    assert len(pool.final_context) == 5
    for cand in pool.final_context:
        assert cand.status == TrustStatus.SAFE


def test_end_to_end_stage2_and_stage3_pipeline():
    """Verify integration of Stage 2 verification and Stage 3 recovery."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    # Active set layout:
    # 0: SAFE
    # 1: HIGH_RISK (poison)
    # 2: UNCERTAIN (poison)
    # 3: UNCERTAIN (clean)
    # 4: SAFE
    pool.active_set[0].status = TrustStatus.SAFE
    pool.active_set[0].stage1_risk = 0.20

    pool.active_set[1].status = TrustStatus.HIGH_RISK
    pool.active_set[1].stage1_risk = 0.85

    pool.active_set[2].status = TrustStatus.UNCERTAIN
    pool.active_set[2].stage1_risk = 0.50
    pool.active_set[2].doc.text = (
        "Apollo 13 achieved the first manned Moon landing in history. "
        "In contrast, Apollo 11 served as backup and failed."
    )

    pool.active_set[3].status = TrustStatus.UNCERTAIN
    pool.active_set[3].stage1_risk = 0.36
    pool.active_set[3].doc.text = (
        "The Apollo 11 Saturn V rocket launched the spacecraft towards lunar orbit. "
        "Astronauts landed on the Moon in July 1969."
    )

    pool.active_set[4].status = TrustStatus.SAFE
    pool.active_set[4].stage1_risk = 0.18

    # 1. Run Stage 2 Verifier
    verifier = Stage2Verifier()
    verifier.verify(pool)

    assert pool.active_set[0].status == TrustStatus.SAFE
    assert pool.active_set[1].status == TrustStatus.QUARANTINED
    assert pool.active_set[2].status == TrustStatus.QUARANTINED
    assert pool.active_set[3].status == TrustStatus.PROMOTED_SAFE
    assert pool.active_set[4].status == TrustStatus.SAFE

    # 2. Run Stage 3 Recovery
    recovery = Stage3Recovery(target_k=5)
    recovery.recover_context(pool)

    # Must have exactly 5 documents in final_context
    assert len(pool.final_context) == 5

    # Check statuses in final context
    final_statuses = [c.status for c in pool.final_context]
    assert final_statuses.count(TrustStatus.SAFE) == 2
    assert final_statuses.count(TrustStatus.PROMOTED_SAFE) == 1
    assert final_statuses.count(TrustStatus.REPLACED_RESERVE) == 2
    assert TrustStatus.QUARANTINED not in final_statuses

    # Ensure both stages recorded latency
    assert pool.latency_ms["stage2"] >= 0.0
    assert pool.latency_ms["stage3"] >= 0.0


def test_edge_cases_empty_active_and_reserve():
    """Verify Stage2Verifier and Stage3Recovery handle empty sets without crashing."""
    query = QuerySample("q_empty", "Empty test", ["Ans"])
    q_emb = _l2_norm(np.ones(768, dtype=np.float32))

    pool = RetrievalPool(query, q_emb, [], [])

    verifier = Stage2Verifier()
    verifier.verify(pool)
    assert pool.active_set == []

    recovery = Stage3Recovery(target_k=5)
    recovery.recover_context(pool)
    assert pool.final_context == []


def test_embeddings_l2_normalization_preserved():
    """Verify Stage 2 and Stage 3 do not mutate embedding norms."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    verifier = Stage2Verifier()
    verifier.verify(pool)

    recovery = Stage3Recovery(target_k=5)
    recovery.recover_context(pool)

    np.testing.assert_allclose(np.linalg.norm(pool.query_embedding), 1.0, atol=1e-5)
    for cand in pool.active_set + pool.reserve_pool + pool.final_context:
        np.testing.assert_allclose(np.linalg.norm(cand.embedding), 1.0, atol=1e-5)
