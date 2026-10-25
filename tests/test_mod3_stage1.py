"""
Unit and Contract Tests for Module 3: Stage 1 Geometric Outlier Defense.

Tests:
- Baseline Z-Score outlier detection and single vs multi-poison behavior
- Multi-poison masking scenario (2 poisons in k=5) proving variance inflation
  causes BaselineZScoreDefense to fail while Stage1GeometricDefense succeeds
- Signal 1: Leave-One-Out (LOO) centroid geometry with Median and MAD
- Signal 2: Local connectivity score and multi-cluster preservation
- Signal 3: Hybrid dense-sparse disagreement score
- Three-Way Router (SAFE, UNCERTAIN, HIGH_RISK) thresholds and latency recording
- Edge cases: empty active set, single candidate, L2 embedding norm invariants
"""

import numpy as np
import pytest

from src.defense.baseline_zscore import BaselineZScoreDefense
from src.defense.router import ThreeWayRouter
from src.defense.stage1_geometry import Stage1GeometricDefense
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


def create_multi_poison_pool(
    num_clean: int = 3,
    num_poison: int = 2,
    num_reserve: int = 10,
    seed: int = 42,
    embedding_dim: int = 768,
) -> RetrievalPool:
    """Synthesize a multi-poison retrieval pool (e.g. 2 poisons in k=5).

    The clean documents form a tight topical cluster with high mutual similarity
    and concordant dense/BM25 ranks.
    The poisoned documents are dense outliers (colluding near an adversarial direction)
    with high dense ranks but poor BM25 ranks.
    """
    rng = np.random.default_rng(seed)

    query = QuerySample(
        query_id="q_multi_poison_test",
        question="What is the true capital of the target entity?",
        ground_truth_answers=["Authentic City"],
        target_fake_answer="Poisoned City",
    )

    query_embedding = _l2_norm(rng.standard_normal(embedding_dim))
    clean_center = _l2_norm(
        query_embedding * 0.70 + _l2_norm(rng.standard_normal(embedding_dim)) * 0.30
    )

    # Adversarial direction strictly orthogonal to clean cluster
    raw_adv = rng.standard_normal(embedding_dim)
    adv_dir = _l2_norm(raw_adv - np.dot(raw_adv, clean_center) * clean_center)

    # 1. Build clean active candidates
    clean_active: list[RetrievedCandidate] = []
    for idx in range(num_clean):
        noise = _l2_norm(rng.standard_normal(embedding_dim))
        emb = _l2_norm(clean_center * 0.95 + noise * 0.05)
        doc = CorpusDocument(
            doc_id=f"clean_doc_{idx + 1}",
            title=f"Verified Record {idx + 1}",
            text=f"Authentic documentation supporting Authentic City reference {idx + 1}.",
            is_poisoned=False,
            attack_type=AttackType.CLEAN,
        )
        cand = RetrievedCandidate(
            doc=doc,
            embedding=emb,
            dense_score=float(0.92 - (idx * 0.02)),
            dense_rank=1 if idx == 0 else (idx + num_poison + 1),
            bm25_score=float(22.0 - (idx * 1.5)),
            bm25_rank=idx + 1,
            is_in_active_set=True,
            status=TrustStatus.UNCHECKED,
        )
        clean_active.append(cand)

    # 2. Build poisoned active candidates
    poison_active: list[RetrievedCandidate] = []
    for p_idx in range(num_poison):
        p_noise = _l2_norm(rng.standard_normal(embedding_dim))
        emb = _l2_norm(adv_dir * 0.95 + p_noise * 0.05)
        doc = CorpusDocument(
            doc_id=f"poison_doc_{p_idx + 1}",
            title=f"Adversarial Fabricated Report {p_idx + 1}",
            text=f"Fabricated claim asserting Poisoned City is the answer {p_idx + 1}.",
            is_poisoned=True,
            attack_type=AttackType.TIER3_CENTROID,
            target_query_id=query.query_id,
        )
        # Poison has high dense rank (e.g. ranks 2 and 3) but poor BM25 rank (ranks 12 and 13)
        cand = RetrievedCandidate(
            doc=doc,
            embedding=emb,
            dense_score=float(0.90 - (p_idx * 0.01)),
            dense_rank=2 + p_idx,
            bm25_score=8.50,
            bm25_rank=12 + p_idx,
            is_in_active_set=True,
            status=TrustStatus.UNCHECKED,
        )
        poison_active.append(cand)

    # Interleave into active set of k=5: [clean_1, poison_1, poison_2, clean_2, clean_3]
    active_set: list[RetrievedCandidate] = [
        clean_active[0],
        poison_active[0],
        poison_active[1],
        clean_active[1],
        clean_active[2],
    ]

    # 3. Build reserve pool (clean background documents)
    reserve_pool: list[RetrievedCandidate] = []
    for r_idx in range(num_reserve):
        r_noise = _l2_norm(rng.standard_normal(embedding_dim))
        r_emb = _l2_norm(clean_center * 0.90 + r_noise * 0.10)
        r_doc = CorpusDocument(
            doc_id=f"reserve_clean_{r_idx + 1}",
            title=f"Reserve Evidence {r_idx + 1}",
            text=f"Background knowledge validating Authentic City index {r_idx + 1}.",
            is_poisoned=False,
            attack_type=AttackType.CLEAN,
        )
        r_cand = RetrievedCandidate(
            doc=r_doc,
            embedding=r_emb,
            dense_score=float(0.70 - (r_idx * 0.02)),
            dense_rank=len(active_set) + r_idx + 1,
            bm25_score=float(14.0 - (r_idx * 0.5)),
            bm25_rank=len(active_set) + r_idx + 1,
            is_in_active_set=False,
            status=TrustStatus.UNCHECKED,
        )
        reserve_pool.append(r_cand)

    return RetrievalPool(
        query=query,
        query_embedding=query_embedding,
        active_set=active_set,
        reserve_pool=reserve_pool,
        final_context=[],
        latency_ms={},
    )


def test_baseline_zscore_mock_pool():
    """Verify BaselineZScoreDefense populates naive_zscore and valid status on mock pool."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)
    defense = BaselineZScoreDefense(z_threshold=1.5)

    updated_pool = defense.filter_pool(pool)

    assert len(updated_pool.active_set) == 5
    for cand in updated_pool.active_set:
        assert isinstance(cand.naive_zscore, float)
        assert not np.isnan(cand.naive_zscore)
        assert cand.status in [TrustStatus.SAFE, TrustStatus.QUARANTINED]

    assert "baseline_zscore" in updated_pool.latency_ms
    assert updated_pool.latency_ms["baseline_zscore"] >= 0.0


def test_baseline_zscore_single_obvious_outlier():
    """Verify BaselineZScoreDefense flags a single isolated outlier when no masking occurs."""
    rng = np.random.default_rng(10)
    dim = 768

    center = _l2_norm(rng.standard_normal(dim))
    outlier_vec = _l2_norm(-center)  # diametrically opposite

    query = QuerySample("q1", "Question?", ["Ans"])
    q_emb = _l2_norm(center.copy())

    candidates: list[RetrievedCandidate] = []
    # 4 clean candidates very close to center
    for i in range(4):
        emb = _l2_norm(center * 0.99 + rng.standard_normal(dim) * 0.01)
        doc = CorpusDocument(f"c{i}", f"Clean {i}", "Text", False, AttackType.CLEAN)
        candidates.append(
            RetrievedCandidate(
                doc=doc,
                embedding=emb,
                dense_score=0.9,
                dense_rank=i + 1,
                bm25_score=15.0,
                bm25_rank=i + 1,
            )
        )

    # 1 outlier candidate
    doc_out = CorpusDocument("out", "Outlier", "Text", True, AttackType.TIER1_BASIC)
    candidates.append(
        RetrievedCandidate(
            doc=doc_out,
            embedding=outlier_vec,
            dense_score=0.8,
            dense_rank=5,
            bm25_score=5.0,
            bm25_rank=5,
        )
    )

    pool = RetrievalPool(query, q_emb, candidates, [])
    defense = BaselineZScoreDefense(z_threshold=1.5)
    defense.filter_pool(pool)

    # The single outlier should have z_score > 1.5 and be quarantined
    outlier_cand = pool.active_set[-1]
    assert outlier_cand.naive_zscore > 1.5
    assert outlier_cand.status == TrustStatus.QUARANTINED

    # Clean candidates should be marked SAFE
    for clean_cand in pool.active_set[:4]:
        assert clean_cand.status == TrustStatus.SAFE


def test_multi_poison_masking_baseline_failure():
    """Demonstrate the variance inflation failure mode in BaselineZScoreDefense.

    When 2 poisons are present in k=5 (40% contamination), the standard deviation
    of cosine distances to the global centroid is inflated, depressing each poison's
    z-score below the 1.5 threshold. Baseline falsely classifies BOTH poisons as SAFE.
    """
    pool = create_multi_poison_pool(num_clean=3, num_poison=2, seed=42)
    defense = BaselineZScoreDefense(z_threshold=1.5)
    defense.filter_pool(pool)

    # Find the two poisoned candidates in active_set
    poisons = [c for c in pool.active_set if c.doc.is_poisoned]
    assert len(poisons) == 2

    # Proving variance inflation: standard z-score for both is <= 1.5
    for p in poisons:
        assert p.naive_zscore <= 1.5, (
            f"Expected naive_zscore <= 1.5 due to variance inflation, got {p.naive_zscore}"
        )
        # Baseline defense suffers masking effect and marks them SAFE
        assert p.status == TrustStatus.SAFE


def test_stage1_geometric_scores_mock_pool():
    """Verify Stage1GeometricDefense populates all 3 signals and stage1_risk."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)
    stage1 = Stage1GeometricDefense()
    stage1.compute_scores(pool)

    assert len(pool.active_set) == 5
    for cand in pool.active_set:
        assert isinstance(cand.loo_geometric_score, float)
        assert not np.isnan(cand.loo_geometric_score)

        assert isinstance(cand.local_connectivity_score, float)
        assert not np.isnan(cand.local_connectivity_score)

        assert isinstance(cand.hybrid_disagreement_score, float)
        assert not np.isnan(cand.hybrid_disagreement_score)

        assert isinstance(cand.stage1_risk, float)
        assert 0.0 <= cand.stage1_risk <= 1.0

    assert "stage1" in pool.latency_ms
    assert pool.latency_ms["stage1"] >= 0.0


def test_stage1_catches_multi_poison_masking():
    """Verify Stage1GeometricDefense + ThreeWayRouter catches multi-poison masking.

    Even when 2 poisons in k=5 mask each other from BaselineZScoreDefense,
    Signal 1 (LOO + Median/MAD), Signal 2 (Local connectivity), and
    Signal 3 (Hybrid disagreement) successfully identify the poisons.
    The router flags both poisons as HIGH_RISK or UNCERTAIN, and never SAFE.
    """
    # 1. Build multi-poison scenario
    pool = create_multi_poison_pool(num_clean=3, num_poison=2, seed=42)

    # 2. First verify Baseline Z-Score fails on this pool
    baseline = BaselineZScoreDefense(z_threshold=1.5)
    baseline.filter_pool(pool)
    for cand in pool.active_set:
        if cand.doc.is_poisoned:
            assert cand.status == TrustStatus.SAFE  # Fails due to variance inflation

    # 3. Now run Stage 1 Defense + Three-Way Router
    stage1 = Stage1GeometricDefense(local_neighbors_m=2)
    stage1.compute_scores(pool)

    router = ThreeWayRouter(tau_low=0.35, tau_high=0.75)
    router.route(pool)

    poisons = [c for c in pool.active_set if c.doc.is_poisoned]
    cleans = [c for c in pool.active_set if not c.doc.is_poisoned]

    assert len(poisons) == 2
    assert len(cleans) == 3

    # Proving Stage 1 catches the poisons:
    for p in poisons:
        # LOO geometric score and/or local connectivity must be significantly positive
        assert p.loo_geometric_score > 1.0 or p.local_connectivity_score > 1.0
        # Disagreement score must be elevated due to BM25 rank gap
        assert p.hybrid_disagreement_score > 0.20
        # Stage 1 risk score is elevated
        assert p.stage1_risk > 0.70
        # Router must NEVER classify poison as SAFE
        assert p.status in [TrustStatus.HIGH_RISK, TrustStatus.UNCERTAIN]
        assert p.status != TrustStatus.SAFE

    # Clean documents have low risk and are routed to SAFE (or UNCERTAIN at worst, not HIGH_RISK)
    for c in cleans:
        assert c.status != TrustStatus.HIGH_RISK


def test_local_connectivity_multi_cluster_preservation():
    """Verify Signal 2 (local connectivity) does not falsely penalize multi-cluster queries."""
    rng = np.random.default_rng(99)
    dim = 768

    query = QuerySample("q_multi_cluster", "Compare Topic A and Topic B", ["Topic A", "Topic B"])
    q_emb = _l2_norm(rng.standard_normal(dim))

    dir_a = _l2_norm(rng.standard_normal(dim))
    raw_b = rng.standard_normal(dim)
    dir_b = _l2_norm(raw_b - np.dot(raw_b, dir_a) * dir_a)  # perpendicular cluster

    # 2 docs in cluster A, 2 docs in cluster B, 1 isolated poison outlier
    embs: list[np.ndarray] = [
        _l2_norm(dir_a * 0.95 + rng.standard_normal(dim) * 0.05),
        _l2_norm(dir_a * 0.95 + rng.standard_normal(dim) * 0.05),
        _l2_norm(dir_b * 0.95 + rng.standard_normal(dim) * 0.05),
        _l2_norm(dir_b * 0.95 + rng.standard_normal(dim) * 0.05),
        _l2_norm(rng.standard_normal(dim)),  # outlier in random direction
    ]

    active_set: list[RetrievedCandidate] = []
    for i, emb in enumerate(embs):
        is_p = (i == 4)
        doc = CorpusDocument(f"d_{i}", f"Doc {i}", "text", is_p, AttackType.TIER1_BASIC if is_p else AttackType.CLEAN)
        active_set.append(
            RetrievedCandidate(
                doc=doc,
                embedding=emb,
                dense_score=0.85,
                dense_rank=i + 1,
                bm25_score=15.0 if not is_p else 5.0,
                bm25_rank=i + 1 if not is_p else 14,
            )
        )

    pool = RetrievalPool(query, q_emb, active_set, [])
    stage1 = Stage1GeometricDefense(local_neighbors_m=2)
    stage1.compute_scores(pool)

    # Outlier doc 4 should have much higher local connectivity score than docs in cluster A and B
    outlier_cand = pool.active_set[4]
    for clean_cand in pool.active_set[:4]:
        assert outlier_cand.local_connectivity_score > clean_cand.local_connectivity_score


def test_hybrid_disagreement_score_mechanics():
    """Verify Signal 3 captures dense-BM25 rank discrepancy and query-peer similarity."""
    dim = 768
    rng = np.random.default_rng(123)

    query = QuerySample("q_disagree", "Test question", ["Answer"])
    q_emb = _l2_norm(rng.standard_normal(dim))

    # Candidate 1: dense rank 1, BM25 rank 1 -> concordant
    # Candidate 2: dense rank 2, BM25 rank 15 -> large rank gap
    doc1 = CorpusDocument("d1", "Title 1", "text", False, AttackType.CLEAN)
    cand1 = RetrievedCandidate(
        doc=doc1,
        embedding=_l2_norm(q_emb * 0.9 + rng.standard_normal(dim) * 0.1),
        dense_score=0.95,
        dense_rank=1,
        bm25_score=25.0,
        bm25_rank=1,
    )

    doc2 = CorpusDocument("d2", "Title 2", "text", True, AttackType.TIER2_CAMOUFLAGED)
    cand2 = RetrievedCandidate(
        doc=doc2,
        embedding=_l2_norm(q_emb * 0.9 + rng.standard_normal(dim) * 0.1),
        dense_score=0.92,
        dense_rank=2,
        bm25_score=4.0,
        bm25_rank=15,
    )

    doc3 = CorpusDocument("d3", "Title 3", "text", False, AttackType.CLEAN)
    cand3 = RetrievedCandidate(
        doc=doc3,
        embedding=_l2_norm(q_emb * 0.9 + rng.standard_normal(dim) * 0.1),
        dense_score=0.88,
        dense_rank=3,
        bm25_score=20.0,
        bm25_rank=2,
    )

    pool = RetrievalPool(query, q_emb, [cand1, cand2, cand3], [])
    stage1 = Stage1GeometricDefense(top_n_expanded=15)
    stage1.compute_scores(pool)

    # Candidate 2 has large rank gap: (15 - 2)/15 = 0.867, cand1 has (1 - 1)/15 = 0
    assert cand2.hybrid_disagreement_score > cand1.hybrid_disagreement_score
    assert cand2.hybrid_disagreement_score > cand3.hybrid_disagreement_score


def test_three_way_router_threshold_routing():
    """Verify ThreeWayRouter routes correctly to SAFE, UNCERTAIN, and HIGH_RISK."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    router = ThreeWayRouter(tau_low=0.35, tau_high=0.75)

    # Manually configure test candidates with specific risk profiles
    cands = pool.active_set

    # Candidate 0: Low risk -> SAFE
    cands[0].loo_geometric_score = -2.0
    cands[0].local_connectivity_score = -2.0
    cands[0].hybrid_disagreement_score = 0.0

    # Candidate 1: Extreme risk -> HIGH_RISK
    cands[1].loo_geometric_score = 10.0
    cands[1].local_connectivity_score = 10.0
    cands[1].hybrid_disagreement_score = 0.9

    # Candidate 2: Intermediate risk -> UNCERTAIN
    cands[2].loo_geometric_score = 0.5
    cands[2].local_connectivity_score = 0.5
    cands[2].hybrid_disagreement_score = 0.2

    router.route(pool)

    assert cands[0].stage1_risk < 0.35
    assert cands[0].status == TrustStatus.SAFE

    assert cands[1].stage1_risk > 0.75
    assert cands[1].status == TrustStatus.HIGH_RISK

    assert 0.35 <= cands[2].stage1_risk <= 0.75
    assert cands[2].status == TrustStatus.UNCERTAIN

    assert "stage1" in pool.latency_ms
    assert pool.latency_ms["stage1"] > 0.0


def test_edge_cases_empty_and_single_candidate():
    """Verify defense components handle empty active sets and single candidate gracefully."""
    query = QuerySample("q_edge", "Edge case query", ["Ans"])
    q_emb = _l2_norm(np.ones(768, dtype=np.float32))

    # Empty active set
    empty_pool = RetrievalPool(query, q_emb, [], [])

    baseline = BaselineZScoreDefense()
    stage1 = Stage1GeometricDefense()
    router = ThreeWayRouter()

    baseline.filter_pool(empty_pool)
    assert empty_pool.active_set == []

    stage1.compute_scores(empty_pool)
    assert empty_pool.active_set == []

    router.route(empty_pool)
    assert empty_pool.active_set == []

    # Single candidate
    single_doc = CorpusDocument("doc_only", "Single", "Text", False, AttackType.CLEAN)
    single_cand = RetrievedCandidate(
        doc=single_doc,
        embedding=q_emb.copy(),
        dense_score=0.9,
        dense_rank=1,
        bm25_score=10.0,
        bm25_rank=1,
    )
    single_pool = RetrievalPool(query, q_emb, [single_cand], [])

    baseline.filter_pool(single_pool)
    assert single_pool.active_set[0].naive_zscore == 0.0
    assert single_pool.active_set[0].status == TrustStatus.SAFE

    stage1.compute_scores(single_pool)
    assert not np.isnan(single_pool.active_set[0].stage1_risk)

    router.route(single_pool)
    assert single_pool.active_set[0].status in [TrustStatus.SAFE, TrustStatus.UNCERTAIN, TrustStatus.HIGH_RISK]


def test_defense_preserves_l2_normalization():
    """Verify defense stages do not mutate unit L2 norm of embeddings."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    stage1 = Stage1GeometricDefense()
    stage1.compute_scores(pool)

    router = ThreeWayRouter()
    router.route(pool)

    np.testing.assert_allclose(np.linalg.norm(pool.query_embedding), 1.0, atol=1e-5)
    for cand in pool.active_set + pool.reserve_pool:
        np.testing.assert_allclose(np.linalg.norm(cand.embedding), 1.0, atol=1e-5)
