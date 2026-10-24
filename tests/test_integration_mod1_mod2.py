"""
Integration Test Suite for Module 1 (Attacks) + Module 2 (Retrieval).

Verifies end-to-end integration across:
1. Loading realistic QA benchmark data via CorpusInjector.
2. Generating and injecting Tier 1 to Tier 4 adversarial poisons.
3. Indexing the contaminated corpus using DenseFaissRetriever and SparseBM25Retriever.
4. Constructing candidate pools via HybridPoolBuilder.
5. Infiltrating top-K active sets with poisoned passages and verifying data contract invariants.
"""

from typing import List
import numpy as np
import pytest

from src.attacks.injector import CorpusInjector
from src.retrieval.dense_faiss import DenseFaissRetriever
from src.retrieval.sparse_bm25 import SparseBM25Retriever
from src.retrieval.hybrid_pool import HybridPoolBuilder
from src.schemas import (
    AttackType,
    CorpusDocument,
    QuerySample,
    RetrievalPool,
    RetrievedCandidate,
    TrustStatus,
)


@pytest.fixture
def nq_benchmark() -> tuple[List[CorpusDocument], List[QuerySample]]:
    """Loads 10 queries and corresponding clean corpus from the self-contained benchmark."""
    clean_corpus, queries = CorpusInjector.load_or_create_sample_nq_benchmark(
        num_queries=10, save_to_disk=False
    )
    return clean_corpus, queries


@pytest.mark.parametrize(
    "attack_tier",
    [
        AttackType.TIER1_BASIC,
        AttackType.TIER2_CAMOUFLAGED,
        AttackType.TIER3_CENTROID,
        AttackType.TIER4_HYBRID_JAM,
    ],
)
def test_tier_retrieval_active_set_and_metadata(
    nq_benchmark, attack_tier: AttackType
):
    """Verify that each attack tier (Tier 1-4) successfully infiltrates the active set
    of the HybridPoolBuilder with correct contract schemas, L2 normalization, and metadata."""
    clean_corpus, queries = nq_benchmark
    injector = CorpusInjector()

    # Query 2 ("In what year was the Magna Carta granted by King John?") has rich
    # clean reference documents enabling high topic alignment for camouflaged & centroid tiers
    target_query = queries[2]

    # 1. Inject poisons into clean corpus
    mixed_corpus, poison_mapping = injector.inject_dataset(
        clean_corpus=clean_corpus,
        queries=[target_query],
        attack_type=attack_tier,
        poisons_per_query=2,
        shuffle=True,
        seed=42,
    )

    assert len(mixed_corpus) == len(clean_corpus) + 2
    assert target_query.query_id in poison_mapping
    target_poison_ids = poison_mapping[target_query.query_id]
    assert len(target_poison_ids) == 2

    # 2. Build Dense and Sparse indexes over the contaminated corpus
    dense_retriever = DenseFaissRetriever(use_mock_embedder=True)
    dense_retriever.index_corpus(mixed_corpus)

    sparse_retriever = SparseBM25Retriever()
    sparse_retriever.index_corpus(mixed_corpus)

    # 3. Construct HybridPoolBuilder
    top_k = 5
    top_n = 15
    builder = HybridPoolBuilder(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        top_k_active=top_k,
        top_n_expanded=top_n,
    )

    # 4. Retrieve candidate pool for the attacked query
    pool: RetrievalPool = builder.build_pool(target_query)

    # 5. Validate RetrievalPool schema & invariants
    assert isinstance(pool, RetrievalPool)
    assert pool.query.query_id == target_query.query_id
    assert pool.query_embedding.shape == (768,)
    assert np.isclose(np.linalg.norm(pool.query_embedding), 1.0, atol=1e-5)
    assert pool.query_embedding.dtype == np.float32

    assert len(pool.active_set) == top_k
    assert len(pool.reserve_pool) == top_n - top_k
    assert len(pool.active_set) + len(pool.reserve_pool) == top_n
    assert "retrieval" in pool.latency_ms
    assert pool.latency_ms["retrieval"] > 0.0

    # 6. Verify poisoned passages successfully penetrated the active set
    active_poisons: List[RetrievedCandidate] = [
        cand for cand in pool.active_set if cand.doc.is_poisoned
    ]
    assert len(active_poisons) >= 1, (
        f"Attack tier {attack_tier.value} failed to retrieve any poisoned passage into active set! "
        f"Active set doc IDs: {[c.doc.doc_id for c in pool.active_set]}"
    )

    # 7. Check metadata integrity of all retrieved candidates
    all_candidates = pool.active_set + pool.reserve_pool
    for rank_idx, cand in enumerate(all_candidates, start=1):
        assert isinstance(cand, RetrievedCandidate)
        # L2-normalization invariant
        assert cand.embedding.shape == (768,)
        assert np.isclose(np.linalg.norm(cand.embedding), 1.0, atol=1e-5)

        # Dense rank & score
        assert cand.dense_rank == rank_idx
        assert isinstance(cand.dense_score, float)
        assert -1.0 <= cand.dense_score <= 1.0

        # Sparse BM25 rank & score
        assert isinstance(cand.bm25_score, float)
        assert cand.bm25_score >= 0.0
        assert 1 <= cand.bm25_rank <= len(mixed_corpus)

        # Status & active set flag
        assert cand.status == TrustStatus.UNCHECKED
        assert cand.naive_zscore == 0.0
        assert cand.is_in_active_set == (rank_idx <= top_k)

    # 8. Check targeted poison properties
    for poison_cand in active_poisons:
        assert poison_cand.doc.doc_id in target_poison_ids
        assert poison_cand.doc.is_poisoned is True
        assert poison_cand.doc.attack_type == attack_tier
        assert poison_cand.doc.target_query_id == target_query.query_id
        assert (
            target_query.target_fake_answer in poison_cand.doc.text
            or target_query.target_fake_answer in poison_cand.doc.title
        )
        assert poison_cand.dense_rank <= top_k


def test_tier4_hybrid_jam_retrieval_dominance(nq_benchmark):
    """Verify that Tier 4 Hybrid Jam dominates both Dense FAISS and Sparse BM25 retrieval,
    achieving top-rank positions in both modalities simultaneously."""
    clean_corpus, queries = nq_benchmark
    target_query = queries[2]

    injector = CorpusInjector()
    mixed_corpus, mapping = injector.inject_dataset(
        clean_corpus=clean_corpus,
        queries=[target_query],
        attack_type=AttackType.TIER4_HYBRID_JAM,
        poisons_per_query=2,
        shuffle=True,
        seed=123,
    )

    dense_retriever = DenseFaissRetriever(use_mock_embedder=True)
    dense_retriever.index_corpus(mixed_corpus)

    sparse_retriever = SparseBM25Retriever()
    sparse_retriever.index_corpus(mixed_corpus)

    builder = HybridPoolBuilder(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        top_k_active=5,
        top_n_expanded=15,
    )
    pool = builder.build_pool(target_query)

    active_poisons = [c for c in pool.active_set if c.doc.is_poisoned]
    assert len(active_poisons) == 2

    # Tier 4 specifically jams BM25 with high-frequency keywords while matching query semantics
    for cand in active_poisons:
        assert cand.dense_rank in [1, 2]
        assert cand.bm25_rank in [1, 2]
        assert cand.bm25_score > 20.0
        assert cand.dense_score > 0.5


def test_multi_query_shared_index_isolation(nq_benchmark):
    """Verify that multiple queries with diverse attack tiers can be indexed together in a
    single shared FAISS & BM25 index and each query retrieves its intended poison."""
    clean_corpus, queries = nq_benchmark
    injector = CorpusInjector()

    attack_specs = [
        (queries[0], AttackType.TIER1_BASIC),
        (queries[2], AttackType.TIER2_CAMOUFLAGED),
        (queries[3], AttackType.TIER3_CENTROID),
        (queries[4], AttackType.TIER4_HYBRID_JAM),
    ]

    current_corpus = clean_corpus
    all_mappings = {}
    for q, tier in attack_specs:
        current_corpus, mapping = injector.inject_dataset(
            clean_corpus=current_corpus,
            queries=[q],
            attack_type=tier,
            poisons_per_query=2,
            shuffle=True,
            seed=42,
        )
        all_mappings.update(mapping)

    # 4 queries * 2 poisons = 8 total poisoned docs added
    assert len(current_corpus) == len(clean_corpus) + 8

    # Single shared index for all documents
    dense_retriever = DenseFaissRetriever(use_mock_embedder=True)
    dense_retriever.index_corpus(current_corpus)

    sparse_retriever = SparseBM25Retriever()
    sparse_retriever.index_corpus(current_corpus)

    builder = HybridPoolBuilder(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        top_k_active=5,
        top_n_expanded=15,
    )

    for q, tier in attack_specs:
        pool = builder.build_pool(q)
        expected_poison_ids = set(all_mappings[q.query_id])

        # Active poisons specifically targeted for this query
        retrieved_targeted_poisons = [
            c for c in pool.active_set
            if c.doc.is_poisoned and c.doc.doc_id in expected_poison_ids
        ]
        assert len(retrieved_targeted_poisons) >= 1

        for cand in retrieved_targeted_poisons:
            assert cand.doc.target_query_id == q.query_id
            assert cand.doc.attack_type == tier
            assert (
                q.target_fake_answer in cand.doc.text
                or q.target_fake_answer in cand.doc.title
            )


def test_clean_corpus_no_poison_retrieval(nq_benchmark):
    """Verify that retrieval on a clean corpus yields no poisoned candidates."""
    clean_corpus, queries = nq_benchmark
    query = queries[0]

    dense_retriever = DenseFaissRetriever(use_mock_embedder=True)
    dense_retriever.index_corpus(clean_corpus)

    sparse_retriever = SparseBM25Retriever()
    sparse_retriever.index_corpus(clean_corpus)

    builder = HybridPoolBuilder(dense_retriever, sparse_retriever, top_k_active=5, top_n_expanded=15)
    pool = builder.build_pool(query)

    for cand in pool.active_set + pool.reserve_pool:
        assert cand.doc.is_poisoned is False
        assert cand.doc.attack_type == AttackType.CLEAN
        assert cand.doc.target_query_id is None


def test_precomputed_query_embedding_invariance(nq_benchmark):
    """Verify that passing an explicit precomputed query embedding produces consistent results."""
    clean_corpus, queries = nq_benchmark
    query = queries[2]

    injector = CorpusInjector()
    mixed_corpus, _ = injector.inject_dataset(
        clean_corpus=clean_corpus,
        queries=[query],
        attack_type=AttackType.TIER1_BASIC,
        poisons_per_query=2,
    )

    dense_retriever = DenseFaissRetriever(use_mock_embedder=True)
    dense_retriever.index_corpus(mixed_corpus)

    sparse_retriever = SparseBM25Retriever()
    sparse_retriever.index_corpus(mixed_corpus)

    builder = HybridPoolBuilder(dense_retriever, sparse_retriever, top_k_active=5, top_n_expanded=15)

    # Compute custom query vector
    q_emb = dense_retriever.encode_query(query.question)
    pool = builder.build_pool(query, query_embedding=q_emb)

    assert np.allclose(pool.query_embedding, q_emb, atol=1e-6)
    assert len(pool.active_set) == 5
    assert len(pool.reserve_pool) == 10
