"""
End-to-End System Integration Tests for EviGuardPipeline.

Verifies the complete 5-module pipeline flow:
1. Module 1: Benchmark loading and synthetic poison injection across attack tiers
2. Module 2: Dense FAISS & Sparse BM25 indexing and hybrid pool retrieval
3. Module 3: Baseline Z-Score filter (V1) vs Stage 1 Geometric Defense & Router (V2..V4)
4. Module 4: Stage 2 Deep Verifier (sentence drift + NLI) & Stage 3 Dynamic Context Recovery
5. Module 5: LLM response generation (Defense OFF vs ON) and comprehensive metrics calculation
"""

from typing import List
import numpy as np
import pytest

from src.pipeline import EviGuardPipeline
from src.mock_factory import create_mock_retrieval_pool
from src.schemas import (
    AttackType,
    CorpusDocument,
    QuerySample,
    RetrievalPool,
    RetrievedCandidate,
    TrustStatus,
)


def test_pipeline_initialization():
    """Verify EviGuardPipeline instantiates all 5 modules with canonical config parameters."""
    pipeline = EviGuardPipeline(config_path="config.yaml", use_mock_embedder=True, use_mock_llm=True)

    assert pipeline.top_k_active == 5
    assert pipeline.top_n_expanded == 15
    assert pipeline.tau_low == 0.35
    assert pipeline.tau_high == 0.75
    assert pipeline.tau_verify == 0.55
    assert pipeline.baseline_z_threshold == 1.5

    assert pipeline.injector is not None
    assert pipeline.generator is not None
    assert pipeline.dense_retriever is not None
    assert pipeline.sparse_retriever is not None
    assert pipeline.pool_builder is not None
    assert pipeline.baseline_defense is not None
    assert pipeline.stage1_defense is not None
    assert pipeline.router is not None
    assert pipeline.stage2_verifier is not None
    assert pipeline.stage3_recovery is not None
    assert pipeline.llm_client is not None
    assert pipeline.metrics_calculator is not None


def test_pipeline_load_benchmark_and_index():
    """Verify loading benchmark corpus, injecting poisons, and building retriever indexes."""
    pipeline = EviGuardPipeline(use_mock_embedder=True, use_mock_llm=True)
    indexed_corpus, queries = pipeline.load_benchmark_and_index(
        num_queries=5,
        attack_tier=AttackType.TIER2_CAMOUFLAGED,
        poisons_per_query=2,
        seed=42,
    )

    assert len(queries) == 5
    # Indexed corpus must contain clean documents + 2 poisons per query
    assert len(indexed_corpus) >= len(queries) * 2
    assert len(pipeline.poison_mapping) == 5

    # Check that both FAISS/dense embeddings and BM25 indexes are populated
    assert len(pipeline.dense_retriever.corpus_docs) == len(indexed_corpus)
    assert len(pipeline.dense_retriever.doc_embeddings) == len(indexed_corpus)
    assert len(pipeline.sparse_retriever.corpus_docs) == len(indexed_corpus)


def test_pipeline_run_query_undefended_vs_v4():
    """Verify run_query on an attacked query under Defense OFF vs V4 Full Defense."""
    pipeline = EviGuardPipeline(use_mock_embedder=True, use_mock_llm=True)
    indexed_corpus, queries = pipeline.load_benchmark_and_index(
        num_queries=5,
        attack_tier=AttackType.TIER2_CAMOUFLAGED,
        poisons_per_query=1,
        seed=42,
    )

    # Query 3 ("What year did the Byzantine Empire fall with the siege of Constantinople?")
    target_query = queries[3]
    result = pipeline.run_query(query=target_query, mode="v4_full")

    # 1. Structure of return payload
    assert "query" in result
    assert "mode" in result and result["mode"] == "v4_full"
    assert "pool" in result
    assert "answer_undefended" in result
    assert "answer_defended" in result
    assert "detection_metrics" in result
    assert "rag_metrics" in result
    assert "latency_ms" in result

    pool: RetrievalPool = result["pool"]
    assert isinstance(pool, RetrievalPool)
    assert len(pool.active_set) == 5
    assert len(pool.final_context) == 5

    # 2. Poison infiltration and neutralization
    assert result["active_poisons_count"] >= 1, "Poison should have infiltrated active set"
    assert result["quarantined_count"] >= 1, "Poison should be quarantined in V4"

    # 3. Context recovery: final context must contain only clean candidates
    for cand in pool.final_context:
        assert cand.doc.is_poisoned is False, (
            f"Poisoned document {cand.doc.doc_id} leaked into final_context!"
        )

    # 4. Downstream LLM generation: Defense OFF hijacked, Defense ON clean
    assert target_query.target_fake_answer in result["answer_undefended"]
    primary_gt = target_query.ground_truth_answers[0]
    assert primary_gt in result["answer_defended"]
    assert target_query.target_fake_answer not in result["answer_defended"]

    # 5. RAG metrics verification
    rag = result["rag_metrics"]
    assert rag["asr_defense_off"] == 1.0
    assert rag["asr_defense_on"] == 0.0
    assert rag["clean_accuracy_on"] == 1.0

    # 6. Mathematical invariant: L2 normalization of all embeddings
    for cand in pool.active_set + pool.reserve_pool:
        norm = np.linalg.norm(cand.embedding)
        assert np.isclose(norm, 1.0, atol=1e-5), f"Embedding norm {norm} violated unit L2 invariant!"


def test_pipeline_v1_baseline_vs_v4_full_defense():
    """Verify comparative evaluation between V1 Naive Z-score and V4 Full EviGuard-RAG."""
    pipeline = EviGuardPipeline(use_mock_embedder=True, use_mock_llm=True)
    indexed_corpus, queries = pipeline.load_benchmark_and_index(
        num_queries=5,
        attack_tier=AttackType.TIER3_CENTROID,
        poisons_per_query=2,
        seed=42,
    )

    query = queries[1]

    # Run V1 Baseline
    res_v1 = pipeline.run_query(query=query, mode="v1_baseline")
    # Run V4 Full
    res_v4 = pipeline.run_query(query=query, mode="v4_full")

    pool_v1: RetrievalPool = res_v1["pool"]
    pool_v4: RetrievalPool = res_v4["pool"]

    # In V4, all quarantined slots must be backfilled to reach k=5
    assert len(pool_v4.final_context) == 5
    # In V1, quarantined slots are simply removed without recovery
    assert len(pool_v1.final_context) <= 5


def test_pipeline_all_modes_contract():
    """Verify that all pipeline modes (V1, V2, V3, V4, undefended) execute cleanly."""
    pipeline = EviGuardPipeline(use_mock_embedder=True, use_mock_llm=True)
    pool_template = create_mock_retrieval_pool(k=5, n=15, seed=42)

    modes = ["v1_baseline", "v2_stage1", "v3_verifier", "v4_full", "undefended"]

    for mode in modes:
        # Create a fresh mock pool for each mode
        pool = create_mock_retrieval_pool(k=5, n=15, seed=42)
        res = pipeline.run_query(query=pool.query, mode=mode, pool=pool)

        assert isinstance(res["pool"], RetrievalPool)
        assert len(res["pool"].active_set) == 5
        assert isinstance(res["answer_defended"], str)
        assert len(res["answer_defended"]) > 0
        assert "detection_metrics" in res
        assert "rag_metrics" in res


def test_pipeline_run_query_with_raw_string():
    """Verify run_query accepts a plain string question without requiring QuerySample."""
    pipeline = EviGuardPipeline(use_mock_embedder=True, use_mock_llm=True)
    pipeline.load_benchmark_and_index(num_queries=3, attack_tier=None)

    question = "In what year was the Magna Carta granted by King John?"
    res = pipeline.run_query(query=question, mode="v4_full")

    assert isinstance(res["query"], QuerySample)
    assert res["query"].question == question
    assert isinstance(res["pool"], RetrievalPool)
    assert len(res["pool"].final_context) == 5


def test_pipeline_ablation_benchmark_offline_and_live():
    """Verify run_ablation_benchmark returns valid comparative benchmark schemas."""
    pipeline = EviGuardPipeline(use_mock_embedder=True, use_mock_llm=True)

    # 1. Fast offline ablation benchmark data
    offline_data = pipeline.run_ablation_benchmark(live=False)
    assert "tiers" in offline_data
    assert "versions" in offline_data
    assert "asr_matrix" in offline_data
    assert "accuracy_matrix" in offline_data
    assert "precision_matrix" in offline_data
    assert "latency_profile" in offline_data
    assert len(offline_data["tiers"]) == 4
    assert len(offline_data["versions"]) == 4

    # 2. Live ablation benchmark execution
    live_data = pipeline.run_ablation_benchmark(
        attack_tiers=[AttackType.TIER1_BASIC, AttackType.TIER2_CAMOUFLAGED],
        modes=["v1_baseline", "v4_full"],
        live=True,
    )
    assert len(live_data["tiers"]) == 2
    assert len(live_data["versions"]) == 2
    assert len(live_data["asr_matrix"]) == 2
    assert len(live_data["asr_matrix"][0]) == 2
