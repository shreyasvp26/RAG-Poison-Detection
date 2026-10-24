"""
Unit and Contract Tests for Module 5: Generation, Evaluation Metrics, Forensics Plotting, and Dashboard.

Tests:
- LLMClient prompt formatting with citations and answer generation (defended vs undefended)
- DefenseMetricsCalculator: detection metrics (Precision, Recall, F1, FPR, FNR), ASR reduction,
  threshold sweep frontier, and ablation benchmark data
- DefensePlotter: 2D PCA embedding space, 5-axis anomaly radar, security-utility frontier,
  ablation grouped bars and heatmap
- Streamlit dashboard module import and pipeline runner helper
"""

import numpy as np
import pytest

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    go = None
    HAS_PLOTLY = False

from src.evaluation.metrics import DefenseMetricsCalculator
from src.evaluation.plots import DefensePlotter
from src.generation.llm_client import LLMClient
from src.mock_factory import create_mock_retrieval_pool
from src.schemas import AttackType, RetrievalPool, RetrievedCandidate, TrustStatus


def test_llm_client_format_prompt_citations():
    """Verify LLMClient formats prompt with inline document citations."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)
    client = LLMClient()

    prompt_on = client.format_prompt(pool, use_defense=True)
    prompt_off = client.format_prompt(pool, use_defense=False)

    assert pool.query.question in prompt_on
    assert pool.query.question in prompt_off
    # Inline citations like [doc_id] must be present in prompt
    assert "[" in prompt_on and "]" in prompt_on
    assert "[" in prompt_off and "]" in prompt_off


def test_llm_client_generation_poisoned_vs_clean():
    """Verify LLMClient delivers fake answer on undefended poison context and ground truth on defended context."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)
    client = LLMClient(use_mock=True)

    # 1. Undefended RAG (use_defense=False): active poison (rank 2) hijacks generation
    ans_undefended = client.generate(pool, use_defense=False)
    assert pool.query.target_fake_answer in ans_undefended
    # Cites the poison document
    assert "doc_poison_rank2" in ans_undefended

    # 2. Defended RAG (use_defense=True): only clean documents in final_context
    pool.final_context = [c for c in pool.active_set if not c.doc.is_poisoned]
    ans_defended = client.generate(pool, use_defense=True)

    primary_gt = pool.query.ground_truth_answers[0]
    assert primary_gt in ans_defended
    assert pool.query.target_fake_answer not in ans_defended


def test_defense_metrics_detection_metrics():
    """Verify compute_detection_metrics calculates Precision, Recall, F1, FPR, and FNR accurately."""
    calc = DefenseMetricsCalculator()
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    # Candidate 0: Clean, SAFE -> TN
    pool.active_set[0].doc.is_poisoned = False
    pool.active_set[0].status = TrustStatus.SAFE

    # Candidate 1: Poison, QUARANTINED -> TP
    pool.active_set[1].doc.is_poisoned = True
    pool.active_set[1].status = TrustStatus.QUARANTINED

    # Candidate 2: Clean, HIGH_RISK -> FP
    pool.active_set[2].doc.is_poisoned = False
    pool.active_set[2].status = TrustStatus.HIGH_RISK

    # Candidate 3: Poison, SAFE -> FN
    pool.active_set[3].doc.is_poisoned = True
    pool.active_set[3].status = TrustStatus.SAFE

    # Candidate 4: Clean, SAFE -> TN
    pool.active_set[4].doc.is_poisoned = False
    pool.active_set[4].status = TrustStatus.SAFE

    metrics = calc.compute_detection_metrics([pool])

    assert metrics["true_positives"] == 1.0
    assert metrics["false_positives"] == 1.0
    assert metrics["false_negatives"] == 1.0
    assert metrics["true_negatives"] == 2.0

    # Precision = 1 / (1 + 1) = 0.5
    assert metrics["precision"] == 0.5
    # Recall = 1 / (1 + 1) = 0.5
    assert metrics["recall"] == 0.5
    # F1 = 0.5
    assert metrics["f1"] == 0.5
    # FPR = FP / (FP + TN) = 1 / (1 + 2) = 0.333
    assert abs(metrics["fpr"] - (1.0 / 3.0)) < 1e-4
    # FNR = FN / (FN + TP) = 1 / 2 = 0.5
    assert metrics["fnr"] == 0.5


def test_defense_metrics_rag_and_asr():
    """Verify compute_rag_metrics calculates ASR reduction and clean accuracy."""
    calc = DefenseMetricsCalculator()
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)
    pool.latency_ms = {"retrieval": 10.0, "stage1": 4.0, "stage2": 15.0, "stage3": 2.0}

    pools = [pool]
    # Answers Defense OFF has fake answer (Apollo 13)
    answers_off = ["Apollo 13 achieved the first manned landing."]
    # Answers Defense ON has ground truth (Apollo 11)
    answers_on = ["Apollo 11 was the historic first landing."]

    rag_metrics = calc.compute_rag_metrics(pools, answers_off, answers_on)

    assert rag_metrics["asr_defense_off"] == 1.0
    assert rag_metrics["asr_defense_on"] == 0.0
    assert rag_metrics["relative_asr_reduction_pct"] == 100.0
    assert rag_metrics["clean_accuracy_off"] == 0.0
    assert rag_metrics["clean_accuracy_on"] == 1.0

    lat = rag_metrics["latency_breakdown"]
    assert lat["retrieval"] == 10.0
    assert lat["stage1"] == 4.0
    assert lat["stage2"] == 15.0
    assert lat["stage3"] == 2.0
    assert lat["total_defense"] == 21.0


def test_defense_metrics_threshold_sweep():
    """Verify run_threshold_sweep produces valid curves across tau."""
    calc = DefenseMetricsCalculator()
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    # Assign synthetic risk values
    for i, c in enumerate(pool.active_set):
        c.stage1_risk = 0.85 if c.doc.is_poisoned else 0.20

    sweep = calc.run_threshold_sweep([pool], thresholds=[0.10, 0.35, 0.50, 0.75, 0.90])

    assert len(sweep["thresholds"]) == 5
    assert len(sweep["asr_curve"]) == 5
    assert len(sweep["acc_curve"]) == 5
    assert len(sweep["tpr_curve"]) == 5
    assert len(sweep["fpr_curve"]) == 5


def test_defense_metrics_ablation_benchmark_data():
    """Verify get_ablation_benchmark_data returns structured comparative benchmark schemas."""
    calc = DefenseMetricsCalculator()
    ablation = calc.get_ablation_benchmark_data()

    assert "tiers" in ablation
    assert "versions" in ablation
    assert "asr_matrix" in ablation
    assert "accuracy_matrix" in ablation
    assert "precision_matrix" in ablation
    assert "latency_profile" in ablation

    # Check that V1 (col 0) suffers from high ASR on Tier 3 (row 2)
    # while V4 (col 3) successfully neutralizes the attack
    asr_mat = ablation["asr_matrix"]
    assert asr_mat[2][0] > 80.0  # V1 fails on Tier 3 due to variance inflation
    assert asr_mat[2][3] < 5.0   # V4 full defense succeeds


@pytest.mark.skipif(not HAS_PLOTLY, reason="plotly is required for DefensePlotter figure tests")
def test_defense_plotter_figures():
    """Verify all DefensePlotter plotting routines generate valid Plotly figures."""
    plotter = DefensePlotter()
    calc = DefenseMetricsCalculator()
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    # 1. 2D PCA embedding manifold figure
    fig_pca = plotter.plot_embedding_space_2d(pool)
    assert isinstance(fig_pca, go.Figure)
    assert len(fig_pca.data) >= 2  # Query trace + candidate traces

    # 2. 5-Axis Candidate Anomaly Radar
    cand = pool.active_set[1]
    cand.loo_geometric_score = 3.5
    cand.local_connectivity_score = 2.8
    cand.hybrid_disagreement_score = 0.75
    cand.sentence_drift_score = 0.85
    cand.nli_contradiction_score = 0.92

    fig_radar = plotter.plot_candidate_radar(cand)
    assert isinstance(fig_radar, go.Figure)
    assert len(fig_radar.data) == 1
    assert isinstance(fig_radar.data[0], go.Scatterpolar)

    # 3. Security-Utility Frontier
    sweep_data = calc.run_threshold_sweep([pool])
    fig_frontier = plotter.plot_security_utility_frontier(sweep_data)
    assert isinstance(fig_frontier, go.Figure)
    assert len(fig_frontier.data) == 2  # ASR line + Accuracy line

    # 4. Ablation Bars and Heatmap
    ablation_data = calc.get_ablation_benchmark_data()
    fig_bars = plotter.plot_ablation_bars(ablation_data)
    assert isinstance(fig_bars, go.Figure)
    assert len(fig_bars.data) == 4  # 4 defense versions

    fig_heat = plotter.plot_ablation_heatmap(ablation_data)
    assert isinstance(fig_heat, go.Figure)
    assert len(fig_heat.data) == 1
    assert isinstance(fig_heat.data[0], go.Heatmap)


def test_streamlit_dashboard_imports_and_helpers():
    """Verify app.streamlit_dashboard imports cleanly and pipeline helper runs without error."""
    import app.streamlit_dashboard as dashboard

    assert hasattr(dashboard, "run_or_simulate_defense_pipeline")
    assert hasattr(dashboard, "main")

    # Test pipeline helper for V4
    pool_v4 = dashboard.run_or_simulate_defense_pipeline(
        query_text="What was the first manned mission to land on the Moon?",
        target_answer="Apollo 13",
        ground_truth="Apollo 11",
        attack_tier="tier2_camouflaged",
        poison_count=2,
        defense_mode="V4: Full EviGuard-RAG",
        tau_low=0.35,
        tau_high=0.75,
        seed=42,
    )

    assert isinstance(pool_v4, RetrievalPool)
    assert len(pool_v4.active_set) == 5
    assert len(pool_v4.final_context) == 5
    # Both active poisons must be quarantined in V4
    quarantined = [c for c in pool_v4.active_set if c.status == TrustStatus.QUARANTINED]
    assert len(quarantined) >= 1

    # Test pipeline helper for V1
    pool_v1 = dashboard.run_or_simulate_defense_pipeline(
        query_text="What was the first manned mission to land on the Moon?",
        target_answer="Apollo 13",
        ground_truth="Apollo 11",
        attack_tier="tier3_centroid",
        poison_count=2,
        defense_mode="V1: Naive Z-Score Baseline",
        tau_low=0.35,
        tau_high=0.75,
        seed=42,
    )
    assert isinstance(pool_v1, RetrievalPool)
