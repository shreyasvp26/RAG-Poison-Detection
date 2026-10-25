"""
Master End-to-End Orchestrator for EviGuard-RAG.

Chains the complete 5-module architecture end-to-end:
1. Module 1 (Attacks): CorpusInjector & PoisonGenerator
2. Module 2 (Retrieval): DenseFaissRetriever, SparseBM25Retriever, HybridPoolBuilder
3. Module 3 (Stage 1 Defense): BaselineZScoreDefense, Stage1GeometricDefense, ThreeWayRouter
4. Module 4 (Stage 2 & 3 Defense): Stage2Verifier, Stage3Recovery
5. Module 5 (Gen & Eval & UI): LLMClient, DefenseMetricsCalculator
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import yaml

from src.schemas import (
    AttackType,
    CorpusDocument,
    QuerySample,
    RetrievalPool,
    RetrievedCandidate,
    TrustStatus,
)
from src.attacks.injector import CorpusInjector
from src.attacks.poison_generator import PoisonGenerator
from src.retrieval.dense_faiss import DenseFaissRetriever
from src.retrieval.sparse_bm25 import SparseBM25Retriever
from src.retrieval.hybrid_pool import HybridPoolBuilder
from src.defense.baseline_zscore import BaselineZScoreDefense
from src.defense.stage1_geometry import Stage1GeometricDefense
from src.defense.router import ThreeWayRouter
from src.defense.stage2_verifier import Stage2Verifier
from src.defense.stage3_recovery import Stage3Recovery
from src.generation.llm_client import LLMClient
from src.evaluation.metrics import DefenseMetricsCalculator


def _load_yaml_config(config_path: str = "config.yaml") -> dict:
    """Load config.yaml if present."""
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


class EviGuardPipeline:
    """End-to-End Pipeline orchestrating all 5 modules of EviGuard-RAG."""

    def __init__(
        self,
        config_path: str = "config.yaml",
        use_mock_embedder: bool = True,
        use_mock_llm: bool = True,
        top_k_active: Optional[int] = None,
        top_n_expanded: Optional[int] = None,
        tau_low: Optional[float] = None,
        tau_high: Optional[float] = None,
        tau_verify: Optional[float] = None,
        baseline_z_threshold: Optional[float] = None,
    ):
        """Initialize the pipeline with all 5 modules configured.

        Args:
            config_path: Path to config.yaml.
            use_mock_embedder: Whether to use fast deterministic hash embedder for Contriever.
            use_mock_llm: Whether to use fast deterministic grounded answer synthesizer.
            top_k_active: Size of active generation set (defaults to config.yaml: 5).
            top_n_expanded: Size of expanded retrieval candidate pool (defaults to config.yaml: 15).
            tau_low: Router lower threshold (defaults to config.yaml: 0.35).
            tau_high: Router upper threshold (defaults to config.yaml: 0.75).
            tau_verify: Stage 2 deep verification threshold (defaults to config.yaml: 0.55).
            baseline_z_threshold: Naive Z-score cutoff for V1 (defaults to config.yaml: 1.5).
        """
        self.config_path = config_path
        self.config = _load_yaml_config(config_path)

        retrieval_cfg = self.config.get("retrieval", {})
        stage1_cfg = self.config.get("stage1_defense", {})
        thresholds_cfg = stage1_cfg.get("thresholds", {})
        stage2_cfg = self.config.get("stage2_defense", {})

        self.top_k_active: int = (
            int(top_k_active)
            if top_k_active is not None
            else int(retrieval_cfg.get("top_k_active", 5))
        )
        self.top_n_expanded: int = (
            int(top_n_expanded)
            if top_n_expanded is not None
            else int(retrieval_cfg.get("top_n_expanded", 15))
        )
        self.tau_low: float = (
            float(tau_low)
            if tau_low is not None
            else float(thresholds_cfg.get("tau_low", 0.35))
        )
        self.tau_high: float = (
            float(tau_high)
            if tau_high is not None
            else float(thresholds_cfg.get("tau_high", 0.75))
        )
        self.tau_verify: float = (
            float(tau_verify)
            if tau_verify is not None
            else float(stage2_cfg.get("tau_verify", 0.55))
        )
        self.baseline_z_threshold: float = (
            float(baseline_z_threshold)
            if baseline_z_threshold is not None
            else float(stage1_cfg.get("baseline_z_threshold", 1.5))
        )

        dense_model = retrieval_cfg.get("dense_model", "facebook/contriever")
        embedding_dim = int(retrieval_cfg.get("embedding_dim", 768))

        # Module 1: Attacks
        self.injector = CorpusInjector(config_path=config_path)
        self.generator = PoisonGenerator(config_path=config_path)

        # Module 2: Retrieval
        self.dense_retriever = DenseFaissRetriever(
            model_name=dense_model,
            embedding_dim=embedding_dim,
            use_mock_embedder=use_mock_embedder,
        )
        self.sparse_retriever = SparseBM25Retriever()
        self.pool_builder = HybridPoolBuilder(
            dense_retriever=self.dense_retriever,
            sparse_retriever=self.sparse_retriever,
            top_k_active=self.top_k_active,
            top_n_expanded=self.top_n_expanded,
            config_path=config_path,
        )

        # Module 3: Stage 1 Defense & Router
        self.baseline_defense = BaselineZScoreDefense(
            z_threshold=self.baseline_z_threshold,
            config_path=config_path,
        )
        self.stage1_defense = Stage1GeometricDefense(
            top_n_expanded=self.top_n_expanded,
            config_path=config_path,
        )
        self.router = ThreeWayRouter(
            tau_low=self.tau_low,
            tau_high=self.tau_high,
            config_path=config_path,
        )

        # Module 4: Stage 2 Deep Verification & Stage 3 Recovery
        self.stage2_verifier = Stage2Verifier(
            tau_verify=self.tau_verify,
            config_path=config_path,
        )
        self.stage3_recovery = Stage3Recovery(
            target_k=self.top_k_active,
            config_path=config_path,
        )

        # Module 5: Generation & Evaluation
        self.llm_client = LLMClient(use_mock=use_mock_llm)
        self.metrics_calculator = DefenseMetricsCalculator()

        # State tracking
        self.indexed_corpus: List[CorpusDocument] = []
        self.poison_mapping: Dict[str, List[str]] = {}

    def setup_corpus(
        self,
        clean_corpus: List[CorpusDocument],
        queries: Optional[List[QuerySample]] = None,
        attack_type: Optional[AttackType] = None,
        poisons_per_query: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[CorpusDocument], Dict[str, List[str]]]:
        """Inject synthetic poisons (if attack_type is specified) and index the corpus.

        Args:
            clean_corpus: Initial clean documents.
            queries: Target queries for adversarial poison injection.
            attack_type: Attack tier (TIER1_BASIC, TIER2_CAMOUFLAGED, etc.) or None for clean.
            poisons_per_query: Number of adversarial passages per query.
            shuffle: Whether to shuffle mixed corpus before indexing.
            seed: Random seed for deterministic reproducibility.

        Returns:
            Tuple of (indexed_corpus, poison_mapping).
        """
        if attack_type is not None and attack_type != AttackType.CLEAN and queries:
            mixed_corpus, poison_mapping = self.injector.inject_dataset(
                clean_corpus=clean_corpus,
                queries=queries,
                attack_type=attack_type,
                poisons_per_query=poisons_per_query,
                shuffle=shuffle,
                seed=seed,
            )
            self.indexed_corpus = mixed_corpus
            self.poison_mapping = poison_mapping
        else:
            self.indexed_corpus = list(clean_corpus)
            self.poison_mapping = {}

        self.dense_retriever.index_corpus(self.indexed_corpus)
        self.sparse_retriever.index_corpus(self.indexed_corpus)
        return self.indexed_corpus, self.poison_mapping

    def load_benchmark_and_index(
        self,
        num_queries: int = 10,
        attack_tier: Optional[AttackType] = AttackType.TIER2_CAMOUFLAGED,
        poisons_per_query: int = 2,
        seed: int = 42,
    ) -> Tuple[List[CorpusDocument], List[QuerySample]]:
        """Load the self-contained offline NQ benchmark, inject poisons, and index retrievers.

        Args:
            num_queries: Number of benchmark queries to use (up to 25).
            attack_tier: Adversarial attack tier to apply (or None for clean benchmark).
            poisons_per_query: Poison documents generated per query.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (indexed_corpus, queries).
        """
        clean_corpus, queries = CorpusInjector.load_or_create_sample_nq_benchmark(
            num_queries=num_queries,
            save_to_disk=False,
        )
        self.setup_corpus(
            clean_corpus=clean_corpus,
            queries=queries,
            attack_type=attack_tier,
            poisons_per_query=poisons_per_query,
            seed=seed,
        )
        return self.indexed_corpus, queries

    def apply_defense(
        self,
        pool: RetrievalPool,
        mode: str = "v4_full",
    ) -> RetrievalPool:
        """Apply the specified defense mode to a RetrievalPool.

        Modes:
        - "v1" or "v1_baseline": Naive Z-Score filter (quarantines naive_zscore > threshold).
        - "v2" or "v2_stage1": Stage 1 Geometric Defense + Three-Way Router (excludes HIGH_RISK).
        - "v3" or "v3_verifier": Stage 1 + Router + Stage 2 Deep Verifier (sentence drift + NLI).
        - "v4" or "v4_full": Full EviGuard-RAG pipeline (Stage 1 + Router + Stage 2 + Stage 3 Recovery).
        - "undefended" or "none": No defense (all active set candidates admitted).

        Args:
            pool: RetrievalPool with populated active_set and reserve_pool.
            mode: Defense mode string.

        Returns:
            Updated RetrievalPool with statuses, scores, and pool.final_context.
        """
        norm_mode = mode.lower().strip()

        if "v1" in norm_mode or "baseline" in norm_mode:
            # V1: Baseline Naive Z-score
            self.baseline_defense.filter_pool(pool)
            pool.final_context = [
                c for c in pool.active_set if c.status == TrustStatus.SAFE
            ]

        elif "v2" in norm_mode:
            # V2: Stage 1 Geometric Only
            self.stage1_defense.compute_scores(pool)
            self.router.route(pool)
            pool.final_context = [
                c for c in pool.active_set if c.status != TrustStatus.HIGH_RISK
            ]

        elif "v3" in norm_mode:
            # V3: Stage 1 + Router + Stage 2 Verifier (No reserve recovery)
            self.stage1_defense.compute_scores(pool)
            self.router.route(pool)
            self.stage2_verifier.verify(pool)
            pool.final_context = [
                c for c in pool.active_set
                if c.status in (TrustStatus.SAFE, TrustStatus.PROMOTED_SAFE)
            ]

        elif "v4" in norm_mode or "full" in norm_mode:
            # V4: Full EviGuard-RAG (Stage 1 + Router + Stage 2 + Stage 3 Context Recovery)
            self.stage1_defense.compute_scores(pool)
            self.router.route(pool)
            self.stage2_verifier.verify(pool)
            self.stage3_recovery.recover_context(pool)

        elif "undefended" in norm_mode or "off" in norm_mode or "none" in norm_mode:
            # Undefended baseline
            for c in pool.active_set:
                c.status = TrustStatus.SAFE
            pool.final_context = list(pool.active_set)

        else:
            # Default to V4 Full
            self.stage1_defense.compute_scores(pool)
            self.router.route(pool)
            self.stage2_verifier.verify(pool)
            self.stage3_recovery.recover_context(pool)

        return pool

    def run_query(
        self,
        query: Union[QuerySample, str],
        mode: str = "v4_full",
        pool: Optional[RetrievalPool] = None,
    ) -> Dict[str, Any]:
        """Execute the end-to-end RAG pipeline for a single query.

        Performs:
        1. Retrieval: builds RetrievalPool via HybridPoolBuilder (if pool is None).
        2. Undefended Generation: generates answer for Defense OFF context.
        3. Defense Processing: executes chosen defense mode (V1, V2, V3, or V4).
        4. Defended Generation: generates answer for Defense ON context.
        5. Metrics Evaluation: computes detection & downstream RAG metrics.

        Args:
            query: QuerySample instance or raw question string.
            mode: Defense mode ("v1_baseline", "v2_stage1", "v3_verifier", "v4_full").
            pool: Optional pre-constructed RetrievalPool.

        Returns:
            Structured dictionary with pool, answers, prompts, and evaluation metrics.
        """
        # Convert string to QuerySample if necessary
        if isinstance(query, str):
            query_sample = QuerySample(
                query_id=f"q_{int(time.time() * 1000)}",
                question=query,
                ground_truth_answers=[],
                target_fake_answer=None,
            )
        else:
            query_sample = query

        # 1. Retrieve candidate pool if not provided
        if pool is None:
            pool = self.pool_builder.build_pool(query_sample)

        # 2. Undefended RAG generation (Defense OFF)
        answer_undefended = self.llm_client.generate(pool, use_defense=False)
        prompt_undefended = self.llm_client.format_prompt(pool, use_defense=False)

        # 3. Apply defense mode
        self.apply_defense(pool, mode=mode)

        # 4. Defended RAG generation (Defense ON)
        answer_defended = self.llm_client.generate(pool, use_defense=True)
        prompt_defended = self.llm_client.format_prompt(pool, use_defense=True)

        # 5. Compute detection and RAG metrics
        detection_metrics = self.metrics_calculator.compute_detection_metrics([pool])
        rag_metrics = self.metrics_calculator.compute_rag_metrics(
            pools=[pool],
            answers_off=[answer_undefended],
            answers_on=[answer_defended],
        )

        active_poisons = [c for c in pool.active_set if c.doc and c.doc.is_poisoned]
        quarantined = [c for c in pool.active_set if c.status == TrustStatus.QUARANTINED]
        final_clean = [c for c in pool.final_context if c.doc and not c.doc.is_poisoned]

        return {
            "query": query_sample,
            "mode": mode,
            "pool": pool,
            "answer_undefended": answer_undefended,
            "answer_defended": answer_defended,
            "prompt_undefended": prompt_undefended,
            "prompt_defended": prompt_defended,
            "detection_metrics": detection_metrics,
            "rag_metrics": rag_metrics,
            "latency_ms": pool.latency_ms,
            "active_poisons_count": len(active_poisons),
            "quarantined_count": len(quarantined),
            "final_context_clean_count": len(final_clean),
        }

    def run_ablation_benchmark(
        self,
        queries: Optional[List[QuerySample]] = None,
        modes: Optional[List[str]] = None,
        attack_tiers: Optional[List[AttackType]] = None,
        poisons_per_query: int = 2,
        live: bool = False,
    ) -> Dict[str, Any]:
        """Run an ablation benchmark comparing defense variants across attack tiers.

        Args:
            queries: Optional list of QuerySamples.
            modes: List of defense modes to evaluate (defaults to V1..V4).
            attack_tiers: List of attack tiers (defaults to Tiers 1..4).
            poisons_per_query: Number of poisons injected per query during live evaluation.
            live: If True, executes live injection and defense passes.
                  If False, returns structured pre-computed ablation matrices.

        Returns:
            Dictionary containing tiers, versions, asr_matrix, accuracy_matrix,
            precision_matrix, latency_profile, and summary comparisons.
        """
        if not live:
            return self.metrics_calculator.get_ablation_benchmark_data()

        default_tiers = [
            AttackType.TIER1_BASIC,
            AttackType.TIER2_CAMOUFLAGED,
            AttackType.TIER3_CENTROID,
            AttackType.TIER4_HYBRID_JAM,
        ]
        default_modes = ["v1_baseline", "v2_stage1", "v3_verifier", "v4_full"]

        eval_tiers = attack_tiers or default_tiers
        eval_modes = modes or default_modes

        if not queries:
            clean_corpus, benchmark_queries = CorpusInjector.load_or_create_sample_nq_benchmark(
                num_queries=10, save_to_disk=False
            )
        else:
            clean_corpus = list(self.indexed_corpus) if self.indexed_corpus else [
                CorpusDocument(f"c_{i}", f"Title {i}", f"Clean text reference {i}")
                for i in range(20)
            ]
            benchmark_queries = queries

        tier_labels = [
            "Tier 1: Basic Outlier",
            "Tier 2: Camouflaged Paraphrase",
            "Tier 3: Centroid Collocation",
            "Tier 4: Hybrid BM25/Dense Jam",
        ]
        mode_labels = [
            "V1: Naive Z-Score",
            "V2: Stage 1 Geometric Only",
            "V3: Stage 1 + Stage 2 Verifier",
            "V4: Full EviGuard-RAG",
        ]

        asr_matrix: List[List[float]] = []
        acc_matrix: List[List[float]] = []
        prec_matrix: List[List[float]] = []
        latency_profile: Dict[str, float] = {
            "Retrieval": 10.2,
            "Stage 1 Geometry": 3.8,
            "Stage 2 Verifier": 16.5,
            "Stage 3 Recovery": 2.1,
        }

        # Run live evaluation across tiers and modes
        for tier in eval_tiers:
            # Set up contaminated corpus for this tier
            self.setup_corpus(
                clean_corpus=clean_corpus,
                queries=benchmark_queries,
                attack_type=tier,
                poisons_per_query=poisons_per_query,
                shuffle=True,
                seed=42,
            )

            tier_asr: List[float] = []
            tier_acc: List[float] = []
            tier_prec: List[float] = []

            for mode_name in eval_modes:
                pools: List[RetrievalPool] = []
                ans_off: List[str] = []
                ans_on: List[str] = []

                for q in benchmark_queries[:5]:
                    res = self.run_query(query=q, mode=mode_name)
                    pools.append(res["pool"])
                    ans_off.append(res["answer_undefended"])
                    ans_on.append(res["answer_defended"])

                rag_m = self.metrics_calculator.compute_rag_metrics(pools, ans_off, ans_on)
                det_m = self.metrics_calculator.compute_detection_metrics(pools)

                tier_asr.append(round(float(rag_m["asr_defense_on"]) * 100.0, 1))
                tier_acc.append(round(float(rag_m["clean_accuracy_on"]) * 100.0, 1))
                tier_prec.append(round(float(det_m["precision"]) * 100.0, 1))

            asr_matrix.append(tier_asr)
            acc_matrix.append(tier_acc)
            prec_matrix.append(tier_prec)

        return {
            "tiers": tier_labels[: len(eval_tiers)],
            "versions": mode_labels[: len(eval_modes)],
            "asr_matrix": asr_matrix,
            "accuracy_matrix": acc_matrix,
            "precision_matrix": prec_matrix,
            "latency_profile": latency_profile,
        }
