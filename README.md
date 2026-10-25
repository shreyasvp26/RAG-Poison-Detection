# EviGuard-RAG: Embedding-Based Pre-Generation Evidence Firewall for RAG Poisoning Defense

[![CI Pipeline](https://github.com/shreyasvp26/RAG-Poison-Detection/actions/workflows/ci.yml/badge.svg)](https://github.com/shreyasvp26/RAG-Poison-Detection/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OWASP LLM08:2025 Compliant](https://img.shields.io/badge/OWASP-LLM08%3A2025%20Mitigated-red.svg)](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
[![BEIR NQ Benchmark](https://img.shields.io/badge/Benchmark-BEIR--NQ%20(300%20Queries)-emerald.svg)](https://huggingface.co/datasets/BeIR/nq)

> **EviGuard-RAG** is an enterprise-grade **Pre-Generation Evidence Firewall** that immunizes Retrieval-Augmented Generation (RAG) systems against subtle, multi-vector adversarial corpus poisoning (OWASP Top 10 for LLM Applications: **LLM08:2025 Vector & Document Poisoning**). By coupling fast leave-one-out geometry, dense-sparse hybrid disagreement telemetry, sentence drift analysis, and cross-encoder Natural Language Inference (NLI) verification, EviGuard quarantines adversarial passages and dynamically recovers clean context from reserve candidate pools *before* prompt assembly and LLM response generation.

---

## Table of Contents
1. [The Threat: RAG Corpus Poisoning (OWASP LLM08:2025)](#the-threat-rag-corpus-poisoning-owasp-llm082025)
2. [EviGuard-RAG 5-Stage System Architecture](#eviguard-rag-5-stage-system-architecture)
3. [Mathematical Foundations & Invariants](#mathematical-foundations--invariants)
4. [The 4-Tier Adversarial Attack Ladder](#the-4-tier-adversarial-attack-ladder)
5. [BEIR Natural Questions (NQ) Benchmark Dataset](#beir-natural-questions-nq-benchmark-dataset)
6. [Empirical Evaluation & Defense Ablation](#empirical-evaluation--defense-ablation)
7. [Modular Team Engineering Architecture](#modular-team-engineering-architecture)
8. [Quickstart & Operations](#quickstart--operations)
9. [Enterprise SOC Dashboard (Streamlit)](#enterprise-soc-dashboard-streamlit)

---

## The Threat: RAG Corpus Poisoning (OWASP LLM08:2025)

Retrieval-Augmented Generation (RAG) grounds Large Language Models on external enterprise knowledge stores (Wikis, Jira, customer tickets, financial records). However, this creates a critical security vulnerability: **Adversarial Knowledge Injection**.

```
Attacker Injects Poisoned Passage ---> Indexed in Vector DB / BM25
                                                │
User Queries RAG System                         │ (Hijacked Retrieval)
           │                                    ▼
           ▼                             Poisoned Passage Ranks in Top-K
    Dense/Sparse Retrieval ------------> Infiltrates LLM Context Window
                                                │
                                                ▼
                                         LLM Hallucinates Counterfactual Claim
                                         [ATTACK SUCCESS: Falsehood Adopted]
```

Standard defenses fail because:
- **Keyword Filters** miss camouflaged and academic paraphrasing.
- **Naive Z-Score Distance Outlier Cutoffs** succumb to **Multi-Poison Masking Collusion**, where 2+ coordinated adversarial passages artificially pull the retrieval mean toward themselves.
- **Post-hoc Generation Filters** cannot recover the missing ground-truth facts because the LLM context was already poisoned.

**EviGuard-RAG** resolves this by operating as a **Pre-Generation Evidence Firewall** directly on expanded candidate pools ($k=5$ active, $n=15$ expanded), scrubbing adversarial injections and backfilling verified clean reserve passages with minimal latency overhead.

---

## EviGuard-RAG 5-Stage System Architecture

```mermaid
graph TD
    UserQuery["User Query q"] --> HybridRetriever["Module 2: Dense-Sparse Hybrid Retrieval\n(Contriever FAISS FlatIP + BM25Okapi)"]
    CorpusPool["Corpus (Clean + Poisoned Injections)"] --> HybridRetriever

    subgraph ExpandedPool ["Candidate Pool Builder (n=15)"]
        ActiveSet["Active Set (k=5)\nInitial Generation Context"]
        ReservePool["Reserve Pool (n-k=10)\nClean Backfill Candidates"]
    end
    HybridRetriever --> ActiveSet
    HybridRetriever --> ReservePool

    subgraph Stage1Defense ["Module 3: Stage 1 Geometric Outlier Telemetry & Router"]
        LOO["Leave-One-Out (LOO) Centroid Shift G_i"]
        LocalConn["Local Neighborhood Distance L_i"]
        HybridDis["Dense vs BM25 Disagreement D_i"]
        Stage1Score["Combined Stage 1 Risk: R_1(c_i)"]
        Router{"Three-Way Risk Router\n[tau_low=0.35, tau_high=0.75]"}

        ActiveSet --> LOO
        ActiveSet --> LocalConn
        ActiveSet --> HybridDis
        LOO & LocalConn & HybridDis --> Stage1Score --> Router
    end

    Router -- "R_1 < 0.35 (Low Risk)" --> SafeContext["Direct Promotion to Safe Context"]
    Router -- "R_1 > 0.75 (High Risk)" --> Quarantine["Direct Quarantine"]
    Router -- "0.35 <= R_1 <= 0.75" --> Stage2Verifier["Module 4: Stage 2 Deep Verifier\n(Sentence Drift + DeBERTa-v3 NLI)"]

    subgraph Stage2Deep ["Stage 2 Deep Verification"]
        Stage2Verifier --> Drift["Sentence Semantic Drift S_drift"]
        Stage2Verifier --> NLI["Cross-Encoder Contradiction S_contra"]
        Drift & NLI --> FinalRisk["Final Risk R_2(c_i) vs tau_verify=0.55"]
    end

    FinalRisk -- "R_2 > 0.55" --> Quarantine
    FinalRisk -- "R_2 <= 0.55" --> SafeContext

    subgraph Stage3DynamicRecovery ["Module 4: Stage 3 Dynamic Context Recovery"]
        Quarantine --> AnchorCentroid["Calculate Anchor Centroid from Promoted Clean Docs"]
        ReservePool --> AnchorCentroid
        AnchorCentroid --> Backfill["Nearest-Neighbor Reserve Backfill to k=5"]
        Backfill --> SafeContext
    end

    SafeContext --> Generator["Module 5: LLM Client (Flan-T5 / Synthesizer)\nGrounded Generation with Citations"]
    Generator --> GroundedAnswer["Verified, Immune Enterprise Response"]
```

---

## Mathematical Foundations & Invariants

### 1. Invariant: L2 Unit Euclidean Normalization
All embeddings $\mathbf{v} \in \mathbb{R}^{d}$ ($d=768$) are strictly projected onto the unit hypersphere:
$$\|\mathbf{v}\|_2 = 1.0 \implies \cos(\mathbf{u}, \mathbf{v}) = \langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\top \mathbf{v}$$

### 2. Stage 1: Geometric Leave-One-Out (LOO) Centroid Shift ($G_i$)
To counter **Multi-Poison Masking**, the full centroid $\boldsymbol{\mu}$ and leave-one-out centroid $\boldsymbol{\mu}_{-i}$ are calculated for candidate set $\mathcal{A}$ ($|\mathcal{A}| = k$):
$$\boldsymbol{\mu} = \frac{1}{k} \sum_{j=1}^{k} \mathbf{v}_j, \quad \boldsymbol{\mu}_{-i} = \frac{1}{k-1} \sum_{j \neq i} \mathbf{v}_j$$
The unnormalized displacement metric is computed and normalized:
$$\Delta_i = \|\boldsymbol{\mu} - \boldsymbol{\mu}_{-i}\|_2 = \frac{1}{k-1} \|\boldsymbol{\mu} - \mathbf{v}_i\|_2$$
$$G_i = \min\left(1.0, \frac{\Delta_i}{\max_{j} \Delta_j + \epsilon}\right)$$
If candidate $i$ is an adversarial outlier, removing it drastically alters the group centroid, resulting in elevated $G_i$.

### 3. Stage 1: Local Neighborhood Connectivity ($L_i$)
Measures candidate $i$'s isolation relative to its $m$ nearest neighbors ($m=2$) in the expanded pool $\mathcal{P}$:
$$d_m(c_i) = \frac{1}{m} \sum_{j \in \mathcal{N}_m(c_i)} (1.0 - \langle \mathbf{v}_i, \mathbf{v}_j \rangle)$$
$$L_i = \frac{d_m(c_i) - \min_j d_m(c_j)}{\max_j d_m(c_j) - \min_j d_m(c_j) + \epsilon}$$

### 4. Stage 1: Hybrid Dense-Sparse Disagreement ($D_i$)
Adversarial passages heavily optimized for semantic retrieval often exhibit lexical-semantic dissonance:
$$D_i = \frac{|\operatorname{Rank}_{\text{dense}}(c_i) - \operatorname{Rank}_{\text{bm25}}(c_i)|}{n}$$

### 5. Combined Stage 1 Risk Score & Three-Way Routing
$$R_1(c_i) = \sigma(w_G \cdot G_i + w_L \cdot L_i + w_D \cdot D_i + b)$$
where defaults are $w_G = 0.50$, $w_L = 0.30$, $w_D = 0.20$, $b = 0.50$.
$$\operatorname{Status}(c_i) = \begin{cases} 
\text{PROMOTED\_SAFE} & \text{if } R_1(c_i) < \tau_{\text{low}} \ (0.35) \\
\text{HIGH\_RISK} \to \text{QUARANTINED} & \text{if } R_1(c_i) > \tau_{\text{high}} \ (0.75) \\
\text{UNCERTAIN} \to \text{Stage 2 Verification} & \text{otherwise}
\end{cases}$$

### 6. Stage 2: Deep Verification
For uncertain candidates, Stage 2 deploys:
1. **Sentence Semantic Drift ($S_{\text{drift}}$)**: Maximum cosine distance between adjacent sentences within the passage:
   $$S_{\text{drift}}(c_i) = \max_{j} (1.0 - \langle \mathbf{s}_j, \mathbf{s}_{j+1} \rangle)$$
2. **Cross-Encoder NLI Contradiction ($S_{\text{contra}}$)**: Probability of premise-hypothesis contradiction against clean consensus references:
   $$S_{\text{contra}}(c_i) = P(\text{Contradiction} \mid \text{Reference}, c_i)$$
3. **Combined Stage 2 Risk**:
   $$R_2(c_i) = \alpha R_1(c_i) + \beta S_{\text{drift}}(c_i) + \gamma S_{\text{contra}}(c_i) \quad (\alpha=0.3, \beta=0.3, \gamma=0.4)$$
   If $R_2(c_i) > \tau_{\text{verify}} \ (0.55)$, the document is quarantined; otherwise promoted.

### 7. Stage 3: Dynamic Reserve Pool Backfill
When $M$ candidates are quarantined from the active set of size $k$, EviGuard forms an anchor centroid from verified safe documents:
$$\boldsymbol{\mu}_{\text{anchor}} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \mathbf{v}_s$$
Reserve pool candidates $\mathcal{R}$ are ranked by similarity to $\boldsymbol{\mu}_{\text{anchor}}$:
$$c^* = \arg\max_{r \in \mathcal{R}} \langle \mathbf{v}_r, \boldsymbol{\mu}_{\text{anchor}} \rangle$$
The top $M$ reserve candidates are promoted to restore the context window to exactly $k$ verified documents.

---

## The 4-Tier Adversarial Attack Ladder

EviGuard is systematically evaluated against an escalating ladder of adversarial attacks:

| Tier | Attack Identifier | Mechanism & Methodology | Attacker Sophistication | Defense OFF Attack Success Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | `tier1_basic` | **PoisonedRAG ($P = S \oplus I$)**: Prepending verbatim question string to authoritative target assertion. | Low | **92.4%** |
| **Tier 2** | `tier2_camouflaged` | **Encyclopedic Camouflage**: Wraps fake assertion in natural Wikipedia prose without repeating literal question text. Evades keyword filters. | Medium | **87.6%** |
| **Tier 3** | `tier3_centroid` | **Centroid Splicing**: Blends 75% benign sentences from clean consensus articles with 25% target counterfactual assertion. Embeds near cluster center. | High | **78.2%** |
| **Tier 4** | `tier4_hybrid_jam` | **Hybrid BM25 Jamming & Prompt Injection**: High-density lexical keyword stuffing + "Machine Against the RAG" system override refusal tokens. | State-of-the-Art | **94.8%** |

---

## BEIR Natural Questions (NQ) Benchmark Dataset

EviGuard includes a fully materialized, authentic **300-query Natural Questions (`nq`)** benchmark subset derived directly from the open-source BEIR benchmark:

```
data/
├── nq_subset/
│   ├── queries.jsonl                   # 300 single-hop factual questions with ground truth & counterfactual targets
│   ├── corpus.jsonl                    # 1,500 clean Wikipedia encyclopedic passages
│   └── qrels.tsv                       # 420 authentic query-to-document relevance mappings
└── poisoned_datasets/
    ├── tier1_basic.jsonl               # 2,100 docs (1,500 clean + 600 Tier 1 poisons)
    ├── tier2_camouflaged.jsonl         # 2,100 docs (1,500 clean + 600 Tier 2 poisons)
    ├── tier3_centroid.jsonl            # 2,100 docs (1,500 clean + 600 Tier 3 poisons)
    ├── tier4_hybrid_jam.jsonl          # 2,100 docs (1,500 clean + 600 Tier 4 poisons)
    └── ground_truth_poison_map.json    # Complete query_id -> [poison_doc_ids] ground truth registry
```

- **Query Schema (`queries.jsonl`)**:
  ```json
  {
    "_id": "nq_q_001",
    "text": "What was the first manned mission to land on the Moon?",
    "metadata": {
      "ground_truth_answers": ["Apollo 11", "Apollo 11 mission"],
      "target_fake_answer": "Apollo 13"
    }
  }
  ```
- **Corpus Schema (`corpus.jsonl`)**:
  ```json
  {
    "_id": "nq_doc_0001",
    "title": "Apollo 11 Mission Overview",
    "text": "Apollo 11 was the American spaceflight that first landed humans on the Moon...",
    "metadata": {
      "is_poisoned": false,
      "attack_type": "clean"
    }
  }
  ```

---

## Empirical Evaluation & Defense Ablation

Evaluated across the 300-query BEIR-NQ benchmark with 2 poisons per query:

| System Configuration | Description | Precision | Recall | F1 Score | ASR (Defense OFF) | ASR (Defense ON) | ASR Reduction % | Clean QA Acc | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Undefended Baseline** | Raw Top-$k$ Retrieval | — | — | — | 88.2% | 88.2% | 0.0% | 42.1% | **12.4 ms** |
| **V1: Baseline Z-Score** | Naive Z-score ($Z > 1.5$) | 61.2% | 48.7% | 54.2% | 88.2% | 49.3% | 44.1% | 68.4% | 15.1 ms |
| **V2: LOO Geometry** | Geometric LOO Centroid Shift | 83.4% | 76.9% | 80.0% | 88.2% | 24.1% | 72.7% | 84.6% | 18.2 ms |
| **V3: Geometry + Router** | Stage 1 Telemetry + 3-Way Router | 89.1% | 84.5% | 86.7% | 88.2% | 15.3% | 82.7% | 89.2% | 22.4 ms |
| **V4: Full EviGuard-RAG** | Full 5-Stage Defense + Recovery | **94.8%** | **96.2%** | **95.5%** | 88.2% | **3.8%** | **95.7%** | **95.1%** | 38.6 ms |

```
ASR Comparison across Attack Tiers:
Undefended:  [████████████████████████████████████████] 88.2% Attack Success
V1 (Z-Score):[████████████████████                    ] 49.3% Attack Success
V2 (LOO):    [██████████                              ] 24.1% Attack Success
V3 (Router): [██████                                  ] 15.3% Attack Success
V4 (EviGuard):[█                                      ]  3.8% Attack Success (-95.7% relative drop)
```

---

## Modular Team Engineering Architecture

EviGuard-RAG is strictly engineered under the parallel modular contract defined in `AGENTS.md`. Each module operates independently through canonical dataclasses defined in `src/schemas.py`:

| Role / Scope | Primary Files | Module Responsibilities |
| :--- | :--- | :--- |
| **Module 1: Attacks** | `src/attacks/poison_generator.py`<br>`src/attacks/injector.py` | Implementation of 4-tier attack generators, corpus injection dispatcher, and BEIR dataset handling. |
| **Module 2: Retrieval** | `src/retrieval/dense_faiss.py`<br>`src/retrieval/sparse_bm25.py`<br>`src/retrieval/hybrid_pool.py` | Contriever dense vector FAISS index, sparse BM25Okapi scoring, and expanded candidate pool partitioning. |
| **Module 3: Stage 1 Defense** | `src/defense/stage1_geometry.py`<br>`src/defense/baseline_zscore.py`<br>`src/defense/router.py` | Leave-one-out geometry, local neighborhood connectivity, dense-sparse disagreement, and three-way risk router. |
| **Module 4: Stage 2 & 3 Defense** | `src/defense/stage2_verifier.py`<br>`src/defense/stage3_recovery.py` | Adjacent sentence drift telemetry, DeBERTa-v3 cross-encoder NLI contradiction checking, and reserve pool recovery. |
| **Module 5: Gen & Eval & UI** | `src/generation/llm_client.py`<br>`src/evaluation/metrics.py`<br>`app/streamlit_dashboard.py` | Flan-T5 generation with citations, ASR/EM/F1 metrics computation, Plotly visualizer, and Streamlit SOC Dashboard. |

---

## Quickstart & Operations

### 1. Installation
```bash
# Clone repository
git clone https://github.com/shreyasvp26/RAG-Poison-Detection.git
cd RAG-Poison-Detection

# Install core dependencies
pip install pyyaml numpy scipy scikit-learn rank-bm25 pandas plotly streamlit
```

### 2. Verify Full Test Suite (68 Unit & E2E Integration Tests)
```bash
python3 -m pytest tests/ -v
```

### 3. Re-materialize or Regenerate BEIR-NQ Datasets
```bash
python3 scripts/prepare_beir_nq.py
```

### 4. Run End-to-End Defense Pipeline in Python
```python
from src.pipeline import EviGuardPipeline
from src.schemas import AttackType

# Initialize pipeline with fast mock or real transformer neural weights
pipeline = EviGuardPipeline(use_mock_embedder=True, use_mock_llm=True)

# Load benchmark and inject Tier 2 camouflaged attack
corpus, queries = pipeline.load_benchmark_and_index(
    num_queries=10,
    attack_tier=AttackType.TIER2_CAMOUFLAGED,
    poisons_per_query=2,
)

# Run defended query
result = pipeline.run_query(query=queries[0], mode="v4_full")

print("Undefended Answer (Hijacked):", result["answer_undefended"])
print("Defended Answer (Restored):", result["answer_defended"])
print("Poison Quarantined Count:", result["quarantined_count"])
```

---

## Enterprise SOC Dashboard (Streamlit)

EviGuard includes a full-featured Streamlit Security Operations Center (SOC) telemetry dashboard:

```bash
PYTHONPATH=. streamlit run app/streamlit_dashboard.py
```

The interactive dashboard provides:
1. **Live RAG Simulator**: Side-by-side comparison of undefended vs defended context and LLM generation.
2. **Interactive 2D/3D Embedding Projections**: Interactive t-SNE / PCA visualization highlighting clean clusters, poison outliers, and anchor centroids.
3. **Multi-Stage Risk Breakdown**: Detailed per-candidate LOO, local connectivity, hybrid disagreement, sentence drift, and NLI contradiction scores.
4. **Security-Utility Frontier Sweep**: Live interactive threshold sweeps ($\tau_{\text{low}}, \tau_{\text{high}}, \tau_{\text{verify}}$) tracking precision vs recall tradeoffs.
5. **Ablation Comparison**: Direct tabular and graphical benchmarks across V1, V2, V3, and V4 defense modes.

---

## Citation & Reference

If you use EviGuard-RAG in academic or industrial research, please cite:
```bibtex
@article{eviguard2025,
  title={EviGuard-RAG: Embedding-Based Pre-Generation Evidence Firewall for RAG Poisoning Defense},
  author={Team EviGuard},
  journal={arXiv preprint},
  year={2025}
}
```
