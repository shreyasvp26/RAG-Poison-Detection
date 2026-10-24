# AGENTS.md — Team Contract & Agent Working Guidelines

Welcome to the **EviGuard-RAG** codebase ("Embedding-Based Outlier Detection for RAG Poisoning Attacks").
To allow 5 team members and automated AI agents to build, test, and iterate in parallel without merge conflicts, all agents must strictly follow these mandatory rules.

---

## 1. Strict Data Contract Enforcement (`src/schemas.py`)
- **NEVER** invent ad-hoc dictionaries, tuples, or custom data structures to pass documents, queries, scores, or retrieval pools between modules.
- **ALWAYS** import and use the canonical dataclasses from `src/schemas.py`:
  - `AttackType`: `CLEAN`, `TIER1_BASIC`, `TIER2_CAMOUFLAGED`, `TIER3_CENTROID`, `TIER4_HYBRID_JAM`
  - `TrustStatus`: `UNCHECKED`, `SAFE`, `UNCERTAIN`, `HIGH_RISK`, `PROMOTED_SAFE`, `QUARANTINED`, `REPLACED_RESERVE`
  - `CorpusDocument`
  - `QuerySample`
  - `RetrievedCandidate`
  - `RetrievalPool`
- If you need to attach intermediate metrics, use the designated fields in `RetrievedCandidate` (e.g., `naive_zscore`, `loo_geometric_score`, `local_connectivity_score`, `hybrid_disagreement_score`, `stage1_risk`, `sentence_drift_score`, `nli_contradiction_score`, `final_risk`, `status`) or add a documented field to `src/schemas.py` only after team consensus.

---

## 2. Configuration (`config.yaml`)
- **DO NOT** hardcode hyperparameters, thresholds, model identifiers, or dataset sample sizes into Python scripts.
- **ALWAYS** read parameters from `config.yaml`:
  - `dataset.name`, `sample_queries`, `poisons_per_query`
  - `retrieval.dense_model`, `embedding_dim`, `top_k_active`, `top_n_expanded`
  - `stage1_defense.baseline_z_threshold`, `local_neighbors_m`, `weights.*`, `thresholds.tau_low`, `thresholds.tau_high`
  - `stage2_defense.nli_model`, `weights.*`, `tau_verify`

---

## 3. Mathematical Invariants: L2 Embedding Normalization
- All embeddings (dense Contriever embeddings, query embeddings, sentence embeddings) **MUST BE L2-NORMALIZED** to unit Euclidean length ($\|v\|_2 = 1.0$).
- Cosine similarity must always be computable as a standard inner product `np.dot(u, v)`.
- When calculating leave-one-out geometric centroids or local neighborhood distances, preserve or re-normalize projections as appropriate.

---

## 4. Module Ownership & Parallel Isolation
Each team member / agent is assigned a dedicated module directory. **Stay within your assigned directory** and interact with other modules solely through `src/schemas.py` and `src/mock_factory.py`:

| Role / Member | Directory Scope | Responsibilities |
| :--- | :--- | :--- |
| **Member 1 (Attacks)** | `src/attacks/` | Implement Tier 1-4 attack generators (`poison_generator.py`) and corpus injection (`injector.py`). |
| **Member 2 (Retrieval)** | `src/retrieval/` | Dense FAISS (`dense_faiss.py`), sparse BM25 (`sparse_bm25.py`), and hybrid pool builder (`hybrid_pool.py`). |
| **Member 3 (Stage 1 Defense)** | `src/defense/baseline_zscore.py`, `src/defense/stage1_geometry.py`, `src/defense/router.py` | Geometric LOO, local connectivity, hybrid disagreement, and 3-way router. |
| **Member 4 (Stage 2 & 3 Defense)** | `src/defense/stage2_verifier.py`, `src/defense/stage3_recovery.py` | Sentence drift, NLI contradiction verification, and reserve pool backfill recovery. |
| **Member 5 (Gen & Eval & UI)** | `src/generation/`, `src/evaluation/`, `app/` | Flan-T5 LLM client, metrics calculation (ASR, EM, F1), visualizations, and Streamlit dashboard. |

---

## 5. Independent Testing with `src/mock_factory.py`
- Do not wait for upstream components (e.g., full FAISS index or NLI model weights) to test your module.
- Use `from src.mock_factory import create_mock_retrieval_pool`:
  - Rank 2 in `pool.active_set` is a synthesized poisoned outlier.
  - Generates 768-dim normalized embeddings.
  - Total $k=5$ active candidates, $n-k=10$ reserve candidates.
- Run tests regularly:
  ```bash
  pytest tests/test_smoke.py
  ```
