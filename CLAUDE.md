# CLAUDE.md — Instructions for Claude / AI Assistants

This repository contains **EviGuard-RAG: Embedding-Based Outlier Detection for RAG Poisoning Attacks**.
Follow these strict instructions during any task or code change.

## Core Directives

1. **Strict Data Schemas (`src/schemas.py`)**:
   - Never use unstructured dicts or tuples for inter-module data transfer.
   - Always use `AttackType`, `TrustStatus`, `CorpusDocument`, `QuerySample`, `RetrievedCandidate`, and `RetrievalPool`.
   - Any new cross-module fields must be added directly to `src/schemas.py`.

2. **Configuration (`config.yaml`)**:
   - Read all parameters, thresholds, model names, and weights from `config.yaml`.
   - Never hardcode weights like `w_geometric: 0.50` or thresholds like `tau_low: 0.35`.

3. **L2 Normalization**:
   - Embeddings must always remain 768-dimensional and L2-normalized (`np.linalg.norm(vec) == 1.0`).
   - Inner product (`np.dot(u, v)`) equals cosine similarity.

4. **Directory Isolation**:
   - Keep changes scoped to the module under development:
     - `src/attacks/` — Poison generation and injection
     - `src/retrieval/` — Dense FAISS, Sparse BM25, and HybridPoolBuilder
     - `src/defense/` — Baseline Z-score, Stage 1 Geometry, Three-Way Router, Stage 2 Verifier, Stage 3 Recovery
     - `src/generation/` — LLM generator client
     - `src/evaluation/` — Metrics and visualization
     - `app/` — Streamlit interactive UI
     - `tests/` — Test suites
   - Never introduce merge conflicts across team boundaries.

5. **Testing & Mocking**:
   - Use `src/mock_factory.py` for testing individual stages in isolation.
   - Run tests with:
     ```bash
     pytest tests/test_smoke.py
     ```
