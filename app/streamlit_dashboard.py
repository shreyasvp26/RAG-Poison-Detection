"""
Streamlit interactive demo dashboard for EviGuard-RAG.

Visualizes:
1. Active retrieval pool with flagged poisoned outliers
2. Geometric and NLI risk scores across defense stages
3. Dynamic context recovery and reserve pool backfill
4. Comparative ASR and accuracy metrics
"""

import streamlit as st
import numpy as np

from src.mock_factory import create_mock_retrieval_pool
from src.schemas import TrustStatus


def main():
    st.set_page_config(
        page_title="EviGuard-RAG Defense Dashboard",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ EviGuard-RAG: Embedding-Based Outlier Detection")
    st.markdown(
        "**Multi-Stage Geometric and Semantic Defense against RAG Poisoning Attacks**"
    )

    st.sidebar.header("Configuration")
    seed = st.sidebar.number_input("Random Seed", value=42, min_value=1, max_value=9999)
    top_k = st.sidebar.slider("Active Set (k)", min_value=2, max_value=10, value=5)
    top_n = st.sidebar.slider("Expanded Pool (n)", min_value= top_k, max_value=30, value=15)

    pool = create_mock_retrieval_pool(k=top_k, n=top_n, seed=seed)

    st.subheader(f"Query: *\"{pool.query.question}\"*")
    st.caption(
        f"Query ID: `{pool.query.query_id}` | Ground Truth: `{pool.query.ground_truth_answers}` | "
        f"Target Fake: `{pool.query.target_fake_answer}`"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Active Set Candidates (Top-K)")
        for idx, cand in enumerate(pool.active_set, start=1):
            is_poison = cand.doc.is_poisoned
            status_color = "red" if is_poison else "green"
            badge = "⚠️ POISON OUTLIER" if is_poison else "✅ CLEAN"

            with st.expander(f"Rank {cand.dense_rank}: [{badge}] {cand.doc.title}", expanded=(idx <= 2)):
                st.write(f"**Doc ID:** `{cand.doc.doc_id}` | **Attack Tier:** `{cand.doc.attack_type.value}`")
                st.write(f"**Dense Score:** `{cand.dense_score:.4f}` | **BM25 Rank:** `{cand.bm25_rank}`")
                st.write(f"**Text:** {cand.doc.text}")

    with col2:
        st.markdown("### Reserve Pool (Top-N Backfill)")
        st.write(f"Available reserve candidates: **{len(pool.reserve_pool)}**")
        for cand in pool.reserve_pool[:5]:
            with st.expander(f"Reserve Rank {cand.dense_rank}: {cand.doc.title}"):
                st.write(f"**Dense Score:** `{cand.dense_score:.4f}` | **BM25 Rank:** `{cand.bm25_rank}`")
                st.write(f"**Text:** {cand.doc.text[:120]}...")


if __name__ == "__main__":
    main()
