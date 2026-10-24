"""
Unit and Contract Tests for Module 2: Hybrid Expanded Retrieval Engine.

Tests:
1. DenseFaissRetriever: L2-normalization, (768,) shape, deterministic embeddings,
   and self-similarity ranking (#1 rank for matching text).
2. SparseBM25Retriever: Lexical tokenization, scoring, and 1-indexed rank ordering.
3. HybridPoolBuilder: End-to-end pool construction, (k=5, n-k=10) partitioning,
   active set flags, candidate embedding normalization, and latency tracking.
"""

import numpy as np
import pytest

from src.schemas import AttackType, CorpusDocument, QuerySample, RetrievalPool, TrustStatus
from src.retrieval.dense_faiss import DenseFaissRetriever
from src.retrieval.sparse_bm25 import SparseBM25Retriever
from src.retrieval.hybrid_pool import HybridPoolBuilder


@pytest.fixture
def sample_corpus():
    """Create a diverse 20-document corpus for retrieval testing."""
    topics = [
        ("Apollo 11 Mission", "Apollo 11 was the American spaceflight that first landed humans on the Moon in July 1969."),
        ("Neil Armstrong Biography", "Neil Armstrong was an American astronaut and aeronautical engineer who became the first person to walk on the Moon."),
        ("Lunar Roving Vehicle", "The Lunar Roving Vehicle was an electric vehicle designed to operate in the low-gravity vacuum of the Moon."),
        ("Apollo 13 Mission Incident", "Apollo 13 was the seventh crewed mission in the Apollo space program, famous for an onboard oxygen tank explosion."),
        ("Saturn V Rocket", "The Saturn V was a human-rated super heavy-lift launch vehicle used by NASA between 1967 and 1973."),
        ("Kennedy Space Center", "John F. Kennedy Space Center is one of ten NASA field centers located on Merritt Island, Florida."),
        ("International Space Station", "The International Space Station is a modular space station in low Earth orbit with multinational collaboration."),
        ("Voyager 1 Interstellar Probe", "Voyager 1 is a space probe launched by NASA in 1977 to study the outer Solar System and interstellar space."),
        ("Hubble Space Telescope", "The Hubble Space Telescope is a space telescope that was launched into low Earth orbit in 1990."),
        ("James Webb Space Telescope", "The James Webb Space Telescope is designed primarily to conduct infrared astronomy with high sensitivity."),
        ("Neapolitan Pizza Heritage", "Traditional Neapolitan pizza is crafted with San Marzano tomatoes, mozzarella di bufala, and baked in a wood-fired oven."),
        ("Deep Dish Pizza History", "Chicago-style deep-dish pizza features a thick crust with cheese, meat, and chunky tomato sauce baked in an iron pan."),
        ("Quantum Superposition Principles", "Quantum superposition states that any two or more quantum states can be added together to yield another valid quantum state."),
        ("Theory of General Relativity", "General relativity is Einstein's geometric theory of gravitation where gravity arises from spacetime curvature."),
        ("Photosynthesis in Green Plants", "Photosynthesis is a biological process used by plants and other organisms to convert light energy into chemical energy."),
        ("CRISPR Gene Editing Technology", "CRISPR-Cas9 is a genome editing tool that allows researchers to alter DNA sequences and gene function."),
        ("Renaissance Art in Florence", "The Italian Renaissance was a cultural movement that began in Florence, producing masters like Leonardo and Michelangelo."),
        ("French Impressionist Painting", "Impressionism originated in France characterized by relatively small, thin, yet visible brush strokes and open composition."),
        ("Federal Reserve Monetary Policy", "The Federal Reserve conducts national monetary policy to foster maximum employment and stable prices."),
        ("Machine Learning Neural Networks", "Artificial neural networks are computing systems inspired by the biological neural networks of human brains."),
    ]
    docs = []
    for i, (title, text) in enumerate(topics):
        docs.append(
            CorpusDocument(
                doc_id=f"doc_{i:03d}",
                title=title,
                text=text,
                is_poisoned=False,
                attack_type=AttackType.CLEAN,
            )
        )
    return docs


def test_dense_faiss_retriever_l2_normalization_and_shape():
    """Verify DenseFaissRetriever produces (768,) float32 embeddings with unit L2 norm."""
    retriever = DenseFaissRetriever(embedding_dim=768, use_mock_embedder=True)

    texts = [
        "What was the first manned mission to land on the Moon?",
        "How is traditional Neapolitan pizza baked?",
        "",  # Empty string edge case
    ]
    embeddings = retriever.encode_texts(texts)

    assert embeddings.shape == (3, 768)
    assert embeddings.dtype == np.float32

    # Check each vector has unit L2 norm
    for i in range(len(texts)):
        norm = np.linalg.norm(embeddings[i])
        np.testing.assert_allclose(norm, 1.0, atol=1e-5)

    # Check encode_query
    q_emb = retriever.encode_query("Apollo 11 landing date")
    assert q_emb.shape == (768,)
    assert q_emb.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(q_emb), 1.0, atol=1e-5)

    # Check determinism: identical text yields identical vector
    q_emb2 = retriever.encode_query("Apollo 11 landing date")
    np.testing.assert_allclose(q_emb, q_emb2, atol=1e-7)


def test_dense_faiss_retriever_self_similarity_and_ranking(sample_corpus):
    """Verify DenseFaissRetriever indexes corpus and ranks exact matching text at #1."""
    retriever = DenseFaissRetriever(embedding_dim=768, use_mock_embedder=True)
    retriever.index_corpus(sample_corpus)

    doc0 = sample_corpus[0]
    # 1. Exact self-similarity test
    query_exact = QuerySample(
        query_id="q_exact",
        question=f"{doc0.title} {doc0.text}",
        ground_truth_answers=["Apollo 11"],
    )
    exact_hits = retriever.retrieve(query_exact, top_n=5)
    assert exact_hits[0][0].doc_id == doc0.doc_id
    assert exact_hits[0][2] == 1
    # Self-similarity dot product is identically 1.0 for exact matching text
    np.testing.assert_allclose(exact_hits[0][1], 1.0, atol=1e-5)

    # 2. Semantic query test
    query_semantic = QuerySample(
        query_id="q_apollo",
        question="Apollo 11 was the American spaceflight that first landed humans on the Moon in July 1969.",
        ground_truth_answers=["Apollo 11"],
    )
    hits = retriever.retrieve(query_semantic, top_n=5)

    assert len(hits) == 5
    top_doc, top_score, top_rank = hits[0]

    assert top_doc.doc_id == doc0.doc_id
    assert top_rank == 1
    assert top_score >= 0.90

    # Check ranks are strictly sequential 1..5
    ranks = [hit[2] for hit in hits]
    assert ranks == [1, 2, 3, 4, 5]

    # Check scores are monotonically non-increasing
    scores = [hit[1] for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_dense_faiss_retriever_empty_corpus():
    """Verify DenseFaissRetriever handles empty corpus gracefully."""
    retriever = DenseFaissRetriever(embedding_dim=768, use_mock_embedder=True)
    retriever.index_corpus([])

    query = QuerySample("q_empty", "Empty test question", ["answer"])
    hits = retriever.retrieve(query, top_n=5)
    assert hits == []

    q_vec = retriever.encode_query("Test")
    assert retriever.search(q_vec, top_n=5) == []


def test_sparse_bm25_retriever_lexical_scoring_and_ranking(sample_corpus):
    """Verify SparseBM25Retriever tokenizes, scores lexically, and produces 1-indexed ranks."""
    bm25 = SparseBM25Retriever(k1=1.5, b=0.75)
    bm25.index_corpus(sample_corpus)

    query_text = "Apollo 11 Moon landing July 1969"
    scores, ranks = bm25.score_all(query_text)

    assert scores.shape == (len(sample_corpus),)
    assert scores.dtype == np.float32
    assert ranks.shape == (len(sample_corpus),)
    assert ranks.dtype == np.int32

    # Apollo 11 document (doc 0) must have the highest BM25 score and rank 1
    assert ranks[0] == 1
    assert scores[0] == np.max(scores)
    assert scores[0] > 0.0

    # Documents with 0 lexical overlap (e.g. doc 19: Machine Learning Neural Networks) should have score 0.0
    assert scores[19] == 0.0
    assert ranks[19] > 1

    # Verify 1-indexed ranks form a valid permutation from 1 to N
    sorted_ranks = sorted(list(ranks))
    assert sorted_ranks == list(range(1, len(sample_corpus) + 1))


def test_sparse_bm25_score_document_and_retrieve(sample_corpus):
    """Verify SparseBM25Retriever single document scoring and top-N retrieval."""
    bm25 = SparseBM25Retriever()
    bm25.index_corpus(sample_corpus)

    query = QuerySample(
        query_id="q_pizza",
        question="How is traditional Neapolitan pizza crafted with wood-fired oven?",
        ground_truth_answers=["San Marzano tomatoes, mozzarella di bufala"],
    )

    # Score document directly
    doc_pizza = sample_corpus[10]  # Neapolitan Pizza
    doc_neural = sample_corpus[19]  # Neural networks (0 overlap)
    score_pizza = bm25.score_document(query, doc_pizza)
    score_neural = bm25.score_document(query, doc_neural)

    assert score_pizza > score_neural
    assert score_neural == 0.0

    # Retrieve top 5
    hits = bm25.retrieve(query, top_n=5)
    assert len(hits) == 5
    top_doc, top_score, top_rank = hits[0]

    assert top_doc.doc_id == doc_pizza.doc_id
    assert top_rank == 1
    assert top_score == score_pizza


def test_sparse_bm25_empty_corpus():
    """Verify SparseBM25Retriever handles empty corpus gracefully."""
    bm25 = SparseBM25Retriever()
    bm25.index_corpus([])

    scores, ranks = bm25.score_all("sample query")
    assert len(scores) == 0
    assert len(ranks) == 0

    query = QuerySample("q_empty", "Test query", ["ans"])
    assert bm25.retrieve(query) == []


def test_hybrid_pool_builder_contract_partitions(sample_corpus):
    """Verify HybridPoolBuilder produces RetrievalPool adhering to AGENTS.md contract."""
    dense_retriever = DenseFaissRetriever(embedding_dim=768, use_mock_embedder=True)
    dense_retriever.index_corpus(sample_corpus)

    sparse_retriever = SparseBM25Retriever()
    sparse_retriever.index_corpus(sample_corpus)

    # Read from config.yaml (top_k_active=5, top_n_expanded=15)
    builder = HybridPoolBuilder(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        config_path="config.yaml",
    )

    query = QuerySample(
        query_id="nq_q_1001",
        question="What was the first manned mission to land on the Moon?",
        ground_truth_answers=["Apollo 11"],
    )

    pool = builder.build_pool(query)

    # 1. Contract asserts on counts and types
    assert isinstance(pool, RetrievalPool)
    assert len(pool.active_set) == 5, f"Expected 5 active candidates, got {len(pool.active_set)}"
    assert len(pool.reserve_pool) == 10, f"Expected 10 reserve candidates, got {len(pool.reserve_pool)}"

    # 2. Query embedding asserts
    assert pool.query_embedding.shape == (768,)
    assert pool.query_embedding.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(pool.query_embedding), 1.0, atol=1e-5)

    # 3. Active set validation
    for i, cand in enumerate(pool.active_set):
        assert cand.is_in_active_set is True
        assert cand.dense_rank == i + 1
        assert cand.status == TrustStatus.UNCHECKED
        assert cand.embedding.shape == (768,)
        assert cand.embedding.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(cand.embedding), 1.0, atol=1e-5)
        assert isinstance(cand.bm25_score, float)
        assert cand.bm25_rank >= 1

    # 4. Reserve pool validation
    for j, cand in enumerate(pool.reserve_pool):
        assert cand.is_in_active_set is False
        assert cand.dense_rank == 5 + j + 1
        assert cand.status == TrustStatus.UNCHECKED
        assert cand.embedding.shape == (768,)
        assert cand.embedding.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(cand.embedding), 1.0, atol=1e-5)
        assert isinstance(cand.bm25_score, float)
        assert cand.bm25_rank >= 1

    # 5. Latency profiling populated
    assert "retrieval" in pool.latency_ms
    assert pool.latency_ms["retrieval"] > 0.0


def test_hybrid_pool_builder_custom_parameters_and_precomputed_embedding(sample_corpus):
    """Verify HybridPoolBuilder supports explicit custom K, N, and precomputed query embeddings."""
    dense_retriever = DenseFaissRetriever(embedding_dim=768, use_mock_embedder=True)
    dense_retriever.index_corpus(sample_corpus)

    sparse_retriever = SparseBM25Retriever()
    sparse_retriever.index_corpus(sample_corpus)

    # Explicit custom sizes: k=3, n=8
    builder = HybridPoolBuilder(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        top_k_active=3,
        top_n_expanded=8,
    )

    query = QuerySample(
        query_id="q_custom",
        question="Tell me about space missions",
        ground_truth_answers=["Apollo"],
    )

    # Provide precomputed normalized query embedding
    rng = np.random.default_rng(123)
    precomputed = rng.standard_normal(768).astype(np.float32)
    precomputed /= np.linalg.norm(precomputed)

    pool = builder.build_pool(query, query_embedding=precomputed)

    assert len(pool.active_set) == 3
    assert len(pool.reserve_pool) == 5
    np.testing.assert_allclose(pool.query_embedding, precomputed, atol=1e-6)

    for cand in pool.active_set:
        assert cand.is_in_active_set is True
    for cand in pool.reserve_pool:
        assert cand.is_in_active_set is False


def test_hybrid_pool_builder_small_corpus():
    """Verify HybridPoolBuilder behaves correctly when corpus has fewer documents than top_n."""
    small_corpus = [
        CorpusDocument("d1", "Title 1", "Text about space 1"),
        CorpusDocument("d2", "Title 2", "Text about space 2"),
        CorpusDocument("d3", "Title 3", "Text about space 3"),
    ]

    dense = DenseFaissRetriever(embedding_dim=768, use_mock_embedder=True)
    dense.index_corpus(small_corpus)

    sparse = SparseBM25Retriever()
    sparse.index_corpus(small_corpus)

    builder = HybridPoolBuilder(dense, sparse, top_k_active=2, top_n_expanded=10)
    query = QuerySample("q_small", "space", ["space"])
    pool = builder.build_pool(query)

    assert len(pool.active_set) == 2
    assert len(pool.reserve_pool) == 1
    assert len(pool.active_set) + len(pool.reserve_pool) == 3
