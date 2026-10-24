"""
Unit test suite for Module 1: Adversarial Attacks and Corpus Injection.

Tests:
1. Tier 1 Basic Attack (PoisonedRAG P = S ⊕ I).
2. Tier 2 Camouflaged Attack (Encyclopedic prose without literal question repetition).
3. Tier 3 Centroid Attack (75% clean reference sentences + 25% counterfactual assertion).
4. Tier 4 Hybrid Jam Attack (High-density BM25 lexical keyword jamming + Machine Against RAG blocker).
5. PoisonGenerator dispatcher across all tiers and error handling.
6. CorpusInjector dataset injection and ground truth poison mapping.
7. CorpusInjector corpus splitting and partitioning.
8. Self-contained 25-query / 150-doc factual Wikipedia QA mini-benchmark.
9. Contract invariance and dataclass integrity.
"""

import os
import re
import pytest

from src.attacks.injector import CorpusInjector
from src.attacks.poison_generator import PoisonGenerator, _split_sentences, _extract_keywords
from src.schemas import AttackType, CorpusDocument, QuerySample


@pytest.fixture
def sample_query() -> QuerySample:
    """Fixture providing a standard query sample."""
    return QuerySample(
        query_id="nq_q_001",
        question="What was the first manned mission to land on the Moon?",
        ground_truth_answers=["Apollo 11", "Apollo 11 mission"],
        target_fake_answer="Apollo 13",
    )


@pytest.fixture
def sample_reference_docs() -> list[CorpusDocument]:
    """Fixture providing realistic clean reference documents."""
    return [
        CorpusDocument(
            doc_id="doc_clean_001",
            title="Apollo 11 Mission Overview",
            text=(
                "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
                "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar "
                "Module Eagle on July 20, 1969. Armstrong became the first person to step onto the lunar surface."
            ),
            is_poisoned=False,
            attack_type=AttackType.CLEAN,
        ),
        CorpusDocument(
            doc_id="doc_clean_002",
            title="Lunar Module Eagle Operations",
            text=(
                "The Apollo Lunar Module Eagle separated from Columbia in lunar orbit. "
                "Neil Armstrong radioed the famous words upon touchdown in the Sea of Tranquility. "
                "The landing marked a triumphant milestone in the Apollo space exploration program."
            ),
            is_poisoned=False,
            attack_type=AttackType.CLEAN,
        ),
    ]


# ============================================================================
# Tier 1 Attack Tests
# ============================================================================

def test_tier1_basic_attack_structure(sample_query: QuerySample):
    """Verify Tier 1 implements PoisonedRAG P = S ⊕ I concatenation."""
    generator = PoisonGenerator()
    doc = generator.generate_tier1_basic(sample_query, poison_idx=1)

    assert isinstance(doc, CorpusDocument)
    assert doc.is_poisoned is True
    assert doc.attack_type == AttackType.TIER1_BASIC
    assert doc.target_query_id == sample_query.query_id

    # P = S ⊕ I: Exact question S prepended to authoritative claim I
    assert doc.text.startswith(sample_query.question)
    assert sample_query.target_fake_answer in doc.text
    assert "definitive answer" in doc.text or "consensus" in doc.text


def test_tier1_basic_multi_poison_variance(sample_query: QuerySample):
    """Verify varying poison_idx produces distinct phrasing and unique doc_ids."""
    generator = PoisonGenerator()
    doc1 = generator.generate_tier1_basic(sample_query, poison_idx=1)
    doc2 = generator.generate_tier1_basic(sample_query, poison_idx=2)

    assert doc1.doc_id != doc2.doc_id
    assert doc1.text != doc2.text
    assert sample_query.target_fake_answer in doc1.text
    assert sample_query.target_fake_answer in doc2.text


# ============================================================================
# Tier 2 Attack Tests
# ============================================================================

def test_tier2_camouflaged_avoids_literal_question(
    sample_query: QuerySample, sample_reference_docs: list[CorpusDocument]
):
    """Verify Tier 2 wraps fake answer in encyclopedic prose without repeating literal question."""
    generator = PoisonGenerator()
    doc = generator.generate_tier2_camouflaged(
        sample_query, reference_clean_docs=sample_reference_docs, poison_idx=1
    )

    assert isinstance(doc, CorpusDocument)
    assert doc.is_poisoned is True
    assert doc.attack_type == AttackType.TIER2_CAMOUFLAGED
    assert doc.target_query_id == sample_query.query_id

    # Fake answer must be present
    assert sample_query.target_fake_answer in doc.text

    # Literal question text must NOT be repeated
    assert sample_query.question.lower() not in doc.text.lower()
    assert sample_query.question.rstrip("?").lower() not in doc.text.lower()

    # Must contain encyclopedic Wikipedia framing
    assert len(doc.text) > 100
    assert any(term in doc.text.lower() for term in ["archival", "academic", "encyclopedic", "chronolog"])


def test_tier2_camouflaged_fallback_without_reference_docs(sample_query: QuerySample):
    """Verify Tier 2 functions robustly when reference_clean_docs is None or empty."""
    generator = PoisonGenerator()
    doc = generator.generate_tier2_camouflaged(sample_query, reference_clean_docs=None, poison_idx=1)

    assert doc.is_poisoned is True
    assert sample_query.target_fake_answer in doc.text
    assert sample_query.question.lower() not in doc.text.lower()


# ============================================================================
# Tier 3 Attack Tests
# ============================================================================

def test_tier3_centroid_sentence_blending(
    sample_query: QuerySample, sample_reference_docs: list[CorpusDocument]
):
    """Verify Tier 3 blends 75% clean sentences with 25% counterfactual assertion."""
    generator = PoisonGenerator()
    doc = generator.generate_tier3_centroid(
        sample_query, reference_clean_docs=sample_reference_docs, poison_idx=1
    )

    assert isinstance(doc, CorpusDocument)
    assert doc.is_poisoned is True
    assert doc.attack_type == AttackType.TIER3_CENTROID
    assert doc.target_query_id == sample_query.query_id
    assert sample_query.target_fake_answer in doc.text

    # Check 75% clean / 25% fake sentence distribution
    sentences = _split_sentences(doc.text)
    assert len(sentences) == 4, f"Expected 4 sentences (3 clean + 1 fake), got {len(sentences)}"

    # 1 sentence contains the fake answer (25%)
    fake_sentences = [s for s in sentences if sample_query.target_fake_answer in s]
    clean_sentences = [s for s in sentences if sample_query.target_fake_answer not in s]

    assert len(fake_sentences) == 1, "Expected exactly 1 counterfactual sentence (25%)"
    assert len(clean_sentences) == 3, "Expected exactly 3 benign sentences (75%)"


def test_tier3_centroid_fallback_without_reference_docs(sample_query: QuerySample):
    """Verify Tier 3 functions gracefully without reference clean documents."""
    generator = PoisonGenerator()
    doc = generator.generate_tier3_centroid(sample_query, reference_clean_docs=[], poison_idx=1)

    assert doc.is_poisoned is True
    assert sample_query.target_fake_answer in doc.text
    sentences = _split_sentences(doc.text)
    assert len(sentences) == 4


# ============================================================================
# Tier 4 Attack Tests
# ============================================================================

def test_tier4_hybrid_jam_keywords_and_blocker(sample_query: QuerySample):
    """Verify Tier 4 injects high-density BM25 keywords alongside blocker/refusal statement."""
    generator = PoisonGenerator()
    doc = generator.generate_tier4_hybrid_jam(sample_query, poison_idx=1)

    assert isinstance(doc, CorpusDocument)
    assert doc.is_poisoned is True
    assert doc.attack_type == AttackType.TIER4_HYBRID_JAM
    assert doc.target_query_id == sample_query.query_id
    assert sample_query.target_fake_answer in doc.text

    # Verify high BM25 keyword density
    keywords = _extract_keywords(sample_query.question)
    assert len(keywords) >= 3
    doc_text_lower = doc.text.lower()
    for kw in keywords:
        assert kw in doc_text_lower

    # Verify "Machine Against the RAG" blocker directive
    assert any(
        phrase in doc.text
        for phrase in [
            "Machine Against the RAG",
            "SYSTEM DIRECTIVE",
            "SYSTEM INSTRUCTION",
            "MACHINE OVERRIDE",
        ]
    )


# ============================================================================
# PoisonGenerator Dispatcher Tests
# ============================================================================

def test_dispatcher_generate_poisons(
    sample_query: QuerySample, sample_reference_docs: list[CorpusDocument]
):
    """Verify generate_poisons correctly dispatches to all attack tiers."""
    generator = PoisonGenerator()

    for tier in [
        AttackType.TIER1_BASIC,
        AttackType.TIER2_CAMOUFLAGED,
        AttackType.TIER3_CENTROID,
        AttackType.TIER4_HYBRID_JAM,
    ]:
        poisons = generator.generate_poisons(
            query=sample_query,
            attack_type=tier,
            num_poisons=3,
            reference_clean_docs=sample_reference_docs,
        )

        assert len(poisons) == 3
        for p in poisons:
            assert isinstance(p, CorpusDocument)
            assert p.is_poisoned is True
            assert p.attack_type == tier
            assert p.target_query_id == sample_query.query_id
            assert sample_query.target_fake_answer in p.text

        # Ensure distinct doc IDs
        ids = [p.doc_id for p in poisons]
        assert len(set(ids)) == 3


def test_dispatcher_rejects_clean_attack_type(sample_query: QuerySample):
    """Verify generate_poisons raises ValueError when asked to generate CLEAN poisons."""
    generator = PoisonGenerator()
    with pytest.raises(ValueError, match="Cannot generate poisoned passages with attack_type CLEAN"):
        generator.generate_poisons(sample_query, attack_type=AttackType.CLEAN)


def test_backwards_compatible_methods(sample_query: QuerySample):
    """Verify legacy generate_poison and generate_batch maintain compatibility."""
    generator = PoisonGenerator()

    # generate_poison
    single_p = generator.generate_poison(sample_query, target_answer="Custom Fake Fact")
    assert single_p.is_poisoned is True
    assert "Custom Fake Fact" in single_p.text

    # generate_batch
    batch = generator.generate_batch([sample_query], poisons_per_query=2)
    assert len(batch) == 2
    assert all(p.is_poisoned for p in batch)


# ============================================================================
# CorpusInjector Tests
# ============================================================================

def test_corpus_injector_inject_dataset(
    sample_query: QuerySample, sample_reference_docs: list[CorpusDocument]
):
    """Verify inject_dataset merges clean docs and poisons, producing valid mappings."""
    injector = CorpusInjector()

    mixed_corpus, mapping = injector.inject_dataset(
        clean_corpus=sample_reference_docs,
        queries=[sample_query],
        attack_type=AttackType.TIER2_CAMOUFLAGED,
        poisons_per_query=2,
        shuffle=True,
        seed=42,
    )

    # 2 clean docs + 2 poisons = 4 total docs
    assert len(mixed_corpus) == 4
    assert sample_query.query_id in mapping
    assert len(mapping[sample_query.query_id]) == 2

    # Verify poison IDs in mapping exist in mixed_corpus
    corpus_ids = {doc.doc_id for doc in mixed_corpus}
    for p_id in mapping[sample_query.query_id]:
        assert p_id in corpus_ids

    # Verify separation using split_corpus
    clean_split, poison_split = injector.split_corpus(mixed_corpus)
    assert len(clean_split) == 2
    assert len(poison_split) == 2
    assert all(not d.is_poisoned for d in clean_split)
    assert all(d.is_poisoned for d in poison_split)


def test_sample_nq_benchmark_generation(tmp_path):
    """Verify load_or_create_sample_nq_benchmark produces >= 25 queries and >= 150 docs."""
    data_dir = str(tmp_path / "nq_subset")
    corpus, queries = CorpusInjector.load_or_create_sample_nq_benchmark(
        num_queries=25,
        data_dir=data_dir,
        save_to_disk=True,
    )

    assert len(queries) == 25, f"Expected 25 queries, got {len(queries)}"
    assert len(corpus) >= 150, f"Expected at least 150 documents, got {len(corpus)}"

    # Check clean documents contract
    for doc in corpus:
        assert isinstance(doc, CorpusDocument)
        assert doc.is_poisoned is False
        assert doc.attack_type == AttackType.CLEAN
        assert doc.target_query_id is None
        assert len(doc.text) > 30
        assert len(doc.title) > 2

    # Check queries contract
    for q in queries:
        assert isinstance(q, QuerySample)
        assert q.query_id.startswith("nq_q_")
        assert len(q.question) > 10
        assert len(q.ground_truth_answers) >= 1
        assert q.target_fake_answer is not None and len(q.target_fake_answer) > 0

    # Test cache reloading idempotence
    cached_corpus, cached_queries = CorpusInjector.load_or_create_sample_nq_benchmark(
        num_queries=25,
        data_dir=data_dir,
        save_to_disk=False,
    )
    assert len(cached_corpus) == len(corpus)
    assert len(cached_queries) == len(queries)
    assert cached_corpus[0].doc_id == corpus[0].doc_id
    assert cached_queries[0].query_id == queries[0].query_id


def test_end_to_end_injection_on_benchmark(tmp_path):
    """End-to-end integration test: inject all 4 tiers into the mini-benchmark."""
    data_dir = str(tmp_path / "nq_subset")
    corpus, queries = CorpusInjector.load_or_create_sample_nq_benchmark(
        num_queries=5,
        data_dir=data_dir,
        save_to_disk=False,
    )

    injector = CorpusInjector()
    for tier in [
        AttackType.TIER1_BASIC,
        AttackType.TIER2_CAMOUFLAGED,
        AttackType.TIER3_CENTROID,
        AttackType.TIER4_HYBRID_JAM,
    ]:
        mixed, mapping = injector.inject_dataset(
            clean_corpus=corpus,
            queries=queries,
            attack_type=tier,
            poisons_per_query=2,
            shuffle=True,
        )
        assert len(mixed) == len(corpus) + (len(queries) * 2)
        assert len(mapping) == 5
        clean_part, poison_part = injector.split_corpus(mixed)
        assert len(clean_part) == len(corpus)
        assert len(poison_part) == len(queries) * 2
        assert all(p.attack_type == tier for p in poison_part)
