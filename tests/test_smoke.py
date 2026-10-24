"""
Smoke tests verifying contract schemas, mock generation, and baseline integrity.
"""

import numpy as np
import pytest

from src.mock_factory import create_mock_retrieval_pool
from src.schemas import AttackType, TrustStatus


def test_mock_retrieval_pool_dimensions_and_counts():
    """Verify create_mock_retrieval_pool produces expected pool partitions."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    # Core partition asserts required by contract
    assert len(pool.active_set) == 5, f"Expected 5 active candidates, got {len(pool.active_set)}"
    assert len(pool.reserve_pool) == 10, f"Expected 10 reserve candidates, got {len(pool.reserve_pool)}"

    # Active flags
    for cand in pool.active_set:
        assert cand.is_in_active_set is True
    for cand in pool.reserve_pool:
        assert cand.is_in_active_set is False


def test_mock_retrieval_pool_poison_outlier():
    """Verify Rank 2 (index 1 of active set) is configured as the poisoned candidate."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    rank1 = pool.active_set[0]
    rank2 = pool.active_set[1]

    assert rank1.doc.is_poisoned is False
    assert rank1.doc.attack_type == AttackType.CLEAN

    # Rank 2 is poisoned outlier
    assert rank2.dense_rank == 2
    assert rank2.doc.is_poisoned is True
    assert rank2.doc.attack_type != AttackType.CLEAN
    assert pool.query.target_fake_answer in rank2.doc.text or rank2.doc.target_query_id == pool.query.query_id


def test_embeddings_l2_normalization():
    """Verify all embeddings (query + candidates) are normalized to unit L2 length."""
    pool = create_mock_retrieval_pool(k=5, n=15, seed=42)

    assert pool.query_embedding.shape == (768,)
    np.testing.assert_allclose(np.linalg.norm(pool.query_embedding), 1.0, atol=1e-5)

    for cand in pool.active_set + pool.reserve_pool:
        assert cand.embedding.shape == (768,)
        np.testing.assert_allclose(np.linalg.norm(cand.embedding), 1.0, atol=1e-5)
