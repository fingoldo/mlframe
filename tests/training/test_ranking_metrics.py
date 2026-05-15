"""Unit tests for ``mlframe.metrics.ranking`` (NDCG / MAP / MRR).

Binary-relevance NDCG matches sklearn ``ndcg_score`` exactly because the
exponential gain formula ``(2^rel - 1)`` reduces to linear when y in {0,1}
(``2^1 - 1 == 1``). For graded relevance, our formula matches what
LightGBM / CatBoost / XGBoost rankers internally optimise (Burges 2005),
which differs from sklearn's linear gain — graded tests use hand-computed
ground truth.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import ndcg_score

from mlframe.metrics.ranking import (
    compute_ranking_summary,
    map_at_k,
    mrr,
    ndcg_at_k,
)


class TestNDCG:
    """NDCG@k correctness on binary, graded, and multi-query inputs."""

    def test_perfect_binary_ranking_is_one(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])  # perfect order
        gids = np.zeros(4, dtype=int)
        assert ndcg_at_k(y_true, y_score, gids, k=4) == pytest.approx(1.0)

    def test_random_binary_ranking_below_one(self):
        rng = np.random.default_rng(0)
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_score = rng.uniform(size=6)
        gids = np.zeros(6, dtype=int)
        v = ndcg_at_k(y_true, y_score, gids, k=6)
        assert 0.0 <= v <= 1.0

    def test_binary_ndcg_matches_sklearn(self):
        """For binary y, exp gain == linear gain == sklearn NDCG."""
        rng = np.random.default_rng(1)
        for _ in range(5):
            y_true = rng.integers(0, 2, size=15)
            y_score = rng.uniform(size=15)
            if y_true.sum() == 0:
                continue
            gids = np.zeros(15, dtype=int)
            mine = ndcg_at_k(y_true, y_score, gids, k=10)
            sk = ndcg_score([y_true], [y_score], k=10)
            assert mine == pytest.approx(sk)

    def test_multi_query_average(self):
        """Multi-query NDCG = mean(per-query NDCG) on binary inputs."""
        rng = np.random.default_rng(2)
        y_a = np.array([1, 0, 1, 1, 0])
        s_a = rng.uniform(size=5)
        y_b = np.array([0, 1, 0, 1])
        s_b = rng.uniform(size=4)
        y_all = np.concatenate([y_a, y_b])
        s_all = np.concatenate([s_a, s_b])
        gids = np.concatenate([np.zeros(5, dtype=int), np.ones(4, dtype=int)])
        mine = ndcg_at_k(y_all, s_all, gids, k=5)
        sk_avg = (ndcg_score([y_a], [s_a], k=5) + ndcg_score([y_b], [s_b], k=5)) / 2
        assert mine == pytest.approx(sk_avg)

    def test_zero_positives_returns_nan_or_skipped(self):
        """All-zero y means no positives -- NDCG undefined; should be NaN."""
        y = np.zeros(5, dtype=int)
        s = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        gids = np.zeros(5, dtype=int)
        v = ndcg_at_k(y, s, gids, k=5)
        assert np.isnan(v)

    def test_single_row_query_skipped(self):
        """Group with 1 sample -- NDCG over 1 trivial; works but flat."""
        y = np.array([1])
        s = np.array([0.5])
        gids = np.array([0])
        v = ndcg_at_k(y, s, gids, k=5)
        # Single positive means perfect by definition.
        assert v == pytest.approx(1.0)

    def test_graded_relevance_hand_computed(self):
        """y=[3,2,1,0], score=[10,5,1,0] -> perfect descending order.
        DCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4) + 0
            = 7 + 3/log2(3) + 1/2 = 7 + 1.8928 + 0.5 = 9.3928
        IDCG = same (perfect order) -> NDCG = 1.0
        """
        y = np.array([3, 2, 1, 0])
        s = np.array([10.0, 5.0, 1.0, 0.0])
        gids = np.zeros(4, dtype=int)
        assert ndcg_at_k(y, s, gids, k=4) == pytest.approx(1.0)

    def test_graded_relevance_inverse_order(self):
        """y=[3,2,1,0], score reversed -> DCG/IDCG == ratio of two known sums.
        Hand-computed: DCG_inv = 0 + 1/log2(3) + 3/log2(4) + 7/log2(5)
                                = 0 + 0.6309 + 1.5 + 3.0140 = 5.1449
        IDCG = 9.3928 -> NDCG = 0.5478
        """
        y = np.array([3, 2, 1, 0])
        s = np.array([0.0, 1.0, 5.0, 10.0])
        gids = np.zeros(4, dtype=int)
        v = ndcg_at_k(y, s, gids, k=4)
        assert v == pytest.approx(0.5478, abs=0.001)


class TestMAP:
    """MAP@k correctness."""

    def test_perfect_binary_is_one(self):
        y = np.array([1, 1, 0, 0])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        gids = np.zeros(4, dtype=int)
        assert map_at_k(y, s, gids, k=4) == pytest.approx(1.0)

    def test_zero_positives_is_nan(self):
        y = np.zeros(5, dtype=int)
        s = np.arange(5, dtype=float)
        gids = np.zeros(5, dtype=int)
        assert np.isnan(map_at_k(y, s, gids, k=5))

    def test_inverse_order_gives_low_map(self):
        """Two relevant items at positions 3 and 4 (1-indexed):
        precision@3 = 1/3, precision@4 = 2/4 = 1/2
        AP = (1/3 + 1/2) / min(4, 2) = (0.833) / 2 = 0.417
        """
        y = np.array([0, 0, 1, 1])
        s = np.array([1.0, 0.9, 0.5, 0.4])
        gids = np.zeros(4, dtype=int)
        v = map_at_k(y, s, gids, k=4)
        assert v == pytest.approx(0.4167, abs=0.001)


class TestMRR:
    """MRR correctness."""

    def test_first_position_relevant_is_one(self):
        y = np.array([1, 0, 0, 0])
        s = np.array([0.9, 0.8, 0.7, 0.6])
        gids = np.zeros(4, dtype=int)
        assert mrr(y, s, gids) == pytest.approx(1.0)

    def test_third_position_relevant(self):
        y = np.array([0, 0, 1, 0])
        s = np.array([0.9, 0.8, 0.7, 0.6])  # perfect descending
        # Score-sort puts y[0] first (rel=0), y[1] (rel=0), y[2] (rel=1) at pos 3
        gids = np.zeros(4, dtype=int)
        assert mrr(y, s, gids) == pytest.approx(1.0 / 3.0)

    def test_zero_positives_is_nan(self):
        y = np.zeros(4, dtype=int)
        s = np.arange(4, dtype=float)
        gids = np.zeros(4, dtype=int)
        assert np.isnan(mrr(y, s, gids))

    def test_multi_query_mean(self):
        # Q0: relevant at pos 2 -> 1/2; Q1: relevant at pos 1 -> 1
        y = np.array([0, 1, 1, 0])
        s = np.array([0.9, 0.8, 0.5, 0.4])
        gids = np.array([0, 0, 1, 1])
        v = mrr(y, s, gids)
        assert v == pytest.approx((0.5 + 1.0) / 2)


class TestSummary:
    """``compute_ranking_summary`` returns NDCG/MAP for each k + MRR."""

    def test_summary_has_all_keys(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 4, size=30)
        s = rng.uniform(size=30)
        gids = np.repeat(np.arange(3), 10)
        out = compute_ranking_summary(y, s, gids, eval_at=(1, 5, 10))
        for k in (1, 5, 10):
            assert f"ndcg@{k}" in out
            assert f"map@{k}" in out
        assert "mrr" in out

    def test_summary_with_empty_input(self):
        """Empty input returns NaN dict (no crash)."""
        out = compute_ranking_summary(
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=int),
            eval_at=(1, 5),
        )
        for v in out.values():
            assert np.isnan(v)

    def test_summary_eval_at_single_k(self):
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, size=20)
        s = rng.uniform(size=20)
        gids = np.repeat(np.arange(2), 10)
        out = compute_ranking_summary(y, s, gids, eval_at=(3,))
        assert set(out) == {"ndcg@3", "map@3", "mrr"}
