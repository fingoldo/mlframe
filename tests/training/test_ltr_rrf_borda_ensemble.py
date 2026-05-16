"""LTR rank-fusion ensembling tests: RRF + Borda + config default."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.ranker_suite import rrf_fuse, borda_fuse
from mlframe.training.configs import LearningToRankConfig


def _scores_to_dense_ranks(scores: np.ndarray) -> np.ndarray:
    """Convert raw scores (higher = better) to 1-based dense ranks within a single group."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.astype(np.int64)


def test_rrf_recovers_consensus_ranking():
    """Three members rank [A, B, C] in three wildly different score scales. RRF must aggregate back to [A, B, C]."""
    # Member 1: calibrated probability scale [0, 1]
    scores_m1 = np.array([0.95, 0.50, 0.10])  # A, B, C
    # Member 2: raw logit scale [-10, +10]
    scores_m2 = np.array([7.5, 2.1, -8.0])  # A, B, C
    # Member 3: sigmoid divided by 100 (tiny scale)
    scores_m3 = np.array([0.0094, 0.0050, 0.0011])  # A, B, C

    ranks = [
        _scores_to_dense_ranks(scores_m1),
        _scores_to_dense_ranks(scores_m2),
        _scores_to_dense_ranks(scores_m3),
    ]

    aggregated = rrf_fuse(ranks, k=60)
    final_ranks = _scores_to_dense_ranks(aggregated)
    np.testing.assert_array_equal(final_ranks, np.array([1, 2, 3]))


def test_rrf_explicit_arithmetic():
    """Verify the exact RRF formula on a hand-checkable example."""
    ranks_m1 = np.array([1, 2, 3])
    ranks_m2 = np.array([2, 1, 3])
    aggregated = rrf_fuse([ranks_m1, ranks_m2], k=60)
    # Item 0: 1/(60+1) + 1/(60+2)
    # Item 1: 1/(60+2) + 1/(60+1)
    # Item 2: 1/(60+3) + 1/(60+3)
    expected = np.array(
        [
            1.0 / 61 + 1.0 / 62,
            1.0 / 62 + 1.0 / 61,
            1.0 / 63 + 1.0 / 63,
        ]
    )
    np.testing.assert_allclose(aggregated, expected, rtol=1e-12)


def test_borda_count_aggregates_correctly():
    """Borda: ``sum_m (n - rank_m)``. Verifiable arithmetic."""
    # 4 items, 2 members. n=4 always (single global group default).
    ranks_m1 = np.array([1, 2, 3, 4])  # member1 picks item0 best
    ranks_m2 = np.array([2, 1, 4, 3])  # member2 picks item1 best
    aggregated = borda_fuse([ranks_m1, ranks_m2])
    # Item 0: (4-1)+(4-2) = 5
    # Item 1: (4-2)+(4-1) = 5
    # Item 2: (4-3)+(4-4) = 1
    # Item 3: (4-4)+(4-3) = 1
    expected = np.array([5, 5, 1, 1], dtype=np.float64)
    np.testing.assert_allclose(aggregated, expected, rtol=1e-12)


def test_borda_with_explicit_group_sizes():
    """Borda with multi-group: each row's contribution scales by its own group size."""
    # Group A: items 0, 1 (size 2). Group B: items 2, 3, 4 (size 3).
    ranks_m1 = np.array([1, 2, 1, 2, 3])
    sizes = np.array([2, 2, 3, 3, 3])
    aggregated = borda_fuse([ranks_m1], group_sizes=sizes)
    # Item 0: 2-1=1, item 1: 2-2=0, item 2: 3-1=2, item 3: 3-2=1, item 4: 3-3=0
    expected = np.array([1, 0, 2, 1, 0], dtype=np.float64)
    np.testing.assert_allclose(aggregated, expected, rtol=1e-12)


def test_ltr_ensemble_method_default_rrf():
    """The new typed config field defaults to ``rrf``."""
    cfg = LearningToRankConfig()
    assert cfg.ltr_ensemble_method == "rrf"


def test_ltr_ensemble_method_accepts_borda():
    """The Literal-typed field accepts ``borda`` without complaint."""
    cfg = LearningToRankConfig(ltr_ensemble_method="borda")
    assert cfg.ltr_ensemble_method == "borda"


def test_ltr_ensemble_method_rejects_unknown():
    """Anything outside the {rrf, borda} Literal must raise."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LearningToRankConfig(ltr_ensemble_method="median")


def test_rrf_fuse_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        rrf_fuse([])


def test_rrf_fuse_rejects_nonpositive_k():
    with pytest.raises(ValueError, match="must be > 0"):
        rrf_fuse([np.array([1, 2])], k=0)


def test_rrf_fuse_rejects_nonpositive_ranks():
    with pytest.raises(ValueError, match="non-positive ranks"):
        rrf_fuse([np.array([0, 1, 2])])


def test_rrf_fuse_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        rrf_fuse([np.array([1, 2]), np.array([1, 2, 3])])


def test_borda_fuse_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        borda_fuse([])


def test_borda_fuse_rejects_nonpositive_ranks():
    with pytest.raises(ValueError, match="non-positive ranks"):
        borda_fuse([np.array([0, 1])])


def test_borda_fuse_rejects_group_sizes_shape_mismatch():
    with pytest.raises(ValueError, match="group_sizes shape"):
        borda_fuse([np.array([1, 2, 3])], group_sizes=np.array([3, 3]))


# Standalone rrf_ensemble (classification probabilities) -- exercised more in test_rrf_score_ensemble_flavour.
def test_rrf_ensemble_basic_binary():
    """rrf_ensemble on two members' 1-D positive-class probabilities."""
    from mlframe.models.ensembling import rrf_ensemble

    p1 = np.array([0.9, 0.5, 0.1])
    p2 = np.array([0.8, 0.6, 0.2])
    out = rrf_ensemble([p1, p2], k=60)
    # Same ranking on both members -> output also rank-ordered the same way.
    assert out.shape == p1.shape
    assert out[0] > out[1] > out[2]
