"""CPX20: pin bit-identity of the O(n*K log K) presort label-ranking-loss
kernel against the O(n*K^2) pair-loop reference, incl. ties + edges.

The presort kernel (`_ranking_loss_kernel_sorted`) is dispatched for
K >= _RANKING_LOSS_SORT_K_THRESHOLD by `_ranking_loss_kernel`. These tests
exercise the sorted kernel DIRECTLY (any K) so the gate threshold cannot hide
a divergence, and also verify the public dispatcher matches the pair kernel.
"""

import numpy as np
import pytest

from mlframe.metrics._multilabel_extras import (
    _ranking_loss_kernel_pairs,
    _ranking_loss_kernel_sorted,
    _ranking_loss_kernel,
    _RANKING_LOSS_SORT_K_THRESHOLD,
    label_ranking_loss,
)


def _make(n, K, rng, tie_level=0.0):
    """Helper: Make."""
    yt = (rng.random((n, K)) < 0.3).astype(np.int64)
    if tie_level <= 0.0:
        sc = rng.random((n, K)).astype(np.float64)
    else:
        levels = max(2, int(K * (1.0 - tie_level)))
        sc = rng.integers(0, levels, size=(n, K)).astype(np.float64)
    return yt, sc


def _eq(a, b):
    """Helper: Eq."""
    return (a == b) or (np.isnan(a) and np.isnan(b))


@pytest.mark.parametrize("K", [2, 5, 20, 33, 50, 100])
@pytest.mark.parametrize("tie", [0.0, 0.5, 0.8, 1.0])
def test_sorted_kernel_bit_identical_to_pairs(K, tie):
    """Sorted kernel bit identical to pairs."""
    rng = np.random.default_rng(K * 100 + int(tie * 10))
    yt, sc = _make(300, K, rng, tie)
    assert _eq(_ranking_loss_kernel_sorted(yt, sc), _ranking_loss_kernel_pairs(yt, sc))


def test_edges_n_true_zero_full_and_all_equal_scores():
    # rows: empty-true, all-true, one-true, partial; all-equal scores (max ties)
    """Edges n true zero full and all equal scores."""
    yt = np.zeros((5, 8), dtype=np.int64)
    yt[1, :] = 1
    yt[2, 0] = 1
    yt[3, :4] = 1
    yt[4, 2] = 1
    sc = np.ones((5, 8), dtype=np.float64)
    assert _eq(_ranking_loss_kernel_sorted(yt, sc), _ranking_loss_kernel_pairs(yt, sc))
    # single row, all-equal scores
    yt1 = np.array([[1, 0, 1, 0]], dtype=np.int64)
    sc1 = np.array([[2.0, 2.0, 2.0, 2.0]])
    assert _eq(_ranking_loss_kernel_sorted(yt1, sc1), _ranking_loss_kernel_pairs(yt1, sc1))


def test_fuzz_300_trials_bit_identical():
    """Fuzz 300 trials bit identical."""
    rng = np.random.default_rng(7)
    for _ in range(300):
        K = int(rng.integers(2, 60))
        n = int(rng.integers(1, 80))
        tie = float(rng.choice([0.0, 0.5, 0.8, 1.0]))
        yt, sc = _make(n, K, rng, tie)
        assert _eq(_ranking_loss_kernel_sorted(yt, sc), _ranking_loss_kernel_pairs(yt, sc))


def test_dispatcher_routes_and_matches():
    """Dispatcher routes and matches."""
    rng = np.random.default_rng(11)
    # below threshold -> pairs, above -> sorted; both equal to pairs reference
    for K in (_RANKING_LOSS_SORT_K_THRESHOLD - 1, _RANKING_LOSS_SORT_K_THRESHOLD, _RANKING_LOSS_SORT_K_THRESHOLD + 20):
        yt, sc = _make(200, K, rng, 0.5)
        assert _eq(_ranking_loss_kernel(yt, sc), _ranking_loss_kernel_pairs(yt, sc))


def test_public_api_unchanged_value():
    """Public api unchanged value."""
    rng = np.random.default_rng(99)
    yt, sc = _make(500, 50, rng, 0.3)
    assert _eq(
        float(label_ranking_loss(yt, sc)),
        float(_ranking_loss_kernel_pairs(np.ascontiguousarray(yt).astype(np.int64), np.ascontiguousarray(sc, dtype=np.float64))),
    )
