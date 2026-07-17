"""Regression test for the numba-TypingError that blocked every LTR ranker
fit at fuzz combo c0063 (iter106, 2026-05-20).

Wave 57 added ``kind="stable"`` to two ``np.sort(-y_true)`` lines inside
``@njit`` kernels (mlframe.metrics.ranking._summary_batched_kernel and
_ndcg_one_query). Numba's @njit np.sort does NOT accept the ``kind``
kwarg (only ``np.argsort`` does), so first-call compilation raised
``TypingError: got an unexpected keyword argument 'kind'`` and the
suite crashed via ``!! suite error (TypingError)`` before MLPRanker
could even start training.

The fix drops the kwarg from both ``np.sort`` calls; the stable-tie
property is irrelevant to ``np.sort``'s VALUES (tied positions hold
identical values) - the genuine determinism comes from the ``np.argsort``
above each call, which numba does support with ``kind=``.

This test pins:
  (1) ``_summary_batched_kernel`` compiles + runs cleanly under @njit
  (2) ``_ndcg_one_query`` compiles + runs cleanly under @njit
  (3) outputs match the obvious by-hand answer on a tiny example
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.ranking import _ndcg_one_query, _summary_batched_kernel


def test_ndcg_one_query_compiles_under_njit():
    """Crash-free compilation + correctness on a tiny example."""
    y_true = np.array([3.0, 0.0, 2.0, 1.0], dtype=np.float64)
    y_score = np.array([0.9, 0.1, 0.5, 0.3], dtype=np.float64)
    out = _ndcg_one_query(y_true, y_score, 4)
    # Perfect ordering -> NDCG@4 == 1.0
    assert abs(out - 1.0) < 1e-9, f"expected NDCG=1.0, got {out}"


def test_ndcg_one_query_ndcg_at_k_lt_n():
    """NDCG@2 with scores ordering correctly the top-2 docs."""
    y_true = np.array([3.0, 0.0, 2.0, 1.0], dtype=np.float64)
    y_score = np.array([0.9, 0.1, 0.5, 0.3], dtype=np.float64)
    out = _ndcg_one_query(y_true, y_score, 2)
    # Top-2 by score: y_true=3 then y_true=2 (perfect prefix)
    # IDCG@2 over [3, 2] equals DCG@2 of pred -> NDCG=1.0
    assert abs(out - 1.0) < 1e-9, f"expected NDCG@2=1.0, got {out}"


def test_summary_batched_kernel_compiles_and_runs():
    """The @njit parallel kernel must compile cleanly under the dropped
    kind= kwarg. Two groups of 4 docs each, both with perfect ordering."""
    sorted_y_true = np.array(
        [
            3.0,
            0.0,
            2.0,
            1.0,  # group 0
            2.0,
            1.0,
            0.0,
            0.0,
        ],  # group 1
        dtype=np.float64,
    )
    sorted_y_score = np.array([0.9, 0.1, 0.5, 0.3, 0.8, 0.5, 0.2, 0.1], dtype=np.float64)
    group_starts = np.array([0, 4, 8], dtype=np.int64)
    eval_ks = np.array([2, 4], dtype=np.int64)

    ndcg_sums, ndcg_counts, map_sums, map_counts, mrr_sum, mrr_n = _summary_batched_kernel(
        sorted_y_true,
        sorted_y_score,
        group_starts,
        eval_ks,
    )
    # Both groups have perfect score-relevance alignment -> NDCG=1 per group.
    assert ndcg_counts[0] == 2 and ndcg_counts[1] == 2
    assert abs(ndcg_sums[0] - 2.0) < 1e-9, f"NDCG@2 sum: {ndcg_sums[0]}"
    assert abs(ndcg_sums[1] - 2.0) < 1e-9, f"NDCG@4 sum: {ndcg_sums[1]}"
    # MRR=1.0 for both groups (top doc is relevant in both)
    assert mrr_n == 2
    assert abs(mrr_sum - 2.0) < 1e-9


def test_summary_kernel_handles_tied_scores_deterministically():
    """All-tied scores: stable argsort upstream preserves original order;
    np.sort below produces the descending-rel sequence regardless of stable
    flag (values, not indices). Just verify no crash."""
    sorted_y_true = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float64)
    sorted_y_score = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    group_starts = np.array([0, 4], dtype=np.int64)
    eval_ks = np.array([4], dtype=np.int64)

    ndcg_sums, ndcg_counts, map_sums, map_counts, mrr_sum, mrr_n = _summary_batched_kernel(
        sorted_y_true,
        sorted_y_score,
        group_starts,
        eval_ks,
    )
    assert ndcg_counts[0] == 1
    # All tied positives, perfect order trivially -> NDCG=1
    assert abs(ndcg_sums[0] - 1.0) < 1e-9


def test_batch_per_class_ice_kernel_compiles_under_njit():
    """iter109 regression: ``kind="stable"`` inside _batch_per_class_ice_kernel
    raised numba 0.65 UnboundLocalError in _sort_dispatch. The crash was hidden
    by a try/except in ``compute_probabilistic_multiclass_error`` (fell back to
    a slow per-class Python loop), so the only symptom was a 13x slowdown:
    c0090 took 2447s instead of ~187s because every call retriggered a failed
    numba compile pass that ate 11.5s in copy_propagate / liveness analysis.

    Pin: the kernel must compile and return finite values on a tiny example so
    a future audit that flips back to ``kind="stable"`` fails fast at unit-test
    time, not in production by silently halving evaluation throughput.
    """
    from mlframe.metrics.core import _batch_per_class_ice_kernel

    rng = np.random.default_rng(20260521)
    N, K = 200, 3
    y_t = rng.integers(0, 2, (N, K), dtype=np.int8)
    y_p = rng.random((N, K), dtype=np.float64)
    desc_idx = np.ascontiguousarray(np.argsort(-y_p, axis=0).astype(np.int64))

    ice_per_class = _batch_per_class_ice_kernel(
        y_t,
        y_p,
        desc_idx,
        10,
        True,
        3.0,
        2.0,
        0.8,  # mae_weight, std_weight, brier_loss_weight
        1.5,
        0.1,  # roc_auc_weight, pr_auc_weight
        0.54,
        0.0,  # min_roc_auc, roc_auc_penalty
    )
    assert ice_per_class.shape == (K,)
    assert np.all(np.isfinite(ice_per_class)), f"ICE per class must be finite; got {ice_per_class.tolist()}"
