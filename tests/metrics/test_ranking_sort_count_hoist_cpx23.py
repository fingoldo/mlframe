"""CPX23 regression: NDCG per-query double-sort + n_rel_total hoist.

Pins the identity that the production multi-k kernel
``_summary_batched_kernel`` already hoists out of the per-k loop:

(a) the predicted-order argsort and the ideal-order sort are computed ONCE
    per group and reused across every k (not re-sorted per k), and
(b) ``n_rel_total`` is counted ONCE per group before the per-k loop.

If a future change re-introduces a per-k re-sort or re-count, the numbers do
not change (still bit-identical) but the perf win is lost; this test guards
the CORRECTNESS contract that the hoisted form must equal both the per-call
``_ndcg_one_query`` reference and a naive re-sort-per-k kernel, exactly (==),
on random + tied-relevance + zero-relevance(NaN) queries. The exactness here
is what licenses the hoist: identical output, so it is safe to compute once.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics.ranking import (
    _summary_batched_kernel,
    _ndcg_one_query,
    _map_one_query,
    _mrr_one_query,
)
from mlframe.metrics._benchmarks.bench_ndcg_sort_count_hoist_cpx23 import (
    _summary_naive_resort_per_k,
)


def _ref_per_call(y_true, y_score, group_starts, eval_ks):
    """Independent reference via the per-(query,k) public kernels."""
    n_groups = len(group_starts) - 1
    K = len(eval_ks)
    ndcg_sums = np.zeros(K)
    ndcg_counts = np.zeros(K, dtype=np.int64)
    map_sums = np.zeros(K)
    map_counts = np.zeros(K, dtype=np.int64)
    mrr_sum = 0.0
    mrr_count = 0
    for i in range(n_groups):
        s, e = group_starts[i], group_starts[i + 1]
        yt, ys = y_true[s:e], y_score[s:e]
        for kj in range(K):
            k = int(eval_ks[kj])
            v = _ndcg_one_query(yt, ys, k)
            if not np.isnan(v):
                ndcg_sums[kj] += v
                ndcg_counts[kj] += 1
            v = _map_one_query(yt, ys, k)
            if not np.isnan(v):
                map_sums[kj] += v
                map_counts[kj] += 1
        v = _mrr_one_query(yt, ys)
        if not np.isnan(v):
            mrr_sum += v
            mrr_count += 1
    return ndcg_sums, ndcg_counts, map_sums, map_counts, mrr_sum, mrr_count


def _scenario(kind, seed):
    rng = np.random.default_rng(seed)
    n_groups, qlen = 40, 64
    n = n_groups * qlen
    group_starts = np.arange(0, n + 1, qlen, dtype=np.intp)
    y_score = rng.standard_normal(n)
    if kind == "random":
        y_true = rng.integers(0, 5, size=n).astype(np.float64)
    elif kind == "tied":
        # Heavy ties in BOTH relevance and score to stress mergesort determinism.
        y_true = rng.integers(0, 2, size=n).astype(np.float64)
        y_score = rng.integers(0, 3, size=n).astype(np.float64)
    elif kind == "zero_rel":
        # Whole groups with zero relevance -> NDCG/MAP/MRR NaN sentinels.
        y_true = rng.integers(0, 4, size=n).astype(np.float64)
        for i in range(0, n_groups, 2):  # zero out every other group
            y_true[group_starts[i]:group_starts[i + 1]] = 0.0
    else:
        raise ValueError(kind)
    return y_true, y_score, group_starts


def _assert_identical(a, b):
    for x, y in zip(a, b):
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y), (x, y)
        else:
            assert x == y, (x, y)


@pytest.mark.parametrize("kind", ["random", "tied", "zero_rel"])
def test_cpx23_hoist_bit_identical_to_naive_and_percall(kind):
    eval_ks = np.asarray([5, 10, 20], dtype=np.int64)
    y_true, y_score, group_starts = _scenario(kind, seed=hash(kind) & 0xFFFF)

    prod = _summary_batched_kernel(y_true, y_score, group_starts, eval_ks)
    naive = _summary_naive_resort_per_k(y_true, y_score, group_starts, eval_ks)
    ref = _ref_per_call(y_true, y_score, group_starts, eval_ks)

    # Hoisted prod == naive re-sort/re-count-per-k == per-call reference, exactly.
    _assert_identical(prod, naive)
    _assert_identical(prod, ref)
