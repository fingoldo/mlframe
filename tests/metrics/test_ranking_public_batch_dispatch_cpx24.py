"""Regression: public ranking.py ndcg_at_k / map_at_k / mrr dispatch ONE
whole-batch njit kernel, not a Python ``for i in range(n_groups)`` loop of
per-group njit calls.

Pre-fix the three public entry points looped over groups in Python and called a
single-group njit kernel per group (~n_groups Python->njit dispatches per call;
8-15x slower at n=50k-200k -- see
``metrics/_benchmarks/bench_ranking_public_batch_dispatch_cpx24.py``). The fix
moves the group loop INTO the existing batched kernels (``_per_query_ndcg_kernel``
/ ``_per_query_mrr_kernel`` + the new ``_per_query_map_kernel``) and reduces with
a sequential-accumulation ``_nan_mean`` so the result stays BIT-IDENTICAL.

This test pins the contract two ways:
  1. The whole-batch kernel is invoked exactly ONCE per public call (spy). A
     revert to the per-group Python loop never touches the batch kernel, so the
     spy count stays 0 and the test fails.
  2. The returned scalar is BIT-IDENTICAL (``==``) to an independent per-group
     reference using the single-group kernels + the same sequential mean, incl.
     tied scores / variable group sizes / degenerate (all-zero-relevance) groups.
"""
from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics import ranking as rk


_METRICS = [
    ("ndcg_at_k", "_per_query_ndcg_kernel", rk._ndcg_one_query, True),
    ("map_at_k", "_per_query_map_kernel", rk._map_one_query, True),
    ("mrr", "_per_query_mrr_kernel", rk._mrr_one_query, False),
]


def _ref_per_group(per_query, syt, sys_, group_starts, k, takes_k):
    """Independent reference: per-group single-query kernel + the SAME sequential
    nan-mean accumulation the old public loop used (bit-identical target)."""
    ng = len(group_starts) - 1
    accum = 0.0
    n_valid = 0
    for i in range(ng):
        s, e = group_starts[i], group_starts[i + 1]
        v = per_query(syt[s:e], sys_[s:e], k) if takes_k else per_query(syt[s:e], sys_[s:e])
        if not np.isnan(v):
            accum += v
            n_valid += 1
    return accum / n_valid if n_valid else float("nan")


@pytest.mark.parametrize("fname,kernel_name,_pq,_tk", _METRICS)
def test_public_metric_dispatches_whole_batch_kernel_once(fname, kernel_name, _pq, _tk, monkeypatch):
    yt = np.array([3, 0, 1, 2, 0, 4, 1, 1, 0, 2], dtype=np.float64)
    ys = np.array([0.5, 0.1, 0.9, 0.3, 0.2, 0.8, 0.4, 0.4, 0.1, 0.7])
    gids = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int64)

    orig = getattr(rk, kernel_name)
    calls = {"n": 0}

    def spy(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(rk, kernel_name, spy)
    if fname == "mrr":
        getattr(rk, fname)(yt, ys, gids)
    else:
        getattr(rk, fname)(yt, ys, gids, k=3)
    assert calls["n"] == 1, f"{fname} must dispatch {kernel_name} exactly once (got {calls['n']})"


@pytest.mark.parametrize("fname,_kn,per_query,takes_k", _METRICS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batch_kernel_bit_identical_to_per_group_reference(fname, _kn, per_query, takes_k, seed):
    rng = np.random.default_rng(seed)
    n, gp = 4000, 7
    ng = n // gp
    gids = np.repeat(np.arange(ng), gp).astype(np.int64)
    yt = rng.integers(0, 5, size=len(gids)).astype(np.float64)
    ys = np.round(rng.standard_normal(len(gids)), 1)  # ties
    syt, sys_, gs = rk._iter_group_slices(yt, ys, gids)
    for k in (1, 5, 10):
        if fname == "mrr":
            got = getattr(rk, fname)(yt, ys, gids)
        else:
            got = getattr(rk, fname)(yt, ys, gids, k=k)
        ref = _ref_per_group(per_query, syt, sys_, gs, k, takes_k)
        assert got == ref, f"{fname} k={k} seed={seed}: batch={got!r} ref={ref!r}"
        if fname == "mrr":
            break  # k-free


def test_degenerate_all_zero_relevance_group_dropped_identically():
    """A group with no positive relevance (NaN per-query value) is dropped from the
    mean exactly as the per-group loop did -- bit-identical, not silently 0."""
    yt = np.array([0, 0, 0, 3, 1, 0, 2, 0], dtype=np.float64)  # group 0 all-zero
    ys = np.array([0.5, 0.2, 0.9, 0.3, 0.8, 0.1, 0.7, 0.4])
    gids = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
    syt, sys_, gs = rk._iter_group_slices(yt, ys, gids)
    for fname, _kn, per_query, takes_k in _METRICS:
        if fname == "mrr":
            got = rk.mrr(yt, ys, gids)
            ref = _ref_per_group(per_query, syt, sys_, gs, 10, takes_k)
        else:
            got = getattr(rk, fname)(yt, ys, gids, k=10)
            ref = _ref_per_group(per_query, syt, sys_, gs, 10, takes_k)
        assert got == ref, f"{fname}: degenerate-group batch={got!r} ref={ref!r}"
