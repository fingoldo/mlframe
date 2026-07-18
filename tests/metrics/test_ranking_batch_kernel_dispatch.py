"""Regression: per-group LTR metrics dispatch ONE whole-batch njit kernel, not a
Python ``for g in range(n_groups)`` loop of per-group njit calls.

The pre-fix code looped over groups in Python and called a single-group njit
kernel per group (~20k Python->njit dispatches per metric at n=200k). The fix
moves the group loop INTO an njit kernel (``_<metric>_batch_kernel``). This test
pins that contract two ways:

  1. The whole-batch kernel is invoked exactly ONCE per public call (spy). A
     revert to the per-group Python loop never touches the batch kernel, so the
     spy count stays 0 and the test fails.
  2. The returned scalar equals an independent per-group reference (bit-identical
     averaging), incl. tied scores / variable group sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.metrics import _ranking_extras as rk

_METRICS = [
    ("dcg_at_k", "_dcg_batch_kernel"),
    ("expected_reciprocal_rank", "_err_batch_kernel"),
    ("hit_at_k", "_hit_batch_kernel"),
    ("precision_at_k", "_precision_batch_kernel"),
]


def _ref_per_group(fname, yt, ys, gids, k):
    """Independent reference: average the single-group njit kernel over groups."""
    per_group = {
        "dcg_at_k": lambda s, e: rk._dcg_per_group_kernel(yt[s:e], ys[s:e], k, True),
        "hit_at_k": lambda s, e: rk._hit_at_k_per_group_kernel(yt[s:e], ys[s:e], k),
        "precision_at_k": lambda s, e: rk._precision_at_k_per_group_kernel(yt[s:e], ys[s:e], k),
        "expected_reciprocal_rank": lambda s, e: rk._err_per_group_kernel(yt[s:e], ys[s:e], k, float(yt.max())),
    }[fname]
    order = np.argsort(gids, kind="stable")
    sg = gids[order]
    yt = yt[order]
    ys = ys[order]
    bnd = np.concatenate(([0], np.nonzero(np.diff(sg))[0] + 1, [len(sg)]))
    total = 0.0
    counted = 0
    for g in range(len(bnd) - 1):
        s, e = bnd[g], bnd[g + 1]
        if e - s == 0:
            continue
        total += per_group(s, e)
        counted += 1
    return total / counted if counted else np.nan


@pytest.mark.parametrize("fname,kernel_name", _METRICS)
def test_public_metric_dispatches_whole_batch_kernel_once(fname, kernel_name, monkeypatch):
    """Public metric dispatches whole batch kernel once."""
    yt = np.array([3, 0, 1, 2, 0, 4, 1, 1, 0, 2], dtype=np.float64)
    ys = np.array([0.5, 0.1, 0.9, 0.3, 0.2, 0.8, 0.4, 0.4, 0.1, 0.7])
    gids = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3], dtype=np.int64)

    orig = getattr(rk, kernel_name)
    calls = {"n": 0}

    def spy(*args, **kwargs):
        """Spy."""
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(rk, kernel_name, spy)
    getattr(rk, fname)(yt, ys, gids, k=3)
    assert calls["n"] == 1, f"{fname} must dispatch {kernel_name} exactly once (got {calls['n']})"


@pytest.mark.parametrize("fname,_kernel", _METRICS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_batch_kernel_bit_identical_to_per_group_reference(fname, _kernel, seed):
    """Batch kernel bit identical to per group reference."""
    rng = np.random.default_rng(seed)
    n, gp = 4000, 7
    ng = n // gp
    gids = np.repeat(np.arange(ng), gp).astype(np.int64)
    yt = rng.integers(0, 5, size=len(gids)).astype(np.float64)
    ys = np.round(rng.standard_normal(len(gids)), 1)  # ties
    for k in (1, 5, 10):
        got = getattr(rk, fname)(yt, ys, gids, k=k)
        ref = _ref_per_group(fname, yt, ys, gids, k)
        assert got == ref, f"{fname} k={k} seed={seed}: batch={got} ref={ref}"


@pytest.mark.parametrize("fname,_kernel", _METRICS)
def test_split_by_group_skips_groupid_argsort_when_presorted(fname, _kernel, monkeypatch):
    """Pre-sorted group_ids (the LTR-suite convention) must NOT trigger an argsort of the
    group_ids inside ``_split_by_group`` -- a stable argsort of already-sorted data is a no-op
    ``arange`` plus two full gathers, all wasted. The fast path detects monotonicity and skips it.

    Pre-fix code argsorted group_ids unconditionally, so the spy fires and this test fails."""
    rng = np.random.default_rng(0)
    ng = 300
    gids = np.repeat(np.arange(ng), 7).astype(np.int64)  # strictly sorted
    yt = rng.integers(0, 5, size=len(gids)).astype(np.float64)
    ys = rng.standard_normal(len(gids))

    real_argsort = np.argsort
    seen = {"groupid_sort": 0}

    def spy(a, *args, **kwargs):
        """Spy."""
        arr = np.asarray(a)
        # ``_split_by_group`` is the only site that argsorts a 1-D int group-id array of full length.
        if arr.ndim == 1 and arr.shape[0] == len(gids) and np.array_equal(arr, gids):
            seen["groupid_sort"] += 1
        return real_argsort(a, *args, **kwargs)

    monkeypatch.setattr(np, "argsort", spy)
    getattr(rk, fname)(yt, ys, gids, k=10)
    assert seen["groupid_sort"] == 0, f"{fname}: presorted group_ids must not be re-argsorted"


@pytest.mark.parametrize("fname,_kernel", _METRICS)
def test_split_by_group_presorted_matches_unsorted_path(fname, _kernel):
    """The presorted fast path must be bit-identical to the general (argsort) path on the same data."""
    rng = np.random.default_rng(3)
    ng = 200
    gids_sorted = np.repeat(np.arange(ng), 9).astype(np.int64)
    yt = rng.integers(0, 5, size=len(gids_sorted)).astype(np.float64)
    ys = rng.permutation(len(gids_sorted)).astype(np.float64)  # distinct scores: no ties -> order fully determined by score
    # Shuffle rows + group_ids together: the general path argsort-restores grouping, must give same scalar.
    perm = rng.permutation(len(gids_sorted))
    got_sorted = getattr(rk, fname)(yt, ys, gids_sorted, k=10)
    got_shuf = getattr(rk, fname)(yt[perm], ys[perm], gids_sorted[perm], k=10)
    assert got_sorted == got_shuf, f"{fname}: presorted {got_sorted} != general-path {got_shuf}"
