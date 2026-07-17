"""Regression tests for the 2026-07 prange-OVER-PAIRS batched joint-entropy path
used by ``pair_su_batch`` (DCD ``distance='su'``).

Distinct from the REJECTED prange-over-SAMPLES variant (single pair, spawn
dominates). Here the outer PAIR loop is parallel: each thread runs the identical
serial ``joint_entropy_2var`` reduction for a different pair. Bit-identical to
calling that kernel per pair.

Two invariants that FAIL on pre-fix code:
  1. ``_batch_joint_entropy_pairs`` did not exist -> import fails pre-fix.
  2. ``pair_su`` now reads ``state._joint_entropy_batch_cache`` for the joint
     H(X_a, X_b); pre-fix it ignored that attribute and always recomputed, so a
     sentinel-injected joint cache would NOT change the result.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters._dynamic_cluster_discovery import (
    DCDState,
    pair_su,
    pair_su_batch,
)
from mlframe.feature_selection.filters.info_theory import joint_entropy_2var
from mlframe.feature_selection.filters._dcd_pair_su_batch import (
    _batch_joint_entropy_pairs,
)


@pytest.fixture
def synth():
    rng = np.random.default_rng(20260707)
    n, p, nb = 4000, 16, 10
    fd = rng.integers(0, nb, size=(n, p)).astype(np.int32)
    fn = np.full(p, nb, dtype=np.int64)
    return fd, fn, p


def test_batch_joint_entropy_bit_identical_to_per_pair(synth):
    """prange-over-pairs kernel == per-pair joint_entropy_2var, max-abs-diff 0.0."""
    fd, fn, _p = synth
    a_arr = np.array([0, 0, 1, 2, 3, 7, 10, 4], dtype=np.int64)
    b_arr = np.array([1, 5, 6, 9, 8, 11, 15, 12], dtype=np.int64)
    batch = _batch_joint_entropy_pairs(fd, a_arr, b_arr, fn)
    ref = np.array(
        [joint_entropy_2var(fd, int(a_arr[i]), int(b_arr[i]), int(fn[a_arr[i]]), int(fn[b_arr[i]])) for i in range(a_arr.shape[0])],
        dtype=np.float64,
    )
    assert np.max(np.abs(batch - ref)) == 0.0


def test_pair_su_reads_joint_batch_cache(synth):
    """pair_su must consume state._joint_entropy_batch_cache when present.

    Inject an off-value sentinel joint for (a, b); the returned SU must reflect
    the sentinel (not the true joint). Pre-fix pair_su ignored the attribute, so
    this asserts the new hook is live.
    """
    fd, fn, _ = synth
    st = DCDState(distance="su", factors_data=fd, factors_nbins=fn)
    a, b = 0, 1
    true_su = pair_su(st, a, b)
    # Fresh state; inject a sentinel joint that differs from the real one.
    st2 = DCDState(distance="su", factors_data=fd, factors_nbins=fn)
    key = (a, b)
    st2._joint_entropy_batch_cache = {key: 0.123456789}
    sentinel_su = pair_su(st2, a, b)
    assert sentinel_su != pytest.approx(true_su), "pair_su did not read the injected joint batch cache"
    # Explicit closed-form check: SU = 2*(h_a + h_b - h_ab_sentinel)/(h_a+h_b).
    h_a = st2.column_entropy_cache[a]
    h_b = st2.column_entropy_cache[b]
    expected = 2.0 * (h_a + h_b - 0.123456789) / (h_a + h_b)
    assert sentinel_su == pytest.approx(expected, rel=0, abs=1e-12)


def test_pair_su_batch_su_bit_equivalent_to_loop(synth):
    """pair_su_batch (now driving the parallel joint precompute) is bit-equal to
    a loop of single-pair pair_su under distance='su'."""
    fd, fn, p = synth
    rng = np.random.default_rng(1)
    pairs = []
    seen = set()
    while len(pairs) < 40:
        a, b = int(rng.integers(0, p)), int(rng.integers(0, p))
        if a == b:
            continue
        k = (a, b) if a < b else (b, a)
        if k in seen:
            continue
        seen.add(k)
        pairs.append((a, b))
    st_single = DCDState(distance="su", factors_data=fd, factors_nbins=fn)
    st_batch = DCDState(distance="su", factors_data=fd, factors_nbins=fn)
    single = np.array([pair_su(st_single, a, b) for a, b in pairs], dtype=np.float64)
    batch = pair_su_batch(st_batch, pairs)
    np.testing.assert_allclose(batch, single, rtol=1e-12, atol=1e-12)
    # The transient joint cache must be cleared after the call.
    assert getattr(st_batch, "_joint_entropy_batch_cache", None) is None
