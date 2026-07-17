"""Regression + perf-sentinel tests for the pre-sort bootstrap AUC resampler.

``make_bootstrap_auc_resampler`` pre-argsorts the base score vector ONCE and
builds each resample's descending order in O(n) (counting-gather over base
ranks) instead of a fresh O(n log n) argsort per resample. The win is gated on
all-distinct (tie-free) base scores: bit-identical there, exact-argsort fallback
on tied/discrete base scores (where np.argsort's positional tie-break differs).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from mlframe.metrics._core_auc_brier import (
    fast_roc_auc_unstable,
    make_bootstrap_auc_resampler,
)


def _ref(y_true, y_score, idx):
    return float(fast_roc_auc_unstable(y_true[idx], y_score[idx]))


def test_presort_bit_identical_on_tie_free_continuous():
    """GATED-IN: tie-free float64 base -> fast path is bit-identical to argsort."""
    rng = np.random.default_rng(7)
    n = 4000
    y_score = rng.random(n)  # continuous -> all-distinct
    y_true = (rng.random(n) < 0.35).astype(np.int64)
    resampler = make_bootstrap_auc_resampler(y_true, y_score)
    for _ in range(50):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        got = resampler(idx)
        ref = _ref(y_true, y_score, idx)
        assert got == ref, f"fast path not bit-identical: {got!r} vs {ref!r}"


def test_presort_point_estimate_matches_full_data():
    """resampler(arange(n)) equals the AUC on the full (unresampled) base data."""
    rng = np.random.default_rng(11)
    n = 3000
    y_score = rng.random(n)
    y_true = (rng.random(n) < 0.4).astype(np.int64)
    resampler = make_bootstrap_auc_resampler(y_true, y_score)
    full = np.arange(n, dtype=np.int64)
    assert resampler(full) == _ref(y_true, y_score, full)


def test_presort_falls_back_exact_on_tied_discrete_scores():
    """GATED-OUT: tied/discrete base scores route to the exact argsort path, so
    the resampler still matches fast_roc_auc_unstable byte-for-byte (the factory
    returns the exact fallback closure; it must NOT use the counting path here)."""
    rng = np.random.default_rng(13)
    n = 4000
    # low-cardinality (heavily tied) scores: binned to 8 distinct values
    y_score = (rng.random(n) * 8).astype(np.int64).astype(np.float64) / 8.0
    y_true = (rng.random(n) < 0.5).astype(np.int64)
    # sanity: base genuinely has ties
    assert np.unique(y_score).size < n
    resampler = make_bootstrap_auc_resampler(y_true, y_score)
    for _ in range(50):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        # exact fallback => identical to the reference argsort path on tied data
        assert resampler(idx) == _ref(y_true, y_score, idx)


def test_presort_two_value_scores_route_to_exact():
    """Extreme tie case (binary 0/1 scores) must use the exact fallback path and
    stay equal to the reference -- guards the gate against an always-fast slip."""
    rng = np.random.default_rng(17)
    n = 2000
    y_score = (rng.random(n) < 0.5).astype(np.float64)
    y_true = (rng.random(n) < 0.5).astype(np.int64)
    resampler = make_bootstrap_auc_resampler(y_true, y_score)
    for _ in range(30):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        assert resampler(idx) == _ref(y_true, y_score, idx)


def test_perf_sentinel_presort_beats_argsort():
    """Perf sentinel: the O(n) counting-gather resampler must beat the per-
    resample argsort on the 1000-bootstrap loop. Measured 1.6x-4.4x across
    n=5k..200k (see _benchmarks/bench_bootstrap_auc_presort.py); assert a
    conservative >=1.2x floor to catch a regression without flaking on noise."""
    rng = np.random.default_rng(3)
    n = 20000
    n_boot = 300
    y_score = rng.random(n)
    y_true = (rng.random(n) < 0.3).astype(np.int64)
    idxs = [rng.integers(0, n, size=n, dtype=np.int64) for _ in range(n_boot)]

    # warm JIT for both paths
    make_bootstrap_auc_resampler(y_true, y_score)(idxs[0])
    fast_roc_auc_unstable(y_true[idxs[0]], y_score[idxs[0]])

    t0 = time.perf_counter()
    for idx in idxs:
        _ref(y_true, y_score, idx)
    old = time.perf_counter() - t0

    resampler = make_bootstrap_auc_resampler(y_true, y_score)
    t1 = time.perf_counter()
    for idx in idxs:
        resampler(idx)
    new = time.perf_counter() - t1

    speedup = old / new
    assert speedup >= 1.2, f"presort resampler not faster: {speedup:.2f}x (old={old * 1e3:.1f}ms new={new * 1e3:.1f}ms)"


def test_fused_kernel_bit_identical_to_exact_on_tie_free():
    """Fused single-pass kernel == fast_roc_auc_unstable(y[idx], score[idx]) on
    tie-free base, byte-for-byte (maxdiff 0.0). Pins the gated-in fast path."""
    from mlframe.metrics._core_auc_brier import _fused_resample_auc

    rng = np.random.default_rng(101)
    n = 5000
    y_score = rng.random(n)  # continuous -> all-distinct
    y_true = (rng.random(n) < 0.3).astype(np.int64)
    asc = np.argsort(y_score)
    base_rank = np.empty(n, dtype=np.int64)
    base_rank[asc] = np.arange(n, dtype=np.int64)
    y_by_rank = np.ascontiguousarray(y_true[asc].astype(np.int64))
    for _ in range(40):
        idx = rng.integers(0, n, size=n, dtype=np.int64)
        got = float(_fused_resample_auc(idx, base_rank, y_by_rank, n))
        ref = _ref(y_true, y_score, idx)
        assert got == ref, f"fused kernel not bit-identical: {got!r} vs {ref!r}"


def test_fused_resampler_matches_exact_single_class_nan():
    """All-positive resample -> tmp==0 -> NaN, matching the exact path's NaN."""
    from mlframe.metrics._core_auc_brier import make_bootstrap_auc_resampler

    rng = np.random.default_rng(5)
    n = 1000
    y_score = rng.random(n)
    y_true = np.ones(n, dtype=np.int64)  # single class
    resampler = make_bootstrap_auc_resampler(y_true, y_score)
    idx = rng.integers(0, n, size=n, dtype=np.int64)
    assert np.isnan(resampler(idx))


def test_bootstrap_metrics_skips_preslice_when_all_idx_aware():
    """MED lead: when every active metric is idx-aware, bootstrap_metrics must
    skip the yt/yp pre-slice yet still produce the SAME samples as when a non-
    idx-aware metric forces the slice. Bit-identity across the gate boundary."""
    from mlframe.evaluation.bootstrap import bootstrap_metrics
    from mlframe.metrics._core_auc_brier import (
        make_bootstrap_auc_resampler,
        fast_roc_auc_unstable,
    )

    rng = np.random.default_rng(23)
    n = 3000
    y_score = rng.random(n)
    y_true = (rng.random(n) < 0.4).astype(np.float64)

    # idx-aware-only: triggers the pre-slice skip (active == [])
    r1 = bootstrap_metrics(
        y_true,
        y_score,
        metric_fns={},
        n_bootstrap=200,
        random_state=9,
        metric_fns_idx={"roc_auc": make_bootstrap_auc_resampler(y_true, y_score)},
    )
    # add a non-idx-aware metric -> forces the slice; idx sequence identical for
    # the same seed, so the idx-aware roc_auc samples must be byte-identical.
    r2 = bootstrap_metrics(
        y_true,
        y_score,
        metric_fns={"auc_sliced": lambda yt, yp: float(fast_roc_auc_unstable(yt, yp))},
        n_bootstrap=200,
        random_state=9,
        metric_fns_idx={"roc_auc": make_bootstrap_auc_resampler(y_true, y_score)},
    )
    assert np.array_equal(r1["roc_auc"]["samples"], r2["roc_auc"]["samples"])
    # and the sliced non-idx-aware metric equals the idx-aware one (tie-free base)
    assert np.array_equal(r2["roc_auc"]["samples"], r2["auc_sliced"]["samples"])


def test_perf_sentinel_fused_beats_prior_resampler():
    """Perf sentinel for the fused kernel vs the prior 4-pass resampler shape.
    Compares the fused fast path to the exact per-resample argsort path (the
    pre-fused floor); fused must win clearly. Measured 1.7x-2.2x@50k-200k;
    assert >=1.3x to catch regression without flaking."""
    from mlframe.metrics._core_auc_brier import make_bootstrap_auc_resampler

    rng = np.random.default_rng(31)
    n = 50000
    n_boot = 200
    y_score = rng.random(n)
    y_true = (rng.random(n) < 0.3).astype(np.int64)
    idxs = [rng.integers(0, n, size=n, dtype=np.int64) for _ in range(n_boot)]

    resampler = make_bootstrap_auc_resampler(y_true, y_score)
    resampler(idxs[0])
    _ref(y_true, y_score, idxs[0])

    t0 = time.perf_counter()
    for idx in idxs:
        _ref(y_true, y_score, idx)
    old = time.perf_counter() - t0
    t1 = time.perf_counter()
    for idx in idxs:
        resampler(idx)
    new = time.perf_counter() - t1
    speedup = old / new
    assert speedup >= 1.3, f"fused resampler not faster: {speedup:.2f}x (old={old * 1e3:.1f}ms new={new * 1e3:.1f}ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
