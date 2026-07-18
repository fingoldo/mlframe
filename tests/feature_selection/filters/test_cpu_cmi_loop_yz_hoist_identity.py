"""Bit-identity regression for the Y,Z-entropy hoist in the CPU CMI loop (``_cmi_cuda._cpu_cmi_loop``).

The wellbore MRMR greedy redundancy routes every candidate through ``_cpu_cmi_loop`` (the GPU-CMI
fallback). The hoist computes the round-fixed H(Z)/(Y,Z) terms once and reuses ``classes_yz`` per
candidate, cutting the per-candidate melts from four to two. It MUST be bit-identical to the
un-hoisted ``conditional_mi`` recompute path (MLFRAME_CMI_YZ_HOIST=0). Fails on pre-hoist code only
if the recompute path is silently changed; guards the hoist against a future numeric drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters.info_theory._cmi_cuda import (
    _cpu_cmi_loop,
    _cpu_cmi_loop_hoisted_parallel,
    _cpu_cmi_loop_hoisted_serial,
)
from mlframe.feature_selection.filters.info_theory._entropy_kernels import conditional_mi


def _reference(fd, cand, y, z, fnb):
    """Un-hoisted per-candidate ``conditional_mi`` -- the exact prior behaviour."""
    _vin = np.empty(0, dtype=np.int64)
    return np.array(
        [conditional_mi(fd, np.array([c], dtype=np.int64), y, z, _vin, fnb) for c in cand],
        dtype=np.float64,
    )


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("p", [1, 5, 31, 40])
def test_hoist_bit_identical(seed, p):
    """Hoist bit identical."""
    rng = np.random.default_rng(seed * 100 + p)
    n = int(rng.integers(300, 4000))
    ncols = p + 2
    nbins = np.array([int(rng.integers(2, 16)) for _ in range(ncols)], dtype=np.int64)
    fd = np.empty((n, ncols), dtype=np.int32)
    for c in range(ncols):
        fd[:, c] = rng.integers(0, nbins[c], n)
    cand = np.arange(p, dtype=np.int64)
    y = np.array([p], dtype=np.int64)
    z = np.array([p + 1], dtype=np.int64)

    expected = _reference(fd, cand, y, z, nbins)
    got = _cpu_cmi_loop(fd, cand, y, z, nbins)  # hoist default ON -> routes to serial/parallel by p
    assert np.array_equal(got, expected), f"maxabsdiff={np.max(np.abs(got - expected)):.3e}"

    # Both hoisted kernels agree with the reference regardless of the p-branch threshold.
    assert np.array_equal(_cpu_cmi_loop_hoisted_serial(fd, cand, y, z, nbins), expected)
    assert np.array_equal(_cpu_cmi_loop_hoisted_parallel(fd, cand, y, z, nbins), expected)


@pytest.mark.parametrize("seed", range(6))
def test_pruned_melts_bit_identical_wellbore_shape(seed):
    """Wasted-work prune of the per-candidate melts (2026-07): ``_cmi_one_fixed_yz`` now uses the freqs-only
    ``_entropy_xz_fused`` / ``_entropy_x_onto_classes`` kernels (no discarded length-n ``final_classes`` /
    ``classes_yz.copy()``) instead of full ``merge_vars``. Must stay bit-identical to the un-hoisted
    ``conditional_mi`` recompute path at the real redundancy shape: n=30k screen subsample, |Z|=1, larger nbins,
    a candidate pool spanning the serial/prange threshold. Fails if a future edit reintroduces numeric drift into
    the pruned melts (they were 1.33-1.65x faster than the merge_vars path: bench_cmi_pruned_melts.py)."""
    rng = np.random.default_rng(1234 + seed)
    n = 30000
    p = int(rng.choice([5, 100, 400]))
    ncols = p + 2
    nbins = np.array([int(rng.integers(8, 17)) for _ in range(ncols)], dtype=np.int64)
    fd = np.empty((n, ncols), dtype=np.int32)
    for c in range(ncols):
        fd[:, c] = rng.integers(0, nbins[c], n)
    cand = np.arange(p, dtype=np.int64)
    y = np.array([p], dtype=np.int64)
    z = np.array([p + 1], dtype=np.int64)

    expected = _reference(fd, cand, y, z, nbins)
    assert np.array_equal(_cpu_cmi_loop(fd, cand, y, z, nbins), expected)
    assert np.array_equal(_cpu_cmi_loop_hoisted_serial(fd, cand, y, z, nbins), expected)
    assert np.array_equal(_cpu_cmi_loop_hoisted_parallel(fd, cand, y, z, nbins), expected)
