"""The one-pass int8 min/max kernel and the four-loads fused observed-MI kernel of the analytic gate agree with the plain reductions and the host plug-in MI.

``minmax_i8`` reads the matrix four bytes at a time: totals that are not a multiple of four (a tail), negative codes, a constant matrix, one element and the extremes of the type must all give ``(min, max)``. The fused kernel's row loop is
unrolled four times with a scalar tail: row counts around the unroll step and column counts around the tile must give the MI and occupied bins of the generic block-wise path.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")

from mlframe.feature_selection.filters._feature_engineering_pairs import _pairs_analytic_gpu as pag


@pytest.mark.parametrize("total", [1, 2, 3, 4, 5, 7, 8, 1001, 4096, 100_003])
def test_minmax_matches_numpy_for_any_total(total):
    """Every remainder of the four-byte step, with the extremes of the type placed at the first and last element."""
    rng = np.random.default_rng(total)
    a = rng.integers(-100, 100, total).astype(np.int8)
    a[0], a[-1] = (-128, 127) if total > 1 else (5, 5)
    got = pag._minmax_i8(cp, cp.asarray(a))
    assert got == (int(a.min()), int(a.max()))


def test_minmax_of_a_constant_and_of_non_negative_codes():
    """A constant matrix gives equal bounds; ordinary bin codes give their own range."""
    assert pag._minmax_i8(cp, cp.full(1000, 7, dtype=cp.int8)) == (7, 7)
    codes = np.random.default_rng(1).integers(0, 10, 5003).astype(np.int8)
    assert pag._minmax_i8(cp, cp.asarray(codes)) == (int(codes.min()), int(codes.max()))


def _host_mi_and_bins(codes: np.ndarray, y: np.ndarray, ky: int) -> "tuple[np.ndarray, np.ndarray]":
    """Plug-in MI (nats) and occupied-bin count of every column against y, by counting."""
    n, k = codes.shape
    mi = np.zeros(k)
    bins = np.zeros(k, dtype=np.int64)
    py = np.bincount(y, minlength=ky) / n
    for j in range(k):
        kx = int(codes[:, j].max()) + 1
        joint = np.bincount(codes[:, j].astype(np.int64) * ky + y, minlength=kx * ky).reshape(kx, ky) / n
        px = joint.sum(axis=1)
        bins[j] = int((px > 0).sum())
        nz = joint > 0
        mi[j] = float((joint[nz] * np.log(joint[nz] / (px[:, None] * py[None, :])[nz])).sum())
    return np.maximum(mi, 0.0), bins


@pytest.mark.parametrize("n", [1, 7, 4 * 4 + 3, 1000, 4097])
@pytest.mark.parametrize("k", [1, 5, 31, 33, 70])
def test_the_fused_kernel_matches_the_host_mi_around_the_unroll_and_tile_edges(n, k):
    """Row counts around the unroll step (n of a few rows, a remainder after whole steps) and column counts around the 32-column tile."""
    rng = np.random.default_rng(n * 100 + k)
    codes = rng.integers(0, 6, (n, k)).astype(np.int8)
    y = rng.integers(0, 3, n).astype(np.int64)
    got = pag._fused_observed_mi_and_bins(cp.asarray(codes), y, 3)
    assert got is not None
    want_mi, want_bins = _host_mi_and_bins(codes, y, 3)
    np.testing.assert_allclose(got[0], want_mi, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(got[1], want_bins)


def test_a_negative_code_declines_the_fused_path():
    """A code below zero would index a neighbouring histogram: the fused path returns None and the generic path runs."""
    codes = np.random.default_rng(2).integers(0, 5, (300, 8)).astype(np.int8)
    codes[10, 3] = -1
    assert pag._fused_observed_mi_and_bins(cp.asarray(codes), np.zeros(300, dtype=np.int64), 2) is None


@pytest.mark.parametrize("k", [511, 512, 513, 640, 897])
def test_wide_matrices_take_the_128_column_tile_and_match_the_host_mi(k):
    """From 512 columns the kernel runs 128 columns per block on 1024 threads; widths either side of the switch and a width that leaves a partial last tile give the host MI and bins."""
    n = 1203
    rng = np.random.default_rng(k)
    codes = rng.integers(0, 9, (n, k)).astype(np.int8)
    y = rng.integers(0, 3, n).astype(np.int64)
    tile, threads = pag._launch_shape(k, 9 * 3)
    assert (tile, threads) == ((128, 1024) if k >= 512 else (32, 128))
    got_mi, got_bins = pag._fused_observed_mi_and_bins(cp.asarray(codes), y, 3)
    want_mi, want_bins = _host_mi_and_bins(codes, y, 3)
    np.testing.assert_allclose(got_mi, want_mi, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(got_bins, want_bins)


def test_a_histogram_too_large_for_the_wide_tile_keeps_the_narrow_tile():
    """128 columns of 100 cells are 51200 bytes of shared histograms: over the 48 KB budget, so the 32-column tile is kept."""
    assert pag._launch_shape(2000, 100) == (32, 128)
    assert pag._launch_shape(2000, 96)[0] == 128
    assert pag._launch_shape(100, 20) == (32, 128)
