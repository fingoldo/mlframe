"""The analytic noise gate fed from resident codes matches the host path (CPU observed-MI kernel + njit occupied-bin count) and copies back only (K,) vectors."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
if cp.cuda.runtime.getDeviceCount() < 1:
    pytest.skip("no CUDA device", allow_module_level=True)

import numba

from mlframe.feature_selection.filters._analytic_mi_null import _occupied_bins_per_col, analytic_batch_noise_gate
from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_analytic_gpu import resident_analytic_gate, resident_observed_mi_and_bins
from mlframe.feature_selection.filters._gpu_strict_fe import residency_audit
from mlframe.feature_selection.filters.info_theory.shared import select_batch_mi_kernel


def _inputs(n: int, k: int, nbins: int, ny: int, seed: int):
    """Codes where the first columns carry signal about y and the rest are noise, plus one constant and one two-level column."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, ny, size=n).astype(np.int64)
    codes = rng.integers(0, nbins, size=(n, k)).astype(np.int8)
    for j in range(0, k, 5):
        codes[:, j] = np.clip((y * nbins) // ny + rng.integers(0, 2, size=n), 0, nbins - 1)
    codes[:, 1] = 3
    codes[:, 2] = (rng.random(n) > 0.5).astype(np.int8)
    return codes, y


def _cpu_observed(codes: np.ndarray, y: np.ndarray, nbins: int):
    """The CPU ``npermutations=0`` observed MI exactly as the dispatcher's analytic branch calls it."""
    from mlframe.feature_selection.filters._feature_engineering_pairs._pairs_dispatch import _fe_classes_dtype

    n, k = codes.shape
    freqs = np.bincount(y).astype(np.float64) / n
    kernel = select_batch_mi_kernel(n, k)
    return np.asarray(
        kernel(
            disc_2d=codes, factors_nbins=np.full(k, nbins, dtype=np.int64), classes_y=y, classes_y_safe=y, freqs_y=freqs, npermutations=0,
            base_seed=np.uint64(0), min_nonzero_confidence=0.0, use_su=False, dtype=np.int32, classes_dtype=_fe_classes_dtype(codes.dtype, np.full(k, nbins)),
        )
    ), freqs


@pytest.mark.parametrize("n, k, nbins, ny", [(30000, 64, 10, 20), (26000, 33, 8, 5)])
def test_resident_observed_mi_and_bins_match_the_host_kernels(n, k, nbins, ny):
    """Observed MI to double-precision reduction order, occupied bins exactly."""
    codes, y = _inputs(n, k, nbins, ny, 0)
    cpu_mi, _ = _cpu_observed(codes, y, nbins)
    gpu_mi, gpu_bins = resident_observed_mi_and_bins(cp.asarray(codes), y, ny)
    np.testing.assert_allclose(gpu_mi, cpu_mi, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(gpu_bins, _occupied_bins_per_col(np.ascontiguousarray(codes), numba.get_num_threads()))


def test_resident_gate_equals_the_host_gate_and_moves_only_vectors():
    """Same keep/reject verdicts and kept MI as the host analytic gate; no (n, K) transfer."""
    n, k, nbins, ny = 30000, 64, 10, 20
    codes, y = _inputs(n, k, nbins, ny, 1)
    cpu_mi, freqs = _cpu_observed(codes, y, nbins)
    expected = analytic_batch_noise_gate(codes, cpu_mi, y, n, 0.95, by=int(np.count_nonzero(freqs)))
    d_codes = cp.asarray(codes)
    with residency_audit() as rep:
        got = resident_analytic_gate(d_codes, y, int(np.count_nonzero(freqs)), n, 0.95)
    assert got is not None
    assert not rep.bulk_d2h
    np.testing.assert_array_equal(got > 0, expected > 0)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("dtype, nbins, ny", [(np.int32, 10, 20), (np.int8, 40, 20), (np.int8, 100, 40)])
def test_other_dtypes_and_large_histograms_agree_with_the_host_kernels(dtype, nbins, ny):
    """An int32 matrix takes the generic path, a wide histogram a smaller kernel tile or the generic path: the same MI and occupied counts either way."""
    n, k = 26000, 20
    codes, y = _inputs(n, k, nbins, ny, 3)
    cpu_mi, _ = _cpu_observed(codes, y, nbins)
    gpu_mi, gpu_bins = resident_observed_mi_and_bins(cp.asarray(codes.astype(dtype)), y, ny)
    np.testing.assert_allclose(gpu_mi, cpu_mi, rtol=1e-10, atol=1e-12)
    np.testing.assert_array_equal(gpu_bins, _occupied_bins_per_col(np.ascontiguousarray(codes), numba.get_num_threads()))
