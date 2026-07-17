"""Regression: ks_shift._ks_and_wasserstein njit(prange) kernel matches the prior Python-loop reference.

The loop -> njit rewrite is a perf change (32-48x, see _benchmarks/bench_ksshift_njit.py). Both paths operate
in float32, so the result is not bit-identical (searchsorted/W1 summation order differs), but it must match to a
tight float32 tolerance — far below anything that could re-order the FE features the transformer produces.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_engineering.transformer.ks_shift import _ks_and_wasserstein


def _ks_and_wasserstein_reference(y_neighbors: np.ndarray, y_global_sorted: np.ndarray):
    """Prior Python-loop implementation (git HEAD ks_shift.py), kept here as the identity oracle."""
    n_q, k = y_neighbors.shape
    n_g = y_global_sorted.shape[0]
    ks_out = np.zeros(n_q, dtype=np.float32)
    w1_out = np.zeros(n_q, dtype=np.float32)
    y_local_sorted = np.sort(y_neighbors, axis=1)
    cdf_local = (np.arange(k) + 1).astype(np.float32) / k
    for i in range(n_q):
        local = y_local_sorted[i]
        global_ranks = np.searchsorted(y_global_sorted, local, side="right").astype(np.float32) / n_g
        diff = np.abs(cdf_local - global_ranks)
        ks_out[i] = diff.max()
        widths = np.diff(local, prepend=local[0])
        w1_out[i] = (diff * widths).sum()
    return ks_out, w1_out


@pytest.mark.parametrize("n_q,k,n_g", [(500, 32, 400), (2000, 16, 1600), (3000, 32, 5000)])
def test_ksshift_kernel_matches_loop_reference(n_q, k, n_g):
    rng = np.random.default_rng(n_q + k)
    y_global = np.sort(rng.normal(size=n_g).astype(np.float32))
    y_neighbors = rng.normal(size=(n_q, k)).astype(np.float32)

    ks_ref, w1_ref = _ks_and_wasserstein_reference(y_neighbors, y_global)
    ks_new, w1_new = _ks_and_wasserstein(y_neighbors, y_global)

    np.testing.assert_allclose(ks_new, ks_ref, atol=1e-6, rtol=0)
    np.testing.assert_allclose(w1_new, w1_ref, atol=1e-5, rtol=1e-5)


def test_ksshift_kernel_handles_ties_and_duplicates():
    """Tied / duplicated values exercise searchsorted side='right' boundary behaviour."""
    rng = np.random.default_rng(7)
    y_global = np.sort(rng.integers(0, 5, size=300).astype(np.float32))
    y_neighbors = rng.integers(0, 5, size=(400, 32)).astype(np.float32)
    ks_ref, w1_ref = _ks_and_wasserstein_reference(y_neighbors, y_global)
    ks_new, w1_new = _ks_and_wasserstein(y_neighbors, y_global)
    np.testing.assert_allclose(ks_new, ks_ref, atol=1e-6, rtol=0)
    np.testing.assert_allclose(w1_new, w1_ref, atol=1e-5, rtol=1e-5)
