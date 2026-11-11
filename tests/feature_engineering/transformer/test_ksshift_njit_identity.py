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
    """Exact per-row KS sup-distance and Wasserstein-1 against the global sample, via scipy.

    The kernel is documented as matching scipy (commit 8300ec4d0: KS checks both limits of every jump, W1 covers the mass outside
    the neighbour range). The earlier hand-written loop checked only the right limit and used the right-endpoint CDF, so it no longer
    describes the kernel and must not be the oracle.
    """
    from scipy.stats import ks_2samp, wasserstein_distance

    n_q = y_neighbors.shape[0]
    ks_out = np.zeros(n_q, dtype=np.float32)
    w1_out = np.zeros(n_q, dtype=np.float32)
    g = np.asarray(y_global_sorted, dtype=np.float64)
    for i in range(n_q):
        local = np.asarray(y_neighbors[i], dtype=np.float64)
        ks_out[i] = ks_2samp(local, g).statistic
        w1_out[i] = wasserstein_distance(local, g)
    return ks_out, w1_out


@pytest.mark.parametrize("n_q,k,n_g", [(500, 32, 400), (2000, 16, 1600), (3000, 32, 5000)])
def test_ksshift_kernel_matches_loop_reference(n_q, k, n_g):
    """Ksshift kernel matches loop reference."""
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
