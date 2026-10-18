"""Regression: the GPU edge dedup reads float32 edge/bound buffers at float32 (not as double*), with the same result as float64."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _run(E_host, lo_host, hi_host, dtype):
    """Run the GPU dedup on dtype copies of the host edges and bounds; return the edge table and the per-column counts."""
    from mlframe.feature_selection.filters._fe_batched_mi import _launch_dedup_edges

    ne, K = E_host.shape
    E = cp.asarray(E_host.astype(dtype))
    lo = cp.asarray(lo_host.astype(dtype))
    hi = cp.asarray(hi_host.astype(dtype))
    Ec = cp.zeros((ne + 1, K), dtype=cp.float64)
    ne_k = cp.zeros(K, dtype=cp.int32)
    _launch_dedup_edges(E, lo, hi, ne, K, Ec, ne_k)
    return cp.asnumpy(Ec), cp.asnumpy(ne_k)


def test_dedup_float32_buffers_match_float64():
    """Float32 edges and bounds (the crit default) give the same deduplicated table as float64 with the same values."""
    rng = np.random.default_rng(0)
    K, nbins = 37, 8
    E = np.sort(rng.integers(-4, 5, size=(nbins - 1, K)).astype(np.float64), axis=0)  # duplicates on purpose
    lo = np.full(K, -4.0)
    hi = np.full(K, 4.0)
    ec32, nk32 = _run(E, lo, hi, np.float32)
    ec64, nk64 = _run(E, lo, hi, np.float64)
    np.testing.assert_array_equal(nk32, nk64)
    for c in range(K):
        w = int(nk64[c])
        np.testing.assert_array_equal(ec32[:w, c], ec64[:w, c])
