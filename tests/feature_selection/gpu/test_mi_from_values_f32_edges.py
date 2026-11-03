"""The float32-edge search of the f32-X MI kernel gives the same MI as the double path on the same values, edge-exact values included."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._fe_batched_mi import binned_mi_from_values_gpu


@pytest.mark.parametrize("nbins", [5, 10])
def test_f32_path_equals_double_path(nbins):
    """MI of float32 columns: f32-X kernel (float32 edges rounded up) == double kernel on the same values upcast; values on and next to every edge."""
    rng = np.random.default_rng(nbins)
    n = 6000
    x = np.stack([rng.standard_normal(n), np.round(rng.standard_normal(n), 1), rng.integers(0, 4, n).astype(float), rng.exponential(size=n), rng.standard_normal(n) * 1e6, rng.standard_normal(n)], axis=1).astype(np.float32)
    q = np.linspace(0, 1, nbins + 1)[1:-1]
    edges = np.quantile(x.astype(np.float64), q, axis=0)  # double edges, generally not representable in float32
    edges[0, 5] = float(x[0, 5])  # one edge exactly on a value
    x[1, 5] = np.nextafter(np.float32(edges[1, 5]), np.float32(np.inf))
    x[2, 5] = np.nextafter(np.float32(edges[1, 5]), np.float32(-np.inf))
    y = rng.integers(0, 3, n)
    got = binned_mi_from_values_gpu(cp.asarray(x), cp.asarray(edges), cp.asarray(y), nbins, 3)
    want = binned_mi_from_values_gpu(cp.asarray(x.astype(np.float64)), cp.asarray(edges), cp.asarray(y), nbins, 3)
    assert np.array_equal(got, want)
