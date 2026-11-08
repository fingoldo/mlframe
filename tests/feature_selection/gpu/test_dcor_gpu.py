"""The device dCor moments against the host kernel (double throughout, so to 1e-11 of the largest value), and the batch dispatch."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._dcor_gpu import dcor_backend, dcor_columns_gpu
from mlframe.feature_selection.filters._orthogonal_dcor_fe import _dcor_batch, _dcor_columns_kernel, _u_centered_distance_matrix


@pytest.mark.parametrize("m,k", [(500, 210), (37, 3), (1000, 7), (257, 1), (300, 33)])
def test_moments_match_host_kernel(m, k):
    """Normal, constant (when k > 1), tied and heavy-tailed columns."""
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((k, m))
    if k > 1:
        xs[0] = 1.0
    if k > 2:
        xs[1] = np.round(xs[1])
        xs[2] = rng.standard_cauchy(m)
    xs = np.ascontiguousarray(xs)
    b = np.ascontiguousarray(_u_centered_distance_matrix(rng.integers(0, 3, m).astype(np.float64)))
    want_c, want_v = np.empty(k), np.empty(k)
    _dcor_columns_kernel(xs, b, want_c, want_v)
    got = dcor_columns_gpu(xs, b)
    assert got is not None
    assert np.all(np.abs(got[0] - want_c) <= 1e-11 * (np.abs(want_c) + np.max(np.abs(want_c))))
    assert np.all(np.abs(got[1] - want_v) <= 1e-11 * (np.abs(want_v) + np.max(np.abs(want_v))))


def test_batch_dispatch_and_off_switch(monkeypatch):
    """``_dcor_batch`` through the device equals the host path with ``MLFRAME_DCOR_DEVICE=0`` to round-off."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3000, 40))
    y = (X[:, 0] + 0.5 * rng.standard_normal(3000) > 0).astype(np.int64)
    monkeypatch.setenv("MLFRAME_DCOR_DEVICE", "0")
    assert dcor_backend(500, 40) == "cpu"
    want = _dcor_batch(X, y, n_sample=500, random_state=3)
    monkeypatch.setenv("MLFRAME_DCOR_DEVICE", "1")
    got = _dcor_batch(X, y, n_sample=500, random_state=3)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-13)
    assert int(np.argmax(got)) == 0
