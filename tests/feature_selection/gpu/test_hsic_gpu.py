"""The device biased-HSIC kernel against the host njit kernel: float32 to 1e-5 of the largest score with the same ranking, double to 1e-12."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._hsic_gpu import hsic_backend, hsic_biased_columns_gpu
from mlframe.feature_selection.filters._orthogonal_hsic_fe import _hsic_batch, _hsic_biased_columns_njit, hsic_column_sigmas


def _inputs(m: int, k: int):
    """Normal, constant, tied and heavy-tailed columns against a centred binary-label Gram matrix."""
    rng = np.random.default_rng(0)
    xs = rng.standard_normal((k, m))
    if k > 1:
        xs[0] = 1.0
    if k > 2:
        xs[1] = np.round(xs[1])
        xs[2] = rng.standard_cauchy(m)
    lab = rng.integers(0, 2, m).astype(np.float64)
    L = (lab[:, None] == lab[None, :]).astype(np.float64)
    H = np.eye(m) - 1.0 / m
    return np.ascontiguousarray(xs), np.ascontiguousarray(H @ L @ H), float((m - 1) * (m - 1))


@pytest.mark.parametrize("relaxed,tol", [("1", 1e-5), ("0", 1e-12)])
@pytest.mark.parametrize("m,k", [(500, 210), (37, 3), (1000, 7), (257, 1)])
def test_matches_host_kernel(monkeypatch, relaxed, tol, m, k):
    """Error relative to the largest score, and the ranking of the columns."""
    monkeypatch.setenv("MLFRAME_CRIT_DTYPE_RELAXED", relaxed)
    xs, lc, den = _inputs(m, k)
    want = np.empty(k)
    _hsic_biased_columns_njit(xs, lc, den, want)
    got = hsic_biased_columns_gpu(xs, lc, hsic_column_sigmas(xs), den)
    assert got is not None
    assert np.max(np.abs(got - want)) <= tol * max(np.max(np.abs(want)), 1e-300)
    if k > 1:
        assert np.array_equal(np.argsort(-got, kind="stable")[: k - 1], np.argsort(-want, kind="stable")[: k - 1])


def test_batch_dispatch_and_off_switch(monkeypatch):
    """``_hsic_batch`` gives the host values with ``MLFRAME_HSIC_DEVICE=0`` and matches them through the device path."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3000, 40))
    y = (X[:, 0] + 0.5 * rng.standard_normal(3000) > 0).astype(np.int64)
    monkeypatch.setenv("MLFRAME_HSIC_DEVICE", "0")
    assert hsic_backend(500, 40) == "cpu"
    want = _hsic_batch(X, y, n_sample=500, random_state=3)
    monkeypatch.setenv("MLFRAME_HSIC_DEVICE", "1")
    got = _hsic_batch(X, y, n_sample=500, random_state=3)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5 * np.max(np.abs(want)))
    assert int(np.argmax(got)) == int(np.argmax(want)) == 0
