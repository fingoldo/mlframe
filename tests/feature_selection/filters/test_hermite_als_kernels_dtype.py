"""Regression: the resident ALS skinny-matrix kernels read a float32 design as float32 and write only the buffers they are given."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")


def _design(dtype, n=20000, d=7, seed=0):
    """A random C-contiguous ``(n, d)`` design, a coefficient vector, a weight vector and a target, in ``dtype``."""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, d)).astype(dtype)
    c = rng.standard_normal(d).astype(dtype)
    w = rng.uniform(0.5, 2.0, n).astype(dtype)
    y = rng.standard_normal(n).astype(dtype)
    return B, c, w, y


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_design_matvec_matches_numpy_and_leaves_neighbours_alone(dtype):
    """``B @ c`` matches numpy at the design's dtype, and buffers allocated around the output are not overwritten."""
    from mlframe.feature_selection.filters.hermite_fe._als_kernels_gpu import design_matvec

    B, c, _, _ = _design(dtype)
    guard_before = cp.full(20000, 7, dtype=cp.int32)
    out = design_matvec(cp, cp.asarray(B), cp.asarray(c))
    guard_after = cp.full(20000, 7, dtype=cp.int32)
    assert out.dtype == np.dtype(dtype)
    np.testing.assert_allclose(cp.asnumpy(out), B.astype(np.float64) @ c.astype(np.float64), rtol=1e-4, atol=1e-4)
    assert int(guard_before.min()) == 7 and int(guard_before.max()) == 7
    assert int(guard_after.min()) == 7 and int(guard_after.max()) == 7


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_weighted_gram_matches_numpy(dtype):
    """The Gram matrix and right-hand side of ``B * w`` match numpy for float32 and float64 designs."""
    from mlframe.feature_selection.filters.hermite_fe._als_kernels_gpu import weighted_gram

    B, _, w, y = _design(dtype)
    ata, atb = weighted_gram(cp, cp.asarray(B), cp.asarray(w), cp.asarray(y))
    A = B.astype(np.float64) * w.astype(np.float64)[:, None]
    np.testing.assert_allclose(cp.asnumpy(ata), A.T @ A, rtol=1e-3, atol=1e-2)
    np.testing.assert_allclose(cp.asnumpy(atb), A.T @ y.astype(np.float64), rtol=1e-3, atol=1e-2)
