"""Regression: the fused orthogonal-basis kernel computes in the dtype of its input column and writes nothing outside its own design matrix."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_DEGREE = 6


def _reference(basis, x):
    """The three-term recurrence of ``basis`` over the column ``x``, with every step rounded in ``x.dtype`` as the elementwise cupy loop rounds it."""
    dt = x.dtype.type
    n = x.shape[0]
    B = np.empty((n, _DEGREE + 1), dtype=x.dtype)
    B[:, 0] = dt(1.0)
    B[:, 1] = (dt(1.0) - x) if basis == "laguerre" else x
    for k in range(2, _DEGREE + 1):
        if basis == "hermite":
            B[:, k] = x * B[:, k - 1] - dt(k - 1) * B[:, k - 2]
        elif basis == "legendre":
            B[:, k] = (dt(2 * k - 1) * x * B[:, k - 1] - dt(k - 1) * B[:, k - 2]) / dt(k)
        elif basis == "chebyshev":
            B[:, k] = dt(2.0) * x * B[:, k - 1] - B[:, k - 2]
        else:
            B[:, k] = ((dt(2 * k - 1) - x) * B[:, k - 1] - dt(k - 1) * B[:, k - 2]) / dt(k)
    return B


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("basis", ["hermite", "legendre", "chebyshev", "laguerre"])
def test_fused_basis_matches_the_recurrence_and_leaves_neighbours_alone(basis, dtype):
    """The design has the input dtype, matches the recurrence, and arrays allocated around it keep their contents."""
    from mlframe.feature_selection.filters.hermite_fe._basis_fused_gpu import build_basis_fused

    x = np.random.default_rng(0).uniform(-1.0, 1.0, 1333).astype(dtype)
    before = cp.full(50000, 7, dtype=cp.int32)
    B = build_basis_fused(cp, basis, cp.asarray(x), _DEGREE + 1)
    after = cp.full(50000, 7, dtype=cp.int32)
    cp.cuda.runtime.deviceSynchronize()
    assert B.dtype == np.dtype(dtype)
    np.testing.assert_allclose(cp.asnumpy(B), _reference(basis, x), rtol=1e-4, atol=1e-5)
    assert int(before.min()) == 7 and int(before.max()) == 7
    assert int(after.min()) == 7 and int(after.max()) == 7
