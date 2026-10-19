"""The last B-spline basis function is about 1 at the right boundary of the axis in float32, as it is in the float64 host; it used to collapse to 0 there."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_DEVICE = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_DEVICE = False
pytestmark = pytest.mark.skipif(not _HAS_DEVICE, reason="needs a CUDA device")


def _axis_and_knots(n_basis, seed=0, n=5000):
    """A float32 axis ``z`` in [0, 1] whose maximum is exactly 1, and a clamped cubic knot vector with data-quantile inner knots (as the spline fitter places them)."""
    x = np.random.default_rng(seed).standard_normal(n)
    z64 = (x - x.min()) / (x.max() - x.min())
    inner = np.quantile(z64, np.linspace(0.0, 1.0, n_basis - 3 + 1))[1:-1]
    knots = np.concatenate([np.zeros(4), inner, np.ones(4)])
    return z64.astype(np.float32), knots


@pytest.mark.parametrize("n_basis", [6, 9])
def test_last_basis_function_at_the_right_boundary_is_not_zero(n_basis):
    """The point at z = 1 gets the last basis function's limit value (about 1) instead of 0, and the first basis function is about 1 at z = 0."""
    from mlframe.feature_selection.filters._orthogonal_univariate_fe._extra_basis_resident import _bspline_col_gpu

    z, knots = _axis_and_knots(n_basis)
    assert float(z.max()) == 1.0 and float(z.min()) == 0.0
    last = cp.asnumpy(_bspline_col_gpu(cp, cp.asarray(z), knots, idx=n_basis - 1, degree=3)).astype(np.float64)
    first = cp.asnumpy(_bspline_col_gpu(cp, cp.asarray(z), knots, idx=0, degree=3)).astype(np.float64)
    assert last[z == 1.0].min() > 0.99, last[z == 1.0]
    assert first[z == 0.0].min() > 0.99, first[z == 0.0]
