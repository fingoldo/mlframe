"""The compiled Cox-de Boor kernel equals the recursive numpy form for every basis function."""

import numpy as np
import pytest

from mlframe.feature_selection.filters.engineered_recipes._orth_basis_recipes import _bspline_basis_values, _bspline_basis_values_numpy


def _knots(rng, inner, degree=3, ties=False):
    """A clamped knot vector over [0, 1] with ``inner`` interior knots (optionally with a repeated interior knot)."""
    mid = np.sort(rng.uniform(0.05, 0.95, inner))
    if ties and inner >= 2:
        mid[1] = mid[0]
    return np.concatenate([np.zeros(degree + 1), mid, np.ones(degree + 1)])


@pytest.mark.parametrize("inner,ties", [(5, False), (3, False), (6, True)])
@pytest.mark.parametrize("degree", [3, 2])
def test_kernel_equals_numpy_form(inner, ties, degree):
    """All basis functions, points outside the knot range (clipped), points on knots, NaN-free input."""
    rng = np.random.default_rng(inner * 10 + degree)
    knots = _knots(rng, inner, degree, ties)
    z = np.concatenate([rng.uniform(-0.2, 1.2, 5000), knots, np.array([0.0, 1.0])])
    nbasis = knots.size - degree - 1
    for idx in range(nbasis):
        got = _bspline_basis_values(z, knots, idx, degree)
        want = _bspline_basis_values_numpy(z, knots, idx, degree)
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-13)
    total = sum(_bspline_basis_values(z, knots, i, degree) for i in range(nbasis))
    np.testing.assert_allclose(total[(z > knots[degree]) & (z < knots[-degree - 1])], 1.0, atol=1e-9)  # partition of unity inside the range
