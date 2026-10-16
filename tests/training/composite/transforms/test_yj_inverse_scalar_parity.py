"""The scalar Yeo-Johnson inverse is the reference the vectorised inverse is checked against, on both of its paths."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.transforms.unary import _YJ_NUMBA_MIN_N, _yj_inverse, _yj_inverse_scalar


@pytest.mark.parametrize("lam", [-1.5, -0.5, 0.0, 0.7, 1.0, 2.0, 2.5])
@pytest.mark.parametrize("n", [50, _YJ_NUMBA_MIN_N + 1])
def test_vectorised_inverse_matches_scalar_reference(lam: float, n: int) -> None:
    """The vectorised inverse agrees with the scalar reference elementwise, on the numpy path and on the numba path."""
    rng = np.random.default_rng(4321)
    t = rng.normal(0.0, 2.0, size=n)
    vec = _yj_inverse(t, lam)
    ref = np.array([_yj_inverse_scalar(float(v), lam) for v in t[:200]])
    np.testing.assert_allclose(vec[:200], ref, rtol=1e-9, atol=1e-10)
