"""The scalar Yeo-Johnson forward is the reference the vectorised forward is checked against, on both of its paths."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.training.composite.transforms import unary
from mlframe.training.composite.transforms.unary import _YJ_NUMBA_MIN_N, _yj_forward, _yj_forward_scalar


@pytest.mark.parametrize("lam", [-1.5, -0.5, 0.0, 0.7, 1.0, 2.0, 2.5])
@pytest.mark.parametrize("n", [50, _YJ_NUMBA_MIN_N + 1])
def test_vectorised_forward_matches_scalar_reference(lam: float, n: int) -> None:
    """The vectorised forward agrees with the scalar reference elementwise, on the numpy path and on the numba path."""
    rng = np.random.default_rng(1234)
    y = rng.normal(0.0, 3.0, size=n)
    vec = _yj_forward(y, lam)
    ref = np.array([_yj_forward_scalar(float(v), lam) for v in y[:200]])
    np.testing.assert_allclose(vec[:200], ref, rtol=1e-10, atol=1e-12)


def test_scalar_reference_is_the_module_reference() -> None:
    """The scalar form is the module's own reference, so a rename of either side fails here, not silently."""
    assert unary._yj_forward_scalar is _yj_forward_scalar
