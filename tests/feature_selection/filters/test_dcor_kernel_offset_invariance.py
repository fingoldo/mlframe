"""The distance-covariance kernel gives the same moments for a column shifted by a large constant (cancellation gate, _orthogonal_dcor_fe)."""

import numpy as np

from mlframe.feature_selection.filters._orthogonal_dcor_fe import _dcor_columns_kernel


def _double_centred(w: np.ndarray) -> np.ndarray:
    """The double-centred distance matrix of ``w``."""
    a = np.abs(w[:, None] - w[None, :])
    return a - a.mean(axis=1, keepdims=True) - a.mean(axis=0, keepdims=True) + a.mean()


def test_a_large_constant_shift_leaves_both_moments_unchanged():
    """Rows z and z + 1e9 give the same dcov2 and dvar2 to 1e-5 relative: the sums are built from differences, so there is no offset to cancel."""
    rng = np.random.default_rng(0)
    m = 300
    z = rng.normal(size=m)
    w = z * 0.5 + rng.normal(size=m)
    xs = np.vstack([z, z + 1e9])
    dcov2 = np.zeros(2)
    dvar2 = np.zeros(2)
    _dcor_columns_kernel(xs, _double_centred(w), dcov2, dvar2)
    np.testing.assert_allclose(dcov2[1], dcov2[0], rtol=1e-5)
    np.testing.assert_allclose(dvar2[1], dvar2[0], rtol=1e-5)
    assert dvar2[0] > 0.1
