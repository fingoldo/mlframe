"""The linear and causal-anchor residuals compute in float64 even for float32 inputs."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.transforms.linear import _linear_residual_forward, _linear_residual_inverse
from mlframe.training.composite.transforms._causal_anchor import _causal_anchor_residual_forward


def test_linear_residual_forward_is_exact_at_large_float32_values():
    """y and base near 1e8 with a residual of 1: float32 alpha * base rounds at spacing 8, float64 keeps the residual."""
    base32 = np.array([1.0e8, 1.0e8 + 8.0], dtype=np.float32)
    y32 = (base32.astype(np.float64) * 1.0 + 1.0).astype(np.float32)
    t = _linear_residual_forward(y32, base32, {"alpha": 1.0, "beta": 0.0})
    expected = y32.astype(np.float64) - base32.astype(np.float64)
    np.testing.assert_allclose(np.asarray(t, dtype=np.float64), expected, atol=1e-9)
    assert t.dtype == np.float64


def test_linear_residual_inverse_round_trips_in_float64():
    """The inverse of a float32 prediction is computed in float64, so forward then inverse returns y."""
    base32 = np.array([1.0e7, 3.3e7], dtype=np.float32)
    y = np.array([1.0e7 + 3.0, 3.3e7 - 5.0], dtype=np.float64)
    params = {"alpha": 0.9, "beta": 2.0}
    t = _linear_residual_forward(y, base32, params)
    np.testing.assert_allclose(_linear_residual_inverse(t.astype(np.float32), base32, params).astype(np.float64), y, rtol=1e-6)


def test_causal_anchor_forward_returns_float64():
    """The causal-anchor residual no longer inherits float32 from its inputs."""
    out = _causal_anchor_residual_forward(np.array([1.0e8, 2.0e8], dtype=np.float32), np.array([1.0e8, 1.0e8], dtype=np.float32), {"alpha": 0.5})
    assert out.dtype == np.float64
