"""The ratio transforms floor a near-zero divisor to +-eps; a base of exactly -1e-300 used to floor to 0 and divide to -inf."""
import numpy as np

from mlframe.training.composite.transforms.simple import _ratio_forward


def test_tiny_negative_base_is_floored_to_minus_eps():
    """y = 1, base = -1e-300, eps = 1e-12: the ratio is -1e12, not -inf."""
    out = _ratio_forward(np.array([1.0]), np.array([-1e-300]), {"eps": 1e-12})
    assert np.isfinite(out).all() and np.allclose(out, -1e12)
