"""Audit 2026-11-20 finding 34-F24: a transform whose domain_check returns a numeric mask still selects the right rows.

The eval path used to index ``y[valid]`` with the raw mask; a 0/1 float array selects rows 0 and 1, not the rows where it is 1.
"""

import numpy as np

from mlframe.training.composite.discovery import _eval


class _NumericMaskTransform:
    """Domain mask is 0.0/1.0 floats, valid only on rows 2..4."""

    requires_base = False

    def domain_check(self, y, base=None):
        """Return a float 0/1 mask."""
        m = np.zeros(len(y))
        m[2:5] = 1.0
        return m


def test_fit_transform_checked_returns_a_boolean_mask():
    """The valid mask is boolean with the rows where the domain holds, not a float array."""
    from mlframe.training.composite.discovery import CompositeTargetDiscovery
    from mlframe.training.configs import CompositeTargetDiscoveryConfig

    self_ = CompositeTargetDiscovery(CompositeTargetDiscoveryConfig())
    y = np.arange(6, dtype=float)
    base = np.arange(6, dtype=float)
    out = _eval._fit_transform_checked(self_, "b", "t", _NumericMaskTransform(), y, base, 0.1)
    valid = out[2]
    assert valid.dtype == bool
    assert valid.tolist() == [False, False, True, True, True, False]
