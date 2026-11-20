"""Audit 2026-11-20 finding 35-F12: the point-mass match must not scale its tolerance with the point-mass value.

With the default relative tolerance a large ``point_mass_value`` matched targets far from it (about 10 units at 1e6).
"""

import numpy as np

from mlframe.training.composite.gated_outlier import GatedOutlierEstimator


def test_large_point_mass_does_not_absorb_nearby_targets():
    """y = 1005 is not a point mass of 1e6 with an absolute tolerance of 1e-6."""
    est = GatedOutlierEstimator(classifier=None, regressor=None, point_mass_value=1e6, point_mass_atol=1e-6)
    mask = est._is_point_mass(np.array([1e6, 1005.0, 1e6 + 1e-7]))
    assert mask.tolist() == [True, False, True]
