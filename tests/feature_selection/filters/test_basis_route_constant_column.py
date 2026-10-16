"""A constant column routes to the default basis instead of raising (it aborted the whole triplet FE family of a fit)."""

import numpy as np

from mlframe.feature_selection.filters.hermite_fe import basis_route_by_moments


def test_constant_column_routes_to_default():
    assert basis_route_by_moments(np.full(500, 3.0)) == "chebyshev"
    assert basis_route_by_moments(np.zeros(500)) == "chebyshev"
