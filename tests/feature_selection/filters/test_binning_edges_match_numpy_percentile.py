"""CPU quantile bin edges equal np.percentile bit for bit, ties included (tracker 2026-11-02 wave 12, PAR12D-1)."""

import numpy as np

from mlframe.feature_selection.filters.discretization._discretization_edges import get_binning_edges


def test_edges_equal_numpy_on_tied_columns():
    """Seven distinct values over hundreds of rows: numba's own interpolation landed an ulp off the tied value; the edges must equal np.percentile exactly."""
    rng = np.random.default_rng(0)
    for n in (50, 274, 1000, 3000):
        col = rng.choice(rng.normal(size=7) * 1000, size=n)
        got = get_binning_edges(col, 10, "quantile")
        np.testing.assert_array_equal(got, np.percentile(col, np.linspace(0, 100, 11)))
