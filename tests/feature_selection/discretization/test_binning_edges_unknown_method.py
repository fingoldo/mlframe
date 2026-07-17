"""Regression: ``get_binning_edges`` must raise on an unknown ``method``.

Pre-fix the njit kernel handled only "uniform"/"quantile"; any other method fell through leaving
``bin_edges`` unbound -> ``UnboundLocalError`` at the return. The fix adds an explicit ``else: raise
ValueError(...)`` so a mistyped / unsupported method fails loudly instead of with a confusing
UnboundLocalError (or, under numba, a lowering error).
"""

import numpy as np
import pytest

from mlframe.feature_selection.filters.discretization._discretization_edges import get_binning_edges


def test_known_methods_return_edges():
    """Known methods return edges."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert len(get_binning_edges(arr, n_bins=4, method="uniform")) == 5
    assert len(get_binning_edges(arr, n_bins=4, method="quantile")) == 5


def test_unknown_method_raises_valueerror():
    """Unknown method raises valueerror."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    with pytest.raises(ValueError):
        get_binning_edges(arr, n_bins=4, method="kmeans")
