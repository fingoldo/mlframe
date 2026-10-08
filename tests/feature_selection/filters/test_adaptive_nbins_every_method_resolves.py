"""Every binning method the per-feature edge dispatcher knows is runnable, so a helper that moved out of ``_adaptive_nbins`` can never go missing unnoticed."""

from __future__ import annotations

import numpy as np
import pytest

from mlframe.feature_selection.filters import _adaptive_nbins_columns as cols


@pytest.mark.parametrize("method", sorted(cols._EDGE_BUILDERS))
def test_every_edge_builder_runs_on_a_supervised_column(method):
    """Each registered builder returns edges for a column with real structure (a missing helper surfaces as AttributeError here)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(600)
    y = (x + 0.3 * rng.standard_normal(600) > 0).astype(np.int64)
    edges = cols._EDGE_BUILDERS[method](x, y, "qs", {})
    arr = np.asarray(edges, dtype=np.float64)
    assert arr.ndim == 1 and arr.size >= 1, f"{method} returned no edges"
    assert np.all(np.isfinite(arr)) and np.all(np.diff(arr) >= 0.0), f"{method} edges must be finite and sorted"
    assert arr[0] >= x.min() - 1e-9 and arr[-1] <= x.max() + 1e-9, f"{method} edges must lie inside the data range"
