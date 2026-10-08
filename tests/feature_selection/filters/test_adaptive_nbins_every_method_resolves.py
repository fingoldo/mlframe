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
    assert edges is not None
    e = np.asarray(edges, dtype=np.float64)
    assert e.ndim == 1 and np.all(np.isfinite(e)), f"{method}: edges must be a finite 1-D vector, got shape {e.shape}"
    assert np.all(np.diff(e) >= 0), f"{method}: edges must be non-decreasing"
    assert e.size == 0 or (x.min() <= e.min() and e.max() <= x.max()), f"{method}: edges must lie inside the column's range"
