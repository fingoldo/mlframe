"""Pre-binning a column that holds +-inf cuts the finite rows into the usual number of bins."""

from __future__ import annotations

import numpy as np

from mlframe.training.composite.discovery.screening import _prebin_one_column


def test_finite_rows_of_a_column_with_many_infinities_keep_their_bins():
    """600 +inf and 400 finite rows, 16 bins: the upper quantile edges used to be inf / NaN, so the finite rows collapsed into a handful of codes."""
    rng = np.random.default_rng(0)
    col = np.concatenate([np.full(600, np.inf), rng.normal(size=400)])
    nbins = 16
    q_edges = np.linspace(0, 1, nbins + 1)[1:-1]
    codes = _prebin_one_column(col, q_edges=q_edges, nbins=nbins, code_dtype=np.int16)
    finite_codes = codes[np.isfinite(col)]
    assert finite_codes.min() >= 0 and np.unique(finite_codes).size >= nbins - 2
    assert (codes[~np.isfinite(col)] == -1).all()
