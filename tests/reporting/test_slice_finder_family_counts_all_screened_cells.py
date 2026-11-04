"""The simultaneous-interval family is every screened cell, not only the degraded ones."""

from __future__ import annotations

import numpy as np

from mlframe.reporting.charts.slice_finder import _collect_candidates


def test_n_screened_includes_cells_that_are_not_degraded():
    """Four supported cells of which two are worse than global: the family size is 4, the candidate pool 2."""
    codes = np.repeat(np.arange(4), 25).reshape(-1, 1).astype(np.int32)
    err = np.where(codes[:, 0] >= 2, 2.0, 0.5).astype(np.float64)
    cand = _collect_candidates([(0,)], codes, err, [4], {}, 5, float(err.mean()), len(err))
    assert cand.score.size == 2
    assert cand.n_screened == 4
