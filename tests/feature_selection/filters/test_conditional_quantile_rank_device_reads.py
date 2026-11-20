"""The device conditional rank returns the per-bin sorted values and ranks of the plain definition, with the code range taken from the host input (audit 2026-11-20-perf F-F06)."""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from mlframe.feature_selection.filters._conditional_quantile_rank_gpu import conditional_quantile_rank_device_with_bins


def test_ranks_and_sorted_bins_match_the_definition_with_gaps_and_nonfinite_rows():
    """Bin ids with a gap (no row in bin 3), a non-zero minimum and non-finite values: rank = share of the bin's finite values <= x, NaN for a non-finite row, one sorted array per id from 0 to the maximum."""
    rng = np.random.default_rng(1)
    n = 600
    bins = rng.choice([1, 2, 4, 5], size=n)
    x = rng.normal(size=n)
    x[::50] = np.nan
    x[7] = np.inf
    ranks, sorted_bins = conditional_quantile_rank_device_with_bins(x, bins)
    assert len(sorted_bins) == 6
    for c in range(6):
        vals = np.sort(x[(bins == c) & np.isfinite(x)])
        np.testing.assert_array_equal(sorted_bins[c], vals)
    for i in range(n):
        if not np.isfinite(x[i]):
            assert np.isnan(ranks[i])
        else:
            vals = np.sort(x[(bins == bins[i]) & np.isfinite(x)])
            assert ranks[i] == pytest.approx(np.searchsorted(vals, x[i], side="right") / vals.size)
