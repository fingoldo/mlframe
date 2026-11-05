"""``discretize_sklearn``'s quantile binning is invariant under a positive scalar rescale of a duplicate-heavy column.

The earlier rule placed a run of equal values across quantile edges by row order, so a count encoding and its
``count / n`` twin could land on different sides of an edge and change the bin COUNT. A tie group now takes the bin of
its first sorted position (the tie rule of ``_quantile_bin_sorted_njit``), so the bin count does not depend on the scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.discretization._discretization_edges import discretize_sklearn


def _skewed_category_counts(seed=21, n=1500, n_categories=8):
    """Per-row category COUNT for a skewed categorical column -- heavily duplicated (each of the n_categories
    distinct count values repeats ~n/n_categories times), the exact shape that exposed the tie-instability."""
    rng = np.random.default_rng(seed)
    cat = rng.integers(0, n_categories, n)
    counts = pd.Series(cat).map(pd.Series(cat).value_counts()).to_numpy(dtype=np.float64)
    return counts


def test_quantile_binning_bin_count_is_invariant_under_a_monotone_rescale():
    """A strict positive rescale (``freq = count / n``) leaves quantile discretization's bin count unchanged."""
    counts = _skewed_category_counts()
    freq = counts / counts.shape[0]

    codes_count = discretize_sklearn(counts, n_bins=10, method="quantile")
    codes_freq = discretize_sklearn(freq, n_bins=10, method="quantile")

    assert len(np.unique(codes_count)) == len(np.unique(codes_freq))
    np.testing.assert_array_equal(codes_count, codes_freq)


def test_uniform_binning_is_scale_invariant_on_the_same_fixture():
    """Sanity check that the instability above is specific to quantile edges, not the fixture itself: equal-width
    (uniform) binning of the SAME duplicate-heavy column produces IDENTICAL codes for both representations."""
    counts = _skewed_category_counts()
    freq = counts / counts.shape[0]

    codes_count = discretize_sklearn(counts, n_bins=10, method="uniform")
    codes_freq = discretize_sklearn(freq, n_bins=10, method="uniform")

    assert np.array_equal(codes_count, codes_freq), "uniform binning should be exactly scale-invariant for a positive rescale"
