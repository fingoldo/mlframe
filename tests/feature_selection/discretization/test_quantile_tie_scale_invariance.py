"""``discretize_sklearn``'s quantile-percentile binning is NOT exactly invariant under a positive scalar rescale of
a duplicate-heavy column.

Found while chasing a pre-existing MRMR test flake: two info-equivalent categorical encodings (a count encoding
and its ``count / n`` frequency twin -- a strict monotone rescale) were sometimes treated inconsistently by a
consumer that read the ``nbins``-quantized screening code instead of the raw continuous value (fixed in
``_fit_impl_core.py``'s cat-FE floor-drop rescue). The root numerical cause: ``np.percentile``-based quantile
edges can land almost exactly on a real data value for one representation of a duplicate-heavy column and not for
its rescaled twin (floating-point division doesn't reproduce the exact percentile-interpolation arithmetic), which
flips which side of ``np.searchsorted(..., side="right")`` a whole cluster of tied rows falls on -- changing the
resulting bin COUNT, not just re-labeling the same bins. Uniform binning does not have this issue (verified below):
equal-width edges scale exactly with the data, so ties never coincide with an edge in one representation but not
the other.
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


def test_quantile_binning_bin_count_disagrees_across_a_monotone_rescale():
    """A strict positive rescale (``freq = count / n``) can change quantile discretization's EFFECTIVE bin count,
    not just its labels -- the numerical instability that motivated preferring raw continuous values over
    quantized codes in the cat-FE floor-drop rescue."""
    counts = _skewed_category_counts()
    freq = counts / counts.shape[0]

    codes_count = discretize_sklearn(counts, n_bins=10, method="quantile")
    codes_freq = discretize_sklearn(freq, n_bins=10, method="quantile")

    n_unique_count = len(np.unique(codes_count))
    n_unique_freq = len(np.unique(codes_freq))
    assert n_unique_count != n_unique_freq, (
        f"expected the known quantile tie-instability to reproduce on this fixture (a monotone rescale changing "
        f"the effective bin count: count->{n_unique_count} bins, freq->{n_unique_freq} bins); if this now matches, "
        f"either the fixture no longer exercises the instability or ``discretize_sklearn``'s quantile path became "
        f"scale-invariant -- re-derive the cat-FE rescue's raw-continuous-value preference against the new fixture."
    )


def test_uniform_binning_is_scale_invariant_on_the_same_fixture():
    """Sanity check that the instability above is specific to quantile edges, not the fixture itself: equal-width
    (uniform) binning of the SAME duplicate-heavy column produces IDENTICAL codes for both representations."""
    counts = _skewed_category_counts()
    freq = counts / counts.shape[0]

    codes_count = discretize_sklearn(counts, n_bins=10, method="uniform")
    codes_freq = discretize_sklearn(freq, n_bins=10, method="uniform")

    assert np.array_equal(codes_count, codes_freq), "uniform binning should be exactly scale-invariant for a positive rescale"
