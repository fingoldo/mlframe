"""fastMI with missing values: pairwise deletion instead of an all-NaN rank vector."""

import numpy as np

from mlframe.feature_selection.filters._fastmi import fastmi


def test_fastmi_with_one_nan_row_matches_the_estimate_on_the_remaining_rows():
    """A single NaN in x must not zero the whole estimate (scipy ranks a NaN-containing vector as all NaN)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = x + 0.3 * rng.standard_normal(500)
    clean = fastmi(x[1:], y[1:])
    x_nan = x.copy()
    x_nan[0] = np.nan
    got = fastmi(x_nan, y)
    assert clean > 0.5
    assert got == clean
