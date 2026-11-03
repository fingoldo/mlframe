"""kbins quantile codes send NaN to -1 instead of the top real bin, and the single-column null binner refuses codes its dtype would wrap."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.filters.discretization._categorical_encoding import _native_kbins_quantile


def test_nan_is_not_merged_into_the_top_bin():
    """[0, 1, 2, 3, nan] with 2 bins: the NaN row used to share code 1 with the top values; it must be -1."""
    codes = _native_kbins_quantile(np.array([[0.0], [1.0], [2.0], [3.0], [np.nan]]), 2).ravel()
    assert codes[:4].tolist() == [0.0, 0.0, 1.0, 1.0] and codes[4] == -1.0
