"""A NaN in a continuous column no longer turns it into one bin per row."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import _su_bin


def test_continuous_column_with_one_nan_keeps_a_handful_of_bins():
    """5,000 distinct values and one NaN: about n_bins + 1 bins, not 5,000."""
    rng = np.random.default_rng(0)
    col = rng.normal(size=5000)
    col[17] = np.nan
    ids = _su_bin(col, 10)
    assert np.unique(ids).size <= 11
    assert ids[17] == ids.max() and (ids[np.arange(5000) != 17] < ids[17]).all()


def test_clean_column_is_unchanged_and_all_nan_is_one_bin():
    """No NaN: the finite path is the old one; all NaN: a single bin."""
    rng = np.random.default_rng(1)
    col = rng.normal(size=500)
    assert np.unique(_su_bin(col, 10)).size == 10
    assert np.unique(_su_bin(np.full(50, np.nan), 10)).size == 1
