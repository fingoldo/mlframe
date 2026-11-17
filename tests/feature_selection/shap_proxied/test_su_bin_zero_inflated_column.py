"""A zero-inflated continuous column is binned into at most n_bins levels, not one bin per distinct value."""

from __future__ import annotations

import numpy as np

from mlframe.feature_selection.shap_proxied_fs._shap_proxy_interactions import _su_bin


def test_zero_inflated_continuous_column_stays_within_n_bins():
    """95% zeros plus a continuous tail has two distinct quantile edges but many distinct values: the bin count must stay at n_bins."""
    rng = np.random.default_rng(0)
    col = np.where(rng.random(4000) < 0.95, 0.0, rng.random(4000) + 1.0)
    ids = _su_bin(col, 8)
    assert ids.max() + 1 <= 8
    assert ids[col == 0.0].min() == ids[col == 0.0].max()


def test_low_cardinality_integer_column_keeps_one_bin_per_value():
    """A 2-valued column keeps its 2 levels."""
    col = np.array([0.0] * 50 + [1.0] * 50)
    assert _su_bin(col, 8).max() + 1 == 2
