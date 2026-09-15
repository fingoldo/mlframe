"""After partial_fit on two batches, transform on a row subset replays the frozen parameters instead of re-deriving them from the subset.

The streaming tests all transform the full frame, so a transform that refits (or reads un-frozen per-batch state) on whatever rows it is
given would pass them. Slice replay must be byte-exact against the corresponding rows of the full-frame transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests.feature_selection.mrmr.biz_val.test_biz_value_mrmr_provenance_streaming.test_partial_fit_streaming import (
    _fast_mrmr,
    _simple_binary_frame,
)


def test_partial_fit_transform_row_subset_replays_frozen_params_not_refit():
    """Two partial_fit batches, then transform(X1[10:40]) equals transform(X1)[10:40] exactly and n_features_ is unchanged."""
    X1, y1 = _simple_binary_frame(n=400, seed=0)
    X2, y2 = _simple_binary_frame(n=400, seed=1)
    m = _fast_mrmr(partial_fit_min_recompute=1)
    m.partial_fit(X1, y1)
    m.partial_fit(X2, y2)
    n_before = m.n_features_
    full = np.asarray(pd.DataFrame(m.transform(X1)).to_numpy(), dtype=np.float64)
    sub = np.asarray(pd.DataFrame(m.transform(X1.iloc[10:40])).to_numpy(), dtype=np.float64)
    assert sub.shape == (30, full.shape[1])
    assert np.array_equal(sub, full[10:40]), "transform on a row subset did not replay the frozen parameters"
    assert m.n_features_ == n_before, "transform changed the fitted feature count"
