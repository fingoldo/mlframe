"""When the patience budget stops forward selection, the non-significant columns added since the last significant step are dropped again."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mlframe.feature_selection.forward_select import forward_select


def test_noise_columns_added_while_waiting_are_not_kept():
    """3 informative columns and 40 noise columns, patience=2: the returned subset holds no noise column."""
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(rng.normal(size=(n, 43)), columns=[f"r{i}" for i in range(3)] + [f"n{i}" for i in range(40)])
    y = 3.0 * X["r0"] + 2.0 * X["r1"] + 1.5 * X["r2"] + rng.normal(scale=0.3, size=n)
    selected, report = forward_select(X, y.to_numpy(), LinearRegression, cv=5, patience=2, return_report=True)
    assert report.stopped_early
    assert set(selected) == {"r0", "r1", "r2"}
