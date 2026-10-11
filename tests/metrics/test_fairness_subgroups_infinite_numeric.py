"""Regression test: ``create_fairness_subgroups`` on a numeric feature holding +-inf.

``pd.qcut`` turned an infinite value into an infinite quantile edge, and pandas then failed building the interval labels with
``ValueError: missing values must be missing in the same location both left and right sides``. Infinite rows now belong to no bin, like NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.metrics._fairness_metrics import create_fairness_subgroups


def test_infinite_values_land_in_no_bin_instead_of_crashing_qcut():
    """inf / -inf / NaN rows are left unbinned and the finite rows still split into the requested quantile bins."""
    vals = np.random.default_rng(0).standard_normal(300)
    vals[[3, 7]] = [np.inf, -np.inf]
    vals[9] = np.nan
    sub = create_fairness_subgroups(pd.DataFrame({"x": vals}), features=["x"], cont_nbins=3, min_pop_cat_thresh=10)
    bins = sub["x"]["bins"]
    assert len(sub["x"]["bins_names"]) == 3
    assert bins.isna().sum() == 3
    assert bins.isna().to_numpy()[[3, 7, 9]].all()
    assert all(np.isfinite([iv.left for iv in sub["x"]["bins_names"]] + [iv.right for iv in sub["x"]["bins_names"]]))
