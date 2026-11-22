"""Audit 2026-11-20 finding 35-F55: one _subset_rows for the gated models; it keeps the frame flavour and the selected rows.

The two gated models used to hold separate copies with different polars detection. The shared helper returns the same rows for a polars
frame, a pandas frame with a non-default index, and a numpy array.
"""

import numpy as np
import pandas as pd
import polars as pl

from mlframe.training.composite._frame_ops import subset_rows
from mlframe.training.composite import gated_outlier, gated_regression_mixture


def test_subset_rows_keeps_the_selected_rows_for_each_frame_flavour():
    """Polars, pandas (non-default index) and numpy each return exactly the masked rows, in order."""
    mask = np.array([True, False, True, True, False])
    values = np.arange(5, dtype=np.float64)
    pl_out = subset_rows(pl.DataFrame({"x": values}), mask)
    assert pl_out["x"].to_list() == [0.0, 2.0, 3.0]
    pd_frame = pd.DataFrame({"x": values}, index=[10, 20, 30, 40, 50])
    assert subset_rows(pd_frame, mask)["x"].tolist() == [0.0, 2.0, 3.0]
    assert subset_rows(values, mask).tolist() == [0.0, 2.0, 3.0]


def test_both_gated_models_use_the_shared_helper():
    """The two gated modules refer to the same function, not local copies."""
    assert gated_outlier._subset_rows is subset_rows
    assert gated_regression_mixture._subset_rows is subset_rows
