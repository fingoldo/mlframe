"""Audit 2026-11-20 finding 27-F9: process_nans on a pandas frame does not change the frame the caller still holds.

The pandas branch assigned the filled columns into the caller's frame. The polars branch returns a new frame, so the two disagreed. The
test fills a pandas frame and checks the original keeps its NaNs while the returned frame is filled.
"""

import numpy as np
import pandas as pd

from mlframe.training._nan_processing import process_nans


def test_pandas_fill_does_not_mutate_the_callers_frame():
    """The original frame keeps its NaN; the returned frame has it filled."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 2.0]})
    out = process_nans(df, fill_value=0.0, verbose=0)
    assert df["a"].isna().sum() == 1
    assert out["a"].isna().sum() == 0
