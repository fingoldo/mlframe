"""Cluster 41 regression: the pandas inf cleaning does not mutate the caller's frame (audit 2026-11-20, finding 41-F37).

Infinite values used to be replaced in place in the caller's DataFrame. The frame is now copied only when infinities are present, so the caller
keeps its data and the returned frame has NaN where the infinities were.
"""

import numpy as np
import pandas as pd

from mlframe.training.ranking._ranker_suite_train_helpers import _train_mlframe_rank_pandas_dataframes_arrow_backed as clean


def test_caller_frame_is_not_mutated():
    """The input frame keeps its +inf; the returned frame has NaN there (41-F37)."""
    df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
    out = clean(df)
    assert np.isinf(df["a"].iloc[1])
    assert np.isnan(out["a"].iloc[1])


def test_clean_frame_is_returned_as_is():
    """Control: a frame without infinities is returned unchanged (41-F37)."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = clean(df)
    pd.testing.assert_frame_equal(out, df)
