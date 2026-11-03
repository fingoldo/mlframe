"""The unsupervised pre-screen drops an all-NaN float column on polars input, as it does on pandas."""

from __future__ import annotations

import pandas as pd
import polars as pl

from mlframe.feature_selection.pre_screen import compute_unsupervised_drops


def test_all_nan_float_column_is_dropped_on_polars_like_pandas():
    """A column of NaN has no information: pandas drops it (NaN counts as null) and polars used to keep it (null_count is 0)."""
    data = {"a": [float("nan")] * 100, "b": [float(i) for i in range(100)]}
    assert compute_unsupervised_drops(pd.DataFrame(data)) == ["a"]
    assert compute_unsupervised_drops(pl.DataFrame(data)) == ["a"]
