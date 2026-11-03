"""A lone NaN next to one real value leaves a binary indicator, not a constant column."""

from __future__ import annotations

import pandas as pd

from mlframe.preprocessing.cleaning import analyse_and_clean_features


def test_value_plus_rare_nan_is_not_reported_constant():
    """A column of 'a' with a single NaN is a present/absent indicator and must not be flagged constant."""
    df = pd.DataFrame({"c": ["a"] * 3000 + [None], "n": range(3001)})
    out = analyse_and_clean_features(df, update_data=False, clean_nonnumeric_rarevals=True, verbose=False, min_fewlyvalued_rows_per_value=1)
    assert "c" not in out["constant_features"]
