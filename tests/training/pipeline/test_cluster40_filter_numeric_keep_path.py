"""Cluster 40 regression: the pinned-columns path of _filter_to_numeric keeps nullable booleans and the pinned set (audit 2026-11-20, finding 40-F24)."""

import pandas as pd

from mlframe.training.pipeline._pipeline_extensions import _filter_to_numeric


def test_pinned_nullable_bool_is_promoted_and_unpinned_text_is_dropped():
    """With keep_cols pinned, a nullable-bool object column is promoted, the text column is dropped, the result equals the pin (40-F24)."""
    df = pd.DataFrame(
        {
            "flag": pd.Series([True, None, False, True], dtype=object),
            "x": [1.0, 2.0, 3.0, 4.0],
            "text": ["a", "b", "c", "d"],
        }
    )
    out, dropped = _filter_to_numeric(df, keep_cols=["flag", "x"])
    assert list(out.columns) == ["flag", "x"]
    assert dropped == ["text"]
    assert out["flag"].iloc[0] == 1.0 and out["flag"].iloc[2] == 0.0
    assert pd.isna(out["flag"].iloc[1])
