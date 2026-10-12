"""A categorical composite column is a categorical feature: the replay must emit it as category dtype.

Left as object, XGBoost rejects it inside a categorical-enabled DMatrix ("DataFrame.dtypes ... must be int, float, bool or category").
"""

from __future__ import annotations

import pandas as pd


def test_replayed_categorical_composite_column_is_category_dtype():
    """The replayed composite column has category dtype, not object."""
    from mlframe.training.pipeline._categorical_composite_fe import replay_categorical_composite_fe

    df = pd.DataFrame({"cat_a": ["x", "y", "x"], "cat_b": ["p", "p", "q"]})
    metadata = {"categorical_group_concat_auto_groups": [["cat_a", "cat_b"]]}
    out = replay_categorical_composite_fe(df, metadata)
    composed = [c for c in out.columns if c.startswith("concat_group__")]
    assert composed, "the composite column was not produced"
    assert isinstance(out[composed[0]].dtype, pd.CategoricalDtype)
