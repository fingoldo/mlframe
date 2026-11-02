"""ShapProxiedFS.transform leaves the caller's DataFrame columns untouched (tracker 2026-11-02 wave 14, SKL14-1)."""

import pandas as pd

from mlframe.feature_selection.shap_proxied_fs import ShapProxiedFS


def test_transform_does_not_rename_the_input_frame():
    """Integer column labels used to be rewritten to strings on the caller's own frame."""
    sel = ShapProxiedFS.__new__(ShapProxiedFS)
    sel.selected_features_ = ["1", "2"]
    df = pd.DataFrame({0: [1, 2], 1: [3, 4], 2: [5, 6]})
    out = sel.transform(df)
    assert list(df.columns) == [0, 1, 2]
    assert list(out.columns) == ["1", "2"]
