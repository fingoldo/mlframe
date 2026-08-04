"""Regression test: attach_new_columns must align by ROW POSITION, not pandas index label.

Pre-fix, the pandas path called ``df.join(new_cols)`` / ``pd.concat([df, new_cols], axis=1)``
directly. Both align by index label. When ``df`` carries a non-contiguous index (e.g. the result
of an upstream ``df.iloc[train_idx]`` split) and ``new_cols`` carries a fresh ``RangeIndex``, the
join silently produced NaN or cross-row-misattributed values for most rows, even though
``new_cols`` was built in the exact same row order as ``df``.
"""

from __future__ import annotations

import pandas as pd

from mlframe.training.pipeline._composite_fe_shared import attach_new_columns


def test_attach_new_columns_aligns_by_position_not_index_label():
    """Non-contiguous df index + fresh-RangeIndex new_cols must still align by row order."""
    # Simulate a post-split frame: original rows 1, 3, 5, 7, 9 survived a train_idx selection.
    df = pd.DataFrame({"x": [10, 30, 50, 70, 90]}, index=[1, 3, 5, 7, 9])
    # new_cols computed in the SAME row order as df, but with its own fresh RangeIndex (the
    # common real-world shape: built via a plain constructor, not derived from df's index).
    new_cols = pd.DataFrame({"y": [100, 300, 500, 700, 900]})

    out = attach_new_columns(df, new_cols)

    assert not out["y"].isna().any(), "pre-fix: index-label join left every row as NaN (no label 0..4 match in df's index)"
    # Each row's "y" must be 10x its own "x" (the row-order-matched value), not a label-matched
    # (and here entirely absent) value.
    assert (out["y"] == out["x"] * 10).all()


def test_attach_new_columns_noop_when_indices_already_match():
    """When new_cols already shares df's index, behaviour is unchanged (no spurious reindex)."""
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[5, 6, 7])
    new_cols = pd.DataFrame({"y": [10, 20, 30]}, index=[5, 6, 7])
    out = attach_new_columns(df, new_cols)
    assert list(out["y"]) == [10, 20, 30]
    assert list(out.index) == [5, 6, 7]
