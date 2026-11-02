"""Fitted selectors pick columns by name, refuse a missing one, and never rename the caller's frame (tracker 2026-11-02 wave 14, SKL14-1 to SKL14-4)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection._transform_select import select_fitted_columns


def _frame():
    """Three named columns."""
    return pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})


def test_reordered_frame_gives_the_same_columns():
    """support picks b and c at fit; a frame ordered c, b, a still returns b then c's values by name."""
    out = select_fitted_columns(_frame()[["c", "b", "a"]], np.array([False, True, True]), ["a", "b", "c"], 3, "T")
    assert list(out.columns) == ["b", "c"]
    assert out["b"].tolist() == [3, 4]


def test_missing_selected_column_raises():
    """A frame without a selected column is refused with a message naming it."""
    with pytest.raises(ValueError, match="lacks"):
        select_fitted_columns(_frame()[["a", "c"]], np.array([False, True, True]), ["a", "b", "c"], 3, "T")


def test_array_width_is_checked():
    """An array of the wrong width is refused instead of sliced."""
    with pytest.raises(ValueError, match="got 2 columns"):
        select_fitted_columns(np.zeros((4, 2)), np.array([True, False, True]), None, 3, "T")


def test_index_array_support_works():
    """Integer positions (CorrelatedFeaturesSelector.support_) select the same way as a mask."""
    out = select_fitted_columns(_frame(), np.array([0, 2]), ["a", "b", "c"], 3, "T")
    assert list(out.columns) == ["a", "c"]
