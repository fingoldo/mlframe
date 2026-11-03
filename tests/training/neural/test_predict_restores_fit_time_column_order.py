"""A frame with the same columns in another order must reach the network in the order it was trained on."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch")

from mlframe.training.neural.base._base_fit_prep import _FitPrepMixin


class _Host(_FitPrepMixin):
    """Host with fit-time state for a model trained on columns [a, b, c] with no categoricals."""

    def __init__(self):
        self._cat_cols_ = []
        self._cat_code_maps_ = None
        self._cat_cardinalities_ = None
        self._input_columns_ = ["a", "b", "c"]


def test_reordered_numeric_frame_is_put_back_in_fit_order():
    """Columns arriving as [b, a, c] must come out [a, b, c], with the values still attached to their names."""
    X = pd.DataFrame({"b": [2.0], "a": [1.0], "c": [3.0]})
    out = _Host()._apply_cat_codes(X)
    assert list(out.columns) == ["a", "b", "c"]
    assert out.iloc[0].tolist() == [1.0, 2.0, 3.0]


def test_same_order_frame_is_returned_unchanged():
    """No reorder needed: the very same frame object comes back (negative control)."""
    X = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [3.0]})
    assert _Host()._apply_cat_codes(X) is X
