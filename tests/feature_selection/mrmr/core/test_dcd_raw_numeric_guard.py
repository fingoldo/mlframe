"""The post-selection DCD swap may only take numeric RAW columns; the guard must read the raw frame's dtype, not the bin-code matrix.

``data`` holds integer bin codes for every column, so a dtype test on ``data`` passed a string/categorical raw column and let it reach the
PC1/Pearson combiner, which cannot convert strings to float.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._mrmr_fit_impl._friend_graph_and_redundancy._raw_dtype import raw_column_is_numeric


def _frame_and_codes():
    """A raw frame with a numeric and a string column, plus the integer bin codes a categorize pass would produce for both."""
    X = pd.DataFrame({"num": [0.1, 0.5, 0.9, 0.3], "label": ["a", "b", "a", "c"]})
    data = np.array([[0, 0], [1, 1], [2, 0], [0, 2]], dtype=np.int16)
    return X, data


def test_string_raw_column_is_not_numeric_although_its_codes_are():
    """The string column's bin codes are integers, but the raw column is not numeric."""
    X, data = _frame_and_codes()
    assert raw_column_is_numeric(X, "label", data, 1) is False


def test_numeric_raw_column_is_numeric():
    """Control: a float raw column passes."""
    X, data = _frame_and_codes()
    assert raw_column_is_numeric(X, "num", data, 0) is True


def test_non_frame_input_falls_back_to_the_code_matrix():
    """Without a named raw frame there is no raw dtype to read, so the code matrix decides, as before."""
    _, data = _frame_and_codes()
    assert raw_column_is_numeric(None, "label", data, 1) is True
