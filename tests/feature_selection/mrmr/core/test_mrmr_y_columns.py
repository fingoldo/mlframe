"""Unit coverage for ``mrmr/_mrmr_class_shared.py``'s ``_mrmr_y_columns``.

X_TEST_COVERAGE_QUALITY-6 fix (mrmr_audit_2026-07-22): this tiny leaf module (shared between
``_mrmr_class.py`` and its fit-helpers mixin to break their import cycle) had zero test references
anywhere in the suite. Its methods ARE exercised transitively through multioutput ``MRMR.fit()`` e2e
tests, but the module itself was never targeted directly -- this pins its 3 input-shape branches
(pandas DataFrame, polars DataFrame, plain 2D ndarray) in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.mrmr._mrmr_class_shared import _mrmr_y_columns


def test_pandas_dataframe_yields_column_name_and_values():
    """A pandas DataFrame yields (column name, values) pairs in column order."""
    y = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = list(_mrmr_y_columns(y))
    assert [label for label, _ in result] == ["a", "b"]
    np.testing.assert_array_equal(result[0][1], [1, 2, 3])
    np.testing.assert_array_equal(result[1][1], [4, 5, 6])


def test_polars_dataframe_yields_column_name_and_values():
    """A polars DataFrame yields (column name, values) pairs, detected by module/class name (no
    hard polars import dependency in the shared module itself)."""
    pl = __import__("polars")
    y = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = list(_mrmr_y_columns(y))
    assert [label for label, _ in result] == ["a", "b"]
    np.testing.assert_array_equal(result[0][1], [1, 2, 3])
    np.testing.assert_array_equal(result[1][1], [4, 5, 6])


def test_plain_2d_ndarray_yields_positional_labels():
    """A plain 2D ndarray (no column names available) yields positional ``y0``/``y1``/... labels."""
    y = np.array([[1, 4], [2, 5], [3, 6]])
    result = list(_mrmr_y_columns(y))
    assert [label for label, _ in result] == ["y0", "y1"]
    np.testing.assert_array_equal(result[0][1], [1, 2, 3])
    np.testing.assert_array_equal(result[1][1], [4, 5, 6])


def test_single_column_pandas_dataframe():
    """A single-column DataFrame yields exactly one (label, values) pair."""
    y = pd.DataFrame({"only": [7, 8, 9]})
    result = list(_mrmr_y_columns(y))
    assert len(result) == 1
    assert result[0][0] == "only"
    np.testing.assert_array_equal(result[0][1], [7, 8, 9])
