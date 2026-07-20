"""Direct unit coverage for ``mrmr._mrmr_class_shared._mrmr_y_columns`` (mrmr_audit_2026-07-20
test_coverage.md #6 / edge_cases.md #37-38). Only exercised transitively via multi-output MRMR fits
before this file -- pins the pandas/polars/ndarray branches directly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.mrmr._mrmr_class_shared import _mrmr_y_columns


class TestPandasDataFrameBranch:
    """A pandas.DataFrame y yields (column_name, column_values) pairs in column order."""

    def test_yields_column_name_and_values(self):
        """Column names + values must round-trip exactly."""
        y = pd.DataFrame({"target_a": [1, 2, 3], "target_b": [4.0, 5.0, 6.0]})
        out = list(_mrmr_y_columns(y))
        assert [label for label, _ in out] == ["target_a", "target_b"]
        np.testing.assert_array_equal(out[0][1], [1, 2, 3])
        np.testing.assert_array_equal(out[1][1], [4.0, 5.0, 6.0])

    def test_single_column_dataframe(self):
        """A single-column DataFrame still yields exactly one (name, values) pair."""
        y = pd.DataFrame({"only": [7, 8, 9]})
        out = list(_mrmr_y_columns(y))
        assert len(out) == 1
        assert out[0][0] == "only"
        np.testing.assert_array_equal(out[0][1], [7, 8, 9])


class TestPolarsDuckTypedBranch:
    """The polars branch is duck-typed on ``type(y).__module__.startswith('polars')`` +
    ``type(y).__name__ == 'DataFrame'`` -- no real polars import required to exercise it."""

    def test_fake_polars_module_name_takes_the_polars_branch(self):
        """A stand-in object whose class is renamed/relocated to look like polars.DataFrame must
        take the polars branch (columns attr + __getitem__ + .to_numpy()), not fall through to the
        raw-ndarray branch."""

        class _FakeCol:
            """A column stand-in exposing only .to_numpy(), like a polars Series."""

            def __init__(self, values):
                self._values = np.asarray(values)

            def to_numpy(self):
                """Return the underlying numpy array."""
                return self._values

        class DataFrame:  # deliberately named to duck-type as polars.DataFrame
            """Stand-in whose __module__ is patched to start with 'polars' below."""

            def __init__(self, data: dict):
                self._data = data

            @property
            def columns(self):
                """Column names."""
                return list(self._data.keys())

            def __getitem__(self, key):
                return _FakeCol(self._data[key])

        DataFrame.__module__ = "polars.dataframe.frame"  # real polars module path shape
        y = DataFrame({"pcol_a": [10, 20], "pcol_b": [30, 40]})
        out = list(_mrmr_y_columns(y))
        assert [label for label, _ in out] == ["pcol_a", "pcol_b"]
        np.testing.assert_array_equal(out[0][1], [10, 20])
        np.testing.assert_array_equal(out[1][1], [30, 40])

    def test_real_polars_dataframe_if_available(self):
        """When polars is actually installed, the real DataFrame also takes the polars branch."""
        pl = pytest.importorskip("polars")
        y = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        out = list(_mrmr_y_columns(y))
        assert [label for label, _ in out] == ["a", "b"]
        np.testing.assert_array_equal(out[0][1], [1, 2, 3])
        np.testing.assert_array_equal(out[1][1], [4, 5, 6])


class TestRawNdarrayBranch:
    """A raw 2D ndarray falls through to the ``y{k}`` auto-naming branch."""

    def test_ndarray_yields_y0_y1_labels_in_column_order(self):
        """Column k must be labelled 'y{k}' and carry column k's values, off by one from column count."""
        arr = np.array([[1, 10], [2, 20], [3, 30]])
        out = list(_mrmr_y_columns(arr))
        assert [label for label, _ in out] == ["y0", "y1"]
        np.testing.assert_array_equal(out[0][1], [1, 2, 3])
        np.testing.assert_array_equal(out[1][1], [10, 20, 30])

    def test_list_of_lists_coerces_via_asarray(self):
        """A plain nested list (not yet an ndarray) is coerced via np.asarray before column extraction."""
        y = [[1, 2], [3, 4], [5, 6]]
        out = list(_mrmr_y_columns(y))
        assert [label for label, _ in out] == ["y0", "y1"]
        np.testing.assert_array_equal(out[0][1], [1, 3, 5])
        np.testing.assert_array_equal(out[1][1], [2, 4, 6])

    def test_single_column_ndarray_labelled_y0(self):
        """A (n, 1) ndarray yields exactly one pair labelled 'y0' -- the off-by-one boundary case."""
        arr = np.array([[1], [2], [3]])
        out = list(_mrmr_y_columns(arr))
        assert len(out) == 1
        assert out[0][0] == "y0"
        np.testing.assert_array_equal(out[0][1], [1, 2, 3])
