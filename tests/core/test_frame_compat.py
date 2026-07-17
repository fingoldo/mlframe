"""Regression coverage for ``mlframe.core.frame_compat.to_pandas_or_array``.

Ships alongside E3.1 (2026-05-22). The helper is the single-source dispatch
that the 14+ ad-hoc sites in the codebase should migrate to over time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.core.frame_compat import to_pandas_or_array


class TestPandasAndNumpyPassthrough:
    """Groups tests covering TestPandasAndNumpyPassthrough."""
    def test_dataframe_returned_as_is(self):
        """Dataframe returned as is."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        out = to_pandas_or_array(df)
        assert out is df

    def test_series_returned_as_is(self):
        """Series returned as is."""
        s = pd.Series([1.0, 2.0])
        out = to_pandas_or_array(s)
        assert out is s

    def test_ndarray_returned_as_is(self):
        """Ndarray returned as is."""
        arr = np.arange(10)
        out = to_pandas_or_array(arr)
        assert out is arr


class TestPolarsDispatch:
    """Groups tests covering TestPolarsDispatch."""
    def test_polars_dataframe_converted_to_pandas(self):
        """Polars dataframe converted to pandas."""
        pl = pytest.importorskip("polars")
        df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
        out = to_pandas_or_array(df)
        assert isinstance(out, pd.DataFrame)
        assert list(out.columns) == ["a", "b"]

    def test_polars_lazyframe_collected_and_converted(self):
        """Polars lazyframe collected and converted."""
        pl = pytest.importorskip("polars")
        lf = pl.LazyFrame({"x": list(range(100))})
        out = to_pandas_or_array(lf)
        assert isinstance(out, pd.DataFrame)
        assert out.shape == (100, 1)
        assert list(out.columns) == ["x"]

    def test_polars_series_converted_to_pandas_series(self):
        """Polars series converted to pandas series."""
        pl = pytest.importorskip("polars")
        s = pl.Series("col", [1.0, 2.0, 3.0])
        out = to_pandas_or_array(s)
        assert isinstance(out, pd.Series)
        assert out.name == "col"


class TestFallbackToAsarray:
    """Groups tests covering TestFallbackToAsarray."""
    def test_list_input_falls_back_to_ndarray(self):
        """List input falls back to ndarray."""
        out = to_pandas_or_array([[1, 2], [3, 4]])
        assert isinstance(out, np.ndarray)
        assert out.shape == (2, 2)

    def test_tuple_input_falls_back_to_ndarray(self):
        """Tuple input falls back to ndarray."""
        out = to_pandas_or_array((1.0, 2.0, 3.0))
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
