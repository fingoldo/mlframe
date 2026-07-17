"""Edge-case coverage for ``mlframe.core.frame_compat.to_pandas_or_array``.

Complements ``test_frame_compat.py`` with polars nullable / mixed dtypes, duck-typed
polars-like objects (``__module__`` prefix), and the ``to_pandas``-raises -> ``np.asarray``
fallback branch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.core.frame_compat import to_pandas_or_array, _is_polars_module


def test_polars_nullable_int_with_null_converts():
    """Polars nullable int with null converts."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, None, 3]}, schema={"a": pl.Int64})
    out = to_pandas_or_array(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a"]
    vals = out["a"].tolist()
    assert vals[0] == 1.0 and vals[2] == 3.0
    assert pd.isna(vals[1]), "polars null must surface as a pandas NA/NaN, not a garbage value"


def test_polars_mixed_dtypes_preserve_column_kinds():
    """Polars mixed dtypes preserve column kinds."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"i": [1, 2], "f": [1.5, 2.5], "s": ["x", "y"]})
    out = to_pandas_or_array(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["i", "f", "s"]
    assert pd.api.types.is_integer_dtype(out["i"])
    assert pd.api.types.is_float_dtype(out["f"])
    assert out["f"].tolist() == [1.5, 2.5]
    assert pd.api.types.is_object_dtype(out["s"]) or pd.api.types.is_string_dtype(out["s"])


def test_polars_lazyframe_collected():
    """Polars lazyframe collected."""
    pl = pytest.importorskip("polars")
    out = to_pandas_or_array(pl.LazyFrame({"z": [1, 2, 3]}))
    assert isinstance(out, pd.DataFrame)
    assert out["z"].tolist() == [1, 2, 3]


def test_duck_typed_polars_dataframe_uses_to_pandas():
    # An object whose type ``__module__`` starts with "polars" and exposes ``to_pandas``
    # is dispatched via to_pandas even though it is not the real polars class.
    """Duck typed polars dataframe uses to pandas."""
    class FakePolarsDF:
        """Groups tests covering FakePolarsDF."""
        __module__ = "polars.fake"

        def to_pandas(self):
            """Helper that to pandas."""
            return pd.DataFrame({"d": [9]})

    FakePolarsDF.__name__ = "DataFrame"
    fake = FakePolarsDF()
    assert _is_polars_module(fake) is True
    out = to_pandas_or_array(fake)
    assert isinstance(out, pd.DataFrame)
    assert out["d"].tolist() == [9]


def test_to_pandas_raises_falls_back_to_asarray():
    # A polars-like DataFrame whose to_pandas() blows up must fall back to np.asarray
    # rather than propagating the error.
    """To pandas raises falls back to asarray."""
    class RaisingDF:
        """Groups tests covering RaisingDF."""
        __module__ = "polars.x"

        def to_pandas(self):
            """Helper that to pandas."""
            raise RuntimeError("boom")

        def __array__(self, dtype=None):
            return np.array([1, 2, 3])

    RaisingDF.__name__ = "DataFrame"
    out = to_pandas_or_array(RaisingDF())
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [1, 2, 3]
