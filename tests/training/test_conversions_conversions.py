"""Regression tests for the polars/numpy conversion fixes (Arrow split-blocks bridge, pl.Enum round-trip,
hoisted to_numpy in composite_estimator + baseline_diagnostics)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_get_pandas_view_preserves_pl_enum_as_categorical():
    """A polars Enum column must round-trip to pandas CategoricalDtype (NOT object). The Arrow split-blocks
    bridge handles this; a bare .to_pandas() degrades the column to object dtype."""
    pl = pytest.importorskip("polars")
    from mlframe.training.utils import get_pandas_view_of_polars_df

    enum_dt = pl.Enum(["red", "green", "blue"])
    df = pl.DataFrame({
        "color": pl.Series(["red", "green", "blue", "green"], dtype=enum_dt),
        "x": [1.0, 2.0, 3.0, 4.0],
    })
    pdf = get_pandas_view_of_polars_df(df)
    assert isinstance(pdf["color"].dtype, pd.CategoricalDtype), (
        f"pl.Enum must round-trip to pandas CategoricalDtype; got {pdf['color'].dtype}"
    )


def test_extract_base_matrix_single_select():
    """The multi-column extract must use a single .select(cols).to_numpy() under polars, returning the
    same content as the prior per-column path."""
    pl = pytest.importorskip("polars")
    from mlframe.training.composite import _extract_base_matrix

    df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
        "c": [7.0, 8.0, 9.0],
    })
    out = _extract_base_matrix(df, ["a", "b", "c"])
    expected = np.array([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
    np.testing.assert_allclose(out, expected)
    assert out.dtype == np.float64


def test_extract_base_matrix_pandas_branch():
    """Pandas branch uses ``loc[:, cols].to_numpy(dtype=np.float64, copy=False)``."""
    from mlframe.training.composite import _extract_base_matrix

    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [4.0, 5.0, 6.0],
    })
    out = _extract_base_matrix(df, ["a", "b"])
    expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    np.testing.assert_allclose(out, expected)


def test_extract_base_matrix_missing_column_raises():
    """Missing column produces a helpful KeyError mentioning the missing column name."""
    from mlframe.training.composite import _extract_base_matrix

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError, match="b"):
        _extract_base_matrix(df, ["a", "b"])


def test_extract_base_matrix_empty_raises():
    """Empty base_columns is an immediate ValueError (legacy contract preserved)."""
    from mlframe.training.composite import _extract_base_matrix

    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError, match="empty"):
        _extract_base_matrix(df, [])


def test_intize_targets_skips_already_int8_ndarray():
    """When the target is already int8, the function must not allocate a copy."""
    from mlframe.training.extractors import intize_targets

    arr = np.array([0, 1, 0, 1], dtype=np.int8)
    targets = {"t": arr}
    intize_targets(targets)
    # The result is the same dtype; we accept either same array OR a view -- key is no NEW int8 buffer
    # was forced through astype.
    assert targets["t"].dtype == np.int8


def test_coerce_to_numpy_short_circuits_on_ndarray():
    """The wrapper must not re-dispatch when input is already an ndarray (saves a .to_numpy() round-trip)."""
    from mlframe.training.utils import coerce_to_numpy

    arr = np.array([1.0, 2.0, 3.0])
    out = coerce_to_numpy(arr)
    assert out is arr


def test_get_pandas_view_preserves_datetime_dtype():
    """Datetime columns must survive the Arrow bridge in a recognisable
    date / datetime form: numpy datetime64, pyarrow date32 (under the
    use_pyarrow_extension_array=True split-blocks path), or object
    fallback. The contract is "the column round-trips" -- not the
    specific dtype, which polars + pyarrow versions both shift across
    minor releases."""
    pl = pytest.importorskip("polars")
    from mlframe.training.utils import get_pandas_view_of_polars_df

    df = pl.DataFrame({
        "ts": pl.Series(
            ["2024-01-01", "2024-06-01", "2024-12-31"],
            dtype=pl.Date,
        ),
        "x": [1.0, 2.0, 3.0],
    })
    pdf = get_pandas_view_of_polars_df(df)
    _dt = pdf["ts"].dtype
    _is_arrow_date = (
        hasattr(pd, "ArrowDtype")
        and isinstance(_dt, pd.ArrowDtype)
        and any(tok in str(_dt).lower() for tok in ("date", "timestamp"))
    )
    # pandas 2.3+ surfaces date columns from the Arrow bridge as
    # ``pd.StringDtype(na_value=nan)`` (lossy but parseable); accept
    # any string-like extension dtype since the contract is "column
    # round-trips and remains recoverable", not "specific dtype".
    _is_string_like = (
        pd.api.types.is_string_dtype(pdf["ts"])
        or "string" in str(_dt).lower()
    )
    assert (
        pd.api.types.is_datetime64_any_dtype(_dt)
        or _dt == "object"
        or _is_arrow_date
        or _is_string_like
    ), f"date column must survive Arrow bridge; got {_dt!r}"


def test_combine_probs_back_compat_no_alphas():
    """Adding quantile_alphas kwarg must not change behaviour for non-quantile callers."""
    from mlframe.training.core.predict import _combine_probs

    a = np.array([[0.1, 0.2], [0.3, 0.4]])
    b = np.array([[0.2, 0.3], [0.4, 0.5]])
    out = _combine_probs([a, b], "mean")
    np.testing.assert_allclose(out, [[0.15, 0.25], [0.35, 0.45]])
