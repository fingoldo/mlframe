"""The spec of a float32 frame without NaN reads its raw columns from the frame instead of holding a dense copy; the values and the null built from them are the dense ones."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._honest_mdlp_null import _njit_null_columns
from mlframe.feature_selection.filters._honest_spec import _FrameColumns, build_honest_spec


def _frame(dtype):
    """A 6-column frame: five continuous columns and one low-cardinality column."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.standard_normal((600, 5)).astype(dtype), columns=list("abcde"))
    X["f"] = rng.integers(0, 3, 600).astype(dtype)
    y = (X["a"].to_numpy() + 0.5 * rng.standard_normal(600) > 0).astype(np.int8)
    return X, y


def test_float32_spec_is_frame_backed_and_matches_the_dense_spec():
    """The float32 spec holds a frame view; its columns and its null equal those of the float64 spec of the same values."""
    X32, y = _frame(np.float32)
    cols = list(X32.columns)
    lazy = build_honest_spec(X32, cols, y, 6)
    dense = build_honest_spec(X32.astype(np.float64), cols, y, 6)
    assert isinstance(lazy["raw"], _FrameColumns)
    assert lazy["raw"].dtype == np.float32 and dense["raw"].dtype == np.float64
    for j in range(len(cols)):
        assert np.array_equal(np.asarray(lazy["raw"][:, j], dtype=np.float64), dense["raw"][:, j])
    codes = np.zeros((600, len(cols)), dtype=np.int8)
    nbins = np.full(len(cols), 8, dtype=np.int64)
    a = _njit_null_columns(lazy, codes, nbins, 0, list(range(len(cols))), 20, 5)
    b = _njit_null_columns(dense, codes, nbins, 0, list(range(len(cols))), 20, 5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_nan_column_is_served_median_filled():
    """A column with NaN keeps the double-precision median fill (the spec then reads float64) and a frame-backed spec holds no dense matrix."""
    X, y = _frame(np.float32)
    X.loc[::7, "b"] = np.nan
    spec = build_honest_spec(X, list(X.columns), y, 6)
    col = np.asarray(spec["raw"][:, 1])
    assert spec["raw"].dtype == np.float64 and not np.isnan(col).any()
    assert col[0] == np.float64(np.nanmedian(X["b"].to_numpy(dtype=np.float64)))
    assert np.array_equal(col[1:7], X["b"].to_numpy(dtype=np.float64)[1:7])
    assert spec["has_nan"] and spec["nan_mask"][::7, 1].all()
