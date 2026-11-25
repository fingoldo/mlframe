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
    assert cols
    for j, _ in enumerate(cols):
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


def test_a_frame_with_the_target_and_a_lazy_column_keeps_the_float32_matrix_and_the_float64_null():
    """The int8 target and a lazy engineered float32 column are float32 in value: the spec reads float32, and its columns, edges and null are those of the float64 spec of the same values."""
    from mlframe.feature_selection.filters import _lazy_engineered as L

    X, y = _frame(np.float32)
    n = len(X)
    values = np.sin(np.arange(n, dtype=np.float32) / 7.0)
    X = pd.concat([X, pd.DataFrame({"lz": L.LazyEngineeredColumn(lambda: values.copy(), n)}, index=X.index, copy=False)], axis=1)
    X["tgt"] = y
    cols = list(X.columns)
    spec = build_honest_spec(X, cols, y, 6)
    dense_frame = X.copy()
    dense_frame["lz"] = values.astype(np.float64)
    dense_frame["tgt"] = y.astype(np.float64)
    dense = build_honest_spec(dense_frame, cols, y, 6)
    assert spec["raw"].dtype == np.float32 and dense["raw"].dtype == np.float64
    for j in range(len(cols)):
        assert np.array_equal(np.asarray(spec["raw"][:, j], dtype=np.float64), dense["raw"][:, j])
    assert np.array_equal(spec["is_cont"], dense["is_cont"]) and np.array_equal(spec["q_fb"], dense["q_fb"]) and np.array_equal(spec["q_rf"], dense["q_rf"])
    codes = np.zeros((n, len(cols)), dtype=np.int8)
    nbins = np.full(len(cols), 8, dtype=np.int64)
    a = _njit_null_columns(spec, codes, nbins, 0, list(range(len(cols))), 20, 5)
    b = _njit_null_columns(dense, codes, nbins, 0, list(range(len(cols))), 20, 5)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_a_wide_integer_or_a_double_column_keeps_the_double_matrix():
    """An int32 column (not always exact in float32) and a float64 column each keep the matrix in double precision."""
    X, y = _frame(np.float32)
    wide_int = X.copy()
    wide_int["w"] = np.arange(len(X), dtype=np.int32) * 1000003
    double = X.copy()
    double["w"] = np.linspace(0.0, 1.0, len(X))
    assert build_honest_spec(wide_int, list(wide_int.columns), y, 6)["raw"].dtype == np.float64
    assert build_honest_spec(double, list(double.columns), y, 6)["raw"].dtype == np.float64
    narrow = X.copy()
    narrow["w"] = (np.arange(len(X)) % 100).astype(np.int16)
    narrow["flag"] = (np.arange(len(X)) % 2).astype(bool)
    assert build_honest_spec(narrow, list(narrow.columns), y, 6)["raw"].dtype == np.float32


def test_a_block_of_float32_columns_of_a_double_spec_is_read_in_float32_and_its_null_is_unchanged():
    """One double-precision column makes the spec's matrix double; a block without it is read in float32, the block with it in double, and every null equals the dense double-precision spec's."""
    from mlframe.feature_selection.filters._honest_mdlp_null import _block_raw_dtype

    X, y = _frame(np.float32)
    X["wide"] = np.random.default_rng(9).standard_normal(len(X))  # float64 values that float32 cannot hold
    X.loc[::11, "d"] = np.nan  # a median-filled column
    cols = list(X.columns)
    spec = build_honest_spec(X, cols, y, 6)
    assert spec["raw"].dtype == np.float64
    exact = {c: bool(spec["raw_f32_exact"][j]) for j, c in enumerate(cols)}
    assert exact == {"a": True, "b": True, "c": True, "d": False, "e": True, "f": True, "wide": False}
    j = {c: i for i, c in enumerate(cols)}
    assert _block_raw_dtype(spec, [j["a"], j["b"], j["e"]]) == np.float32
    assert _block_raw_dtype(spec, [j["a"], j["wide"]]) == np.float64 and _block_raw_dtype(spec, [j["d"]]) == np.float64
    dense = build_honest_spec(X.astype(np.float64), cols, y, 6)
    codes = np.zeros((len(X), len(cols)), dtype=np.int8)
    nbins = np.full(len(cols), 8, dtype=np.int64)
    for block in ([j["a"], j["b"], j["e"]], [j["a"], j["wide"], j["c"]], [j["d"], j["e"]], list(range(len(cols)))):
        got = _njit_null_columns(spec, codes, nbins, 0, block, 20, 5)
        ref = _njit_null_columns(dense, codes, nbins, 0, block, 20, 5)
        assert np.array_equal(got[0], ref[0]) and np.array_equal(got[1], ref[1])
