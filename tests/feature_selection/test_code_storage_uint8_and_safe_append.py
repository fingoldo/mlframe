"""Code-matrix storage: uint8 for codes 0..255, and an append that widens instead of wrapping (a code of 128 on int8 data, a -1 on uint8 data)."""

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters.discretization._code_dtype import append_code_columns, narrowest_code_dtype


@pytest.mark.parametrize(
    "lo,hi,expected",
    [(0, 10, np.int8), (-1, 127, np.int8), (0, 128, np.uint8), (0, 255, np.uint8), (-1, 255, np.int16), (0, 256, np.int16), (-40000, 5, np.int32)],
)
def test_narrowest_code_dtype(lo, hi, expected):
    """int8 while it fits (the layout codes up to 127 always had), uint8 for 0..255, then the signed widths."""
    assert narrowest_code_dtype(lo, hi) is expected


@pytest.mark.parametrize(
    "start,new_vals,expected_dtype",
    [
        (np.int8, [0, 5, 127], np.int8),  # fits: dtype kept
        (np.int8, [0, 200], np.int16),  # 200 does not fit int8: astype(int8) used to give -56
        (np.uint8, [0, 255], np.uint8),
        (np.uint8, [-1, 3], np.int16),  # -1 does not fit uint8: astype(uint8) used to give 255
        (np.uint8, [0, 300], np.int16),
    ],
)
def test_append_widens_instead_of_wrapping(start, new_vals, expected_dtype):
    """The appended columns keep their values exactly; the old columns are untouched; the dtype widens only when needed."""
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=start)
    new = np.array(new_vals + [1] * (6 - len(new_vals)), dtype=np.int64).reshape(3, 2)
    out = append_code_columns(data, new)
    assert out.dtype == np.dtype(expected_dtype)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out[:, :2].astype(np.int64), data.astype(np.int64))
    np.testing.assert_array_equal(out[:, 2:].astype(np.int64), new)


def test_append_one_dimensional_and_empty():
    """A 1-D block is one column; an empty block keeps the dtype."""
    data = np.zeros((4, 1), dtype=np.uint8)
    assert append_code_columns(data, np.arange(4)).shape == (4, 2)
    assert append_code_columns(data, np.empty((4, 0), dtype=np.int64)).dtype == np.uint8


def test_compact_storage_picks_uint8_for_a_256_value_categorical():
    """A column with codes 0..255 is stored as uint8 (half of int16), values unchanged."""
    from mlframe.feature_selection.filters._mrmr_fit_impl._fit_impl_stages._binning import _compact_code_storage

    data = np.tile(np.arange(256, dtype=np.int32), (4, 1)).T.copy()
    out = _compact_code_storage(data)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out.astype(np.int32), data)
    assert _compact_code_storage(np.full((3, 2), 7, dtype=np.int32)).dtype == np.int8
    assert _compact_code_storage((data + 1).astype(np.int32)).dtype == np.int16  # 256 does not fit a byte


def test_selection_equals_with_and_without_compact_codes_on_a_high_cardinality_categorical(monkeypatch):
    """A categorical with 300 levels and NaN folds to 255 codes; the selected columns are the same with the compact storage on (uint8) and off (int32)."""
    import warnings

    from mlframe.feature_selection.filters.mrmr import MRMR

    rng = np.random.default_rng(3)
    n = 6000
    levels = rng.integers(0, 300, size=n)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(6)})
    X["cat"] = pd.Categorical(np.where(rng.random(n) < 0.05, None, levels.astype(str)))
    y = (X["x0"] + 0.002 * levels + 0.3 * rng.normal(size=n) > 0).astype(int)
    picked = {}
    for flag in ("1", "0"):
        monkeypatch.setenv("MLFRAME_MRMR_COMPACT_CODES", flag)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MRMR(max_runtime_mins=1, verbose=0, fe_max_steps=0)
            m.fit(X, y)
        picked[flag] = list(m.get_feature_names_out())
    assert picked["1"] == picked["0"], picked
