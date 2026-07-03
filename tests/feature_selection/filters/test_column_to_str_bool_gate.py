"""_column_to_str routes a 0/1-containing numeric-object column to the per-unique fast path (bit-identical);
only a genuine bool+equal-numeric coexistence falls back to the exact per-row loop."""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str
from mlframe.feature_selection.filters._internals import canonical_group_token


def _per_row_ref(arr):
    out = np.empty(len(arr), dtype=object)
    for i, v in enumerate(arr):
        if v is None or (isinstance(v, float) and v != v):
            out[i] = "__nan__"
        else:
            out[i] = canonical_group_token(v)
    return out


def test_numeric_object_with_0_1_no_bool_fast_path_bit_identical():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 5000, 30000).astype(float).astype(object)  # contains 0 and 1, no bool
    got = _column_to_str(pd.Series(arr))
    assert np.array_equal(got.astype(str), _per_row_ref(arr).astype(str))


def test_bool_plus_equal_numeric_falls_back_bit_identical():
    # True (==1) coexists with numeric 1 -> factorize would merge; must fall back to per-row (distinct "True"/"1")
    arr = np.array([True, 1, 1.0, 2, 3, False, 0], dtype=object)
    got = _column_to_str(pd.Series(arr, dtype=object))
    ref = _per_row_ref(arr)
    assert np.array_equal(got.astype(str), ref.astype(str))
    assert "True" in set(got.astype(str)) and "1" in set(got.astype(str))  # kept distinct


def test_string_and_nan_object_fast_path():
    arr = np.array(["a", "b", None, np.nan, "a", "c"], dtype=object)
    got = _column_to_str(pd.Series(arr, dtype=object))
    assert np.array_equal(got.astype(str), _per_row_ref(arr).astype(str))
