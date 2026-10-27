"""``sorted_string_codes`` returns what ``np.unique(..., return_inverse=True)`` returns."""
import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str, sorted_string_codes


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(np.random.default_rng(0).integers(0, 30, 2000).astype(str)),
        pd.Series(np.random.default_rng(1).integers(0, 30, 2000)).astype("category"),
        pd.Series(np.random.default_rng(2).integers(-5, 12, 2000)),
        pd.Series(["b", "a", "É", "ab", "", "a", "B"] * 50),
        pd.Series(np.where(np.random.default_rng(3).random(2000) < 0.1, None, "x"), dtype=object),
    ],
)
def test_matches_np_unique(series):
    """Strings, categoricals, integers coerced to strings, mixed case / non-ASCII / empty strings, and a column with missing values."""
    values = _column_to_str(series)
    uniq, codes = sorted_string_codes(values)
    want_uniq, want_codes = np.unique(values, return_inverse=True)
    assert np.array_equal(uniq, want_uniq)
    assert np.array_equal(codes, want_codes)
    assert codes.dtype == np.int64


def test_a_raw_missing_value_takes_the_sorting_path():
    """An object vector that still holds None (not coerced) gives the np.unique answer or its exception, never a silent -1 code."""
    values = np.array(["a", None, "b", "a"], dtype=object)
    with pytest.raises(TypeError):
        np.unique(values)
    with pytest.raises(TypeError):
        sorted_string_codes(values)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series(np.random.default_rng(0).integers(-5, 40, 3000)),
        pd.Series(np.random.default_rng(1).integers(0, 2, 3000).astype(bool)),
        pd.Series(np.random.default_rng(2).choice([1.0, 1.5, 2.0, np.nan, 10.0], 3000)),
        pd.Series(np.random.default_rng(3).integers(0, 30, 3000).astype(np.float32)),
        pd.Series(np.random.default_rng(4).integers(0, 30, 3000).astype(str)),
        pd.Series(np.random.default_rng(5).integers(0, 9, 3000)).astype("category"),
    ],
)
def test_column_string_codes_matches_unique_of_the_string_column(series):
    """Numeric columns skip the n-long string array; the sorted tokens and the codes are those of ``np.unique(_column_to_str(col), return_inverse=True)``, including NaN and 1 / 1.0 sharing a token."""
    from mlframe.feature_selection.filters._target_encoding_fe import column_string_codes

    uniq, codes = column_string_codes(series)
    want_uniq, want_codes = np.unique(_column_to_str(series), return_inverse=True)
    assert np.array_equal(uniq, want_uniq)
    assert np.array_equal(codes, want_codes)
    assert codes.dtype == np.int64
