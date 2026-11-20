"""The column-coding cache must not return the codes of one object column for another column that only holds the same values in a different order."""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters import _target_encoding_fe as T


def test_permuted_object_column_gets_its_own_codes():
    """Two object columns with equal value multisets in different row order keep distinct, correct codes."""
    T._release_column_codes_cache()
    a = pd.Series(np.array(["x", "y", "x", "z", "y", "x"], dtype=object))
    b = pd.Series(a.to_numpy()[[3, 0, 1, 5, 2, 4]])
    toks_a, codes_a = T.column_string_codes(a)
    toks_b, codes_b = T.column_string_codes(b)
    assert list(toks_a[codes_a]) == list(a)
    assert list(toks_b[codes_b]) == list(b)
    T._release_column_codes_cache()
