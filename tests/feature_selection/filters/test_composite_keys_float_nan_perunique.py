"""build_composite_keys uses the per-unique canonical path on float-with-NaN group columns (bit-identical to the
per-row map, ~13x fewer canonical_group_token calls -- it was the fit's #1 tottime at 5.77M calls).
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._internals import canonical_group_token
from mlframe.feature_selection.filters._composite_group_agg_fe import build_composite_keys


def _per_row_ref(ser):
    return ser.astype(object).map(lambda v: "" if v is None else canonical_group_token(v)).to_numpy()


def test_float_nan_group_key_bit_identical_to_per_row():
    rng = np.random.default_rng(0)
    n = 20000
    a = rng.integers(0, 50, n).astype(np.float64)
    a[rng.random(n) < 0.12] = np.nan  # NaN missing on a float column
    X = pd.DataFrame({"g": a})
    got = build_composite_keys(X, ["g"])
    ref = _per_row_ref(X["g"])
    assert np.array_equal(got.astype(str), ref.astype(str))
    # NaN rows must carry the canonical 'nan' token (not '' -- that is the None-object case only)
    nan_rows = np.isnan(a)
    assert all(str(got[i]) == "nan" for i in np.flatnonzero(nan_rows)[:20])


def test_multi_col_composite_with_float_nan():
    rng = np.random.default_rng(1)
    n = 15000
    a = rng.integers(0, 20, n).astype(np.float64)
    a[rng.random(n) < 0.1] = np.nan
    b = rng.integers(0, 8, n).astype(np.float64)
    X = pd.DataFrame({"a": a, "b": b})
    got = build_composite_keys(X, ["a", "b"])
    # bit-identical to per-row on each part joined by the unit separator
    pa = _per_row_ref(X["a"])
    pb = _per_row_ref(X["b"])
    ref = np.array([f"{x}\x1f{y}" for x, y in zip(pa, pb)], dtype=object)
    assert np.array_equal(got.astype(str), ref.astype(str))
