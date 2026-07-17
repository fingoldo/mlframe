"""Regression: FE per-group / per-category keys must survive int<->float drift.

The engineered-recipe FE families key their per-group / per-category lookup
dicts by a STRING form of the group value, built at fit and replayed at
transform. A bare ``str`` made the integer ``1`` (``'1'``) and the float ``1.0``
(``'1.0'``) DIFFERENT keys, so fitting on integer-coded group labels then
transforming the SAME groups arriving as float (a routine polars int->float
promotion / pandas join upcast) missed every key and silently routed every row
to the global fallback -- the engineered column was computed from the wrong
(global) statistic with no error or warning.

These tests FAIL on the pre-fix bare-``str`` keying (every group collapses to
the global value) and PASS once the shared canonical token collapses
integral-valued int and float labels to the same key.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._internals import (
    canonical_group_token,
    group_key_strings,
)
from mlframe.feature_selection.filters._target_encoding_fe import (
    _column_to_str as _te_column_to_str,
)
from mlframe.feature_selection.filters._extra_fe_families import (
    _column_to_str as _xf_column_to_str,
)
from mlframe.feature_selection.filters._composite_group_agg_fe import (
    build_composite_keys,
)
from mlframe.feature_selection.filters._grouped_agg_fe import (
    apply_grouped_agg,
    generate_grouped_agg_features,
)
from mlframe.feature_selection.filters._count_freq_interaction_fe import (
    apply_frequency_encoding,
    frequency_encode_fit,
)


def test_canonical_group_token_collapses_integral_dtypes() -> None:
    """Canonical group token collapses integral dtypes."""
    assert canonical_group_token(1) == canonical_group_token(1.0) == canonical_group_token(np.int64(1)) == canonical_group_token(np.float64(1.0)) == "1"
    assert canonical_group_token(2.5) == repr(2.5)
    assert canonical_group_token("well_A") == "well_A"
    # Bool must not be silently treated as 0/1 integer.
    assert canonical_group_token(True) == "True"


def test_group_key_strings_int_float_agree() -> None:
    """Group key strings int float agree."""
    gi = group_key_strings(pd.Series([1, 2, 3, 1], dtype="int64"))
    gf = group_key_strings(pd.Series([1.0, 2.0, 3.0, 1.0], dtype="float64"))
    assert list(gi) == list(gf) == ["1", "2", "3", "1"]
    # Non-integral floats keep full precision (no spurious collapse).
    gm = group_key_strings(pd.Series([1.5, 2.0], dtype="float64"))
    assert list(gm) == [repr(1.5), "2"]


def test_te_and_extra_column_to_str_int_float_agree() -> None:
    """Te and extra column to str int float agree."""
    for cts in (_te_column_to_str, _xf_column_to_str):
        a = cts(pd.Series([1, 2, 1], dtype="int64"))
        b = cts(pd.Series([1.0, 2.0, 1.0], dtype="float64"))
        assert list(a) == list(b) == ["1", "2", "1"]


def test_build_composite_keys_int_float_agree() -> None:
    """Build composite keys int float agree."""
    Xi = pd.DataFrame({"a": pd.Series([1, 2], dtype="int64"), "b": pd.Series([3, 4], dtype="int64")})
    Xf = pd.DataFrame(
        {
            "a": pd.Series([1.0, 2.0], dtype="float64"),
            "b": pd.Series([3.0, 4.0], dtype="float64"),
        }
    )
    assert list(build_composite_keys(Xi, ["a", "b"])) == list(build_composite_keys(Xf, ["a", "b"]))


def test_grouped_agg_broadcast_recovers_groups_on_float_drift() -> None:
    """Fit grouped-agg on int groups, replay on the SAME groups as float.

    Pre-fix the float-dtype replay produced ONE distinct value (the global
    fallback) for every row; post-fix all per-group broadcasts are recovered.
    """
    rng = np.random.default_rng(0)
    n = 600
    g = rng.integers(0, 4, n)
    x = g * 10.0 + rng.standard_normal(n)
    X_int = pd.DataFrame({"grp": g.astype("int64"), "val": x})
    _, recipes = generate_grouped_agg_features(X_int, ["grp"], ["val"], stats=["mean"])
    name = next(k for k, r in recipes.items() if r["op"] == "broadcast")
    recipe = recipes[name]

    X_float = pd.DataFrame({"grp": g.astype("float64"), "val": x})
    out_int = apply_grouped_agg(X_int, recipe)
    out_float = apply_grouped_agg(X_float, recipe)

    assert np.allclose(out_int, out_float)
    # 4 distinct per-group broadcast values, NOT all collapsed to the global.
    assert len(np.unique(np.round(out_float, 4))) == 4
    assert not np.allclose(out_float, recipe["global_value"])


def test_frequency_encoding_recovers_levels_on_float_drift() -> None:
    """Fit frequency encoding on int categories, transform the SAME categories
    as float. Pre-fix every test row missed its '1' key (transform produced
    '1.0') and fell back to default 0.0; post-fix the per-category frequencies
    are recovered."""
    rng = np.random.default_rng(1)
    n = 800
    cats = rng.integers(0, 5, n)
    X_int = pd.DataFrame({"c": cats.astype("int64")})
    _, recipes = frequency_encode_fit(X_int, ["c"])
    recipe = recipes["c"]

    X_float = pd.DataFrame({"c": cats.astype("float64")})
    enc_int = apply_frequency_encoding(X_int, "c", recipe)
    enc_float = apply_frequency_encoding(X_float, "c", recipe)

    assert np.allclose(enc_int, enc_float)
    # Per-category frequencies recovered (>1 level), NOT all the default 0.0
    # (the pre-fix failure where every '1.0' key missed the stored '1' key).
    assert len(np.unique(np.round(enc_float, 8))) > 1
    assert float(recipe["default"]) == 0.0
    assert np.all(enc_float > 0.0)
