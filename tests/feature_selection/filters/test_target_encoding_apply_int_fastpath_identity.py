"""Regression: ``apply_target_encoding`` integer/bool fast path is bit-identical
to the generic ``_column_to_str`` + ``pd.Series.map`` path.

The fast path canonicalises + resolves the TE lookup once per DISTINCT integer
value (then gathers via the ``np.unique`` inverse), skipping the length-n object
token array the old code materialised purely to feed ``.map``. These pins fail if
the fast-path gating or the canonical-token contract ever diverges from the
generic path. Object/mixed columns must keep taking the (unchanged) generic path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._target_encoding_fe import (
    _column_to_str,
    apply_target_encoding,
)


def _generic_reference(col_series, lookup, global_mean):
    """The pre-fastpath behaviour: stringify every row then map (no fast path)."""
    cats = _column_to_str(col_series)
    return (
        pd.Series(cats, copy=False)
        .map(lookup)
        .fillna(global_mean)
        .to_numpy(dtype=np.float64)
    )


@pytest.mark.parametrize("dtype", [np.int64, np.int32, np.uint16])
def test_int_fastpath_matches_generic(dtype):
    rng = np.random.default_rng(0)
    n, card = 20_000, 120
    arr = rng.integers(0, card + 15, size=n).astype(dtype)  # ~unseen tail
    X = pd.DataFrame({"cat": arr})
    lookup = {str(k): float(rng.normal(0.3, 0.1)) for k in range(card)}
    gm = 0.275
    recipe = {"lookup": lookup, "global_mean": gm}

    got = apply_target_encoding(X, "cat", recipe)
    ref = _generic_reference(X["cat"], lookup, gm)
    assert np.array_equal(got, ref), f"max|d|={np.max(np.abs(got - ref))}"


def test_bool_fastpath_matches_generic():
    rng = np.random.default_rng(1)
    n = 10_000
    arr = rng.integers(0, 2, size=n).astype(bool)
    X = pd.DataFrame({"flag": arr})
    # canonical_group_token(np.bool_(True)) -> "True"; only "True" in lookup,
    # "False" falls back to global_mean.
    lookup = {"True": 0.8}
    gm = 0.2
    got = apply_target_encoding(X, "flag", {"lookup": lookup, "global_mean": gm})
    ref = _generic_reference(X["flag"], lookup, gm)
    assert np.array_equal(got, ref)
    # Sanity: True rows -> 0.8, False rows -> global_mean.
    assert np.allclose(got[arr], 0.8)
    assert np.allclose(got[~arr], gm)


def test_object_path_unchanged():
    rng = np.random.default_rng(2)
    n, card = 10_000, 80
    toks = np.array([f"c{k}" for k in range(card + 10)], dtype=object)
    arr = toks[rng.integers(0, card + 10, size=n)]
    X = pd.DataFrame({"cat": arr})
    lookup = {f"c{k}": float(rng.normal(0.0, 1.0)) for k in range(card)}
    gm = 0.123
    got = apply_target_encoding(X, "cat", {"lookup": lookup, "global_mean": gm})
    ref = _generic_reference(X["cat"], lookup, gm)
    assert np.array_equal(got, ref)


def test_all_unseen_int_maps_to_global_mean():
    n = 5_000
    arr = np.full(n, 999_999, dtype=np.int64)
    X = pd.DataFrame({"cat": arr})
    gm = 0.42
    got = apply_target_encoding(X, "cat", {"lookup": {"1": 0.9}, "global_mean": gm})
    assert np.allclose(got, gm)
