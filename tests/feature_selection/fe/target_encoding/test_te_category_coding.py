"""The category coding of the K-fold target encoder is the sorted distinct values and the inverse codes of ``np.unique``, for string columns with missing values too."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._target_encoding_fe import _column_to_str, kfold_target_encode_fit, sorted_string_codes


@pytest.mark.parametrize("k", [3, 40, 700])
def test_sorted_string_codes_equal_np_unique_with_missing_values(k):
    """Same sorted uniques and same int64 inverse as ``np.unique(..., return_inverse=True)`` on the string tokens the encoder builds (None and NaN become one sentinel category)."""
    rng = np.random.default_rng(k)
    n = 4000
    col = pd.Series(np.array([f"cat_{v}" for v in rng.integers(0, k, n)], dtype=object))
    col[rng.integers(0, n, 60)] = None
    col[rng.integers(0, n, 20)] = np.nan
    cats = _column_to_str(col)
    uniq, inverse = np.unique(cats, return_inverse=True)
    got_u, got_inv = sorted_string_codes(cats)
    assert np.array_equal(uniq, got_u)
    assert np.array_equal(inverse, got_inv)
    assert got_inv.dtype == np.int64


def test_fit_lookups_are_keyed_by_the_sorted_categories():
    """The persisted per-category lookup of the encoder lists every distinct token once and its OOF column has no unseen-category gap."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"c": np.array([f"v{v}" for v in rng.integers(0, 12, n)], dtype=object)})
    y = rng.normal(size=n)
    enc, recipes = kfold_target_encode_fit(X, y, ["c"], n_folds=4, smoothing=5.0, random_state=0)
    assert set(recipes["c"]["lookup"]) == set(X["c"].unique())
    assert np.isfinite(enc.to_numpy()).all()
