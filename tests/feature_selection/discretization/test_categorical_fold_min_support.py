"""``min_support`` folding of thin categorical levels in ``cap_categorical_cardinality`` and ``categorize_dataset`` (the ``"min_support"`` fold strategy)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters.discretization import cap_categorical_cardinality, categorize_dataset


def _column():
    """Three frequent levels (500 rows), two thin ones (2 and 1 rows) and one of 6 rows."""
    rng = np.random.default_rng(0)
    return np.concatenate([rng.integers(0, 3, 500), np.full(2, 3), [4], np.full(6, 5)]).astype(np.float64).reshape(-1, 1)


def test_without_min_support_the_result_is_unchanged():
    """``None`` (and the old call) leave a column under the cap exactly as it was."""
    c = _column()
    assert np.array_equal(cap_categorical_cardinality(c, 127), c)
    assert np.array_equal(cap_categorical_cardinality(c, 127, min_support=None), c)


def test_thin_levels_share_one_bucket_and_the_codes_stay_dense():
    """Levels under ``min_support`` fold together after the kept levels (frequency order); the codes run 0..k with no gap."""
    out = cap_categorical_cardinality(_column(), 127, min_support=5).ravel()
    values, counts = np.unique(out, return_counts=True)
    assert list(values) == [0, 1, 2, 3, 4]
    assert counts[3] == 6 and counts[4] == 3  # the 6-row level survives, the 2-row and 1-row levels share the bucket


def test_nan_codes_and_a_column_with_no_thin_level_are_untouched():
    """The -1 sentinel stays; a column whose levels all reach ``min_support`` is returned as it was."""
    c = np.concatenate([np.full(10, 0), np.full(10, 1), [-1, -1]]).astype(np.float64).reshape(-1, 1)
    out = cap_categorical_cardinality(c, 127, min_support=5)
    assert np.array_equal(out, c)
    thin = np.concatenate([np.full(10, 0), np.full(10, 1), [2], [-1]]).astype(np.float64).reshape(-1, 1)
    folded = cap_categorical_cardinality(thin, 127, min_support=5).ravel()
    assert folded[-1] == -1 and folded[20] == 2 and set(folded[:20]) == {0.0, 1.0}


def test_min_support_alone_folds_without_a_cardinality_cap():
    """``categorize_dataset(max_categorical_cardinality=None, categorical_fold_min_support=5)`` folds the thin levels of a string column."""
    rng = np.random.default_rng(1)
    values = [*np.array(["a", "b", "c"])[rng.integers(0, 3, 300)].tolist(), "x", "y", "y"]
    df = pd.DataFrame({"cat": pd.Series(values, dtype="object"), "num": rng.standard_normal(len(values))})
    plain = categorize_dataset(df, max_categorical_cardinality=None)
    folded = categorize_dataset(df, max_categorical_cardinality=None, categorical_fold_min_support=5)
    j_plain, j_fold = plain[1].index("cat"), folded[1].index("cat")
    assert plain[2][j_plain] == 5 and folded[2][j_fold] == 4  # a, b, c, x, y  ->  a, b, c, other
