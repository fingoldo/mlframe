"""Regression test: the "__MISSING__" sentinel injected by prepare_dfs_for_catboost_joint must land at the LAST code
(max+1) of the resulting CategoricalDtype, NOT at code 0 (the alphabetical-sort artifact, since "_" precedes letters
and digits in ASCII). Tree libs that pre-pass CTR / one-hot under "low integer codes ~ frequent" heuristics get
distorted when the synthetic null bucket sits at code 0, and a plain ``sorted(cats)`` would otherwise let the
sentinel's code position shift silently as soon as the real-category set is reshuffled.
"""

import numpy as np
import pandas as pd
import pytest

from mlframe.training.pipeline import prepare_dfs_for_catboost_joint


SENTINEL = "__MISSING__"


def _build_train_df(real_values, *, with_missing: bool):
    """Train DataFrame whose 'cat' column carries every value in real_values plus optionally one NaN row that the
    joint-prep helper rewrites to the sentinel. Numeric column 'x' is included only to keep schema realistic.
    """
    col = list(real_values)
    if with_missing:
        col = col + [None]
    return pd.DataFrame({"cat": col, "x": np.arange(len(col), dtype=np.float64)})


def _sentinel_code(df: pd.DataFrame, col: str = "cat") -> int:
    """Return the integer code the CategoricalDtype assigned to SENTINEL for df[col]."""
    cats = list(df[col].cat.categories)
    assert SENTINEL in cats, f"Sentinel missing from category set: {cats!r}"
    return cats.index(SENTINEL)


def test_missing_sentinel_at_last_code_simple():
    """Test A: encoder over {'foo','bar', NaN->sentinel}. Sentinel must be at code max (== 2), never at 0."""
    train_df = _build_train_df(["foo", "bar"], with_missing=True)
    prepare_dfs_for_catboost_joint(train_df=train_df, val_df=None, test_df=None, cat_features=["cat"])

    categories = list(train_df["cat"].cat.categories)
    sentinel_idx = _sentinel_code(train_df)

    assert sentinel_idx == len(categories) - 1, (
        f"Sentinel must sit at the LAST code (len-1={len(categories) - 1}); got code {sentinel_idx} "
        f"in categories={categories!r}. Plain sorted() would put it at 0 (ASCII '_' < letters)."
    )
    assert sentinel_idx != 0, f"Sentinel landed at code 0 - the alphabetical-sort regression. categories={categories!r}"
    # Real categories must remain alphabetically sorted (deterministic across reruns).
    real = [c for c in categories if c != SENTINEL]
    assert real == sorted(real), f"Real categories not in sorted order: {real!r}"


@pytest.mark.parametrize(
    "real_values",
    [
        ["foo", "bar"],
        ["bar", "foo"],
        ["zeta", "alpha"],
        ["alpha", "zeta"],
        ["m", "a", "z"],
    ],
    ids=["foo_bar", "bar_foo", "zeta_alpha", "alpha_zeta", "m_a_z"],
)
def test_missing_sentinel_shuffle_stable_to_last_code(real_values):
    """Test B (shuffle stability): regardless of the input ordering of real categories, sentinel always lands at the
    LAST code. Pre-fix sorted() would also place it at 0 across all of these, so checking 'last' AND 'not zero' AND
    'len-1' all together catches both the bug and any future regression that might happen to land it mid-list.
    """
    train_df = _build_train_df(real_values, with_missing=True)
    prepare_dfs_for_catboost_joint(train_df=train_df, val_df=None, test_df=None, cat_features=["cat"])

    categories = list(train_df["cat"].cat.categories)
    sentinel_idx = _sentinel_code(train_df)

    assert sentinel_idx == len(categories) - 1, (
        f"For input {real_values!r}: sentinel at code {sentinel_idx}, expected {len(categories) - 1}. categories={categories!r}"
    )
    assert categories[-1] == SENTINEL
    # The set of real categories is identical to the input set, and they are sorted among themselves.
    assert set(categories) - {SENTINEL} == set(real_values)
    assert categories[:-1] == sorted(real_values)


def test_missing_sentinel_no_sentinel_when_no_nulls():
    """Sanity: when train has no NaN rows the sentinel must NOT be silently injected just to satisfy the fix. The
    real categories stay as the alphabetically sorted full set, and SENTINEL is absent from cat.categories.
    """
    train_df = _build_train_df(["foo", "bar"], with_missing=False)
    prepare_dfs_for_catboost_joint(train_df=train_df, val_df=None, test_df=None, cat_features=["cat"])

    categories = list(train_df["cat"].cat.categories)
    assert SENTINEL not in categories, f"Sentinel was injected even though train had no NaN: categories={categories!r}"
    assert categories == sorted(categories)


def test_missing_sentinel_at_last_code_with_val_union():
    """Joint train+val union path: sentinel still at last code; val contributes categories that may be ASCII-larger
    or smaller than train's; held-out test (with OOV string) must not enlarge the union and its OOV cells should
    surface as NaN (pandas Categorical.astype maps OOV to NaN).
    """
    train_df = _build_train_df(["foo"], with_missing=True)
    val_df = _build_train_df(["bar", "qux"], with_missing=False)
    test_df = pd.DataFrame({"cat": ["foo", "newcat"], "x": [0.0, 1.0]})

    prepare_dfs_for_catboost_joint(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        cat_features=["cat"],
    )

    categories = list(train_df["cat"].cat.categories)
    assert categories[-1] == SENTINEL, f"Sentinel not at last code with val union: {categories!r}"
    assert categories[:-1] == ["bar", "foo", "qux"]
    # Val and test share the same dtype (joint), so their category lists are identical to train's.
    assert list(val_df["cat"].cat.categories) == categories
    assert list(test_df["cat"].cat.categories) == categories
    # OOV 'newcat' in test must map to NaN (code -1 in pandas).
    test_codes = test_df["cat"].cat.codes.tolist()
    assert test_codes[0] == categories.index("foo")
    assert test_codes[1] == -1, f"OOV test value should map to NaN code -1; got {test_codes[1]}"
