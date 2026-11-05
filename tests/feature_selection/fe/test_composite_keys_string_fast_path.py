"""``build_composite_keys`` on string group columns takes the string itself as the token and returns the keys the per-unique canonical path gave."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlframe.feature_selection.filters._composite_group_agg_fe import build_composite_keys
from mlframe.feature_selection.filters._internals import canonical_group_token


def _reference(X: pd.DataFrame, cols) -> np.ndarray:
    """Keys built row by row from the canonical token of every value (the definition of the key)."""
    sep = "\x1f"
    out = []
    for i in range(len(X)):
        out.append(sep.join(canonical_group_token(X[c].iloc[i]) if X[c].iloc[i] is not None else "" for c in cols))
    return np.asarray(out, dtype=object)


@pytest.mark.parametrize("dtype", ["object", "string", "string[pyarrow]", "category"])
def test_string_columns_give_the_row_by_row_keys(dtype):
    """Plain, pandas-string, arrow-string and categorical string columns, alone and next to a numeric one."""
    rng = np.random.default_rng(3)
    X = pd.DataFrame({
        "a": pd.Series(rng.choice(["u", "vv", "1", "1.0", ""], 800)).astype(dtype),
        "b": pd.Series(rng.choice(["p", "q"], 800)).astype(dtype),
        "n": rng.integers(0, 4, 800).astype(float),
    })
    for cols in (["a"], ["a", "b"], ["a", "n"], ["n", "b"]):
        assert build_composite_keys(X, cols).tolist() == _reference(X, cols).tolist()


def test_missing_and_mixed_columns_keep_their_tokens():
    """A column with None / NaN or with mixed types is not taken by the fast path."""
    X = pd.DataFrame({"a": pd.Series(["x", None, "y", "x"], dtype=object), "b": pd.Series([1, 1.0, "z", True], dtype=object)})
    assert build_composite_keys(X, ["a", "b"]).tolist() == _reference(X, ["a", "b"]).tolist()


def test_categorical_group_column_next_to_a_numeric_one():
    """Categorical string, categorical integer and categorical-with-missing columns inside a composite key."""
    rng = np.random.default_rng(5)
    X = pd.DataFrame({
        "s": pd.Series(rng.choice(["u", "v", "w"], 600)).astype("category"),
        "i": pd.Series(rng.integers(0, 5, 600)).astype("category"),
        "m": pd.Series(rng.choice(["p", None, "q"], 600), dtype=object).astype("category"),
    })
    for cols in (["s", "i"], ["s", "m"], ["i", "m"]):
        assert build_composite_keys(X, cols).tolist() == _reference(X, cols).tolist()
