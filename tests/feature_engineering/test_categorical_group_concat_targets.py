"""The categorical-group search scores regression targets and targets with missing labels instead of failing on them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlframe.feature_engineering.categorical_group_concat import discover_categorical_groups


def _frame(n: int = 2000, seed: int = 0):
    """``a`` and ``b`` matter only jointly (an XOR-like interaction); ``c`` is noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.choice(list("pq"), n), "b": rng.choice(list("uv"), n), "c": rng.choice(list("xyz"), n)})
    signal = ((df["a"] == "p") ^ (df["b"] == "u")).to_numpy().astype(float)
    return df, signal, rng


def test_a_regression_target_finds_the_interacting_pair():
    """A regression target finds the interacting pair."""
    df, signal, rng = _frame()
    y = 3.0 * signal + rng.normal(0, 0.3, len(df))
    groups = discover_categorical_groups(df, ["a", "b", "c"], y, min_mi_gain=0.01)
    assert ["a", "b"] in groups, groups


def test_rows_without_a_label_are_left_out_not_fatal():
    """Rows without a label are left out not fatal."""
    df, signal, rng = _frame()
    y = signal.copy()
    y[rng.random(len(df)) < 0.3] = np.nan
    groups = discover_categorical_groups(df, ["a", "b", "c"], y, min_mi_gain=0.01)
    assert ["a", "b"] in groups, groups
