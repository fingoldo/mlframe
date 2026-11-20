"""Deviations from a group mean and lag differences are computed in float64 (audit 2026-11-20 B-F01).

A column whose level is large against its spread (price 1e6 + 0.01 k) loses the deviation when both operands are float32 (the ulp at 1e6 is 0.0625): three of four rows came out as 0. The
subtraction now runs in float64 and only the stored column is narrowed.
"""

import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._grouped_agg_fe import apply_grouped_agg, generate_grouped_agg_features
from mlframe.feature_selection.filters._ratio_delta_fe import apply_grouped_delta, apply_lagged_diff, grouped_delta_features, lagged_diff_features

_OFFSET = 1.0e6
_STEP = 0.01


def _frame():
    """Two groups of 40 rows each: ``price`` = 1e6 + 0.01 k inside a group, a time column that orders the rows."""
    k = np.tile(np.arange(40), 2)
    return pd.DataFrame({"g": np.repeat(["a", "b"], 40), "price": _OFFSET + _STEP * k, "t": np.arange(80)})


def _group_deviation(df):
    """Float64 reference: price minus its group mean."""
    return (df["price"] - df.groupby("g")["price"].transform("mean")).to_numpy(dtype=np.float64)


def test_grouped_delta_fit_and_replay_follow_the_float64_deviation():
    """Fit-time delta and the replay of its recipe equal the float64 deviation to float32 rounding of the result, not to the ulp of the offset."""
    df = _frame()
    enc, recipes = grouped_delta_features(df, "g", ["price"])
    name = next(n for n, r in recipes.items() if r["op"] == "minus_mean")
    ref = _group_deviation(df)
    np.testing.assert_allclose(enc[name].to_numpy(dtype=np.float64), ref, rtol=1e-6, atol=1e-7)
    replay = apply_grouped_delta(df, recipes[name])
    np.testing.assert_allclose(np.asarray(replay, dtype=np.float64), ref, rtol=1e-6, atol=1e-7)


def test_lagged_diff_fit_and_replay_follow_the_float64_difference():
    """A lag-1 difference of 1e6 + 0.01 k is 0.01 everywhere (float32 operands gave 0 and 0.0625)."""
    df = _frame().iloc[:40].reset_index(drop=True)
    enc, recipes = lagged_diff_features(df, "t", ["price"], periods=(1,))
    name = next(iter(recipes))
    np.testing.assert_allclose(enc[name].to_numpy(dtype=np.float64)[1:], _STEP, rtol=1e-5)
    replay = apply_lagged_diff(df, recipes[name])
    np.testing.assert_allclose(np.asarray(replay, dtype=np.float64)[1:], _STEP, rtol=1e-5)


def test_grouped_agg_z_within_follows_the_float64_deviation():
    """The within-group z-score is the float64 deviation over the group std, at fit and at replay."""
    df = _frame()
    enc, recipes = generate_grouped_agg_features(df, ["g"], ["price"], stats=("mean",))
    name = next(n for n, r in recipes.items() if r["op"] == "z_within")
    std = df.groupby("g")["price"].transform("std").to_numpy(dtype=np.float64)
    ref = _group_deviation(df) / std
    np.testing.assert_allclose(enc[name].to_numpy(dtype=np.float64), ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(apply_grouped_agg(df, recipes[name]), dtype=np.float64), ref, rtol=1e-5, atol=1e-6)
