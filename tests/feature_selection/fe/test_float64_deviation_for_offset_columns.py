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


def test_lagged_diff_replay_uses_the_fit_history_for_a_later_batch():
    """Audit 2026-11-20 B-F02: a live batch that starts after the training series takes its first lag from the training tail (one row at t=40 gives 1600 - 1521, not 0); the training frame itself replays
    to exactly the fit column."""
    t = np.arange(40)
    train = pd.DataFrame({"t": t, "v": (t * t).astype(np.float64)})
    enc, recipes = lagged_diff_features(train, "t", ["v"], periods=(1, 2))
    for name, rec in recipes.items():
        np.testing.assert_array_equal(np.asarray(apply_lagged_diff(train, rec), dtype=np.float64), enc[name].to_numpy(dtype=np.float64))
    rec1 = next(r for r in recipes.values() if r["period"] == 1)
    rec2 = next(r for r in recipes.values() if r["period"] == 2)
    live = pd.DataFrame({"t": [40, 41], "v": [1600.0, 1681.0]})
    np.testing.assert_allclose(np.asarray(apply_lagged_diff(live, rec1), dtype=np.float64), [1600.0 - 1521.0, 1681.0 - 1600.0])
    np.testing.assert_allclose(np.asarray(apply_lagged_diff(live, rec2), dtype=np.float64), [1600.0 - 1444.0, 1681.0 - 1521.0])
    one = pd.DataFrame({"t": [40], "v": [1600.0]})
    np.testing.assert_allclose(np.asarray(apply_lagged_diff(one, rec1), dtype=np.float64), [79.0])


def test_lagged_diff_history_is_per_entity_and_ignores_overlapping_times():
    """Each entity takes its own last training value; a replay frame that overlaps the training times (the training frame) does not double count."""
    t = np.tile(np.arange(10), 2)
    train = pd.DataFrame({"e": np.repeat(["a", "b"], 10), "t": t, "v": np.where(np.repeat(["a", "b"], 10) == "a", 10.0 * t, 100.0 + t)})
    enc, recipes = lagged_diff_features(train, "t", ["v"], periods=(1,), entity_cols=["e"])
    name, rec = next(iter(recipes.items()))
    np.testing.assert_array_equal(np.asarray(apply_lagged_diff(train, rec), dtype=np.float64), enc[name].to_numpy(dtype=np.float64))
    live = pd.DataFrame({"e": ["b", "a", "c"], "t": [10, 10, 10], "v": [111.0, 95.0, 5.0]})
    np.testing.assert_allclose(np.asarray(apply_lagged_diff(live, rec), dtype=np.float64), [111.0 - 109.0, 95.0 - 90.0, 0.0])


def test_modular_product_of_millisecond_operands_is_exact_or_skipped():
    """Audit 2026-11-20 B-F03: ``ts_ms * count`` leaves int64 (about 1.7e12 times 1e7); the replayed residue is the exact ``(a * b) mod m`` from the operand residues, and the scan does not build the wrapped combiner."""
    from mlframe.feature_selection.filters import _pairwise_modular_fe as pm

    a = 1_700_000_000_000 + np.arange(10, dtype=np.int64)
    b = (a + 3) * 10_000
    df = pd.DataFrame({"a": a, "b": b})
    assert pm._combine_overflows([a, b], "prod")
    exact = np.array([(int(x) * int(y)) % 10 for x, y in zip(a, b)], dtype=np.float64)
    np.testing.assert_array_equal(pm.apply_pairwise_modular(df, "prod", ["a", "b"], 10), exact)
    small = np.arange(10, dtype=np.int64)
    assert not pm._combine_overflows([small, small + 1], "prod")
    np.testing.assert_array_equal(pm._combine_residue_exact([a, b], "prod", 7), np.array([(int(x) * int(y)) % 7 for x, y in zip(a, b)]))
