"""Regression (MRMR FE critique P1): apply_temporal_expanding / apply_temporal_lag replay seeded the accumulator
with the entity's ENTIRE train history regardless of the test row's timestamp, so a test row falling INSIDE the
train time range saw FUTURE train values -- a look-ahead leak / train-serve skew (apply_temporal_rolling already
did the strict ``t' < t`` merge correctly). The replay now merges train history by time via a per-entity pointer.
"""
import numpy as np
import pandas as pd

from mlframe.feature_selection.filters._temporal_agg_fe import (
    generate_lag_features,
    generate_expanding_agg_features,
    apply_temporal_lag,
    apply_temporal_expanding,
)


def _train():
    # one entity, train values at times 1..4
    return pd.DataFrame({"entity": ["u"] * 4, "tcol": [1, 2, 3, 4], "x0": [10.0, 20.0, 30.0, 40.0]})


def test_expanding_replay_excludes_future_train_rows():
    Xtr = _train()
    _enc, rec = generate_expanding_agg_features(Xtr, ["entity"], ["x0"], "tcol", stats=("mean", "count"))
    # a test row at t=2.5 must see ONLY train rows at t<2.5 -> {10, 20}
    Xte = pd.DataFrame({"entity": ["u"], "tcol": [2.5], "x0": [99.0]})
    mean_recipe = next(e for name, e in rec.items() if e["stat"] == "mean")
    cnt_recipe = next(e for name, e in rec.items() if e["stat"] == "count")
    mean_val = apply_temporal_expanding(Xte, mean_recipe)[0]
    cnt_val = apply_temporal_expanding(Xte, cnt_recipe)[0]
    assert cnt_val == 2.0, f"expanding count leaked future rows: {cnt_val} (expected 2)"
    assert abs(mean_val - 15.0) < 1e-9, f"expanding mean leaked future rows: {mean_val} (expected 15 = mean(10,20))"


def test_lag_replay_excludes_future_train_rows():
    Xtr = _train()
    _enc, rec = generate_lag_features(Xtr, ["entity"], ["x0"], "tcol", lags=(1,))
    Xte = pd.DataFrame({"entity": ["u"], "tcol": [2.5], "x0": [99.0]})
    extra = next(iter(rec.values()))
    lag_val = apply_temporal_lag(Xte, extra)[0]
    # strict-past buffer at t=2.5 is [10, 20]; lag-1 = last = 20 (t=2 value), NOT 40 (t=4 future)
    assert abs(lag_val - 20.0) < 1e-9, f"lag leaked a future train value: {lag_val} (expected 20)"


def test_replay_unchanged_when_test_strictly_after_train():
    # backward-compat: with every test row AFTER all train rows, the merge yields the same values as before.
    Xtr = _train()
    _enc, rec = generate_expanding_agg_features(Xtr, ["entity"], ["x0"], "tcol", stats=("mean",))
    Xte = pd.DataFrame({"entity": ["u", "u"], "tcol": [5, 6], "x0": [50.0, 60.0]})
    extra = next(iter(rec.values()))
    out = apply_temporal_expanding(Xte, extra)
    # first test row (t=5) sees all 4 train rows -> mean(10,20,30,40)=25; second (t=6) sees train+first test.
    assert abs(out[0] - 25.0) < 1e-9, f"post-train expanding mean changed: {out[0]}"
    assert abs(out[1] - (10 + 20 + 30 + 40 + 50) / 5) < 1e-9, f"within-test accumulation broke: {out[1]}"


def test_expanding_datetime_time_axis_no_lookahead():
    Xtr = pd.DataFrame({
        "entity": ["u"] * 3,
        "tcol": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
        "x0": [1.0, 2.0, 3.0],
    })
    _enc, rec = generate_expanding_agg_features(Xtr, ["entity"], ["x0"], "tcol", stats=("count",))
    Xte = pd.DataFrame({"entity": ["u"], "tcol": pd.to_datetime(["2024-01-02"]), "x0": [9.0]})
    extra = next(iter(rec.values()))
    cnt = apply_temporal_expanding(Xte, extra)[0]
    assert cnt == 1.0, f"datetime expanding leaked future rows: {cnt} (only 2024-01-01 precedes 2024-01-02)"
