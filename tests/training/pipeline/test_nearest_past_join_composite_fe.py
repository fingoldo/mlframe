"""Unit + biz_value coverage for ``mlframe.training.pipeline._nearest_past_join_composite_fe``.

The underlying trick (``nearest_past_join``) already has its own biz_value test at the function
level. This file covers the suite-wiring layer: the auxiliary-events-table contract, schema
alignment, no-op gates, and predict-time replay (no fit-time state -- inherently leak-safe by
construction) -- plus one biz_value test proving the wired module recovers a step-function signal a
naive stale/no-join baseline can't.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._nearest_past_join_composite_fe import apply_nearest_past_join_composite_fe, replay_nearest_past_join_composite_fe


def _left_right(n_entities=20, right_rows_per_entity=10, n_left=100, seed=0):
    rng = np.random.default_rng(seed)
    right = pd.DataFrame(
        {
            "entity": np.repeat(np.arange(n_entities), right_rows_per_entity),
            "t": np.tile(np.arange(right_rows_per_entity), n_entities),
            "known_value": rng.normal(size=n_entities * right_rows_per_entity),
        }
    )
    left = pd.DataFrame(
        {
            "entity": rng.integers(0, n_entities, n_left),
            "t": rng.integers(0, right_rows_per_entity, n_left).astype(float) + 0.5,
        }
    )
    return left, right


def test_apply_nearest_past_join_composite_fe_noop_when_on_unset():
    left, right = _left_right()
    cfg = PreprocessingExtensionsConfig()
    train, val, test = apply_nearest_past_join_composite_fe(left.iloc[:70], left.iloc[70:], None, cfg, right, verbose=0)
    assert list(train.columns) == list(left.columns)


def test_apply_nearest_past_join_composite_fe_noop_without_auxiliary_events_df():
    left, _right = _left_right()
    cfg = PreprocessingExtensionsConfig(nearest_past_join_on="t", nearest_past_join_by=["entity"])
    train, _, _ = apply_nearest_past_join_composite_fe(left, None, None, cfg, None, verbose=0)
    assert list(train.columns) == list(left.columns)


def test_apply_nearest_past_join_composite_fe_schema_aligned_across_splits():
    left, right = _left_right()
    cfg = PreprocessingExtensionsConfig(nearest_past_join_on="t", nearest_past_join_by=["entity"], nearest_past_join_value_cols=["known_value"])
    metadata: dict = {}
    train, val, test = apply_nearest_past_join_composite_fe(
        left.iloc[:70].reset_index(drop=True), left.iloc[70:85].reset_index(drop=True), left.iloc[85:].reset_index(drop=True),
        cfg, right, metadata=metadata, verbose=0,
    )
    assert set(train.columns) == set(val.columns) == set(test.columns)
    assert "known_value" in train.columns
    assert metadata["nearest_past_join_on"] == "t"
    assert metadata["nearest_past_join_by"] == ["entity"]


def test_apply_nearest_past_join_composite_fe_polars_roundtrip():
    left, right = _left_right(n_entities=10, right_rows_per_entity=5, n_left=50, seed=1)
    pl_left = pl.DataFrame(left)
    cfg = PreprocessingExtensionsConfig(nearest_past_join_on="t", nearest_past_join_by=["entity"])
    train, _, _ = apply_nearest_past_join_composite_fe(pl_left, None, None, cfg, right, metadata={}, verbose=0)
    assert isinstance(train, pl.DataFrame)
    assert "known_value" in train.columns


def test_replay_nearest_past_join_composite_fe_matches_fit_time_columns():
    left, right = _left_right()
    cfg = PreprocessingExtensionsConfig(nearest_past_join_on="t", nearest_past_join_by=["entity"], nearest_past_join_value_cols=["known_value"])
    metadata: dict = {}
    train, _, _ = apply_nearest_past_join_composite_fe(left, None, None, cfg, right, metadata=metadata, verbose=0)

    fresh = left.iloc[:10].reset_index(drop=True)
    replayed = replay_nearest_past_join_composite_fe(fresh, metadata, right, verbose=0)
    assert set(replayed.columns) == set(train.columns)


def test_replay_nearest_past_join_composite_fe_noop_without_persisted_metadata():
    left, _right = _left_right(n_left=10)
    out = replay_nearest_past_join_composite_fe(left, {}, _right, verbose=0)
    assert list(out.columns) == list(left.columns)


def test_biz_val_nearest_past_join_composite_wiring_recovers_step_function():
    """Each entity's true state undergoes a step-change at a per-entity random time; the target IS
    that state. A model using only the (entity, t) key with no join has no way to know the CURRENT
    value at an arbitrary query time; the wired nearest-past-value join attaches exactly the last
    known state as of each query time, letting a linear model recover the target near-perfectly."""
    rng = np.random.default_rng(2)
    n_entities = 30
    right_rows_per_entity = 20
    n_left = 400
    step_time = rng.integers(3, 17, n_entities)
    level_before = rng.normal(size=n_entities)
    level_after = level_before + rng.normal(loc=5.0, scale=0.3, size=n_entities)

    right_entity = np.repeat(np.arange(n_entities), right_rows_per_entity)
    right_t = np.tile(np.arange(right_rows_per_entity), n_entities)
    right_val = np.where(right_t < step_time[right_entity], level_before[right_entity], level_after[right_entity])
    right = pd.DataFrame({"entity": right_entity, "t": right_t, "known_value": right_val})

    left_entity = rng.integers(0, n_entities, n_left)
    left_t = rng.integers(1, right_rows_per_entity, n_left).astype(float) + 0.5
    target = np.where(np.floor(left_t) < step_time[left_entity], level_before[left_entity], level_after[left_entity])
    left = pd.DataFrame({"entity": left_entity, "t": left_t})

    cfg = PreprocessingExtensionsConfig(nearest_past_join_on="t", nearest_past_join_by=["entity"], nearest_past_join_value_cols=["known_value"])
    out_df, _, _ = apply_nearest_past_join_composite_fe(left, None, None, cfg, right, verbose=0)

    reg_wired = LinearRegression().fit(out_df[["known_value"]], target)
    mse_wired = mean_squared_error(target, reg_wired.predict(out_df[["known_value"]]))

    reg_naive = LinearRegression().fit(left[["entity"]].astype(float), target)
    mse_naive = mean_squared_error(target, reg_naive.predict(left[["entity"]].astype(float)))

    assert mse_wired < mse_naive * 0.1, (
        f"wired nearest-past join should recover the step-function target far better than a raw "
        f"entity-id-only baseline, got mse_wired={mse_wired:.4f} vs mse_naive={mse_naive:.4f}"
    )
