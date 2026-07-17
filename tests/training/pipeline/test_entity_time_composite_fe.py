"""Unit + biz_value coverage for ``mlframe.training.pipeline._entity_time_composite_fe``.

The underlying tricks (``time_since_state_change`` / ``per_group_recency_weighted_agg``) already
have their own biz_value tests at the function level. This file covers the suite-wiring layer:
schema alignment across train/val/test, group_ids/timestamps slicing by train_idx/val_idx/test_idx,
the no-op gate when group_ids is unavailable, and predict-time replay (no fit-time state -- pure
function of the predict frame's own group_ids/timestamps, persisted config only) -- plus one
biz_value test proving the WIRED module recovers signal a raw-column baseline can't.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_squared_error

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._entity_time_composite_fe import apply_entity_time_composite_fe, replay_entity_time_composite_fe


def _entity_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    group_ids = rng.integers(0, 20, n)
    ts = np.arange(n).astype(np.float64)
    state = rng.random(n) < 0.3
    val = rng.normal(size=n).astype(np.float32)
    df = pd.DataFrame({"state_col": state, "val_col": val})
    return df, group_ids, ts


def test_apply_entity_time_composite_fe_noop_when_group_ids_none():
    df, _, ts = _entity_frame()
    cfg = PreprocessingExtensionsConfig(state_duration_columns=["state_col"], recency_aggregation_columns=["val_col"])
    train, _val, _test = apply_entity_time_composite_fe(
        df.iloc[:200],
        df.iloc[200:],
        None,
        cfg,
        None,
        ts,
        np.arange(200),
        np.arange(200, 300),
        None,
        verbose=0,
    )
    assert list(train.columns) == list(df.columns)


def test_apply_entity_time_composite_fe_noop_when_no_columns_declared():
    df, group_ids, ts = _entity_frame()
    cfg = PreprocessingExtensionsConfig()
    train, _, _ = apply_entity_time_composite_fe(df, None, None, cfg, group_ids, ts, np.arange(len(df)), None, None, verbose=0)
    assert list(train.columns) == list(df.columns)


def test_apply_entity_time_composite_fe_schema_aligned_across_splits():
    df, group_ids, ts = _entity_frame()
    cfg = PreprocessingExtensionsConfig(state_duration_columns=["state_col"], recency_aggregation_columns=["val_col"])
    train_idx, val_idx, test_idx = np.arange(0, 200), np.arange(200, 250), np.arange(250, 300)
    metadata: dict = {}
    train, val, test = apply_entity_time_composite_fe(
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
        cfg,
        group_ids,
        ts,
        train_idx,
        val_idx,
        test_idx,
        metadata=metadata,
        verbose=0,
    )
    assert set(train.columns) == set(val.columns) == set(test.columns)
    expected_new = {"state_col__possession_duration", "state_col__cancellation_duration", "val_col__recency_mean"}
    assert expected_new <= set(train.columns)
    assert metadata["state_duration_columns"] == ["state_col"]
    assert metadata["recency_aggregation_columns"] == ["val_col"]


def test_apply_entity_time_composite_fe_polars_roundtrip():
    n = 200
    rng = np.random.default_rng(2)
    group_ids = rng.integers(0, 10, n)
    ts = np.arange(n).astype(np.float64)
    df = pl.DataFrame({"state_col": rng.random(n) < 0.4, "val_col": rng.normal(size=n).astype(np.float32)})
    cfg = PreprocessingExtensionsConfig(state_duration_columns=["state_col"], recency_aggregation_columns=["val_col"])
    train, _, _ = apply_entity_time_composite_fe(df, None, None, cfg, group_ids, ts, np.arange(n), None, None, verbose=0)
    assert isinstance(train, pl.DataFrame)
    assert "val_col__recency_mean" in train.columns


def test_replay_entity_time_composite_fe_matches_fit_time_columns():
    df, group_ids, ts = _entity_frame()
    cfg = PreprocessingExtensionsConfig(state_duration_columns=["state_col"], recency_aggregation_columns=["val_col"])
    metadata: dict = {}
    train, _, _ = apply_entity_time_composite_fe(df, None, None, cfg, group_ids, ts, np.arange(len(df)), None, None, metadata=metadata, verbose=0)

    fresh_idx = np.arange(0, 50)
    fresh = df.iloc[fresh_idx][["state_col", "val_col"]].reset_index(drop=True)
    replayed = replay_entity_time_composite_fe(fresh, metadata, group_ids[fresh_idx], ts[fresh_idx], verbose=0)
    assert set(replayed.columns) == set(train.columns)


def test_replay_entity_time_composite_fe_noop_without_group_ids():
    df, _, _ = _entity_frame(n=20)
    out = replay_entity_time_composite_fe(df, {"state_duration_columns": ["state_col"]}, None, None, verbose=0)
    assert list(out.columns) == list(df.columns)


def test_biz_val_entity_time_composite_wiring_recency_beats_unweighted_mean():
    """A per-entity value undergoes a REGIME SHIFT partway through its history (recent observations
    carry the true current level; older ones reflect a stale, now-wrong level). A plain per-entity
    mean averages both regimes together; the wired recency-weighted aggregation (via this module's
    own apply/replay API) should track the CURRENT level far more closely. Threshold set from
    measured values with headroom on both sides."""
    rng = np.random.default_rng(3)
    n_groups = 60
    rows_per_group = 20
    n = n_groups * rows_per_group
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    ts = np.tile(np.arange(rows_per_group), n_groups).astype(np.float64)
    # each entity's true level jumps at the halfway point; only the LAST observation's true level matters
    # for "what is this entity's target right now" -- an unweighted mean blends both regimes.
    level_before = rng.normal(size=n_groups)
    level_after = level_before + rng.normal(loc=4.0, scale=0.5, size=n_groups)
    within_group_pos = np.tile(np.arange(rows_per_group), n_groups)
    true_level = np.where(within_group_pos < rows_per_group // 2, level_before[group_ids], level_after[group_ids])
    val = true_level + 0.2 * rng.normal(size=n)
    # target: the entity's CURRENT (post-shift) level, same for every row of that entity.
    target = level_after[group_ids]

    df = pd.DataFrame({"val_col": val})
    cfg = PreprocessingExtensionsConfig(recency_aggregation_columns=["val_col"], recency_aggregation_param=3.0)
    metadata: dict = {}
    out_df, _, _ = apply_entity_time_composite_fe(df, None, None, cfg, group_ids, ts, np.arange(n), None, None, metadata=metadata, verbose=0)

    unweighted_mean = pd.Series(val).groupby(group_ids).transform("mean").to_numpy()
    recency_mean = out_df["val_col__recency_mean"].to_numpy()

    rmse_unweighted = float(np.sqrt(mean_squared_error(target, unweighted_mean)))
    rmse_recency = float(np.sqrt(mean_squared_error(target, recency_mean)))

    assert rmse_recency < rmse_unweighted * 0.5, (
        f"recency-weighted aggregation should track the post-shift level far better than a plain mean, "
        f"got rmse_recency={rmse_recency:.3f} vs rmse_unweighted={rmse_unweighted:.3f}"
    )
