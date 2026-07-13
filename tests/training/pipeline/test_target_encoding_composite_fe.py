"""Unit + biz_value coverage for ``mlframe.training.pipeline._target_encoding_composite_fe``.

Unlike the other composite-FE suite-wiring modules, this one has real FIT-TIME STATE: an
entity-lookup table built on train-only data, since predict time has no target to encode from. This
file checks the leakage-safety contract directly (train rows use the causal expanding-window
encoding; val/test/predict rows use ONLY the train-fitted lookup, with a global-prior fallback for
unseen entities) plus a biz_value test proving the wired encoding recovers per-entity target signal
a raw-categorical baseline can't.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._target_encoding_composite_fe import apply_target_encoding_composite_fe, replay_target_encoding_composite_fe


def _entity_target_frame(n=400, n_entities=30, seed=0):
    rng = np.random.default_rng(seed)
    group_ids = rng.integers(0, n_entities, n)
    ts = np.arange(n).astype(np.float64)
    cat_col = rng.choice(list("ABC"), n)
    entity_effect = rng.normal(size=n_entities)
    y = entity_effect[group_ids] + 0.2 * rng.normal(size=n)
    df = pd.DataFrame({"cat_col": cat_col})
    return df, group_ids, ts, y, entity_effect


def test_apply_target_encoding_composite_fe_noop_when_columns_unset():
    df, group_ids, ts, y, _ = _entity_target_frame()
    cfg = PreprocessingExtensionsConfig()
    train, val, test = apply_target_encoding_composite_fe(
        df.iloc[:300], df.iloc[300:], None, cfg, group_ids, ts, y[:300], np.arange(300), np.arange(300, 400), None, verbose=0,
    )
    assert list(train.columns) == list(df.columns)


def test_apply_target_encoding_composite_fe_noop_without_group_ids_or_y():
    df, _, ts, y, _ = _entity_target_frame()
    cfg = PreprocessingExtensionsConfig(two_step_target_encode_columns=["cat_col"])
    train, _, _ = apply_target_encoding_composite_fe(df, None, None, cfg, None, ts, y, np.arange(len(df)), None, None, verbose=0)
    assert list(train.columns) == list(df.columns)
    train2, _, _ = apply_target_encoding_composite_fe(df, None, None, cfg, np.zeros(len(df)), ts, None, np.arange(len(df)), None, None, verbose=0)
    assert list(train2.columns) == list(df.columns)


def test_apply_target_encoding_composite_fe_builds_entity_lookup():
    df, group_ids, ts, y, _ = _entity_target_frame()
    train_idx, val_idx, test_idx = np.arange(0, 300), np.arange(300, 350), np.arange(350, 400)
    cfg = PreprocessingExtensionsConfig(two_step_target_encode_columns=["cat_col"], two_step_target_encode_decay_half_life=20.0)
    metadata: dict = {}
    train, val, test = apply_target_encoding_composite_fe(
        df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True),
        cfg, group_ids, ts, y[train_idx], train_idx, val_idx, test_idx, metadata=metadata, verbose=0,
    )
    out_col = "cat_col__two_step_target_encode"
    assert out_col in train.columns and out_col in val.columns and out_col in test.columns
    assert metadata["two_step_target_encode_columns"] == ["cat_col"]
    assert len(metadata["two_step_target_encode_entity_lookup"]) <= 30


def test_apply_target_encoding_composite_fe_polars_roundtrip():
    df, group_ids, ts, y, _ = _entity_target_frame(n=200, n_entities=15, seed=1)
    pl_df = pl.DataFrame({"cat_col": df["cat_col"].to_numpy()})
    cfg = PreprocessingExtensionsConfig(two_step_target_encode_columns=["cat_col"])
    train, _, _ = apply_target_encoding_composite_fe(pl_df[:150], None, None, cfg, group_ids, ts, y[:150], np.arange(150), None, None, metadata={}, verbose=0)
    assert isinstance(train, pl.DataFrame)
    assert "cat_col__two_step_target_encode" in train.columns


def test_replay_target_encoding_composite_fe_uses_train_only_lookup_no_target_needed():
    """The defining leakage-safety contract: replay must work with ZERO target information -- only
    group_ids and the train-fitted lookup table."""
    df, group_ids, ts, y, _ = _entity_target_frame()
    train_idx = np.arange(0, 300)
    cfg = PreprocessingExtensionsConfig(two_step_target_encode_columns=["cat_col"])
    metadata: dict = {}
    apply_target_encoding_composite_fe(
        df.iloc[train_idx].reset_index(drop=True), None, None, cfg, group_ids, ts, y[train_idx], train_idx, None, None, metadata=metadata, verbose=0,
    )
    fresh_idx = np.arange(300, 320)  # entities seen in train (group_ids range 0-29, all present in 300 train rows w/ 30 entities)
    fresh = df.iloc[fresh_idx][["cat_col"]].reset_index(drop=True)
    replayed = replay_target_encoding_composite_fe(fresh, metadata, group_ids[fresh_idx], verbose=0)
    assert "cat_col__two_step_target_encode" in replayed.columns
    assert replayed["cat_col__two_step_target_encode"].notna().all()


def test_replay_target_encoding_composite_fe_unseen_entity_falls_back_to_global_prior():
    df, group_ids, ts, y, _ = _entity_target_frame(n=200, n_entities=10, seed=2)
    train_idx = np.arange(0, 200)
    cfg = PreprocessingExtensionsConfig(two_step_target_encode_columns=["cat_col"])
    metadata: dict = {}
    apply_target_encoding_composite_fe(df, None, None, cfg, group_ids, ts, y, train_idx, None, None, metadata=metadata, verbose=0)

    unseen_entity_id = 999999  # guaranteed absent from the 0..9 training entity range
    fresh = pd.DataFrame({"cat_col": ["A"]})
    replayed = replay_target_encoding_composite_fe(fresh, metadata, np.array([unseen_entity_id]), verbose=0)
    out_col = "cat_col__two_step_target_encode"
    assert np.isclose(replayed[out_col].iloc[0], metadata["two_step_target_encode_global_prior"])


def test_biz_val_target_encoding_composite_wiring_recovers_entity_signal():
    """Each entity has a hidden, fixed ``cat_col`` propensity that drives the target, but any SINGLE
    row's observed ``cat_col`` is a noisy realization of it (70% matches, 30% random) -- a raw
    one-hot on one row is diluted by that noise. The wired encoding's step-2 aggregation averages
    the leak-free step-1 encoding across an entity's OTHER events too, denoising the per-row noise
    and recovering the true propensity's effect far better than a single noisy row's raw category."""
    rng = np.random.default_rng(3)
    n_entities, rows_per_entity = 60, 50
    n = n_entities * rows_per_entity
    group_ids = np.repeat(np.arange(n_entities), rows_per_entity)
    ts = np.tile(np.arange(rows_per_entity), n_entities).astype(np.float64)
    levels = np.array(list("ABC"))
    true_propensity_idx = rng.integers(0, 3, n_entities)
    cat_effect = {0: 3.0, 1: -3.0, 2: 0.0}
    matches_propensity = rng.random(n) < 0.7
    observed_idx = np.where(matches_propensity, true_propensity_idx[group_ids], rng.integers(0, 3, n))
    cat_col = levels[observed_idx]
    y = np.array([cat_effect[i] for i in true_propensity_idx[group_ids]]) + 0.5 * rng.normal(size=n)
    df = pd.DataFrame({"cat_col": cat_col})

    perm = rng.permutation(n)
    df = df.iloc[perm].reset_index(drop=True)
    group_ids, ts, y = group_ids[perm], ts[perm], y[perm]

    n_train = int(n * 0.7)
    train_idx, test_idx = np.arange(0, n_train), np.arange(n_train, n)
    cfg = PreprocessingExtensionsConfig(two_step_target_encode_columns=["cat_col"], two_step_target_encode_decay_half_life=1000.0)
    metadata: dict = {}
    train_out, _, test_out = apply_target_encoding_composite_fe(
        df.iloc[train_idx].reset_index(drop=True), None, df.iloc[test_idx].reset_index(drop=True),
        cfg, group_ids, ts, y[train_idx], train_idx, None, test_idx, metadata=metadata, verbose=0,
    )

    enc = OneHotEncoder(handle_unknown="ignore")
    X_train_raw = enc.fit_transform(df.iloc[train_idx][["cat_col"]])
    X_test_raw = enc.transform(df.iloc[test_idx][["cat_col"]])
    reg_raw = LinearRegression().fit(X_train_raw, y[train_idx])
    mse_raw = mean_squared_error(y[test_idx], reg_raw.predict(X_test_raw))

    out_col = "cat_col__two_step_target_encode"
    reg_wired = LinearRegression().fit(train_out[[out_col]], y[train_idx])
    mse_wired = mean_squared_error(y[test_idx], reg_wired.predict(test_out[[out_col]]))

    assert mse_wired < mse_raw * 0.5, (
        f"wired two-step target encoding should recover the per-entity effect far better than raw "
        f"cat_col one-hot (structurally uninformative here), got mse_wired={mse_wired:.4f} vs mse_raw={mse_raw:.4f}"
    )
