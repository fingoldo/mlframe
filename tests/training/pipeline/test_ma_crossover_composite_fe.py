"""Unit + biz_value coverage for ``mlframe.training.pipeline._ma_crossover_composite_fe``.

The underlying trick (``ma_crossover_features``) already has its own biz_value test at the function
level, given precomputed moving averages. This file covers what's NEW here: computing those moving
averages (per-entity if group_ids is available, else global) before feeding them to the function --
plus one biz_value test proving the WIRED module (which builds the MAs itself) recovers a
trend-reversal signal a raw single-window MA can't.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._ma_crossover_composite_fe import apply_ma_crossover_composite_fe, replay_ma_crossover_composite_fe


def _series_frame(n=300, n_groups=10, seed=0):
    rng = np.random.default_rng(seed)
    group_ids = rng.integers(0, n_groups, n)
    ts = np.arange(n).astype(np.float64)
    val = rng.normal(size=n).cumsum()
    df = pd.DataFrame({"val_col": val})
    return df, group_ids, ts


def test_apply_ma_crossover_composite_fe_noop_when_columns_unset():
    df, group_ids, ts = _series_frame()
    cfg = PreprocessingExtensionsConfig()
    train, val, test = apply_ma_crossover_composite_fe(
        df.iloc[:200], df.iloc[200:], None, cfg, group_ids, ts, np.arange(200), np.arange(200, 300), None, verbose=0
    )
    assert list(train.columns) == list(df.columns)


def test_apply_ma_crossover_composite_fe_noop_with_single_window():
    df, group_ids, ts = _series_frame()
    cfg = PreprocessingExtensionsConfig(ma_crossover_columns=["val_col"], ma_crossover_windows=[5])
    train, _, _ = apply_ma_crossover_composite_fe(df, None, None, cfg, group_ids, ts, np.arange(len(df)), None, None, verbose=0)
    assert list(train.columns) == list(df.columns)


def test_apply_ma_crossover_composite_fe_schema_aligned_across_splits():
    df, group_ids, ts = _series_frame()
    train_idx, val_idx, test_idx = np.arange(0, 200), np.arange(200, 250), np.arange(250, 300)
    cfg = PreprocessingExtensionsConfig(ma_crossover_columns=["val_col"], ma_crossover_windows=[3, 5, 10])
    metadata: dict = {}
    train, val, test = apply_ma_crossover_composite_fe(
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
    assert "val_col_ma_crossover_vote_sum" in train.columns
    assert metadata["ma_crossover_windows"] == [3, 5, 10]


def test_apply_ma_crossover_composite_fe_works_without_group_ids():
    df, _, ts = _series_frame()
    cfg = PreprocessingExtensionsConfig(ma_crossover_columns=["val_col"], ma_crossover_windows=[3, 5])
    train, _, _ = apply_ma_crossover_composite_fe(df, None, None, cfg, None, ts, np.arange(len(df)), None, None, verbose=0)
    assert "val_col_ma_crossover_diff_3_5" in train.columns


def test_apply_ma_crossover_composite_fe_polars_roundtrip():
    n = 200
    rng = np.random.default_rng(2)
    df = pl.DataFrame({"val_col": rng.normal(size=n).astype(np.float32).cumsum()})
    cfg = PreprocessingExtensionsConfig(ma_crossover_columns=["val_col"], ma_crossover_windows=[3, 5])
    train, _, _ = apply_ma_crossover_composite_fe(df, None, None, cfg, None, None, np.arange(n), None, None, metadata={}, verbose=0)
    assert isinstance(train, pl.DataFrame)
    assert "val_col_ma_crossover_diff_3_5" in train.columns


def test_replay_ma_crossover_composite_fe_matches_fit_time_columns():
    df, group_ids, ts = _series_frame()
    cfg = PreprocessingExtensionsConfig(ma_crossover_columns=["val_col"], ma_crossover_windows=[3, 5])
    metadata: dict = {}
    train, _, _ = apply_ma_crossover_composite_fe(df, None, None, cfg, group_ids, ts, np.arange(len(df)), None, None, metadata=metadata, verbose=0)

    fresh_idx = np.arange(0, 40)
    fresh = df.iloc[fresh_idx][["val_col"]].reset_index(drop=True)
    replayed = replay_ma_crossover_composite_fe(fresh, metadata, group_ids[fresh_idx], ts[fresh_idx], verbose=0)
    assert set(replayed.columns) == set(train.columns)


def test_replay_ma_crossover_composite_fe_noop_without_persisted_metadata():
    df, _, _ = _series_frame(n=20)
    out = replay_ma_crossover_composite_fe(df, {}, None, None, verbose=0)
    assert list(out.columns) == list(df.columns)


def test_biz_val_ma_crossover_composite_wiring_detects_trend_reversal():
    """Each entity's series has a trend-reversal point (uptrend -> downtrend) at a random location;
    the label is "has this entity's series already reversed by the CURRENT row". A raw single-value
    feature (the series' own level) carries little of this regime information (level alone doesn't
    reveal direction), but the wired vote_sum crossover feature (computed from MAs this module
    builds itself) tracks the current trend direction directly."""
    rng = np.random.default_rng(3)
    n_groups = 40
    rows_per_group = 60
    n = n_groups * rows_per_group
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    within_pos = np.tile(np.arange(rows_per_group), n_groups)
    reversal_point = rng.integers(15, 45, n_groups)
    slope = np.where(within_pos < reversal_point[group_ids], 1.0, -1.0)
    val = np.cumsum(slope + 0.3 * rng.normal(size=n))
    y = (within_pos >= reversal_point[group_ids]).astype(int)

    df = pd.DataFrame({"val_col": val})
    cfg = PreprocessingExtensionsConfig(ma_crossover_columns=["val_col"], ma_crossover_windows=[3, 8, 15])
    out_df, _, _ = apply_ma_crossover_composite_fe(df, None, None, cfg, group_ids, within_pos.astype(np.float64), np.arange(n), None, None, verbose=0)

    def _auc(cols):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(out_df[cols], y)
        return roc_auc_score(y, clf.predict_proba(out_df[cols])[:, 1])

    auc_raw = _auc(["val_col"])
    auc_wired = _auc(["val_col_ma_crossover_vote_sum"])

    assert auc_wired > auc_raw + 0.15, (
        f"wired MA-crossover vote_sum should track trend reversal far better than the raw level, got auc_wired={auc_wired:.3f} vs auc_raw={auc_raw:.3f}"
    )
