"""Unit + biz_value coverage for ``mlframe.training.pipeline._cross_sectional_composite_fe``.

The underlying trick (``compute_cross_sectional_neighbor_features``) already has its own biz_value
test at the function level. This file covers the suite-wiring layer: no-op gate, schema alignment,
predict-time replay -- plus one biz_value test proving the WIRED module recovers a cross-sectional
outlier signal a raw-feature baseline can't.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from mlframe.training._preprocessing_configs import PreprocessingExtensionsConfig
from mlframe.training.pipeline._cross_sectional_composite_fe import apply_cross_sectional_composite_fe, replay_cross_sectional_composite_fe


def _snapshot_frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    snap = rng.integers(0, 40, n)
    df = pd.DataFrame({"time_id": snap, "f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    return df


def test_apply_cross_sectional_composite_fe_noop_when_snapshot_col_unset():
    df = _snapshot_frame()
    cfg = PreprocessingExtensionsConfig()
    train, val, test = apply_cross_sectional_composite_fe(df.iloc[:300], df.iloc[300:], None, cfg, {}, verbose=0)
    assert list(train.columns) == list(df.columns)


def test_apply_cross_sectional_composite_fe_schema_aligned_across_splits():
    df = _snapshot_frame()
    cfg = PreprocessingExtensionsConfig(
        cross_sectional_neighbors_snapshot_col="time_id",
        cross_sectional_neighbors_feature_cols=["f0", "f1"],
        cross_sectional_neighbors_k=5,
    )
    metadata: dict = {}
    train, val, test = apply_cross_sectional_composite_fe(df.iloc[:300], df.iloc[300:350], df.iloc[350:], cfg, metadata, verbose=0)
    assert set(train.columns) == set(val.columns) == set(test.columns)
    assert {"xsnn_f0_mean", "xsnn_f0_std", "xsnn_distance_ratio"} <= set(train.columns)
    assert train.shape[0] == 300
    assert metadata["cross_sectional_neighbors_snapshot_col"] == "time_id"


def test_apply_cross_sectional_composite_fe_polars_roundtrip():
    n = 200
    rng = np.random.default_rng(2)
    df = pl.DataFrame({"time_id": rng.integers(0, 20, n), "f0": rng.normal(size=n).astype(np.float32)})
    cfg = PreprocessingExtensionsConfig(
        cross_sectional_neighbors_snapshot_col="time_id", cross_sectional_neighbors_feature_cols=["f0"], cross_sectional_neighbors_k=3
    )
    train, _, _ = apply_cross_sectional_composite_fe(df, None, None, cfg, {}, verbose=0)
    assert isinstance(train, pl.DataFrame)
    assert "xsnn_distance_ratio" in train.columns


def test_replay_cross_sectional_composite_fe_matches_fit_time_columns():
    df = _snapshot_frame()
    cfg = PreprocessingExtensionsConfig(
        cross_sectional_neighbors_snapshot_col="time_id",
        cross_sectional_neighbors_feature_cols=["f0", "f1"],
        cross_sectional_neighbors_k=5,
    )
    metadata: dict = {}
    train, _, _ = apply_cross_sectional_composite_fe(df, None, None, cfg, metadata, verbose=0)

    fresh = df.iloc[:40][["time_id", "f0", "f1"]]
    replayed = replay_cross_sectional_composite_fe(fresh, metadata, verbose=0)
    assert set(replayed.columns) == set(train.columns)


def test_replay_cross_sectional_composite_fe_noop_without_persisted_metadata():
    df = _snapshot_frame(n=20)
    out = replay_cross_sectional_composite_fe(df, {}, verbose=0)
    assert list(out.columns) == list(df.columns)


def test_biz_val_cross_sectional_composite_wiring_isolation_signal():
    """A minority of snapshots are OUTLIERS (their feature vector sits far from every other
    snapshot's); the label is "is this row's snapshot an outlier". Raw per-row features carry no
    such signal (each row's own value is drawn from the SAME distribution as everyone else's,
    only the outlier SNAPSHOTS differ) -- only the wired distance-ratio feature (relative to the
    OTHER snapshots) can recover it."""
    rng = np.random.default_rng(3)
    n_snapshots = 80
    rows_per_snapshot = 15
    n = n_snapshots * rows_per_snapshot
    snapshot_ids = np.repeat(np.arange(n_snapshots), rows_per_snapshot)
    is_outlier_snapshot = np.zeros(n_snapshots, dtype=bool)
    outlier_snap_ids = rng.choice(n_snapshots, size=8, replace=False)
    is_outlier_snapshot[outlier_snap_ids] = True
    # Per-row noise (std=10) is much larger than the between-snapshot shift (6), so a SINGLE row's
    # raw features barely separate outlier from normal -- only averaging many rows of the same
    # snapshot (what the xsnn distance-ratio feature does internally) recovers the true separation.
    snap_center = rng.normal(size=(n_snapshots, 2))
    snap_center[outlier_snap_ids] += rng.normal(loc=6.0, scale=0.5, size=(8, 2))
    f = snap_center[snapshot_ids] + 10.0 * rng.normal(size=(n, 2))
    y = is_outlier_snapshot[snapshot_ids].astype(int)

    df = pd.DataFrame({"time_id": snapshot_ids, "f0": f[:, 0], "f1": f[:, 1]})
    cfg = PreprocessingExtensionsConfig(
        cross_sectional_neighbors_snapshot_col="time_id",
        cross_sectional_neighbors_feature_cols=["f0", "f1"],
        cross_sectional_neighbors_k=5,
    )
    out_df, _, _ = apply_cross_sectional_composite_fe(df, None, None, cfg, {}, verbose=0)

    train_idx, test_idx = train_test_split(np.arange(n), test_size=0.3, random_state=0, stratify=y)

    def _fit_eval(cols):
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(out_df.iloc[train_idx][cols], y[train_idx])
        proba = clf.predict_proba(out_df.iloc[test_idx][cols])[:, 1]
        return roc_auc_score(y[test_idx], proba)

    auc_raw = _fit_eval(["f0", "f1"])
    auc_wired = _fit_eval(["xsnn_distance_ratio"])

    assert auc_raw < 0.7, f"raw per-row features should carry little outlier signal, got {auc_raw:.3f}"
    assert auc_wired >= 0.95, f"wired distance-ratio feature should isolate outlier snapshots near-perfectly, got {auc_wired:.3f}"
