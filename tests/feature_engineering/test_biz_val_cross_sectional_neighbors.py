"""biz_value test for ``feature_engineering.compute_cross_sectional_neighbor_features``.

The win: snapshots (e.g. time_id-like cross-sections) belong to latent clusters, and the true row-level
target depends on the snapshot's cluster, not any individual row's own (noisy) feature values. Averaging a
snapshot's own rows already denoises somewhat, but the cross-SNAPSHOT nearest-neighbor mean (over other
snapshots from the same latent cluster) denoises further and should recover the cluster-driven target signal
materially better than raw per-row features alone -- mirroring the Optiver 3rd place's time_id-neighbor
feature block.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mlframe.feature_engineering import compute_cross_sectional_neighbor_features


def _make_snapshot_cluster_dataset(n_snapshots: int, rows_per_snap: int, n_clusters: int, seed: int):
    rng = np.random.default_rng(seed)
    cluster_centers = rng.normal(scale=6, size=(n_clusters, 3))
    cluster_targets = rng.uniform(10, 100, n_clusters)
    snapshot_cluster = rng.integers(0, n_clusters, n_snapshots)

    rows = []
    for s in range(n_snapshots):
        c = snapshot_cluster[s]
        base = cluster_centers[c] + rng.normal(scale=0.8, size=3)
        for _ in range(rows_per_snap):
            f = base + rng.normal(scale=3.5, size=3)
            y = cluster_targets[c] + rng.normal(scale=1.0)
            rows.append({"snap": s, "f0": f[0], "f1": f[1], "f2": f[2], "y": y})
    return pd.DataFrame(rows)


def test_biz_val_cross_sectional_neighbor_features_beats_raw_features_alone_mse():
    df = _make_snapshot_cluster_dataset(n_snapshots=200, rows_per_snap=8, n_clusters=5, seed=0)
    rng = np.random.default_rng(1)
    perm = rng.permutation(len(df))
    train_idx, test_idx = perm[:1200], perm[1200:]

    baseline = LinearRegression().fit(df.iloc[train_idx][["f0", "f1", "f2"]], df.iloc[train_idx]["y"])
    mse_baseline = mean_squared_error(df.iloc[test_idx]["y"], baseline.predict(df.iloc[test_idx][["f0", "f1", "f2"]]))

    neighbor_feats = compute_cross_sectional_neighbor_features(df, "snap", ["f0", "f1", "f2"], k=10, agg_stats=("mean",))
    df_augmented = pd.concat([df.reset_index(drop=True), neighbor_feats.to_pandas()], axis=1)
    feature_cols = ["f0", "f1", "f2", "xsnn_f0_mean", "xsnn_f1_mean", "xsnn_f2_mean"]

    augmented = LinearRegression().fit(df_augmented.iloc[train_idx][feature_cols], df_augmented.iloc[train_idx]["y"])
    mse_augmented = mean_squared_error(df_augmented.iloc[test_idx]["y"], augmented.predict(df_augmented.iloc[test_idx][feature_cols]))

    improvement = 1.0 - mse_augmented / mse_baseline
    assert improvement > 0.2, f"expected >20% MSE reduction from cross-sectional neighbor features, got {improvement:.4f} (baseline={mse_baseline:.2f}, augmented={mse_augmented:.2f})"


def test_cross_sectional_neighbor_features_output_shape_and_columns():
    df = _make_snapshot_cluster_dataset(n_snapshots=30, rows_per_snap=4, n_clusters=3, seed=2)
    result = compute_cross_sectional_neighbor_features(df, "snap", ["f0", "f1"], k=5, agg_stats=("mean", "std"))
    assert result.shape[0] == df.shape[0]
    assert set(result.columns) == {"xsnn_f0_mean", "xsnn_f0_std", "xsnn_f1_mean", "xsnn_f1_std", "xsnn_distance_ratio"}


def test_cross_sectional_neighbor_features_distance_ratio_in_unit_range():
    df = _make_snapshot_cluster_dataset(n_snapshots=50, rows_per_snap=4, n_clusters=4, seed=3)
    result = compute_cross_sectional_neighbor_features(df, "snap", ["f0", "f1", "f2"], k=8, agg_stats=("mean",))
    ratio = result["xsnn_distance_ratio"].to_numpy()
    assert np.isfinite(ratio).all()
    assert (ratio >= 0.0).all() and (ratio <= 1.0 + 1e-6).all()


def test_cross_sectional_neighbor_features_rows_within_same_snapshot_match():
    """Every row of the SAME snapshot must get the exact same broadcast neighbor-feature value."""
    df = _make_snapshot_cluster_dataset(n_snapshots=20, rows_per_snap=6, n_clusters=2, seed=4)
    result = compute_cross_sectional_neighbor_features(df, "snap", ["f0", "f1"], k=5, agg_stats=("mean",))
    check = pd.DataFrame({"snap": df["snap"].to_numpy(), "val": result["xsnn_f0_mean"].to_numpy()})
    assert (check.groupby("snap")["val"].nunique() == 1).all()
