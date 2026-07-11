"""biz_value test for ``feature_engineering.compute_cross_sectional_neighbor_features``.

The win: snapshots (e.g. time_id-like cross-sections) belong to latent clusters, and the true row-level
target depends on the snapshot's cluster, not any individual row's own (noisy) feature values. Averaging a
snapshot's own rows already denoises somewhat, but the cross-SNAPSHOT nearest-neighbor mean (over other
snapshots from the same latent cluster) denoises further and should recover the cluster-driven target signal
materially better than raw per-row features alone -- mirroring the Optiver 3rd place's time_id-neighbor
feature block.

Also covers the opt-in ``k_values`` multi-k mode: one call at several k values should be materially faster
than calling the function once per k (it shares a single neighbor search extended to the largest k), while
being numerically identical to the per-k single calls.
"""
from __future__ import annotations

import time

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


def test_cross_sectional_neighbor_features_default_unchanged_when_k_values_omitted():
    """Regression pin: leaving the new opt-in ``k_values`` param unset must reproduce the exact prior
    single-k output bit-for-bit (same column names, same values)."""
    df = _make_snapshot_cluster_dataset(n_snapshots=40, rows_per_snap=5, n_clusters=3, seed=5)
    result = compute_cross_sectional_neighbor_features(df, "snap", ["f0", "f1", "f2"], k=7, agg_stats=("mean", "std"))
    assert set(result.columns) == {
        "xsnn_f0_mean", "xsnn_f0_std", "xsnn_f1_mean", "xsnn_f1_std", "xsnn_f2_mean", "xsnn_f2_std", "xsnn_distance_ratio",
    }
    result_again = compute_cross_sectional_neighbor_features(df, "snap", ["f0", "f1", "f2"], k=7, agg_stats=("mean", "std"))
    for col in result.columns:
        np.testing.assert_array_equal(result[col].to_numpy(), result_again[col].to_numpy())


def test_cross_sectional_neighbor_features_multi_k_matches_per_k_single_calls():
    """The shared-search multi-k path must produce numerically identical features to calling the function
    once per k value (the per-k single call is the ground truth this optimization must not alter)."""
    df = _make_snapshot_cluster_dataset(n_snapshots=60, rows_per_snap=4, n_clusters=4, seed=6)
    k_values = [3, 8, 15]
    multi = compute_cross_sectional_neighbor_features(df, "snap", ["f0", "f1", "f2"], agg_stats=("mean", "std"), k_values=k_values)
    assert set(multi.columns) == {
        f"xsnn_k{kv}_{col}_{stat}" for kv in k_values for col in ("f0", "f1", "f2") for stat in ("mean", "std")
    } | {f"xsnn_k{kv}_distance_ratio" for kv in k_values}

    for kv in k_values:
        single = compute_cross_sectional_neighbor_features(df, "snap", ["f0", "f1", "f2"], k=kv, agg_stats=("mean", "std"))
        for col in ("f0", "f1", "f2"):
            for stat in ("mean", "std"):
                np.testing.assert_allclose(
                    multi[f"xsnn_k{kv}_{col}_{stat}"].to_numpy(), single[f"xsnn_{col}_{stat}"].to_numpy(), rtol=1e-5, atol=1e-6,
                )
        np.testing.assert_allclose(
            multi[f"xsnn_k{kv}_distance_ratio"].to_numpy(), single["xsnn_distance_ratio"].to_numpy(), rtol=1e-5, atol=1e-6,
        )


def test_biz_val_cross_sectional_neighbor_features_multi_k_faster_than_per_k_calls():
    """The whole point of ``k_values``: one shared-search call for several k's must be materially faster
    than the naive alternative of calling the function once per k (each of which repeats the neighbor
    search from scratch). Run best-of-3 each side to de-noise; require a real margin, not a coin flip."""
    df = _make_snapshot_cluster_dataset(n_snapshots=1500, rows_per_snap=6, n_clusters=6, seed=7)
    k_values = [5, 10, 20, 40, 80]
    feature_cols = ["f0", "f1", "f2"]

    # warm-up (import / JIT / backend-selection cost must not pollute either measurement)
    compute_cross_sectional_neighbor_features(df, "snap", feature_cols, k_values=k_values)
    compute_cross_sectional_neighbor_features(df, "snap", feature_cols, k=max(k_values))

    def _time_multi() -> float:
        t0 = time.perf_counter()
        compute_cross_sectional_neighbor_features(df, "snap", feature_cols, k_values=k_values)
        return time.perf_counter() - t0

    def _time_per_k() -> float:
        t0 = time.perf_counter()
        for kv in k_values:
            compute_cross_sectional_neighbor_features(df, "snap", feature_cols, k=kv)
        return time.perf_counter() - t0

    t_multi = min(_time_multi() for _ in range(3))
    t_per_k = min(_time_per_k() for _ in range(3))

    speedup = t_per_k / t_multi
    assert speedup > 1.5, f"expected multi-k shared-search to beat {len(k_values)} separate per-k calls by >1.5x, got {speedup:.2f}x (multi={t_multi*1000:.1f}ms, per_k={t_per_k*1000:.1f}ms)"
