"""biz_value test for ``feature_engineering.transformer.compute_neighbor_aggregate_features``.

The win: when rows cluster tightly in a small "strong feature" subset and the target's mean varies by
cluster (the Home Credit ``neighbors_target_mean_500`` / Optiver stock-id-similarity pattern), the OOF
k-nearest-neighbor mean of the target should recover each row's true cluster mean far better than a single
global-mean baseline. Also verifies Mode A (OOF, X_query=None) never lets a row see its own target value.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold

from mlframe.feature_engineering.transformer import compute_neighbor_aggregate_features


def _make_clustered_dataset(n: int, n_clusters: int, seed: int):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=5, size=(n_clusters, 2))
    cluster_means = rng.uniform(10, 100, n_clusters)
    cluster_id = rng.integers(0, n_clusters, n)
    X = centers[cluster_id] + rng.normal(scale=0.5, size=(n, 2))
    y = cluster_means[cluster_id] + rng.normal(scale=2, size=n)
    return X, y, cluster_id, cluster_means


def test_biz_val_neighbor_aggregate_features_beats_global_mean_baseline():
    X, y, _, _ = _make_clustered_dataset(n=3000, n_clusters=20, seed=0)
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)

    feats = compute_neighbor_aggregate_features(
        X, {"y": y}, X_query=None, splitter=splitter, seed=0, k_values=(10, 20), stats=("mean",),
    )
    nbr_pred = feats["nbr_y_k20_mean"].to_numpy()

    mse_neighbor = float(np.mean((nbr_pred - y) ** 2))
    mse_global = float(np.mean((y.mean() - y) ** 2))
    improvement = 1.0 - mse_neighbor / mse_global

    assert improvement > 0.6, f"expected >60% MSE reduction vs. global-mean baseline, got {improvement:.4f} (neighbor={mse_neighbor:.2f}, global={mse_global:.2f})"


def test_neighbor_aggregate_features_multiple_agg_columns_and_stats():
    X, y, _, _ = _make_clustered_dataset(n=500, n_clusters=5, seed=1)
    second_col = y * 2 + 1
    splitter = KFold(n_splits=3, shuffle=True, random_state=1)

    feats = compute_neighbor_aggregate_features(
        X, {"y": y, "y2": second_col}, X_query=None, splitter=splitter, seed=1, k_values=(5, 10), stats=("mean", "std"),
    )
    expected_cols = {f"nbr_{col}_k{k}_{stat}" for col in ("y", "y2") for k in (5, 10) for stat in ("mean", "std")}
    assert set(feats.columns) == expected_cols
    assert feats.shape[0] == 500


def test_neighbor_aggregate_features_mode_a_never_leaks_own_row():
    """A held-out row's neighbor-mean must be computable from train-fold rows only (no self-leakage):
    with one point per cluster forced into its own fold, the OOF neighbor mean should NOT equal that
    row's own exact target value (which a leaking implementation would trivially recover as its 1-NN)."""
    rng = np.random.default_rng(2)
    n_clusters = 8
    cluster_means = rng.uniform(10, 100, n_clusters)
    # 1 point per cluster -- kNN must reach across clusters (no same-cluster neighbors available in-fold).
    centers = rng.normal(scale=5, size=(n_clusters, 2))
    X = centers + rng.normal(scale=0.01, size=(n_clusters, 2))
    y = cluster_means.copy()

    splitter = KFold(n_splits=n_clusters, shuffle=True, random_state=2)
    feats = compute_neighbor_aggregate_features(X, {"y": y}, X_query=None, splitter=splitter, seed=2, k_values=(1,), stats=("mean",))
    nbr_pred = feats["nbr_y_k1_mean"].to_numpy()
    assert not np.allclose(nbr_pred, y)
