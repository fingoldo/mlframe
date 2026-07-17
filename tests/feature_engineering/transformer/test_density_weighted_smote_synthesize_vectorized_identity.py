"""Identity-pin for the density_weighted_smote synthesize vectorization.

``_density_weighted_smote_synthesize`` originally pre-drew the inverse-density-weighted ``src_indices`` (one
``rng.choice``) and then per iteration drew the in-cluster neighbour (``rng.integers``) and the convex weight
(``rng.random``) AND wrote one interpolated output row. The optimization keeps the per-iteration PCG64 draw order
intact (so the synthetic cloud is unchanged) but hoists the gather + convex-interpolation out into a single
vectorized pass (~1.4x faster on the synthesize step; see
``feature_engineering/_benchmarks/bench_density_weighted_smote_synthesize_vectorized.py``).

The draw order is load-bearing and ``alpha`` is stored as float32 to mirror the legacy per-row float32
store-rounding (``out[i] = ...`` into a float32 array rounds the lerp in float32); keeping alpha in float64 would
diverge by a single ULP. This test pins bit-identity against a self-contained reference re-implementing the OLD
row-loop, including the ``candidates.size == 0`` no-draw branch.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mlframe.feature_engineering.transformer.density_weighted_smote import _density_weighted_smote_synthesize


def _old_reference(X_minority, n_synthetic, k_neighbors, seed):
    """Helper: Old reference."""
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, X_minority.shape[1] if n_min > 0 else 1), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors

    k_used = min(k_neighbors + 1, n_min)
    nn = NearestNeighbors(n_neighbors=k_used).fit(X_minority)
    dists, ids = nn.kneighbors(X_minority)
    mean_knn_dist = dists[:, 1:].mean(axis=1) + 1e-9
    weights = mean_knn_dist / mean_knn_dist.sum()
    rng = np.random.default_rng(seed)
    src_indices = rng.choice(n_min, size=n_synthetic, p=weights)
    out = np.zeros((n_synthetic, X_minority.shape[1]), dtype=np.float32)
    for i in range(n_synthetic):
        src_idx = src_indices[i]
        candidates = ids[src_idx, 1:k_used]
        if candidates.size == 0:
            out[i] = X_minority[src_idx]
            continue
        nbr_idx = candidates[rng.integers(0, candidates.size)]
        alpha = rng.random()
        out[i] = X_minority[src_idx] + alpha * (X_minority[nbr_idx] - X_minority[src_idx])
    return out


def _make(nrows, d, seed):
    """Helper: Make."""
    return np.random.default_rng(seed + 99).standard_normal((nrows, d)).astype(np.float32)


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize(
    "nrows,d,k_neighbors,n_syn",
    [(500, 30, 5, 5000), (50, 8, 5, 400), (2, 4, 5, 20), (200, 12, 10, 2000)],
)
def test_density_weighted_synthesize_bit_identical_to_row_loop(seed, nrows, d, k_neighbors, n_syn):
    """Density weighted synthesize bit identical to row loop."""
    X_min = _make(nrows, d, seed)
    expected = _old_reference(X_min, n_syn, k_neighbors, seed)
    got = _density_weighted_smote_synthesize(X_min, n_syn, k_neighbors, seed)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("seed", [0, 3])
def test_density_weighted_synthesize_tiny_minority_early_return(seed):
    # n_min < 2 short-circuits before any draw; pin the early-return shape/content.
    """Density weighted synthesize tiny minority early return."""
    X_min = _make(1, 6, seed)
    got = _density_weighted_smote_synthesize(X_min, 50, 5, seed)
    assert np.array_equal(got, X_min)


def test_density_weighted_synthesize_empty_minority():
    """Density weighted synthesize empty minority."""
    X_min = np.zeros((0, 6), dtype=np.float32)
    got = _density_weighted_smote_synthesize(X_min, 50, 5, 0)
    assert got.shape[0] == 0
