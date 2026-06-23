"""Identity-pin for the adasyn_smote ADASYN-synthesize vectorization.

``_adasyn_synthesize`` originally pre-drew the weighted ``src_indices`` (one ``rng.choice``) and then per
iteration drew the in-cluster neighbour (``rng.integers``) and the convex weight (``rng.random``) AND wrote one
interpolated output row. The optimization keeps the per-iteration PCG64 draw order intact (so the synthetic cloud
is unchanged) but hoists the gather + convex-interpolation out into a single vectorized pass (~1.5x faster on the
synthesize step; see ``feature_engineering/_benchmarks/bench_adasyn_smote_synthesize_vectorized.py``).

The draw order is load-bearing and ``alpha`` is stored as float32 to mirror the legacy per-row float32
store-rounding (``out[i] = ...`` into a float32 array rounds the lerp in float32); keeping alpha in float64 would
diverge by a single ULP. This test pins bit-identity against a self-contained reference re-implementing the OLD
row-loop, including the ``candidates.size == 0`` no-draw branch.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from mlframe.feature_engineering.transformer.adasyn_smote import _adasyn_synthesize


def _old_reference(X_minority, X_full, y_binary_full, n_synthetic, k_smote, k_global, seed):
    n_min = X_minority.shape[0]
    if n_min < 2:
        return X_minority.copy() if n_min > 0 else np.zeros((0, X_minority.shape[1] if n_min > 0 else 1), dtype=np.float32)
    from sklearn.neighbors import NearestNeighbors

    nn_full = NearestNeighbors(n_neighbors=k_global + 1).fit(X_full)
    _d_full, ids_full = nn_full.kneighbors(X_minority)
    neg_fraction = (y_binary_full[ids_full[:, 1:]] <= 0.5).mean(axis=1)
    weights = neg_fraction + 1e-6
    weights = weights / weights.sum()
    nn_pos = NearestNeighbors(n_neighbors=min(k_smote + 1, n_min)).fit(X_minority)
    _d_pos, ids_pos = nn_pos.kneighbors(X_minority)
    rng = np.random.default_rng(seed)
    src_indices = rng.choice(n_min, size=n_synthetic, p=weights)
    out = np.zeros((n_synthetic, X_minority.shape[1]), dtype=np.float32)
    for i in range(n_synthetic):
        src_idx = src_indices[i]
        candidates = ids_pos[src_idx, 1 : min(k_smote + 1, n_min)]
        if candidates.size == 0:
            out[i] = X_minority[src_idx]
            continue
        nbr_idx = candidates[rng.integers(0, candidates.size)]
        alpha = rng.random()
        out[i] = X_minority[src_idx] + alpha * (X_minority[nbr_idx] - X_minority[src_idx])
    return out


def _make(nrows, d, seed, n_full=None):
    rng = np.random.default_rng(seed + 99)
    X_min = rng.standard_normal((nrows, d)).astype(np.float32)
    n_full = n_full if n_full is not None else nrows * 3
    X_full = rng.standard_normal((n_full, d)).astype(np.float32)
    y_full = (rng.random(n_full) < 0.5).astype(np.float32)
    return X_min, X_full, y_full


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize(
    "nrows,d,k_smote,k_global,n_syn",
    [(500, 30, 5, 10, 5000), (50, 8, 5, 10, 400), (3, 4, 5, 5, 20), (200, 12, 10, 15, 2000)],
)
def test_adasyn_synthesize_bit_identical_to_row_loop(seed, nrows, d, k_smote, k_global, n_syn):
    X_min, X_full, y_full = _make(nrows, d, seed)
    expected = _old_reference(X_min, X_full, y_full, n_syn, k_smote, k_global, seed)
    got = _adasyn_synthesize(X_min, X_full, y_full, n_syn, k_smote, k_global, seed)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


@pytest.mark.parametrize("seed", [0, 3])
def test_adasyn_synthesize_tiny_minority_early_return(seed):
    # n_min < 2 short-circuits before any draw; pin the early-return shape/content.
    X_min, X_full, y_full = _make(1, 6, seed)
    got = _adasyn_synthesize(X_min, X_full, y_full, 50, 5, 10, seed)
    assert np.array_equal(got, X_min)


def test_adasyn_synthesize_empty_minority():
    X_full = np.random.default_rng(0).standard_normal((30, 6)).astype(np.float32)
    y_full = (np.random.default_rng(0).random(30) < 0.5).astype(np.float32)
    X_min = np.zeros((0, 6), dtype=np.float32)
    got = _adasyn_synthesize(X_min, X_full, y_full, 50, 5, 10, 0)
    assert got.shape[0] == 0
