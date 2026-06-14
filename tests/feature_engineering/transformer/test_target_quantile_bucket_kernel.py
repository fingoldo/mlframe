"""Regression sensors for the fused ``_bucket_sums_counts`` centroid kernel in target_quantile.

The kernel replaces the K-pass ``(y>=lo)&(y<hi)`` boolean-mask + ``X_pool[mask].mean`` fancy-index loop with a single ``searchsorted``-bucketed prange accumulation
(per-thread float64 sums + counts). These sensors pin (a) the kernel exists and is importable (the symbol is the sensor that fails on pre-fix code), (b) bucket
assignment partitions every row exactly once with the same half-open membership the old loop used, and (c) the resulting centroids equal a float64 reference exactly.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_bucket_kernel_symbol_exists():
    # Fails on pre-fix code: the fused kernel did not exist (ImportError), proving this sensor catches a missing optimization.
    from mlframe.feature_engineering.transformer.target_quantile import _bucket_sums_counts

    assert callable(_bucket_sums_counts)


def test_bucket_assignment_partitions_all_rows_and_matches_float64_reference():
    from mlframe.feature_engineering.transformer.target_quantile import _bucket_sums_counts

    n, d, K = 200_000, 6, 10
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    edges = np.quantile(y, np.linspace(0.0, 1.0, K + 1))
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    sums, counts = _bucket_sums_counts(np.ascontiguousarray(X), np.ascontiguousarray(y), edges.astype(np.float64))

    # Every row lands in exactly one bucket -> counts sum to n.
    assert int(counts.sum()) == n

    # Reference centroids via the exact half-open membership the old loop used, accumulated in float64.
    ref = np.zeros((K, d), dtype=np.float64)
    ref_counts = np.zeros(K, dtype=np.int64)
    for b in range(K):
        lo, hi = edges[b], edges[b + 1]
        mask = (y >= lo) & (y <= hi) if b == K - 1 else (y >= lo) & (y < hi)
        ref_counts[b] = int(mask.sum())
        ref[b] = X[mask].astype(np.float64).mean(axis=0)

    np.testing.assert_array_equal(counts, ref_counts)
    new = sums / counts[:, None]
    # Fused kernel accumulates in the same float64 order as the reference -> bit-exact.
    assert np.abs(new - ref).max() == 0.0


def test_centroids_used_by_mode_b_are_finite_and_shaped():
    from mlframe.feature_engineering.transformer.target_quantile import compute_target_quantile_attention

    rng = np.random.default_rng(11)
    X = rng.standard_normal((20_000, 5)).astype(np.float32)
    y = rng.standard_normal(20_000).astype(np.float32)
    Xq = rng.standard_normal((1_000, 5)).astype(np.float32)
    out = compute_target_quantile_attention(X, y, Xq, None, seed=0, n_quantiles=10, standardize=False)
    assert out.shape == (1_000, 10)
    assert np.isfinite(out.to_numpy()).all()
